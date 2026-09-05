#!/usr/bin/env python3
"""Pure helpers for Stage 4 FlowCritic parameter unification.

This module does not read files, write state, or launch training.  It validates
full 12-environment unification cohorts and evaluates their scored results
against both the paper target and the current per-environment incumbents.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

try:  # Support both ``import scripts.unification`` and direct scripts-path imports.
    from . import validate_plan as _plan_validator
except ImportError:  # pragma: no cover - exercised by the repository test layout.
    import validate_plan as _plan_validator


ENVIRONMENTS = _plan_validator.ENVIRONMENTS
SEEDS = _plan_validator.SEEDS

REWARD_SCALE_PARAMETERS = frozenset(
    {"reward_scaler", "acf_coef", "vf_coef", "flow_sigma"}
)

# Stage 4 attempts to share stability/normalization/critic parameters while
# keeping task reward-scale parameters environment-specific by default.
DEFAULT_UNIFIED_PARAMETERS = (
    "prior_std",
    "target_norm",
    "cov_floor",
    "target_norm_update",
    "target_norm_ema_tau",
    "cov_weight_clip",
    "ess_floor",
    "alpha_warmup_updates",
    "old_critic_update",
    "old_critic_ema_tau",
    "bootstrap_mean_mode",
)


class UnificationValidationError(ValueError):
    """Raised when a Stage 4 plan violates the local unification contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _selected_parameters(parameters: Iterable[str] | None) -> tuple[str, ...]:
    if parameters is None:
        return DEFAULT_UNIFIED_PARAMETERS
    try:
        raw = tuple(parameters)
    except TypeError as exc:
        raise UnificationValidationError("unified_parameters must be an iterable of names") from exc
    if not raw:
        raise UnificationValidationError("unified_parameters must not be empty")
    if any(not isinstance(name, str) for name in raw):
        raise UnificationValidationError("unified_parameters must contain only strings")
    if len(set(raw)) != len(raw):
        raise UnificationValidationError("unified_parameters must not contain duplicates")
    excluded = sorted(set(raw) & REWARD_SCALE_PARAMETERS)
    if excluded:
        raise UnificationValidationError(
            f"reward-scale parameters are environment-specific by default: {excluded}"
        )
    unsupported = sorted(set(raw) - set(DEFAULT_UNIFIED_PARAMETERS))
    if unsupported:
        raise UnificationValidationError(
            f"unsupported Stage 4 unification parameters: {unsupported}"
        )
    selected = set(raw)
    return tuple(name for name in DEFAULT_UNIFIED_PARAMETERS if name in selected)


def validate_unification_plan(
    plan: Any,
    *,
    mode: str,
    unified_parameters: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate and normalize a full Stage 4 global or two-cluster plan.

    A valid plan schedules exactly one four-seed variant for every paper
    environment.  Every override object must contain exactly the selected
    unified parameters.  Global mode permits one shared override object;
    cluster mode requires exactly two distinct shared override objects.
    """

    if mode not in {"global", "cluster"}:
        raise UnificationValidationError("mode must be 'global' or 'cluster'")
    selected = _selected_parameters(unified_parameters)

    try:
        # Stage 4 carries the complete shared contract so a hard/cumulative
        # switch still records its paired EMA value.  Behavioral no-op changes
        # remain rejected in ordinary Stage 1-3 tuning plans.
        normalized = _plan_validator.validate_plan(
            plan,
            stage=3,
            reject_inactive=False,
        )
    except _plan_validator.PlanValidationError as exc:
        raise UnificationValidationError(str(exc)) from exc

    if normalized["status"] != "continue":
        raise UnificationValidationError("Stage 4 unification plans must have status 'continue'")
    if set(normalized["target_envs"]) != set(ENVIRONMENTS):
        missing = sorted(set(ENVIRONMENTS) - set(normalized["target_envs"]))
        extra = sorted(set(normalized["target_envs"]) - set(ENVIRONMENTS))
        raise UnificationValidationError(
            f"Stage 4 plan must target all 12 environments; missing={missing}, extra={extra}"
        )

    jobs_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in normalized["jobs"]:
        jobs_by_env[job["env"]].append(job)

    expected_keys = set(selected)
    signatures: dict[str, list[str]] = defaultdict(list)
    for env in ENVIRONMENTS:
        cohort = jobs_by_env.get(env, [])
        variants = {job["variant"] for job in cohort}
        cohort_seeds = {job["seed"] for job in cohort}
        if len(cohort) != len(SEEDS) or len(variants) != 1 or cohort_seeds != set(SEEDS):
            raise UnificationValidationError(
                f"{env} must contain exactly one variant with seeds {list(SEEDS)}"
            )

        overrides = cohort[0]["overrides"]
        override_keys = set(overrides)
        if override_keys != expected_keys:
            missing = sorted(expected_keys - override_keys)
            extra = sorted(override_keys - expected_keys)
            raise UnificationValidationError(
                f"{env} overrides must completely and exclusively contain the unified "
                f"parameters; missing={missing}, extra={extra}"
            )
        signature = _canonical_json(overrides)
        signatures[signature].append(env)

    expected_group_count = 1 if mode == "global" else 2
    if len(signatures) != expected_group_count:
        qualifier = "one" if mode == "global" else "exactly two"
        raise UnificationValidationError(
            f"{mode} unification requires {qualifier} shared override configuration; "
            f"found {len(signatures)}"
        )

    return normalized


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_ratio(value: Any, name: str) -> float:
    number = _finite_number(value)
    if number is None or number <= 0.0 or number > 1.0:
        raise UnificationValidationError(f"{name} must be a finite number in (0, 1]")
    return number


def _candidate_records(payload: Any) -> tuple[dict[str, list[Mapping[str, Any]]], list[str]]:
    if isinstance(payload, Mapping) and "results" in payload:
        payload = payload["results"]

    records: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    unexpected: list[str] = []
    if isinstance(payload, Mapping):
        for env, value in payload.items():
            if env not in ENVIRONMENTS:
                unexpected.append(str(env))
                continue
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("env", env)
            else:
                record = {"env": env, "top5_mean": value, "complete": False}
            records[env].append(record)
        return records, sorted(set(unexpected))

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            if not isinstance(value, Mapping):
                unexpected.append(f"record[{index}]")
                continue
            env = value.get("env")
            if env not in ENVIRONMENTS:
                unexpected.append(str(env))
                continue
            records[str(env)].append(value)
        return records, sorted(set(unexpected))

    raise UnificationValidationError(
        "candidate_results must be a result list, an environment map, or a scorer payload"
    )


def _environment_mapping(payload: Any, wrapper_key: str, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise UnificationValidationError(f"{name} must be a mapping")
    wrapped = payload.get(wrapper_key)
    if isinstance(wrapped, Mapping):
        return wrapped
    return payload


def _paper_mean(entry: Any) -> float | None:
    if isinstance(entry, Mapping):
        value = entry.get("paper_mean", entry.get("mean"))
    else:
        value = entry
    number = _finite_number(value)
    return number if number is not None and number > 0.0 else None


def _incumbent_score(entry: Any, score_key: str) -> float | None:
    if isinstance(entry, Mapping):
        value = entry.get(score_key, entry.get("top5_mean", entry.get("score")))
    else:
        value = entry
    return _finite_number(value)


def evaluate_unification_candidate(
    candidate_results: Any,
    paper_targets: Mapping[str, Any],
    incumbents: Mapping[str, Any],
    *,
    score_key: str = "top5_mean",
    target_ratio: float = 0.90,
    incumbent_floor_ratio: float = 0.95,
) -> dict[str, Any]:
    """Evaluate a full unification cohort without mutating loop state.

    Acceptance requires every paper environment to have exactly one complete,
    finite score, reach ``target_ratio`` of its paper mean, and retain at least
    ``incumbent_floor_ratio`` of its current incumbent score.
    """

    if not isinstance(score_key, str) or not score_key:
        raise UnificationValidationError("score_key must be a non-empty string")
    target_ratio = _positive_ratio(target_ratio, "target_ratio")
    incumbent_floor_ratio = _positive_ratio(
        incumbent_floor_ratio, "incumbent_floor_ratio"
    )
    candidates, unexpected = _candidate_records(candidate_results)
    target_map = _environment_mapping(paper_targets, "environments", "paper_targets")
    incumbent_map = _environment_mapping(incumbents, "incumbents", "incumbents")

    per_env: dict[str, dict[str, Any]] = {}
    duplicate_environments: list[str] = []
    for env in ENVIRONMENTS:
        rows = candidates.get(env, [])
        issues: list[str] = []
        if not rows:
            issues.append("missing_candidate_result")
            row: Mapping[str, Any] = {}
        elif len(rows) > 1:
            issues.append("duplicate_candidate_results")
            duplicate_environments.append(env)
            row = {}
        else:
            row = rows[0]

        candidate_score = _finite_number(row.get(score_key))
        candidate_complete = row.get("complete") is True
        if not candidate_complete:
            issues.append("candidate_incomplete")
        if candidate_score is None:
            issues.append("candidate_score_missing_or_nonfinite")

        paper_mean = _paper_mean(target_map.get(env))
        if paper_mean is None:
            issues.append("paper_target_missing_or_invalid")
        incumbent_score = _incumbent_score(incumbent_map.get(env), score_key)
        if incumbent_score is None:
            issues.append("incumbent_score_missing_or_nonfinite")

        target_threshold = paper_mean * target_ratio if paper_mean is not None else None
        incumbent_floor = (
            incumbent_score * incumbent_floor_ratio if incumbent_score is not None else None
        )
        complete_gate = bool(len(rows) == 1 and candidate_complete and candidate_score is not None)
        target_gate = bool(
            complete_gate
            and target_threshold is not None
            and candidate_score is not None
            and candidate_score >= target_threshold
        )
        regression_gate = bool(
            complete_gate
            and incumbent_floor is not None
            and candidate_score is not None
            and candidate_score >= incumbent_floor
        )
        target_achievement = (
            candidate_score / paper_mean
            if candidate_score is not None and paper_mean is not None
            else None
        )
        incumbent_retention = (
            candidate_score / incumbent_score
            if candidate_score is not None and incumbent_score not in (None, 0.0)
            else None
        )
        accepted = complete_gate and target_gate and regression_gate
        per_env[env] = {
            "candidate_score": candidate_score,
            "candidate_complete": candidate_complete,
            "complete_gate": complete_gate,
            "paper_mean": paper_mean,
            "target_threshold": target_threshold,
            "target_achievement_ratio": target_achievement,
            "target_gate": target_gate,
            "incumbent_score": incumbent_score,
            "incumbent_floor": incumbent_floor,
            "incumbent_retention_ratio": incumbent_retention,
            "regression_gate": regression_gate,
            "accepted": accepted,
            "issues": list(dict.fromkeys(issues)),
        }

    structurally_complete = not unexpected and not duplicate_environments
    complete_gate = structurally_complete and all(
        row["complete_gate"] for row in per_env.values()
    )
    target_gate = structurally_complete and all(row["target_gate"] for row in per_env.values())
    regression_gate = structurally_complete and all(
        row["regression_gate"] for row in per_env.values()
    )
    return {
        "schema_version": 1,
        "score_key": score_key,
        "target_ratio": target_ratio,
        "incumbent_floor_ratio": incumbent_floor_ratio,
        "environment_count": len(ENVIRONMENTS),
        "unexpected_environments": unexpected,
        "duplicate_environments": duplicate_environments,
        "complete_gate": complete_gate,
        "target_gate": target_gate,
        "regression_gate": regression_gate,
        "accepted": complete_gate and target_gate and regression_gate,
        "per_env": per_env,
    }
