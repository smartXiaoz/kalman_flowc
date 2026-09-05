#!/usr/bin/env python3
"""Materialize one user-approved post-round search without bypassing GPT audit.

The continuous supervisor still calls the normal GPT planner after a round is
scored.  This module is a narrow, one-shot override for the search that the
user approved in advance: it builds a complete four-seed plan from the latest
incumbent configs, validates it with the same local contract as GPT output,
and keeps the GPT proposal alongside the executed plan for auditability.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    from common import STATE_DIR, iso_now, read_json, sha256_payload, write_json_atomic
    from validate_plan import PlanValidationError, validate_plan
except ImportError:  # imported as scripts.approved_task_plan in tests/tools
    from scripts.common import (  # type: ignore[no-redef]
        STATE_DIR,
        iso_now,
        read_json,
        sha256_payload,
        write_json_atomic,
    )
    from scripts.validate_plan import (  # type: ignore[no-redef]
        PlanValidationError,
        validate_plan,
    )


PENDING_TASK_PATH = STATE_DIR / "pending_approved_task.json"


class ApprovedTaskError(RuntimeError):
    """Raised when an armed approved task cannot produce a safe plan."""


def _base_map(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("configs"), Mapping):
        payload = payload["configs"]
    if not isinstance(payload, Mapping):
        raise ApprovedTaskError("current configs must be an environment-keyed object")
    return {
        str(env): dict(config)
        for env, config in payload.items()
        if isinstance(config, Mapping)
    }


def _jobs(
    env: str,
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    variant = str(candidate["variant"])
    overrides = copy.deepcopy(candidate.get("overrides", {}))
    hypothesis = str(candidate["hypothesis"])
    evidence = str(candidate["evidence"])
    return [
        {
            "env": env,
            "seed": seed,
            "variant": variant,
            "overrides": copy.deepcopy(overrides),
            "hypothesis": hypothesis,
            "evidence": evidence,
        }
        for seed in (1, 2, 3, 4)
    ]


def _single_candidate_plan(
    env: str,
    candidate: Mapping[str, Any],
    *,
    history: Any,
    base_configs: Any,
) -> dict[str, Any]:
    raw = {
        "status": "continue",
        "target_envs": [env],
        "reasoning": [
            "This cohort is part of the user-approved post-round parameter search.",
            "Only the listed candidate overrides differ from the current incumbent.",
        ],
        "jobs": _jobs(env, candidate),
    }
    return validate_plan(
        raw,
        stage=3,
        history=history,
        base_configs=base_configs,
        # The approved task intentionally supersedes the planner's recent
        # family-cooldown hint.  Core fixed parameters, precision, activity,
        # complete seed cohorts, and unique-history checks remain enforced.
        search_constraints=None,
    )


def _candidate_is_duplicate_or_invalid(exc: Exception) -> bool:
    # Duplicate history is a normal reason to move to the next approved
    # candidate (not a supervisor failure).  Other validation failures are
    # still retained in the audit and may cause the task to fail closed.
    return "repeats a configuration already present in unique history" in str(exc)


def build_plan(
    task: Mapping[str, Any],
    *,
    history_path: Path,
    base_configs_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and validate the armed task against current post-round state."""

    history = read_json(history_path)
    base_configs = read_json(base_configs_path)
    base_map = _base_map(base_configs)
    target_envs = [str(item) for item in task.get("target_envs", [])]
    candidates = task.get("candidates")
    if not target_envs or not isinstance(candidates, Mapping):
        raise ApprovedTaskError("approved task has no target environments/candidates")

    selected_jobs: list[dict[str, Any]] = []
    selected: dict[str, list[str]] = {}
    rejected: list[dict[str, Any]] = []
    for env in target_envs:
        env_candidates = candidates.get(env)
        if not isinstance(env_candidates, list) or not env_candidates:
            raise ApprovedTaskError(f"approved task has no candidates for {env}")
        accepted_for_env: list[dict[str, Any]] = []
        for candidate in env_candidates:
            if not isinstance(candidate, Mapping):
                rejected.append({"env": env, "reason": "candidate is not an object"})
                continue
            variant_name = str(candidate.get("variant", ""))
            # The tau cohort is a precision migration/revalidation step.  If
            # the latest incumbent already uses the compliant value, do not
            # waste a run on an unchanged override; proceed directly to the
            # approved reward-scaler candidates.
            if (
                env == "AllegroHand"
                and variant_name == "allegro_post114_tau049"
                and str(base_map.get(env, {}).get("old_critic_update")) == "ema"
                and base_map.get(env, {}).get("old_critic_ema_tau") == 0.49
            ):
                rejected.append(
                    {
                        "env": env,
                        "variant": variant_name,
                        "reason": "compliant old_critic_ema_tau=0.49 is already the current incumbent",
                    }
                )
                continue
            try:
                normalized = _single_candidate_plan(
                    env,
                    candidate,
                    history=history,
                    base_configs=base_configs,
                )
            except (PlanValidationError, ValueError) as exc:
                rejected.append(
                    {
                        "env": env,
                        "variant": candidate.get("variant"),
                        "reason": str(exc),
                    }
                )
                if not _candidate_is_duplicate_or_invalid(exc):
                    # A malformed approved candidate must never be silently
                    # replaced by a guessed configuration.
                    raise ApprovedTaskError(
                        f"approved candidate {env}/{candidate.get('variant')} failed validation: {exc}"
                    ) from exc
                continue
            accepted_for_env.append(
                {
                    "variant": str(candidate["variant"]),
                    "jobs": normalized["jobs"],
                }
            )
            # At most two variants per environment are allowed.  Keep the
            # order in the task file deterministic.
            if len(accepted_for_env) == 2:
                break
            # Keep tau migration isolated.  Reward-scaler exploration starts
            # only after the migration cohort has been evaluated (or when it
            # was already present in unique history and was skipped above).
            if env == "AllegroHand" and variant_name == "allegro_post114_tau049":
                break
        if not accepted_for_env:
            raise ApprovedTaskError(
                f"all approved candidates for {env} repeat history or are unavailable"
            )
        selected[env] = [item["variant"] for item in accepted_for_env]
        for item in accepted_for_env:
            selected_jobs.extend(item["jobs"])

    raw_plan = {
        "status": "continue",
        "target_envs": target_envs,
        "reasoning": [
            "Execute the user-approved post-round parameter search after preserving the GPT planner audit.",
            "Each candidate is a complete four-seed cohort and is validated against current incumbents and unique history.",
            "The search keeps gamma, PPO, rollout truncation, and all other paper-fixed parameters unchanged.",
        ],
        "jobs": selected_jobs,
    }
    try:
        normalized_plan = validate_plan(
            raw_plan,
            stage=3,
            history=history,
            base_configs=base_configs,
            search_constraints=None,
        )
    except PlanValidationError as exc:
        raise ApprovedTaskError(f"combined approved plan failed validation: {exc}") from exc
    audit = {
        "schema_version": 1,
        "task_id": task.get("task_id"),
        "source_round_id": task.get("source_round_id"),
        "generated_at": iso_now(),
        "history_path": str(history_path),
        "base_configs_path": str(base_configs_path),
        "selected_variants": selected,
        "rejected_candidates": rejected,
        "plan_sha256": sha256_payload(normalized_plan),
        "plan": normalized_plan,
    }
    return normalized_plan, audit


def apply_pending_task(
    round_dir: Path,
    *,
    current_configs_path: Path,
    history_path: Path,
    gpt_plan: Any,
) -> Optional[dict[str, Any]]:
    """Return an approved plan when this round is the armed task source.

    The function is idempotent.  If the supervisor or host restarts between
    GPT completion and materialization, the already validated plan is reused
    rather than invoking a different search.
    """

    if not PENDING_TASK_PATH.is_file():
        return None
    try:
        task = read_json(PENDING_TASK_PATH)
    except (OSError, json.JSONDecodeError) as exc:
        raise ApprovedTaskError(f"cannot read {PENDING_TASK_PATH}: {exc}") from exc
    if not isinstance(task, Mapping):
        raise ApprovedTaskError("pending approved task must be a JSON object")
    if str(task.get("source_round_id")) != round_dir.name:
        return None

    audit_path = round_dir / "approved_task_plan.json"
    if audit_path.is_file():
        try:
            existing = read_json(audit_path)
            plan = existing.get("plan") if isinstance(existing, Mapping) else None
            if isinstance(plan, Mapping):
                return dict(plan)
        except (OSError, json.JSONDecodeError):
            # Rebuild below; a partial audit must not be treated as executable.
            pass

    plan, audit = build_plan(
        task,
        history_path=history_path,
        base_configs_path=current_configs_path,
    )
    write_json_atomic(round_dir / "gpt_next_plan.json", gpt_plan)
    write_json_atomic(audit_path, audit)
    task_update = dict(task)
    task_update.update(
        {
            "status": "applied",
            "applied_at": iso_now(),
            "applied_round_id": round_dir.name,
            "applied_plan_sha256": audit["plan_sha256"],
        }
    )
    write_json_atomic(PENDING_TASK_PATH, task_update)
    write_json_atomic(round_dir / "next_plan.json", plan)

    # Keep the normal GPT output/status untouched apart from an explicit
    # execution-source marker.  This makes it obvious that GPT analyzed the
    # round while the user-approved plan was the one actually launched.
    status_path = round_dir / "gpt_planner_status.json"
    if status_path.is_file():
        try:
            status = read_json(status_path)
        except (OSError, json.JSONDecodeError):
            status = {}
        if isinstance(status, dict):
            status["execution_plan_source"] = "user_approved_task"
            status["execution_plan_path"] = str(audit_path)
            write_json_atomic(status_path, status)
    return plan
