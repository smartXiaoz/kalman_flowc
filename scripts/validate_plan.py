#!/usr/bin/env python3
"""Validate an untrusted GPT proposal for the FlowCritic tuning loop.

This module deliberately has no trainer or launcher imports.  Validation turns a
GPT response into another *proposal* (``next_plan.json``); materializing configs
and launching jobs are separate, trusted steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).with_name("next_plan.schema.json")
SCHEMA_VERSION = "1.0"

ENVIRONMENTS = (
    "FrankaCubeStack",
    "Quadcopter",
    "FrankaCabinet",
    "BallBalance",
    "ShadowHandOpenAI_FF",
    "Anymal",
    "Ant",
    "AllegroHand",
    "Humanoid",
    "AllegroKuka",
    "AllegroKukaTwoArmsLSTM",
    "ShadowHand",
)
SEEDS = (1, 2, 3, 4)

# These values are part of the paper contract.  GPT output is never permitted
# to override them.  minibatch_size is the resolved value (1024*16/2).
FIXED_PARAMETERS: dict[str, Any] = {
    "num_envs": 1024,
    "num_steps": 16,
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "num_minibatches": 2,
    "minibatch_size": 8192,
    "update_epochs": 4,
    "norm_adv": True,
    "clip_coef": 0.2,
    "max_grad_norm": 1.0,
    "k_repeat": 10,
    "n_remove": 1,
    "diff_step": 5,
    "exp_alpha": 0.1,
    "velocity_clip_delta": 0.2,
}

# These are protected even though they are not all scalar entries in the paper
# table.  They either select the core algorithm or identify/schedule the run.
PROTECTED_PARAMETERS = frozenset(FIXED_PARAMETERS) | {
    "rollout_mean_mode",
    "mean_mode",
    "model_L",
    "hidden_size",
    "optimizer",
    "adam_eps",
    "lr",
    "seed",
    "env",
    "env_id",
    "total_timesteps",
    "batch_size",
    "num_iterations",
}
DEAD_OR_UNUSED_PARAMETERS = frozenset(
    {"predict_epsilon", "noise_ratio", "vloss_coef", "weight"}
)

STAGE_1_PARAMETERS = frozenset(
    {
        "prior_std",
        "target_norm",
        "cov_floor",
        "reward_scaler",
        "acf_coef",
        "vf_coef",
        "flow_sigma",
    }
)
STAGE_2_PARAMETERS = frozenset(
    {
        "target_norm_update",
        "target_norm_ema_tau",
        "cov_weight_clip",
        "ess_floor",
        "alpha_warmup_updates",
    }
)
STAGE_3_PARAMETERS = frozenset(
    {
        "old_critic_update",
        "old_critic_ema_tau",
        "bootstrap_mean_mode",
    }
)
ALL_TUNABLE_PARAMETERS = STAGE_1_PARAMETERS | STAGE_2_PARAMETERS | STAGE_3_PARAMETERS

# JSON-serializable public contract.  The semantic validator below is the
# authority and additionally checks finite numbers and ordered clip bounds.
PARAMETER_SPECS: dict[str, dict[str, Any]] = {
    "prior_std": {"type": "number", "minimum": 0.1, "maximum": 4.0},
    "target_norm": {"type": "enum", "values": ["none", "running_return"]},
    "cov_floor": {"type": "number", "minimum": 0.0, "maximum": 10.0},
    "reward_scaler": {"type": "number", "exclusive_minimum": 0.0, "maximum": 10.0},
    "acf_coef": {"type": "number", "exclusive_minimum": 0.0, "maximum": 10.0},
    "vf_coef": {"type": "number", "exclusive_minimum": 0.0, "maximum": 10.0},
    "flow_sigma": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "target_norm_update": {
        "type": "enum",
        "values": ["cumulative", "running", "ema"],
    },
    "target_norm_ema_tau": {"type": "number", "minimum": 0.0, "maximum": 0.9999},
    "cov_weight_clip": {
        "type": "null_or_ordered_pair",
        "item_minimum": 0.0,
        "item_maximum": 1.0,
    },
    "ess_floor": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "alpha_warmup_updates": {"type": "integer", "minimum": 0, "maximum": 100000},
    "old_critic_update": {"type": "enum", "values": ["hard", "ema"]},
    "old_critic_ema_tau": {"type": "number", "minimum": 0.0, "maximum": 0.99999},
    "bootstrap_mean_mode": {
        "type": "enum",
        "values": ["normal", "remove_max", "remove_max_min"],
    },
}

SEARCH_MAX_DECIMAL_PLACES = 3
SEARCH_GRID = Decimal("0.005")
SEARCH_PRECISION_EXEMPT_PARAMETERS = frozenset({"reward_scaler"})
FIXED_PRECISION_EXCEPTIONS = {
    "learning_rate": 5e-4,
    "adam_eps": 1e-5,
}
PARAMETER_FAMILIES: dict[str, frozenset[str]] = {
    "distribution": frozenset({"prior_std", "cov_floor", "flow_sigma"}),
    "normalization": frozenset(
        {"target_norm", "target_norm_update", "target_norm_ema_tau"}
    ),
    "weight_stability": frozenset(
        {"cov_weight_clip", "ess_floor", "alpha_warmup_updates"}
    ),
    "loss_balance": frozenset({"reward_scaler", "acf_coef", "vf_coef"}),
    "critic_update": frozenset({"old_critic_update", "old_critic_ema_tau"}),
    "bootstrap": frozenset({"bootstrap_mean_mode"}),
}
PARAMETER_TO_FAMILY = {
    parameter: family
    for family, parameters in PARAMETER_FAMILIES.items()
    for parameter in parameters
}

TOP_LEVEL_KEYS = frozenset({"status", "target_envs", "reasoning", "jobs"})
JOB_KEYS = frozenset({"env", "seed", "variant", "overrides", "hypothesis", "evidence"})
VARIANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class PlanValidationError(ValueError):
    """Raised when a GPT plan violates one or more local safety contracts."""

    def __init__(self, errors: str | Sequence[str]):
        if isinstance(errors, str):
            errors = [errors]
        self.errors = [str(error) for error in errors]
        super().__init__("; ".join(self.errors))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def allowed_parameters(stage: int) -> frozenset[str]:
    if stage == 1:
        return STAGE_1_PARAMETERS
    if stage == 2:
        return STAGE_1_PARAMETERS | STAGE_2_PARAMETERS
    if stage == 3:
        return ALL_TUNABLE_PARAMETERS
    raise PlanValidationError(f"stage must be 1, 2, or 3; got {stage!r}")


def contract_snapshot(stage: int) -> dict[str, Any]:
    allowed = allowed_parameters(stage)
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "environments": list(ENVIRONMENTS),
        "seeds": list(SEEDS),
        "fixed_parameters": dict(FIXED_PARAMETERS),
        "protected_parameters": sorted(PROTECTED_PARAMETERS),
        "dead_or_unused_parameters": sorted(DEAD_OR_UNUSED_PARAMETERS),
        "allowed_parameters": sorted(allowed),
        "parameter_specs": {key: PARAMETER_SPECS[key] for key in sorted(allowed)},
        "search_precision_policy": {
            "max_decimal_places": SEARCH_MAX_DECIMAL_PLACES,
            "required_increment": float(SEARCH_GRID),
            "preferred_increments": [0.01, float(SEARCH_GRID)],
            "exempt_parameters": sorted(SEARCH_PRECISION_EXEMPT_PARAMETERS),
            "fixed_non_search_exceptions": dict(FIXED_PRECISION_EXCEPTIONS),
        },
        "parameter_families": {
            family: sorted(parameters)
            for family, parameters in sorted(PARAMETER_FAMILIES.items())
        },
        "max_variants_per_environment": 2,
        "max_jobs_per_round": 48,
        "required_seeds_per_variant": list(SEEDS),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def normalize_fixed_contract(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    normalized: dict[str, Any] = {}
    for key, expected in FIXED_PARAMETERS.items():
        candidate = raw.get(key, expected)
        if isinstance(expected, bool):
            normalized[key] = bool(candidate) if isinstance(candidate, bool) else candidate
        elif isinstance(expected, int) and not isinstance(expected, bool):
            normalized[key] = int(candidate) if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) else candidate
        elif isinstance(expected, float):
            normalized[key] = float(candidate) if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) else candidate
        else:
            normalized[key] = candidate
    return {key: normalized[key] for key in sorted(normalized)}


def resolved_contract_signature(
    tunables: Mapping[str, Any], fixed_contract: Any = None
) -> str:
    return _canonical_json(
        {
            "fixed_contract": normalize_fixed_contract(fixed_contract),
            "tunables": canonicalize_behavioral_tunables(tunables),
        }
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _strict_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanValidationError(f"{path} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise PlanValidationError(f"{path} must be finite")
    return 0.0 if result == 0.0 else result


def _decimal_number(value: Any, path: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise PlanValidationError(f"{path} must be a finite decimal number") from exc


def _validate_search_number_precision(key: str, value: float, path: str) -> None:
    if key in SEARCH_PRECISION_EXEMPT_PARAMETERS:
        return
    number = _decimal_number(value, path)
    if number.as_tuple().exponent < -SEARCH_MAX_DECIMAL_PLACES:
        raise PlanValidationError(
            f"{path}={value!r} exceeds {SEARCH_MAX_DECIMAL_PLACES} decimal places"
        )
    if number % SEARCH_GRID != 0:
        raise PlanValidationError(
            f"{path}={value!r} must lie on the {SEARCH_GRID} search grid "
            "(prefer 0.01 steps; reward_scaler is exempt)"
        )


def quantize_tunable_value(key: str, value: Any) -> Any:
    """Return the nearest policy-compliant legacy migration value."""

    if value is None or key in SEARCH_PRECISION_EXEMPT_PARAMETERS:
        return value
    spec = PARAMETER_SPECS.get(key)
    if not spec:
        return value
    kind = spec["type"]
    if kind == "number":
        number = _decimal_number(value, key)
        units = (number / SEARCH_GRID).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return float((units * SEARCH_GRID).quantize(Decimal("0.001")))
    if kind == "null_or_ordered_pair" and isinstance(value, list):
        return [
            None
            if item is None
            else float(
                (
                    (_decimal_number(item, key) / SEARCH_GRID).quantize(
                        Decimal("1"), rounding=ROUND_HALF_UP
                    )
                    * SEARCH_GRID
                ).quantize(Decimal("0.001"))
            )
            for item in value
        ]
    return value


def quantize_tunable_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: quantize_tunable_value(key, value)
        for key, value in config.items()
    }


def tunable_precision_violations(
    config: Mapping[str, Any], path: str = "config"
) -> list[str]:
    errors: list[str] = []
    for key in sorted(set(config) & ALL_TUNABLE_PARAMETERS):
        value = config[key]
        spec = PARAMETER_SPECS[key]
        try:
            if spec["type"] == "number" and value is not None:
                _validate_search_number_precision(key, float(value), f"{path}.{key}")
            elif spec["type"] == "null_or_ordered_pair" and isinstance(value, list):
                for index, item in enumerate(value):
                    if item is not None:
                        _validate_search_number_precision(
                            key, float(item), f"{path}.{key}[{index}]"
                        )
        except (PlanValidationError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    return errors


def validate_tunable_config_precision(
    config: Mapping[str, Any], path: str = "config"
) -> None:
    errors = tunable_precision_violations(config, path)
    if errors:
        raise PlanValidationError(errors)


def _bounded_number(
    key: str, value: Any, path: str, *, enforce_precision: bool = True
) -> float:
    number = _strict_number(value, path)
    spec = PARAMETER_SPECS[key]
    lower = spec.get("minimum", spec.get("exclusive_minimum"))
    below = number < lower if "minimum" in spec else number <= lower
    if below or number > spec["maximum"]:
        left = "[" if "minimum" in spec else "("
        raise PlanValidationError(
            f"{path}={number!r} is outside {left}{lower}, {spec['maximum']}]"
        )
    if enforce_precision:
        _validate_search_number_precision(key, number, path)
    return number


def normalize_override_value(
    key: str, value: Any, path: str, *, enforce_precision: bool = True
) -> Any:
    spec = PARAMETER_SPECS[key]
    kind = spec["type"]
    if kind == "number":
        return _bounded_number(key, value, path, enforce_precision=enforce_precision)
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise PlanValidationError(f"{path} must be a JSON integer")
        if value < spec["minimum"] or value > spec["maximum"]:
            raise PlanValidationError(
                f"{path}={value!r} is outside [{spec['minimum']}, {spec['maximum']}]"
            )
        return value
    if kind == "enum":
        if not isinstance(value, str) or value not in spec["values"]:
            raise PlanValidationError(f"{path} must be one of {spec['values']}; got {value!r}")
        if key == "target_norm_update" and value == "running":
            return "cumulative"
        return value
    if kind == "null_or_ordered_pair":
        if value is None:
            return None
        if not isinstance(value, list) or len(value) != 2:
            raise PlanValidationError(f"{path} must be null or a two-number JSON array")
        low = None if value[0] is None else _strict_number(value[0], f"{path}[0]")
        high = None if value[1] is None else _strict_number(value[1], f"{path}[1]")
        if enforce_precision:
            if low is not None:
                _validate_search_number_precision(key, low, f"{path}[0]")
            if high is not None:
                _validate_search_number_precision(key, high, f"{path}[1]")
        numeric = [item for item in (low, high) if item is not None]
        if any(item < spec["item_minimum"] or item > spec["item_maximum"] for item in numeric):
            raise PlanValidationError(
                f"{path} values must lie in [{spec['item_minimum']}, {spec['item_maximum']}]"
            )
        if low is not None and high is not None and low > high:
            raise PlanValidationError(f"{path} lower bound must not exceed upper bound")
        if low is None and high is None:
            return None
        return [low, high]
    raise AssertionError(f"unknown parameter spec type for {key}: {kind}")


def _normalize_overrides(overrides: Any, stage: int, path: str) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise PlanValidationError(f"{path} must be a JSON object")
    allowed = allowed_parameters(stage)
    normalized: dict[str, Any] = {}
    for key, value in overrides.items():
        if not isinstance(key, str):
            raise PlanValidationError(f"{path} contains a non-string key")
        # Strict Responses schemas require every declared property.  The GPT
        # envelope therefore uses JSON null as an "unchanged" placeholder;
        # approved local plans remain sparse delta overrides.
        if value is None:
            continue
        if key in PROTECTED_PARAMETERS:
            raise PlanValidationError(f"{path}.{key} is fixed/protected and cannot be overridden")
        if key in DEAD_OR_UNUSED_PARAMETERS:
            raise PlanValidationError(f"{path}.{key} is dead or unused and cannot be searched")
        if key not in ALL_TUNABLE_PARAMETERS:
            raise PlanValidationError(f"{path}.{key} is not a supported tuning parameter")
        if key not in allowed:
            raise PlanValidationError(f"{path}.{key} is not open in stage {stage}")
        normalized[key] = normalize_override_value(key, value, f"{path}.{key}")
    return {key: normalized[key] for key in sorted(normalized)}


def _normalize_any_tunables(config: Mapping[str, Any], path: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(set(config) & ALL_TUNABLE_PARAMETERS):
        normalized[key] = normalize_override_value(
            key,
            config[key],
            f"{path}.{key}",
            enforce_precision=False,
        )
    return normalized


def canonicalize_behavioral_tunables(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove values that cannot affect behavior under the resolved switches."""

    canonical = dict(config)
    if canonical.get("target_norm") == "none":
        canonical.pop("target_norm_update", None)
        canonical.pop("target_norm_ema_tau", None)
    elif canonical.get("target_norm_update") != "ema":
        canonical.pop("target_norm_ema_tau", None)
    if canonical.get("old_critic_update") != "ema":
        canonical.pop("old_critic_ema_tau", None)
    return {key: canonical[key] for key in sorted(canonical)}


def validate_resolved_override_activity(
    overrides: Mapping[str, Any], resolved: Mapping[str, Any], path: str
) -> None:
    """Reject search dimensions that are inactive in the resolved configuration."""

    target_keys = {"target_norm_update", "target_norm_ema_tau"}
    if set(overrides) & target_keys and resolved.get("target_norm") == "none":
        raise PlanValidationError(
            f"{path} changes target-normalization controls while target_norm=none"
        )
    if (
        "target_norm_ema_tau" in overrides
        and resolved.get("target_norm_update") != "ema"
    ):
        raise PlanValidationError(
            f"{path}.target_norm_ema_tau is inactive unless target_norm_update=ema"
        )
    if (
        "old_critic_ema_tau" in overrides
        and resolved.get("old_critic_update") != "ema"
    ):
        raise PlanValidationError(
            f"{path}.old_critic_ema_tau is inactive unless old_critic_update=ema"
        )


def _changed_parameters(
    base: Mapping[str, Any], resolved: Mapping[str, Any], overrides: Mapping[str, Any]
) -> set[str]:
    return {
        key
        for key in overrides
        if _canonical_json(base.get(key)) != _canonical_json(resolved.get(key))
    }


def validate_search_diversification(
    *,
    env: str,
    base: Mapping[str, Any],
    resolved: Mapping[str, Any],
    overrides: Mapping[str, Any],
    search_constraints: Any,
    path: str,
) -> None:
    if not isinstance(search_constraints, Mapping):
        return
    policies = search_constraints.get("diversification")
    if not isinstance(policies, Mapping):
        return
    policy = policies.get(env)
    if not isinstance(policy, Mapping) or not bool(policy.get("required")):
        return
    cooldown_raw = policy.get("cooldown_families", [])
    if not isinstance(cooldown_raw, list):
        raise PlanValidationError(f"search_constraints.diversification.{env} is malformed")
    cooldown = {str(item) for item in cooldown_raw}
    invalid = sorted(cooldown - set(PARAMETER_FAMILIES))
    if invalid:
        raise PlanValidationError(
            f"search_constraints.diversification.{env} has unknown families: {invalid}"
        )
    changed = _changed_parameters(base, resolved, overrides)
    changed_families = {
        PARAMETER_TO_FAMILY[key]
        for key in changed
        if key in PARAMETER_TO_FAMILY
    }
    if not changed_families - cooldown:
        raise PlanValidationError(
            f"{path} must change at least one parameter outside cooled families "
            f"{sorted(cooldown)}; actual changed parameters={sorted(changed)}"
        )


def _base_config_map(base_configs: Any) -> dict[str, dict[str, Any]]:
    """Accept either {env: config}, {configs: {env: config}}, or record lists."""

    if base_configs is None:
        return {}
    payload = base_configs
    if isinstance(payload, dict) and isinstance(payload.get("configs"), dict):
        payload = payload["configs"]
    result: dict[str, dict[str, Any]] = {}
    if isinstance(payload, dict):
        for env, config in payload.items():
            if env in ENVIRONMENTS and isinstance(config, dict):
                result[env] = _normalize_any_tunables(config, f"base_configs.{env}")
        return result
    if isinstance(payload, list):
        for index, record in enumerate(payload):
            if not isinstance(record, dict) or record.get("env") not in ENVIRONMENTS:
                continue
            config = record.get("tunable_config", record.get("config", record.get("overrides")))
            if isinstance(config, dict):
                env = record["env"]
                result[env] = _normalize_any_tunables(config, f"base_configs[{index}]")
        return result
    raise PlanValidationError("base_configs must be an object or an array of environment records")


def _resolved_candidate(
    env: str, overrides: Mapping[str, Any], base_map: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    resolved = dict(base_map.get(env, {}))
    resolved.update(overrides)
    return canonicalize_behavioral_tunables(resolved)


def _walk_history_records(
    value: Any,
) -> Iterable[tuple[str, Mapping[str, Any], bool, Any]]:
    """Yield (env, config, is_override_only, fixed_contract) history records."""

    if isinstance(value, list):
        for item in value:
            yield from _walk_history_records(item)
        return
    if not isinstance(value, dict):
        return

    env = value.get("env", value.get("env_id"))
    if env in ENVIRONMENTS:
        for key in ("resolved_tunables", "tunable_config", "resolved_config", "config", "overrides"):
            config = value.get(key)
            if isinstance(config, dict):
                yield env, config, key == "overrides", value.get("fixed_contract")
                break

    for child in value.values():
        if isinstance(child, (dict, list)):
            yield from _walk_history_records(child)


def history_signatures(
    history: Any, base_configs: Any = None
) -> set[tuple[str, str]]:
    base_map = _base_config_map(base_configs)
    signatures: set[tuple[str, str]] = set()
    if history is None:
        return signatures
    for env, raw_config, override_only, fixed_contract in _walk_history_records(history):
        config = _normalize_any_tunables(raw_config, f"history.{env}")
        if override_only and env in base_map:
            merged = dict(base_map[env])
            merged.update(config)
            config = merged
        signatures.add((env, resolved_contract_signature(config, fixed_contract)))
    return signatures


def _require_string(value: Any, path: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{path} must be a non-empty string")
    if len(value) > maximum:
        raise PlanValidationError(f"{path} exceeds {maximum} characters")
    return value


def validate_plan(
    plan: Any,
    *,
    stage: int,
    history: Any = None,
    base_configs: Any = None,
    reject_inactive: bool = True,
    search_constraints: Any = None,
) -> dict[str, Any]:
    """Return a normalized plan or raise :class:`PlanValidationError`.

    All semantic checks are local and deterministic.  No files are written and
    no subprocesses are started by this function.
    """

    allowed_parameters(stage)  # validate before touching the plan
    if not isinstance(plan, dict):
        raise PlanValidationError("plan must be a JSON object")
    missing = sorted(TOP_LEVEL_KEYS - set(plan))
    extra = sorted(set(plan) - TOP_LEVEL_KEYS)
    if missing:
        raise PlanValidationError(f"plan is missing required keys: {missing}")
    if extra:
        raise PlanValidationError(f"plan has unsupported keys: {extra}")

    status = plan["status"]
    if status not in {"continue", "stop", "blocked"}:
        raise PlanValidationError("status must be continue, stop, or blocked")

    target_envs = plan["target_envs"]
    if not isinstance(target_envs, list):
        raise PlanValidationError("target_envs must be a JSON array")
    if len(target_envs) > len(ENVIRONMENTS):
        raise PlanValidationError("target_envs contains more than 12 environments")
    if any(not isinstance(env, str) or env not in ENVIRONMENTS for env in target_envs):
        invalid = [env for env in target_envs if env not in ENVIRONMENTS]
        raise PlanValidationError(f"target_envs contains invalid environments: {invalid}")
    if len(set(target_envs)) != len(target_envs):
        raise PlanValidationError("target_envs must not contain duplicates")

    reasoning = plan["reasoning"]
    if not isinstance(reasoning, list) or not reasoning or len(reasoning) > 64:
        raise PlanValidationError("reasoning must contain 1 to 64 strings")
    normalized_reasoning = [
        _require_string(item, f"reasoning[{index}]", 4000)
        for index, item in enumerate(reasoning)
    ]

    jobs = plan["jobs"]
    if not isinstance(jobs, list):
        raise PlanValidationError("jobs must be a JSON array")
    if len(jobs) > 48:
        raise PlanValidationError(f"jobs exceeds the 48-job round limit: {len(jobs)}")
    if status == "continue":
        if not jobs:
            raise PlanValidationError("continue plans must contain complete four-seed jobs")
        if not target_envs:
            raise PlanValidationError("continue plans must name at least one target environment")
    elif jobs:
        raise PlanValidationError(f"{status} plans must not schedule jobs")

    normalized_jobs: list[dict[str, Any]] = []
    for index, raw_job in enumerate(jobs):
        path = f"jobs[{index}]"
        if not isinstance(raw_job, dict):
            raise PlanValidationError(f"{path} must be a JSON object")
        missing_job = sorted(JOB_KEYS - set(raw_job))
        extra_job = sorted(set(raw_job) - JOB_KEYS)
        if missing_job:
            raise PlanValidationError(f"{path} is missing required keys: {missing_job}")
        if extra_job:
            raise PlanValidationError(f"{path} has unsupported keys: {extra_job}")

        env = raw_job["env"]
        if not isinstance(env, str) or env not in ENVIRONMENTS:
            raise PlanValidationError(f"{path}.env is not one of the paper's 12 environments")
        seed = raw_job["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
            raise PlanValidationError(f"{path}.seed must be exactly one of {list(SEEDS)}")
        variant = raw_job["variant"]
        if not isinstance(variant, str) or not VARIANT_RE.fullmatch(variant):
            raise PlanValidationError(
                f"{path}.variant must be 1-64 safe characters: letters, digits, '.', '_' or '-'"
            )
        overrides = _normalize_overrides(raw_job["overrides"], stage, f"{path}.overrides")
        normalized_jobs.append(
            {
                "env": env,
                "seed": seed,
                "variant": variant,
                "overrides": overrides,
                "hypothesis": _require_string(raw_job["hypothesis"], f"{path}.hypothesis", 4000),
                "evidence": _require_string(raw_job["evidence"], f"{path}.evidence", 4000),
            }
        )

    if status == "continue":
        job_envs = {job["env"] for job in normalized_jobs}
        if job_envs != set(target_envs):
            raise PlanValidationError(
                "target_envs must exactly match the environments represented by jobs"
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    variants_by_env: dict[str, set[str]] = defaultdict(set)
    for job in normalized_jobs:
        key = (job["env"], job["variant"])
        grouped[key].append(job)
        variants_by_env[job["env"]].add(job["variant"])

    for env, variants in variants_by_env.items():
        if len(variants) > 2:
            raise PlanValidationError(
                f"{env} proposes {len(variants)} variants; at most 2 are allowed per round"
            )

    base_map = _base_config_map(base_configs)
    historical = history_signatures(history, base_configs)
    current_configs_by_env: dict[str, dict[str, str]] = defaultdict(dict)
    for (env, variant), cohort in grouped.items():
        cohort_seeds = [job["seed"] for job in cohort]
        if len(cohort) != 4 or set(cohort_seeds) != set(SEEDS) or len(set(cohort_seeds)) != 4:
            raise PlanValidationError(
                f"{env}/{variant} must contain exactly one job for each seed {list(SEEDS)}; "
                f"got {sorted(cohort_seeds)}"
            )
        reference = {
            key: cohort[0][key]
            for key in ("env", "variant", "overrides", "hypothesis", "evidence")
        }
        for job in cohort[1:]:
            candidate = {
                key: job[key]
                for key in ("env", "variant", "overrides", "hypothesis", "evidence")
            }
            if candidate != reference:
                raise PlanValidationError(
                    f"{env}/{variant} jobs must be identical apart from seed"
                )

        raw_resolved = dict(base_map.get(env, {}))
        raw_resolved.update(cohort[0]["overrides"])
        validate_tunable_config_precision(
            raw_resolved,
            f"{env}/{variant}.resolved_tunables",
        )
        if reject_inactive:
            validate_resolved_override_activity(
                cohort[0]["overrides"],
                raw_resolved,
                f"{env}/{variant}.overrides",
            )
        validate_search_diversification(
            env=env,
            base=base_map.get(env, {}),
            resolved=raw_resolved,
            overrides=cohort[0]["overrides"],
            search_constraints=search_constraints,
            path=f"{env}/{variant}.overrides",
        )
        resolved = canonicalize_behavioral_tunables(raw_resolved)
        signature = resolved_contract_signature(resolved, FIXED_PARAMETERS)
        if signature in current_configs_by_env[env]:
            other = current_configs_by_env[env][signature]
            raise PlanValidationError(
                f"{env}/{variant} duplicates the same-round configuration of {env}/{other}"
            )
        current_configs_by_env[env][signature] = variant
        if (env, signature) in historical:
            raise PlanValidationError(
                f"{env}/{variant} repeats a configuration already present in unique history"
            )

    normalized_jobs.sort(key=lambda job: (ENVIRONMENTS.index(job["env"]), job["variant"], job["seed"]))
    return {
        "status": status,
        "target_envs": list(target_envs),
        "reasoning": normalized_reasoning,
        "jobs": normalized_jobs,
    }


def make_validation_record(
    *,
    stage: int,
    raw_plan: Any,
    valid: bool,
    approved_plan: Any = None,
    errors: Sequence[str] = (),
    history: Any = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "validated_at": utc_now(),
        "valid": bool(valid),
        "raw_plan_sha256": _sha256_json(raw_plan),
        "approved_plan_sha256": _sha256_json(approved_plan) if approved_plan is not None else None,
        "job_count": len(approved_plan.get("jobs", [])) if isinstance(approved_plan, dict) else 0,
        "history_record_count": sum(1 for _ in _walk_history_records(history)),
        "errors": [str(error) for error in errors],
        "contract": contract_snapshot(stage),
    }


def read_json(path: Path) -> Any:
    def reject_non_finite(token: str) -> None:
        raise ValueError(f"non-finite JSON number is not allowed: {token}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_non_finite)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Untrusted GPT JSON output")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--history", type=Path, action="append", default=[])
    parser.add_argument("--base-configs", type=Path)
    parser.add_argument("--output", type=Path, help="Write the normalized proposal here")
    parser.add_argument("--record", type=Path, help="Write an auditable validation record here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    record_path = args.record or args.plan.with_name(f"{args.plan.stem}.validation.json")
    try:
        raw_plan = read_json(args.plan)
        histories = [read_json(path) for path in args.history]
        history: Any = histories if histories else None
        base_configs = read_json(args.base_configs) if args.base_configs else None
        approved = validate_plan(
            raw_plan,
            stage=args.stage,
            history=history,
            base_configs=base_configs,
        )
        record = make_validation_record(
            stage=args.stage,
            raw_plan=raw_plan,
            valid=True,
            approved_plan=approved,
            history=history,
        )
        write_json_atomic(record_path, record)
        if args.output:
            write_json_atomic(args.output, approved)
        else:
            json.dump(approved, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        return 0
    except Exception as exc:
        raw_plan = locals().get("raw_plan", {"unreadable_plan": str(args.plan)})
        history = locals().get("history")
        errors = exc.errors if isinstance(exc, PlanValidationError) else [str(exc)]
        try:
            write_json_atomic(
                record_path,
                make_validation_record(
                    stage=args.stage,
                    raw_plan=raw_plan,
                    valid=False,
                    errors=errors,
                    history=history,
                ),
            )
        except Exception as record_exc:
            print(f"could not write validation record: {record_exc}", file=sys.stderr)
        print(f"plan rejected: {'; '.join(errors)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
