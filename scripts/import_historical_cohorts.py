#!/usr/bin/env python3
"""Import complete, unique historical seed-1--4 FlowCritic cohorts.

The importer is intentionally read-only with respect to the legacy project. It
streams reward logs, reads only the final JSON object from large metrics files,
and writes a compact planner context file below this standalone project.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    from common import FIXED_PARAMETERS
    from score_plot_protocol import (
        BUDGETS,
        DEFAULT_PAPER_TARGETS,
        FixedBudgetBins,
        PAPER_ENVS,
        expected_terminal_step,
        finite_float,
        integer,
    )
    from validate_plan import (
        ALL_TUNABLE_PARAMETERS,
        ENVIRONMENTS,
        canonicalize_behavioral_tunables,
        normalize_override_value,
    )
except ImportError:  # pragma: no cover - package-style imports in external tests
    from scripts.common import FIXED_PARAMETERS  # type: ignore[no-redef]
    from scripts.score_plot_protocol import (  # type: ignore[no-redef]
        BUDGETS,
        DEFAULT_PAPER_TARGETS,
        FixedBudgetBins,
        PAPER_ENVS,
        expected_terminal_step,
        finite_float,
        integer,
    )
    from scripts.validate_plan import (  # type: ignore[no-redef]
        ALL_TUNABLE_PARAMETERS,
        ENVIRONMENTS,
        canonicalize_behavioral_tunables,
        normalize_override_value,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Historical runs are external to this source repository. Pass --source-root
# explicitly when importing a local archive.
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "legacy_experiments"
DEFAULT_OUTPUT = PROJECT_ROOT / "state" / "historical_unique_cohorts.json"
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
EXPECTED_SEEDS = (1, 2, 3, 4)
STEP_COLUMNS = ("step", "global_step", "run/global_step", "timesteps", "timestep")
REWARD_COLUMNS = ("episodic_return", "episode_return", "return", "reward", "episode/return")

# Values supplied by the trainer through getattr when an older resolved config
# predates the corresponding key. Stage-1 reward-scale keys are expected to be
# explicit in every usable resolved config and are not guessed here.
TUNABLE_DEFAULTS: Dict[str, Any] = {
    "prior_std": 1.0,
    "target_norm": "none",
    "cov_floor": 0.0,
    "target_norm_update": "cumulative",
    "target_norm_ema_tau": 0.99,
    "cov_weight_clip": None,
    "ess_floor": 0.0,
    "alpha_warmup_updates": 0,
    "old_critic_update": "hard",
    "old_critic_ema_tau": 0.995,
    "bootstrap_mean_mode": "normal",
}
REQUIRED_EXPLICIT_TUNABLES = frozenset(
    {"reward_scaler", "acf_coef", "vf_coef", "flow_sigma"}
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def extract_fixed_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    contract: Dict[str, Any] = {}
    for key, expected in FIXED_PARAMETERS.items():
        if key not in config:
            contract[key] = "__missing__"
            continue
        value = config[key]
        if isinstance(expected, bool):
            contract[key] = value if isinstance(value, bool) else value
        elif isinstance(expected, int) and not isinstance(expected, bool):
            contract[key] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        elif isinstance(expected, float):
            contract[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        else:
            contract[key] = value
    return {key: contract[key] for key in sorted(contract)}


def historical_config_signature(
    tunables: Mapping[str, Any], fixed_contract: Mapping[str, Any]
) -> str:
    return sha256_payload(
        {
            "fixed_contract": dict(fixed_contract),
            "tunables": canonicalize_behavioral_tunables(tunables),
        }
    )


def load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def resolve_candidate(
    round_dir: Path,
    job: Mapping[str, Any],
    config: Mapping[str, Any],
    keys: Sequence[str],
    fallback: str,
) -> Path:
    candidates = []
    raw_run_dir = job.get("job_run_dir")
    if raw_run_dir:
        candidates.append(Path(str(raw_run_dir)).expanduser() / fallback)
    candidates.append(round_dir / "runs" / str(job.get("job_id", "")) / fallback)
    for key in keys:
        raw = job.get(key, config.get(key))
        if raw:
            candidates.append(Path(str(raw)).expanduser())
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = round_dir / candidate
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else round_dir / fallback


def resolve_config_path(round_dir: Path, job: Mapping[str, Any]) -> Path:
    return resolve_candidate(
        round_dir,
        job,
        {},
        ("config_path", "config_resolved", "resolved_config"),
        "config.resolved.json",
    )


def normalize_tunables(config: Mapping[str, Any], source: str) -> Dict[str, Any]:
    missing = sorted(REQUIRED_EXPLICIT_TUNABLES - set(config))
    if missing:
        raise ValueError(f"{source} is missing resolved tunables: {missing}")
    normalized: Dict[str, Any] = {}
    for key in sorted(ALL_TUNABLE_PARAMETERS):
        if key in config:
            value = config[key]
        elif key in TUNABLE_DEFAULTS:
            value = TUNABLE_DEFAULTS[key]
        else:
            raise ValueError(f"{source} cannot resolve tunable {key!r}")
        if key == "cov_weight_clip" and isinstance(value, dict):
            value = [value.get("min"), value.get("max")]
        # The local validator performs the range/type checks and canonicalizes
        # the legacy alias target_norm_update=running to cumulative.
        normalized[key] = normalize_override_value(key, value, f"{source}.{key}")
    return normalized


def nested_steps(value: Any) -> Iterable[int]:
    if not isinstance(value, dict):
        return
    for key in ("global_step", "run/global_step", "last_step", "step"):
        parsed = integer(value.get(key))
        if parsed is not None:
            yield parsed
    metrics = value.get("metrics")
    if isinstance(metrics, dict):
        yield from nested_steps(metrics)


def last_valid_json_object(path: Path, max_tail_bytes: int = 4 * 1024 * 1024) -> Dict[str, Any]:
    """Read the last valid JSONL row without scanning a large metrics file."""

    if not path.is_file():
        return {}
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            position = size
            data = b""
            while position > 0 and len(data) < max_tail_bytes:
                width = min(64 * 1024, position, max_tail_bytes - len(data))
                position -= width
                handle.seek(position)
                data = handle.read(width) + data
                for raw_line in reversed(data.splitlines()):
                    raw_line = raw_line.strip().strip(b"\x00")
                    if not raw_line:
                        continue
                    try:
                        value = json.loads(raw_line.decode("utf-8", errors="strict"))
                    except (UnicodeError, json.JSONDecodeError):
                        continue
                    if isinstance(value, dict):
                        return value
    except OSError:
        return {}
    return {}


def first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def score_and_hash_rewards(path: Path, budget: int) -> Dict[str, Any]:
    """Hash and score a rewards CSV in one streaming pass."""

    result: Dict[str, Any] = {
        "missing": not path.is_file(),
        "sha256": None,
        "last_reward_step": None,
        "valid_rows": 0,
        "malformed_rows": 0,
        "step_reset_count": 0,
    }
    accumulator = FixedBudgetBins(budget)
    if result["missing"]:
        result.update(accumulator.summary())
        return result

    digest = hashlib.sha256()
    headers: Optional[Tuple[str, ...]] = None
    previous_step: Optional[int] = None
    first_line = True
    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                text = raw_line.decode("utf-8-sig" if first_line else "utf-8", errors="replace")
                first_line = False
                text = text.strip().strip("\x00")
                if not text:
                    continue
                try:
                    fields = next(csv.reader([text]))
                except (csv.Error, StopIteration):
                    result["malformed_rows"] += 1
                    continue
                if headers is None:
                    headers = tuple(field.strip().lower() for field in fields)
                    continue
                if len(fields) != len(headers):
                    result["malformed_rows"] += 1
                    continue
                row = {headers[index]: fields[index] for index in range(len(headers))}
                step = integer(first_present(row, STEP_COLUMNS))
                reward = finite_float(first_present(row, REWARD_COLUMNS))
                if step is None or reward is None:
                    result["malformed_rows"] += 1
                    continue
                if previous_step is not None and step < previous_step:
                    accumulator.reset()
                    result["valid_rows"] = 0
                    result["last_reward_step"] = None
                    result["step_reset_count"] += 1
                accumulator.add(step, reward)
                result["valid_rows"] += 1
                result["last_reward_step"] = step
                previous_step = step
    except OSError as exc:
        result.update({"missing": True, "read_error": str(exc)})
        result.update(accumulator.summary())
        return result

    result["sha256"] = digest.hexdigest()
    result.update(accumulator.summary())
    return result


def file_cache_key(path: Path, extra: Any = None) -> Tuple[Any, ...]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None, extra)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, extra)


@dataclass(frozen=True)
class JobCandidate:
    manifest_path: Path
    round_dir: Path
    manifest: Mapping[str, Any]
    job: Mapping[str, Any]
    config: Mapping[str, Any]
    config_path: Path
    env: str
    variant: str
    seed: int
    tunables: Mapping[str, Any]
    fixed_contract: Mapping[str, Any]
    config_signature: str


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def relative_source(path: Path, source_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(source_root.resolve()))
    except ValueError:
        return str(path.resolve())


def evaluate_job(
    candidate: JobCandidate,
    reward_cache: Dict[Tuple[Any, ...], Dict[str, Any]],
    metrics_cache: Dict[Tuple[Any, ...], Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    budget = BUDGETS[candidate.env]
    terminal_step = expected_terminal_step(budget)
    configured_budget = integer(candidate.config.get("total_timesteps"))
    if configured_budget != budget:
        return None, "configured_budget_mismatch"

    rewards_path = resolve_candidate(
        candidate.round_dir,
        candidate.job,
        candidate.config,
        ("rewards_path", "trainer_reward_path", "reward_path"),
        "rewards.txt",
    )
    reward_key = file_cache_key(rewards_path, budget)
    rewards = reward_cache.get(reward_key)
    if rewards is None:
        rewards = score_and_hash_rewards(rewards_path, budget)
        reward_cache[reward_key] = rewards
    if rewards.get("missing") or not rewards.get("sha256"):
        return None, "missing_rewards"

    metrics_path = resolve_candidate(
        candidate.round_dir,
        candidate.job,
        candidate.config,
        ("metrics_jsonl", "metrics_path"),
        "metrics.jsonl",
    )
    metrics_key = file_cache_key(metrics_path)
    metrics_tail = metrics_cache.get(metrics_key)
    if metrics_tail is None:
        metrics_tail = last_valid_json_object(metrics_path)
        metrics_cache[metrics_key] = metrics_tail

    status_path = resolve_candidate(
        candidate.round_dir,
        candidate.job,
        candidate.config,
        ("status_json", "status_path"),
        "status.json",
    )
    summary_path = resolve_candidate(
        candidate.round_dir,
        candidate.job,
        candidate.config,
        ("summary_json", "summary_path"),
        "summary.json",
    )
    status = load_json_dict(status_path)
    summary = load_json_dict(summary_path)

    metric_steps = list(nested_steps(metrics_tail))
    status_steps = list(nested_steps(status)) + list(nested_steps(summary))
    if metric_steps:
        progress_step = max(metric_steps)
        progress_source = "metrics_tail"
    elif status_steps:
        progress_step = max(status_steps)
        progress_source = "status_or_summary"
    elif rewards.get("last_reward_step") == terminal_step:
        progress_step = terminal_step
        progress_source = "rewards_exact_terminal"
    else:
        progress_step = None
        progress_source = None
    if progress_step != terminal_step:
        return None, "terminal_step_mismatch"

    exit_code = integer(summary.get("exit_code"))
    if exit_code is None:
        exit_code = integer(status.get("exit_code"))
    teardown_warning = (
        candidate.env == "FrankaCubeStack" and exit_code in (-11, 139)
    )
    if exit_code not in (None, 0) and not teardown_warning:
        return None, "nonzero_exit"

    top5_score = finite_float(rewards.get("top5_bin_tail_mean"))
    all_bin_score = finite_float(rewards.get("all_sample_bin_tail_mean"))
    if top5_score is None or all_bin_score is None:
        return None, "missing_tail_score"

    return {
        "seed": candidate.seed,
        "top5_tail_mean": top5_score,
        "all_bin_tail_mean": all_bin_score,
        "reward_sha256": rewards["sha256"],
        "terminal_step": progress_step,
        "progress_source": progress_source,
        "exit_code": exit_code,
        "complete_with_teardown_warning": teardown_warning,
        "reward_rows": int(rewards.get("valid_rows", 0)),
        "reward_step_resets": int(rewards.get("step_reset_count", 0)),
        "reward_malformed_rows": int(rewards.get("malformed_rows", 0)),
    }, None


def observation_sort_key(observation: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    created = str(observation.get("source_created_at") or "9999")
    return (
        created,
        str(observation.get("source_manifest", "")),
        str(observation.get("variant", "")),
        str(observation.get("config_signature", "")),
    )


def make_observation(
    cohort: Sequence[JobCandidate],
    seed_rows: Sequence[Mapping[str, Any]],
    source_root: Path,
) -> Dict[str, Any]:
    first = cohort[0]
    top5_values = [float(row["top5_tail_mean"]) for row in seed_rows]
    all_values = [float(row["all_bin_tail_mean"]) for row in seed_rows]
    paper_mean = float(DEFAULT_PAPER_TARGETS[first.env]["mean"])
    reward_hashes = {str(row["seed"]): str(row["reward_sha256"]) for row in seed_rows}
    terminal_steps = {str(row["seed"]): int(row["terminal_step"]) for row in seed_rows}
    exit_codes = {str(row["seed"]): row.get("exit_code") for row in seed_rows}
    seed_scores = {str(row["seed"]): float(row["top5_tail_mean"]) for row in seed_rows}
    seed_all_scores = {
        str(row["seed"]): float(row["all_bin_tail_mean"]) for row in seed_rows
    }
    return {
        "env": first.env,
        "variant": first.variant,
        "round_id": str(first.manifest.get("round_id") or first.round_dir.name),
        "source_manifest": relative_source(first.manifest_path, source_root),
        "source_created_at": first.manifest.get("created_at"),
        "resolved_tunables": dict(first.tunables),
        "fixed_contract": dict(first.fixed_contract),
        "config_signature": first.config_signature,
        "top5_mean": statistics.fmean(top5_values),
        "top5_std": statistics.pstdev(top5_values),
        "all_bin_mean": statistics.fmean(all_values),
        "all_bin_std": statistics.pstdev(all_values),
        "paper_mean": paper_mean,
        "paper_ratio": statistics.fmean(top5_values) / paper_mean,
        "seed_scores": seed_scores,
        "seed_all_bin_scores": seed_all_scores,
        "reward_sha256": reward_hashes,
        "reward_fingerprint": sha256_payload(reward_hashes),
        "terminal_steps": terminal_steps,
        "exit_codes": exit_codes,
        "teardown_warning_seeds": [
            int(row["seed"])
            for row in seed_rows
            if row.get("complete_with_teardown_warning")
        ],
    }


def deduplicate_observations(
    observations: Sequence[Mapping[str, Any]],
) -> Tuple[list[Dict[str, Any]], Dict[str, int]]:
    dsu = DisjointSet(len(observations))
    config_owner: Dict[Tuple[str, str], int] = {}
    reward_owner: Dict[Tuple[str, str, str], int] = {}
    for index, observation in enumerate(observations):
        config_key = (str(observation["env"]), str(observation["config_signature"]))
        if config_key in config_owner:
            dsu.union(index, config_owner[config_key])
        else:
            config_owner[config_key] = index
        for seed, reward_hash in observation["reward_sha256"].items():
            reward_key = (str(observation["env"]), str(seed), str(reward_hash))
            if reward_key in reward_owner:
                dsu.union(index, reward_owner[reward_key])
            else:
                reward_owner[reward_key] = index

    components: Dict[int, list[int]] = defaultdict(list)
    for index in range(len(observations)):
        components[dsu.find(index)].append(index)

    cohorts: list[Dict[str, Any]] = []
    duplicate_same_config = 0
    duplicate_reward_hash = 0
    reward_hash_conflict_components = 0
    for members in components.values():
        ordered = sorted(members, key=lambda index: observation_sort_key(observations[index]))
        canonical_index = ordered[0]
        canonical = dict(observations[canonical_index])
        canonical["evidence_status"] = "independent_unique_cohort"
        config_counts = Counter(str(observations[index]["config_signature"]) for index in members)
        reward_counts: Counter[Tuple[str, str]] = Counter()
        for index in members:
            for seed, reward_hash in observations[index]["reward_sha256"].items():
                reward_counts[(str(seed), str(reward_hash))] += 1
        if len(config_counts) > 1 and any(count > 1 for count in reward_counts.values()):
            reward_hash_conflict_components += 1

        duplicates = []
        for index in ordered[1:]:
            observation = observations[index]
            reasons = []
            if config_counts[str(observation["config_signature"])] > 1:
                reasons.append("same_resolved_tunables_and_fixed_contract")
                duplicate_same_config += 1
            repeated_seeds = sorted(
                int(seed)
                for seed, reward_hash in observation["reward_sha256"].items()
                if reward_counts[(str(seed), str(reward_hash))] > 1
            )
            if repeated_seeds:
                reasons.append("repeated_reward_sha256")
                duplicate_reward_hash += 1
            if not reasons:
                reasons.append("transitive_duplicate_component")
            duplicates.append(
                {
                    "env": observation["env"],
                    "variant": observation["variant"],
                    "round_id": observation["round_id"],
                    "source_manifest": observation["source_manifest"],
                    "source_created_at": observation.get("source_created_at"),
                    "resolved_tunables": observation["resolved_tunables"],
                    "fixed_contract": observation.get("fixed_contract"),
                    "config_signature": observation["config_signature"],
                    "top5_mean": observation["top5_mean"],
                    "top5_std": observation["top5_std"],
                    "paper_ratio": observation["paper_ratio"],
                    "reward_sha256": observation["reward_sha256"],
                    "reward_fingerprint": observation["reward_fingerprint"],
                    "duplicate_reasons": reasons,
                    "repeated_reward_hash_seeds": repeated_seeds,
                }
            )
        canonical["duplicate_observations"] = duplicates
        cohorts.append(canonical)

    env_order = {env: index for index, env in enumerate(PAPER_ENVS)}
    cohorts.sort(
        key=lambda record: (
            env_order.get(str(record["env"]), len(env_order)),
            str(record["config_signature"]),
        )
    )
    return cohorts, {
        "same_config_duplicate_observations": duplicate_same_config,
        "reward_hash_duplicate_observations": duplicate_reward_hash,
        "reward_hash_conflict_components": reward_hash_conflict_components,
    }


def import_history(source_root: Path) -> Dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"legacy experiment directory not found: {source_root}")

    stats: Counter[str] = Counter()
    excluded_cohorts: Counter[str] = Counter()
    invalid_jobs: Counter[str] = Counter()
    complete_by_env: Counter[str] = Counter()
    groups: Dict[Tuple[str, str, str, str], list[JobCandidate]] = defaultdict(list)

    manifest_paths = sorted(source_root.rglob("manifest.json"))
    stats["manifests_discovered"] = len(manifest_paths)
    for manifest_path in manifest_paths:
        manifest = load_json_dict(manifest_path)
        if not manifest:
            invalid_jobs["unreadable_manifest"] += 1
            continue
        stats["manifests_scanned"] += 1
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list):
            invalid_jobs["manifest_without_jobs"] += 1
            continue
        round_dir = manifest_path.parent
        for raw_job in jobs:
            stats["jobs_seen"] += 1
            if not isinstance(raw_job, dict):
                invalid_jobs["non_object_job"] += 1
                continue
            seed = integer(raw_job.get("seed"))
            if seed not in EXPECTED_SEEDS:
                invalid_jobs["seed_outside_1_4"] += 1
                continue
            config_path = resolve_config_path(round_dir, raw_job)
            config = load_json_dict(config_path)
            if not config:
                invalid_jobs["missing_or_invalid_config"] += 1
                continue
            overrides = raw_job.get("overrides")
            overrides = overrides if isinstance(overrides, dict) else {}
            env = str(
                raw_job.get("env")
                or overrides.get("env_id")
                or config.get("env_id")
                or ""
            ).strip()
            if env not in ENVIRONMENTS:
                invalid_jobs["non_paper_environment"] += 1
                continue
            try:
                tunables = normalize_tunables(config, str(config_path))
            except (TypeError, ValueError) as exc:
                invalid_jobs[f"invalid_tunables:{type(exc).__name__}"] += 1
                continue
            fixed_contract = extract_fixed_contract(config)
            variant = str(raw_job.get("variant") or config.get("variant") or "baseline").strip()
            config_signature = historical_config_signature(tunables, fixed_contract)
            group_key = (str(manifest_path), env, variant, config_signature)
            groups[group_key].append(
                JobCandidate(
                    manifest_path=manifest_path,
                    round_dir=round_dir,
                    manifest=manifest,
                    job=raw_job,
                    config=config,
                    config_path=config_path,
                    env=env,
                    variant=variant,
                    seed=int(seed),
                    tunables=tunables,
                    fixed_contract=fixed_contract,
                    config_signature=config_signature,
                )
            )
            stats["eligible_seed_jobs"] += 1

    reward_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    metrics_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    observations: list[Dict[str, Any]] = []
    stats["manifest_config_groups"] = len(groups)
    for cohort in groups.values():
        seed_counts = Counter(candidate.seed for candidate in cohort)
        if set(seed_counts) != set(EXPECTED_SEEDS):
            excluded_cohorts["missing_seed"] += 1
            continue
        if any(seed_counts[seed] != 1 for seed in EXPECTED_SEEDS):
            excluded_cohorts["duplicate_seed_jobs"] += 1
            continue
        stats["candidate_four_seed_cohorts"] += 1
        cohort = sorted(cohort, key=lambda candidate: candidate.seed)
        seed_rows = []
        rejection_reason = None
        for candidate in cohort:
            row, rejection_reason = evaluate_job(candidate, reward_cache, metrics_cache)
            if rejection_reason:
                break
            assert row is not None
            seed_rows.append(row)
        if rejection_reason:
            excluded_cohorts[rejection_reason] += 1
            continue
        observation = make_observation(cohort, seed_rows, source_root)
        observations.append(observation)
        complete_by_env[observation["env"]] += 1

    cohorts, duplicate_stats = deduplicate_observations(observations)
    unique_by_env = Counter(str(record["env"]) for record in cohorts)
    duplicate_by_env = Counter(
        str(record["env"])
        for record in cohorts
        for _ in record.get("duplicate_observations", [])
    )
    distinct_configs = {
        (str(observation["env"]), str(observation["config_signature"]))
        for observation in observations
    }
    best_by_env = {
        env: max(
            (record for record in cohorts if record["env"] == env),
            key=lambda record: float(record["top5_mean"]),
            default=None,
        )
        for env in PAPER_ENVS
    }
    per_environment = {}
    for env in PAPER_ENVS:
        best = best_by_env[env]
        per_environment[env] = {
            "complete_observations": complete_by_env[env],
            "independent_unique_cohorts": unique_by_env[env],
            "duplicate_observations": duplicate_by_env[env],
            "best_unique_top5_mean": best.get("top5_mean") if best else None,
            "best_unique_top5_std": best.get("top5_std") if best else None,
            "best_unique_paper_ratio": best.get("paper_ratio") if best else None,
            "best_unique_round_id": best.get("round_id") if best else None,
            "best_unique_config_signature": best.get("config_signature") if best else None,
        }

    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_root": str(source_root),
        "selection_contract": {
            "environments": list(PAPER_ENVS),
            "seeds": list(EXPECTED_SEEDS),
            "budgets": dict(BUDGETS),
            "required_terminal_steps": {
                env: expected_terminal_step(BUDGETS[env]) for env in PAPER_ENVS
            },
            "score_protocol": {
                "bin_size": 200_000,
                "bin_boundaries": "(left,right]",
                "top_k": 5,
                "tail_fraction": 0.10,
                "seed_std": "population",
            },
            "tunable_keys": sorted(ALL_TUNABLE_PARAMETERS),
            "target_norm_update_alias": {"running": "cumulative"},
            "fixed_contract_keys": sorted(FIXED_PARAMETERS),
            "cohort_boundary": "same manifest + env + variant + resolved tunables + fixed contract",
            "deduplication": (
                "connected components over identical env/resolved-tunable signatures or "
                "repeated per-env/per-seed reward SHA-256; only the earliest observation "
                "is independent evidence"
            ),
        },
        "stats": {
            **dict(sorted(stats.items())),
            "complete_four_seed_observations": len(observations),
            "distinct_tried_configs": len(distinct_configs),
            "independent_unique_cohorts": len(cohorts),
            "duplicate_observations": len(observations) - len(cohorts),
            "excluded_cohort_reasons": dict(sorted(excluded_cohorts.items())),
            "excluded_job_reasons": dict(sorted(invalid_jobs.items())),
            "per_environment": per_environment,
            **duplicate_stats,
        },
        "unique_config_count": len(cohorts),
        "cohorts": cohorts,
    }


def encode_with_size(payload: Dict[str, Any]) -> bytes:
    stats = payload.setdefault("stats", {})
    stats["serialized_bytes"] = 0
    previous = None
    for _ in range(8):
        data = (canonical_json(payload) + "\n").encode("utf-8")
        size = len(data)
        if size == previous and stats["serialized_bytes"] == size:
            return data
        stats["serialized_bytes"] = size
        previous = size
    return (canonical_json(payload) + "\n").encode("utf-8")


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-output-bytes", type=int, default=MAX_OUTPUT_BYTES)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        payload = import_history(args.source_root)
        data = encode_with_size(payload)
        if len(data) > args.max_output_bytes:
            raise ValueError(
                f"historical context is {len(data)} bytes, above limit {args.max_output_bytes}"
            )
        write_atomic(args.output.expanduser().resolve(), data)
    except Exception as exc:
        print(f"historical import failed: {exc}", file=sys.stderr)
        return 1

    stats = payload["stats"]
    print(f"wrote {args.output.expanduser().resolve()} ({len(data)} bytes)")
    print(
        "manifests={manifests} jobs={jobs} candidates={candidates} complete={complete} "
        "unique={unique} duplicates={duplicates}".format(
            manifests=stats.get("manifests_scanned", 0),
            jobs=stats.get("jobs_seen", 0),
            candidates=stats.get("candidate_four_seed_cohorts", 0),
            complete=stats.get("complete_four_seed_observations", 0),
            unique=stats.get("independent_unique_cohorts", 0),
            duplicates=stats.get("duplicate_observations", 0),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
