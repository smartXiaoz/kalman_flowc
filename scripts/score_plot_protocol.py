#!/usr/bin/env python3
"""Score FlowCritic runs with the paper plotting protocol.

The primary score intentionally mirrors the user's plotting code:

* fixed 200,000-transition bins with boundaries ``(left, right]``;
* the mean of the largest five episode returns in a bin (or all returns when
  the bin contains at most five samples);
* the mean of each seed's final 10% fixed-budget bins;
* mean and *population* standard deviation across seeds 1, 2, 3, and 4.

Unlike the plotting script, this analyzer never derives the final window from
the largest observed step.  Anymal and FrankaCabinet use an 80M budget; every
other paper environment uses 100M.  A short run may have a diagnostic score,
but it is never included in the complete four-seed result or the 90% gate.

The all-sample-bin score uses the same bins and tail, but averages every
episode in a bin.  It is diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import heapq
import hashlib
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple


BIN_SIZE = 200_000
TOP_K = 5
TAIL_FRACTION = 0.10
EXPECTED_SEEDS: Tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_BATCH_SIZE = 1024 * 16

PAPER_ENVS: Tuple[str, ...] = (
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

# FlowCritic values in the paper's main performance table. The CLI can load an
# external targets JSON, but these defaults keep the release self-contained.
DEFAULT_PAPER_TARGETS: Dict[str, Dict[str, float]] = {
    "FrankaCubeStack": {"mean": 16.30, "std": 1.63},
    "Quadcopter": {"mean": 1393.56, "std": 6.95},
    "FrankaCabinet": {"mean": 3511.01, "std": 33.21},
    "BallBalance": {"mean": 377.87, "std": 1.38},
    "ShadowHandOpenAI_FF": {"mean": 376.99, "std": 30.59},
    "Anymal": {"mean": 69.63, "std": 0.16},
    "Ant": {"mean": 9345.29, "std": 182.16},
    "AllegroHand": {"mean": 1044.12, "std": 103.53},
    "Humanoid": {"mean": 8863.61, "std": 114.08},
    "AllegroKuka": {"mean": 529.30, "std": 50.44},
    "AllegroKukaTwoArmsLSTM": {"mean": 339.87, "std": 27.00},
    "ShadowHand": {"mean": 1025.76, "std": 427.69},
}

BUDGETS: Dict[str, int] = {
    env: (80_000_000 if env in {"Anymal", "FrankaCabinet"} else 100_000_000)
    for env in PAPER_ENVS
}


def finite_float(value: Any) -> Optional[float]:
    """Return a finite float, or ``None`` for missing/malformed values."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().strip("\x00")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> Optional[int]:
    number = finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def budget_for_env(env: str) -> int:
    """Return the immutable paper reproduction budget for ``env``."""

    if env not in BUDGETS:
        raise ValueError(f"unknown paper environment: {env!r}")
    return BUDGETS[env]


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def expected_terminal_step(budget: int, batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """Last step produced by the trainer's floor-divided iteration loop."""

    if budget <= 0 or batch_size <= 0:
        raise ValueError("budget and batch_size must be positive")
    return (budget // batch_size) * batch_size


def _tail_start(number_of_bins: int, tail_fraction: float = TAIL_FRACTION) -> int:
    if number_of_bins <= 0:
        raise ValueError("number_of_bins must be positive")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    # This is exactly ``int(len(plot_steps) * 0.9)`` for the configured 10%
    # tail.  In general it retains ceil(tail_fraction * n) bins.
    start = int(number_of_bins * (1.0 - tail_fraction))
    return min(start, number_of_bins - 1)


@dataclass
class _Bin:
    count: int = 0
    total: float = 0.0
    top: List[float] = field(default_factory=list)

    def add(self, reward: float, top_k: int) -> None:
        self.count += 1
        self.total += reward
        if len(self.top) < top_k:
            heapq.heappush(self.top, reward)
        elif reward > self.top[0]:
            heapq.heapreplace(self.top, reward)

    @property
    def all_mean(self) -> Optional[float]:
        return self.total / self.count if self.count else None

    @property
    def top_mean(self) -> Optional[float]:
        return sum(self.top) / len(self.top) if self.top else None


class FixedBudgetBins:
    """Streaming fixed-budget bin accumulator."""

    def __init__(self, budget: int, bin_size: int = BIN_SIZE, top_k: int = TOP_K) -> None:
        if budget <= 0 or bin_size <= 0 or top_k <= 0:
            raise ValueError("budget, bin_size, and top_k must be positive")
        self.budget = int(budget)
        self.bin_size = int(bin_size)
        self.top_k = int(top_k)
        self.number_of_bins = math.ceil(self.budget / self.bin_size)
        self.bins = [_Bin() for _ in range(self.number_of_bins)]
        self.rows_in_budget = 0
        self.rows_after_budget = 0
        self.rows_at_or_before_zero = 0

    def add(self, step: int, reward: float) -> None:
        # ``(left, right]`` means step 200000 is in bin 0 and 200001 is in
        # bin 1.  Step zero is outside the first bin.
        if step <= 0:
            self.rows_at_or_before_zero += 1
            return
        if step > self.budget:
            self.rows_after_budget += 1
            return
        index = (step - 1) // self.bin_size
        if index >= self.number_of_bins:
            # Only possible when budget is not a multiple of bin_size and a
            # caller passes inconsistent values; keep the operation safe.
            self.rows_after_budget += 1
            return
        self.bins[index].add(reward, self.top_k)
        self.rows_in_budget += 1

    def reset(self) -> None:
        self.bins = [_Bin() for _ in range(self.number_of_bins)]
        self.rows_in_budget = 0
        self.rows_after_budget = 0
        self.rows_at_or_before_zero = 0

    def summary(self, tail_fraction: float = TAIL_FRACTION) -> Dict[str, Any]:
        top_values = [item.top_mean for item in self.bins]
        all_values = [item.all_mean for item in self.bins]
        counts = [item.count for item in self.bins]
        tail_start = _tail_start(self.number_of_bins, tail_fraction)
        top_tail = [value for value in top_values[tail_start:] if value is not None]
        all_tail = [value for value in all_values[tail_start:] if value is not None]
        return {
            "bin_edges": [min((idx + 1) * self.bin_size, self.budget) for idx in range(self.number_of_bins)],
            "bin_counts": counts,
            "top5_bin_means": top_values,
            "all_sample_bin_means": all_values,
            "number_of_bins": self.number_of_bins,
            "valid_bin_count": sum(value is not None for value in top_values),
            "tail_start_index": tail_start,
            "tail_bin_count": self.number_of_bins - tail_start,
            "tail_valid_bin_count": len(top_tail),
            "top5_bin_tail_mean": statistics.fmean(top_tail) if top_tail else None,
            "all_sample_bin_tail_mean": statistics.fmean(all_tail) if all_tail else None,
            "rows_in_budget": self.rows_in_budget,
            "rows_after_budget": self.rows_after_budget,
            "rows_at_or_before_zero": self.rows_at_or_before_zero,
        }


def aggregate_bins(
    steps: Sequence[int],
    rewards: Sequence[float],
    budget: int,
    bin_size: int = BIN_SIZE,
    top_k: int = TOP_K,
    tail_fraction: float = TAIL_FRACTION,
) -> Dict[str, Any]:
    """Pure helper used by tests and callers with in-memory reward arrays."""

    if len(steps) != len(rewards):
        raise ValueError("steps and rewards must have the same length")
    accumulator = FixedBudgetBins(budget, bin_size, top_k)
    for raw_step, raw_reward in zip(steps, rewards):
        step = integer(raw_step)
        reward = finite_float(raw_reward)
        if step is None or reward is None:
            continue
        accumulator.add(step, reward)
    return accumulator.summary(tail_fraction)


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


STEP_COLUMNS = ("step", "global_step", "run/global_step", "timesteps", "timestep")
REWARD_COLUMNS = ("episodic_return", "episode_return", "return", "reward", "episode/return")


def read_rewards_file(
    path: Path,
    budget: int,
    bin_size: int = BIN_SIZE,
    top_k: int = TOP_K,
    tail_fraction: float = TAIL_FRACTION,
) -> Dict[str, Any]:
    """Read the real ``step,episodic_return`` rewards.txt CSV format.

    Relaunches sometimes leave multiple monotonic segments in one file.  On a
    strict step decrease, all aggregation state is reset so only the latest
    attempt is scored.  Repeated steps are valid because many vectorized
    environments can finish an episode at the same global step.
    """

    accumulator = FixedBudgetBins(budget, bin_size, top_k)
    result: Dict[str, Any] = {
        "path": str(path),
        "missing": False,
        "total_valid_rows": 0,
        "malformed_rows": 0,
        "nonfinite_rows": 0,
        "step_reset_count": 0,
        "last_reward_step": None,
    }
    if not path.exists():
        result["missing"] = True
        result.update(accumulator.summary(tail_fraction))
        return result

    previous_step: Optional[int] = None
    try:
        handle = path.open("r", encoding="utf-8-sig", errors="replace", newline="")
    except OSError as exc:
        result.update({"missing": True, "read_error": str(exc)})
        result.update(accumulator.summary(tail_fraction))
        return result

    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [name.strip().strip("\x00") if name else name for name in reader.fieldnames]
        for row in reader:
            raw_step = _first_present(row, STEP_COLUMNS)
            raw_reward = _first_present(row, REWARD_COLUMNS)
            step = integer(raw_step)
            reward = finite_float(raw_reward)
            if step is None or reward is None:
                result["malformed_rows"] += 1
                for raw_value in (raw_step, raw_reward):
                    try:
                        parsed = float(str(raw_value).strip().strip("\x00"))
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if not math.isfinite(parsed):
                        result["nonfinite_rows"] += 1
                        break
                continue
            if previous_step is not None and step < previous_step:
                result["step_reset_count"] += 1
                result["total_valid_rows"] = 0
                result["last_reward_step"] = None
                accumulator.reset()
            accumulator.add(step, reward)
            result["total_valid_rows"] += 1
            result["last_reward_step"] = step
            previous_step = step

    result.update(accumulator.summary(tail_fraction))
    return result


def read_last_metrics_step(path: Path) -> Dict[str, Any]:
    """Read the last monotonic attempt in metrics.jsonl without loading it."""

    result: Dict[str, Any] = {
        "path": str(path),
        "missing": False,
        "last_step": None,
        "valid_rows": 0,
        "malformed_rows": 0,
        "step_reset_count": 0,
    }
    if not path.exists():
        result["missing"] = True
        return result
    previous: Optional[int] = None
    try:
        handle = path.open("r", encoding="utf-8", errors="replace")
    except OSError as exc:
        result.update({"missing": True, "read_error": str(exc)})
        return result
    with handle:
        for line in handle:
            line = line.strip().strip("\x00")
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                result["malformed_rows"] += 1
                continue
            if not isinstance(row, dict):
                result["malformed_rows"] += 1
                continue
            step = integer(row.get("run/global_step", row.get("global_step")))
            if step is None:
                result["malformed_rows"] += 1
                continue
            if previous is not None and step < previous:
                result["step_reset_count"] += 1
                result["valid_rows"] = 0
                result["last_step"] = None
            result["valid_rows"] += 1
            result["last_step"] = step
            previous = step
    return result


def _load_json_dict(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _nested_steps(value: Any) -> Iterator[int]:
    if not isinstance(value, dict):
        return
    for key in ("global_step", "run/global_step", "last_step", "step"):
        parsed = integer(value.get(key))
        if parsed is not None:
            yield parsed
    metrics = value.get("metrics")
    if isinstance(metrics, dict):
        yield from _nested_steps(metrics)


def _resolve_path(round_dir: Path, job: Mapping[str, Any], keys: Sequence[str], fallback: str) -> Path:
    for key in keys:
        raw = job.get(key)
        if not raw:
            continue
        candidate = Path(str(raw)).expanduser()
        if not candidate.is_absolute():
            candidate = round_dir / candidate
        if candidate.exists() or candidate.is_symlink():
            return candidate
    job_run_raw = job.get("job_run_dir")
    if job_run_raw:
        job_run = Path(str(job_run_raw)).expanduser()
        if not job_run.is_absolute():
            job_run = round_dir / job_run
        candidate = job_run / fallback
        if candidate.exists() or candidate.is_symlink():
            return candidate
    return round_dir / "runs" / str(job.get("job_id", "unknown_job")) / fallback


def _configured_batch_size(config: Mapping[str, Any]) -> int:
    explicit = integer(config.get("batch_size"))
    if explicit is not None and explicit > 0:
        return explicit
    num_envs = integer(config.get("num_envs"))
    num_steps = integer(config.get("num_steps"))
    if num_envs and num_steps and num_envs > 0 and num_steps > 0:
        return num_envs * num_steps
    return DEFAULT_BATCH_SIZE


def _job_identity(job: Mapping[str, Any], config: Mapping[str, Any]) -> Tuple[str, str, Optional[int]]:
    overrides = job.get("overrides") if isinstance(job.get("overrides"), dict) else {}
    env = str(job.get("env") or overrides.get("env_id") or config.get("env_id") or "").strip()
    variant = str(job.get("variant") or config.get("variant") or "baseline").strip() or "baseline"
    seed = integer(job.get("seed", config.get("seed")))
    return env, variant, seed


def score_job(round_dir: Path, job: Mapping[str, Any]) -> Dict[str, Any]:
    """Score one manifest job and enforce fixed-budget completeness."""

    config_path = _resolve_path(round_dir, job, ("config_path", "config_resolved"), "config.resolved.json")
    config = _load_json_dict(config_path)
    env, variant, seed = _job_identity(job, config)
    if env not in BUDGETS:
        raise ValueError(f"job {job.get('job_id', '<unknown>')} has unknown environment {env!r}")
    budget = budget_for_env(env)
    batch_size = _configured_batch_size(config)
    terminal_step = expected_terminal_step(budget, batch_size)

    rewards_path = _resolve_path(
        round_dir,
        job,
        ("rewards_path", "trainer_reward_path", "reward_path"),
        "rewards.txt",
    )
    # Prefer the round-local file when the old absolute trainer path is stale.
    local_rewards = round_dir / "runs" / str(job.get("job_id", "")) / "rewards.txt"
    if not rewards_path.exists() and (local_rewards.exists() or local_rewards.is_symlink()):
        rewards_path = local_rewards
    metrics_path = _resolve_path(round_dir, job, ("metrics_jsonl", "metrics_path"), "metrics.jsonl")
    status_path = _resolve_path(round_dir, job, ("status_json", "status_path"), "status.json")
    summary_path = _resolve_path(round_dir, job, ("summary_json", "summary_path"), "summary.json")
    launcher_status_path = _resolve_path(
        round_dir,
        job,
        ("launcher_status_json", "launcher_status_path"),
        "launcher_status.json",
    )

    rewards = read_rewards_file(rewards_path, budget)
    metrics = read_last_metrics_step(metrics_path)
    status = _load_json_dict(status_path)
    summary = _load_json_dict(summary_path)
    launcher_status = _load_json_dict(launcher_status_path)
    status_steps = list(_nested_steps(status)) + list(_nested_steps(summary))

    # Metrics are authoritative for training progress; status/summary are the
    # fallback, then rewards (which can naturally lag when no episode ends in
    # the final update).
    if metrics.get("last_step") is not None:
        progress_step = int(metrics["last_step"])
        progress_source = "metrics.jsonl"
    elif status_steps:
        progress_step = max(status_steps)
        progress_source = "status_or_summary"
    elif rewards.get("last_reward_step") is not None:
        progress_step = int(rewards["last_reward_step"])
        progress_source = "rewards.txt"
    else:
        progress_step = None
        progress_source = None

    configured_budget = integer(config.get("total_timesteps"))
    coverage_complete = progress_step is not None and progress_step >= terminal_step
    configured_short = configured_budget is not None and configured_budget < budget
    top5_score = rewards.get("top5_bin_tail_mean")
    all_score = rewards.get("all_sample_bin_tail_mean")
    has_tail_score = top5_score is not None and all_score is not None
    rewards_integrity_ok = not rewards.get("malformed_rows") and not rewards.get("step_reset_count")
    exit_code = integer(launcher_status.get("exit_code"))
    if exit_code is None:
        exit_code = integer(summary.get("exit_code"))
    if exit_code is None:
        exit_code = integer(status.get("exit_code"))
    allowed_teardown = env == "FrankaCubeStack" and exit_code in (-11, 139) and coverage_complete
    # A full-budget trace is not sufficient evidence by itself.  The detached
    # wrapper must have durably recorded a clean exit, except for the one
    # explicitly documented FrankaCubeStack teardown case.
    exit_acceptable = exit_code == 0 or allowed_teardown
    eligible = bool(
        coverage_complete
        and not configured_short
        and has_tail_score
        and rewards_integrity_ok
        and exit_acceptable
    )

    if rewards.get("missing"):
        completeness = "missing_rewards"
    elif configured_short:
        completeness = "configured_short"
    elif progress_step is None:
        completeness = "no_progress"
    elif not coverage_complete:
        completeness = "short_run"
    elif not has_tail_score:
        completeness = "no_tail_data"
    elif rewards.get("nonfinite_rows"):
        completeness = "nonfinite_rewards"
    elif rewards.get("malformed_rows"):
        completeness = "malformed_rewards"
    elif rewards.get("step_reset_count"):
        completeness = "reward_step_reset"
    elif exit_code is None:
        completeness = "missing_exit_code"
    elif not exit_acceptable:
        completeness = "failed_nonzero"
    else:
        completeness = "complete"

    state = str(
        launcher_status.get("state")
        or status.get("state")
        or status.get("status")
        or summary.get("state")
        or summary.get("status")
        or ""
    )
    warnings: List[str] = []
    if configured_budget is None:
        warnings.append("configured_budget_missing")
    elif configured_budget != budget:
        warnings.append("configured_budget_mismatch")
    if rewards.get("malformed_rows"):
        warnings.append("malformed_reward_rows")
    if rewards.get("step_reset_count"):
        warnings.append("rewards_step_reset")
    if rewards.get("rows_after_budget"):
        warnings.append("reward_rows_after_budget")
    if metrics.get("malformed_rows"):
        warnings.append("malformed_metrics_rows")
    if metrics.get("step_reset_count"):
        warnings.append("metrics_step_reset")
    if exit_code is None:
        warnings.append("missing_exit_code")
    if exit_code not in (None, 0):
        warnings.append(f"nonzero_exit_{exit_code}")
    if allowed_teardown:
        warnings.append("complete_with_teardown_warning")

    return {
        "env": env,
        "variant": variant,
        "seed": seed,
        "job_id": str(job.get("job_id", "")),
        "budget": budget,
        "configured_budget": configured_budget,
        "batch_size": batch_size,
        "expected_terminal_step": terminal_step,
        "progress_step": progress_step,
        "progress_source": progress_source,
        "coverage_ratio": (progress_step / terminal_step) if progress_step is not None else None,
        "coverage_complete": coverage_complete,
        "complete": eligible,
        "completeness": completeness,
        "top5_tail_mean": top5_score,
        "all_bin_tail_mean": all_score,
        "top5_minus_all": (top5_score - all_score) if top5_score is not None and all_score is not None else None,
        "number_of_bins": rewards.get("number_of_bins"),
        "valid_bin_count": rewards.get("valid_bin_count"),
        "tail_bin_count": rewards.get("tail_bin_count"),
        "tail_valid_bin_count": rewards.get("tail_valid_bin_count"),
        "reward_row_count": rewards.get("total_valid_rows", 0),
        "reward_rows_in_budget": rewards.get("rows_in_budget", 0),
        "last_reward_step": rewards.get("last_reward_step"),
        "reward_step_reset_count": rewards.get("step_reset_count", 0),
        "reward_malformed_rows": rewards.get("malformed_rows", 0),
        "reward_nonfinite_rows": rewards.get("nonfinite_rows", 0),
        "metrics_last_step": metrics.get("last_step"),
        "trainer_state": state,
        "exit_code": exit_code,
        "warnings": sorted(set(warnings)),
        "paths": {
            "config": str(config_path),
            "rewards": str(rewards_path),
            "metrics": str(metrics_path),
            "status": str(status_path),
            "summary": str(summary_path),
            "launcher_status": str(launcher_status_path),
        },
    }


def _normalise_target_entry(value: Any) -> Optional[Dict[str, float]]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        mean = finite_float(value)
        return {"mean": mean, "std": 0.0} if mean is not None else None
    if isinstance(value, (list, tuple)) and value:
        mean = finite_float(value[0])
        std = finite_float(value[1]) if len(value) > 1 else 0.0
        return {"mean": mean, "std": std or 0.0} if mean is not None else None
    if not isinstance(value, dict):
        return None
    mean = finite_float(
        value.get("mean", value.get("paper_mean", value.get("target", value.get("flowcritic"))))
    )
    std = finite_float(value.get("std", value.get("paper_std", 0.0)))
    return {"mean": mean, "std": std or 0.0} if mean is not None else None


def load_paper_targets(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    targets = {env: dict(values) for env, values in DEFAULT_PAPER_TARGETS.items()}
    if path is None:
        return targets
    raw = _load_json_dict(path)
    container: Any = raw.get("targets", raw.get("environments", raw))
    loaded: Dict[str, Dict[str, float]] = {}
    if isinstance(container, dict):
        for env, value in container.items():
            entry = _normalise_target_entry(value)
            if entry is not None:
                loaded[str(env)] = entry
    elif isinstance(container, list):
        for value in container:
            if not isinstance(value, dict):
                continue
            env = value.get("env", value.get("name"))
            entry = _normalise_target_entry(value)
            if env and entry is not None:
                loaded[str(env)] = entry
    targets.update(loaded)
    return targets


def _discover_jobs(round_dir: Path) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    runs_dir = round_dir / "runs"
    if not runs_dir.is_dir():
        return jobs
    seed_pattern = re.compile(r"(?:^|_)seed[_-]?(\d+)(?:_|$)", re.IGNORECASE)
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        config_path = run_dir / "config.resolved.json"
        config = _load_json_dict(config_path)
        match = seed_pattern.search(run_dir.name)
        seed = integer(config.get("seed")) or (int(match.group(1)) if match else None)
        jobs.append(
            {
                "job_id": run_dir.name,
                "job_run_dir": str(run_dir),
                "config_path": str(config_path),
                "env": config.get("env_id"),
                "variant": config.get("variant", "baseline"),
                "seed": seed,
            }
        )
    return jobs


def load_round_jobs(round_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest_path = round_dir / "manifest.json"
    manifest = _load_json_dict(manifest_path)
    raw_jobs = manifest.get("jobs")
    if isinstance(raw_jobs, list):
        jobs = [dict(job) for job in raw_jobs if isinstance(job, dict)]
        return jobs, manifest
    return _discover_jobs(round_dir), manifest


def aggregate_results(
    seed_rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, float]],
) -> List[Dict[str, Any]]:
    """Aggregate unique (environment, variant) cohorts across seeds 1--4."""

    grouped: Dict[Tuple[str, str], Dict[int, List[Mapping[str, Any]]]] = {}
    for row in seed_rows:
        env = str(row.get("env", ""))
        variant = str(row.get("variant", "baseline"))
        seed = integer(row.get("seed"))
        if seed is None:
            continue
        grouped.setdefault((env, variant), {}).setdefault(seed, []).append(row)

    results: List[Dict[str, Any]] = []
    env_order = {env: index for index, env in enumerate(PAPER_ENVS)}
    for (env, variant), by_seed in sorted(
        grouped.items(), key=lambda item: (env_order.get(item[0][0], len(env_order)), item[0][1])
    ):
        seeds: List[Dict[str, Any]] = []
        complete_scores: List[float] = []
        complete_all_scores: List[float] = []
        observed_scores: List[float] = []
        duplicate_seeds: List[int] = []
        for seed in EXPECTED_SEEDS:
            candidates = by_seed.get(seed, [])
            if len(candidates) == 1:
                detail = dict(candidates[0])
                score = finite_float(detail.get("top5_tail_mean"))
                all_score = finite_float(detail.get("all_bin_tail_mean"))
                if score is not None:
                    observed_scores.append(score)
                if detail.get("complete") and score is not None and all_score is not None:
                    complete_scores.append(score)
                    complete_all_scores.append(all_score)
            elif len(candidates) > 1:
                duplicate_seeds.append(seed)
                detail = {
                    "env": env,
                    "variant": variant,
                    "seed": seed,
                    "complete": False,
                    "completeness": "duplicate_jobs",
                    "job_ids": [str(candidate.get("job_id", "")) for candidate in candidates],
                    "top5_tail_mean": None,
                    "all_bin_tail_mean": None,
                    "warnings": ["duplicate_env_variant_seed"],
                }
            else:
                detail = {
                    "env": env,
                    "variant": variant,
                    "seed": seed,
                    "complete": False,
                    "completeness": "missing_job",
                    "job_id": "",
                    "top5_tail_mean": None,
                    "all_bin_tail_mean": None,
                    "warnings": ["missing_seed"],
                }
            seeds.append(detail)

        complete = len(complete_scores) == len(EXPECTED_SEEDS) and not duplicate_seeds
        top5_mean = statistics.fmean(complete_scores) if complete else None
        top5_std = statistics.pstdev(complete_scores) if complete else None
        all_mean = statistics.fmean(complete_all_scores) if complete else None
        all_std = statistics.pstdev(complete_all_scores) if complete else None
        paper_entry = targets.get(env, {})
        paper_mean = finite_float(paper_entry.get("mean"))
        paper_std = finite_float(paper_entry.get("std"))
        target_90 = 0.9 * paper_mean if paper_mean is not None else None
        ratio = top5_mean / paper_mean if top5_mean is not None and paper_mean not in (None, 0.0) else None
        passed = bool(complete and top5_mean is not None and target_90 is not None and top5_mean >= target_90)
        results.append(
            {
                "env": env,
                "variant": variant,
                "complete_seed_count": len(complete_scores),
                "expected_seed_count": len(EXPECTED_SEEDS),
                "seeds": seeds,
                "top5_mean": top5_mean,
                "top5_std": top5_std,
                "all_bin_mean": all_mean,
                "all_bin_std": all_std,
                "top5_minus_all": (top5_mean - all_mean) if top5_mean is not None and all_mean is not None else None,
                "observed_top5_mean": statistics.fmean(observed_scores) if observed_scores else None,
                "paper_mean": paper_mean,
                "paper_std": paper_std,
                "target_90": target_90,
                "ratio": ratio,
                "passed_90": passed,
                "complete": complete,
                "duplicate_seeds": duplicate_seeds,
            }
        )
    return results


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else value


CSV_FIELDS: Tuple[str, ...] = (
    "env",
    "variant",
    "complete",
    "complete_seed_count",
    "top5_mean",
    "top5_std",
    "all_bin_mean",
    "all_bin_std",
    "top5_minus_all",
    "paper_mean",
    "paper_std",
    "target_90",
    "ratio",
    "passed_90",
    "seed1_top5",
    "seed2_top5",
    "seed3_top5",
    "seed4_top5",
    "seed1_status",
    "seed2_status",
    "seed3_status",
    "seed4_status",
)


def write_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            row: Dict[str, Any] = {key: _csv_value(result.get(key)) for key in CSV_FIELDS}
            seeds = result.get("seeds") if isinstance(result.get("seeds"), list) else []
            for detail in seeds:
                if not isinstance(detail, dict):
                    continue
                seed = integer(detail.get("seed"))
                if seed in EXPECTED_SEEDS:
                    row[f"seed{seed}_top5"] = _csv_value(detail.get("top5_tail_mean"))
                    row[f"seed{seed}_status"] = _csv_value(detail.get("completeness"))
            writer.writerow(row)


def _fmt(value: Any, digits: int = 3) -> str:
    number = finite_float(value)
    return "—" if number is None else f"{number:.{digits}f}"


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    lines = [
        "# Score: plotting protocol",
        "",
        "Primary metric: 200,000-step `(left, right]` bins, top 5 episode returns per bin, "
        "mean over each seed's final 10% fixed-budget bins, then four-seed mean ± population std.",
        "Incomplete seeds are excluded from the target gate. `All-bin` averages every episode in each bin and is diagnostic.",
        "",
        "| Environment | Variant | Seeds | Top-5/bin | All-bin | Paper | Ratio | ≥90% | Complete |",
        "|---|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for result in results:
        top = "—"
        if result.get("top5_mean") is not None:
            top = f"{_fmt(result.get('top5_mean'))} ± {_fmt(result.get('top5_std'))}"
        ratio = _fmt((result.get("ratio") or 0.0) * 100.0, 2) + "%" if result.get("ratio") is not None else "—"
        lines.append(
            "| {env} | {variant} | {count}/4 | {top} | {all_mean} | {paper} | {ratio} | {passed} | {complete} |".format(
                env=result.get("env", ""),
                variant=result.get("variant", ""),
                count=result.get("complete_seed_count", 0),
                top=top,
                all_mean=_fmt(result.get("all_bin_mean")),
                paper=_fmt(result.get("paper_mean")),
                ratio=ratio,
                passed="yes" if result.get("passed_90") else "no",
                complete="yes" if result.get("complete") else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Per-seed details",
            "",
            "| Environment | Variant | Seed | Status | Progress | Top-5/bin | All-bin | Tail bins | Warnings |",
            "|---|---|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        for seed in result.get("seeds", []):
            if not isinstance(seed, dict):
                continue
            tail = f"{seed.get('tail_valid_bin_count', 0)}/{seed.get('tail_bin_count', 0)}"
            warnings = ", ".join(str(value) for value in seed.get("warnings", [])) or "—"
            lines.append(
                "| {env} | {variant} | {seed_id} | {status} | {progress} | {top} | {all_mean} | {tail} | {warnings} |".format(
                    env=result.get("env", ""),
                    variant=result.get("variant", ""),
                    seed_id=seed.get("seed", ""),
                    status=seed.get("completeness", ""),
                    progress=seed.get("progress_step", "—") if seed.get("progress_step") is not None else "—",
                    top=_fmt(seed.get("top5_tail_mean")),
                    all_mean=_fmt(seed.get("all_bin_tail_mean")),
                    tail=tail,
                    warnings=warnings,
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_round(round_dir: Path, targets_path: Optional[Path] = None) -> Dict[str, Any]:
    round_dir = round_dir.expanduser().resolve()
    jobs, manifest = load_round_jobs(round_dir)
    targets = load_paper_targets(targets_path)
    seed_rows: List[Dict[str, Any]] = []
    ignored_jobs: List[Dict[str, Any]] = []
    for job in jobs:
        try:
            row = score_job(round_dir, job)
        except ValueError as exc:
            ignored_jobs.append({"job_id": str(job.get("job_id", "")), "reason": str(exc)})
            continue
        if row.get("seed") not in EXPECTED_SEEDS:
            ignored_jobs.append(
                {
                    "job_id": row.get("job_id", ""),
                    "env": row.get("env"),
                    "seed": row.get("seed"),
                    "reason": "seed_outside_1_4",
                }
            )
            continue
        seed_rows.append(row)
    results = aggregate_results(seed_rows, targets)
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "round_dir": str(round_dir),
        "round_id": str(manifest.get("round_id") or round_dir.name),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(round_dir / "manifest.json"),
        "targets_sha256": sha256_file(targets_path) if targets_path else None,
        "protocol": {
            "bin_size": BIN_SIZE,
            "bin_boundary": "(left,right]",
            "top_k": TOP_K,
            "tail_fraction": TAIL_FRACTION,
            "seeds": list(EXPECTED_SEEDS),
            "std": "population",
            "budgets": BUDGETS,
            "short_runs_enter_target_gate": False,
        },
        "input_job_count": len(jobs),
        "scored_job_count": len(seed_rows),
        "ignored_jobs": ignored_jobs,
        "results": results,
    }


def write_outputs(payload: Mapping[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": output_dir / "score_plot_protocol.csv",
        "json": output_dir / "score_plot_protocol.json",
        "md": output_dir / "score_plot_protocol.md",
    }
    write_csv(paths["csv"], payload.get("results", []))
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(paths["md"], payload)
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_dir", type=Path, help="Round directory containing manifest.json and runs/")
    parser.add_argument(
        "--targets",
        "--paper-targets",
        dest="targets",
        type=Path,
        default=None,
        help="paper_targets.json (built-in paper targets are used when omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ROUND_DIR)",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit 2 unless every emitted cohort has four complete seeds",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    round_dir = args.round_dir.expanduser().resolve()
    targets = args.targets.expanduser().resolve() if args.targets else None
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else round_dir
    if not round_dir.is_dir():
        print(f"error: round directory does not exist: {round_dir}", file=sys.stderr)
        return 1
    if targets is not None and not targets.is_file():
        print(f"error: targets file does not exist: {targets}", file=sys.stderr)
        return 1
    payload = analyze_round(round_dir, targets)
    paths = write_outputs(payload, output_dir)
    complete_count = sum(bool(result.get("complete")) for result in payload["results"])
    if not args.quiet:
        print(
            "[score-plot-protocol] results={}/{} output={}".format(
                complete_count, len(payload["results"]), paths["json"]
            )
        )
    if args.require_complete:
        all_jobs_scored = (
            int(payload.get("input_job_count", 0)) > 0
            and payload.get("scored_job_count") == payload.get("input_job_count")
            and not payload.get("ignored_jobs")
        )
        if complete_count != len(payload["results"]) or not all_jobs_scored:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
