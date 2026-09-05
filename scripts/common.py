#!/usr/bin/env python3
"""Shared, dependency-free helpers for the paper tuning loop."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
EXPERIMENTS_DIR = ROOT / "experiments"
STATE_DIR = ROOT / "state"
PAPER_TARGETS_PATH = ROOT / "paper" / "paper_targets.json"

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

# These values are re-applied during materialization.  They therefore cannot be
# changed by a planner even if a malformed plan bypasses the schema validator.
FIXED_PARAMETERS: Dict[str, Any] = {
    "num_envs": 1024,
    "num_steps": 16,
    "batch_size": 16384,
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
    "rollout_mean_mode": "remove_max",
    "mean_mode": "remove_max",
}

STAGE_PARAMETERS = {
    1: {
        "prior_std",
        "target_norm",
        "cov_floor",
        "reward_scaler",
        "acf_coef",
        "vf_coef",
        "flow_sigma",
    },
    2: {
        "target_norm_update",
        "target_norm_ema_tau",
        "cov_weight_clip",
        "ess_floor",
        "alpha_warmup_updates",
    },
    3: {"old_critic_update", "old_critic_ema_tau", "bootstrap_mean_mode"},
}
ALL_TUNABLE_PARAMETERS = set().union(*STAGE_PARAMETERS.values())

DEAD_OR_UNUSED_PARAMETERS = {"predict_epsilon", "noise_ratio", "vloss_coef"}
FORBIDDEN_PARAMETERS = set(FIXED_PARAMETERS) | {
    "seed",
    "env_id",
    "total_timesteps",
    "update_epochs",
    "rollout_mean_mode",
    "mean_mode",
}


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path | str, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def write_json_compact_atomic(path: Path | str, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def write_text_atomic(path: Path | str, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def cohort_signature(env: str, overrides: Mapping[str, Any]) -> str:
    relevant = {key: overrides[key] for key in sorted(overrides) if key in ALL_TUNABLE_PARAMETERS}
    return sha256_payload({"env": env, "parameters": relevant})


def expected_final_step(total_timesteps: int, num_envs: int = 1024, num_steps: int = 16) -> int:
    batch_size = int(num_envs) * int(num_steps)
    return (int(total_timesteps) // batch_size) * batch_size


def stage_allowlist(stage: int) -> set[str]:
    allowed: set[str] = set()
    for level in range(1, min(3, int(stage)) + 1):
        allowed.update(STAGE_PARAMETERS[level])
    return allowed


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
