#!/usr/bin/env python3
"""Validate and materialize a planner output into an immutable round manifest."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from common import (
    ALL_TUNABLE_PARAMETERS,
    CONFIG_DIR,
    ENVIRONMENTS,
    EXPERIMENTS_DIR,
    FIXED_PARAMETERS,
    PAPER_TARGETS_PATH,
    ROOT,
    SEEDS,
    cohort_signature,
    expected_final_step,
    iso_now,
    read_json,
    safe_name,
    sha256_payload,
    sha256_file,
    stage_allowlist,
    write_json_atomic,
    write_text_atomic,
)
from validate_plan import (
    PlanValidationError,
    normalize_override_value,
    validate_tunable_config_precision,
    validate_resolved_override_activity,
)


DEFAULT_PLAN = CONFIG_DIR / "round_000_plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--round-id", default=None)
    parser.add_argument(
        "--base-configs",
        default=None,
        help="Optional current_configs.json; GPT overrides are applied on top of accepted tunables.",
    )
    parser.add_argument("--out-root", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--conda-env", default="RLS")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def group_jobs(jobs: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], list[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str], list[Dict[str, Any]]] = {}
    for raw in jobs:
        env = str(raw.get("env", ""))
        variant = safe_name(str(raw.get("variant", "")))
        groups.setdefault((env, variant), []).append(raw)
    return groups


def validate_materialization_input(plan: Dict[str, Any]) -> None:
    if plan.get("status", "continue") != "continue":
        raise ValueError("only a plan with status=continue can be materialized")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan.jobs must be a non-empty list")
    if len(jobs) > 48:
        raise ValueError("a round may contain at most 48 jobs")
    stage = int(plan.get("stage", 1))
    if stage not in {1, 2, 3}:
        raise ValueError(f"stage must be 1, 2, or 3; got {stage}")
    purpose = str(plan.get("purpose", "tuning"))
    if purpose not in {"tuning", "unification_global", "unification_cluster"}:
        raise ValueError(f"unsupported round purpose: {purpose}")
    if purpose.startswith("unification_") and stage != 3:
        raise ValueError("unification rounds must use the complete Stage 3 search contract")
    allowed = stage_allowlist(stage)
    groups = group_jobs(jobs)
    variants_per_env: Dict[str, set[str]] = {}
    for (env, variant), group in groups.items():
        if env not in ENVIRONMENTS:
            raise ValueError(f"unknown paper environment: {env}")
        if not variant:
            raise ValueError(f"empty variant for {env}")
        variants_per_env.setdefault(env, set()).add(variant)
        seeds = sorted(int(job.get("seed", -1)) for job in group)
        if seeds != list(SEEDS):
            raise ValueError(f"{env}/{variant} must contain exactly seeds 1,2,3,4; got {seeds}")
        canonical_overrides = group[0].get("overrides", {})
        if not isinstance(canonical_overrides, dict):
            raise ValueError(f"{env}/{variant} overrides must be an object")
        for job in group:
            if job.get("overrides", {}) != canonical_overrides:
                raise ValueError(f"{env}/{variant} must use identical overrides for all seeds")
        bad = sorted(set(canonical_overrides) - allowed)
        if bad:
            raise ValueError(f"{env}/{variant} contains parameters unavailable in stage {stage}: {bad}")
        try:
            normalized = {
                key: normalize_override_value(
                    key,
                    value,
                    f"{env}/{variant}.overrides.{key}",
                )
                for key, value in canonical_overrides.items()
            }
        except PlanValidationError as exc:
            raise ValueError(str(exc)) from exc
        for job in group:
            job["overrides"] = copy.deepcopy(normalized)
    for env, variants in variants_per_env.items():
        if len(variants) > 2:
            raise ValueError(f"{env} contains more than two variants")


def load_base_tunables(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("configs"), dict):
        payload = payload["configs"]
    if not isinstance(payload, dict):
        raise ValueError("base configs must be an environment-keyed object")
    result: Dict[str, Dict[str, Any]] = {}
    for env, raw in payload.items():
        if env not in ENVIRONMENTS or not isinstance(raw, dict):
            continue
        config = raw.get("tunable_config", raw.get("config", raw))
        if isinstance(config, dict):
            normalized: Dict[str, Any] = {}
            for key, value in config.items():
                if key not in ALL_TUNABLE_PARAMETERS:
                    continue
                try:
                    normalized[key] = normalize_override_value(
                        key,
                        value,
                        f"base_configs.{env}.{key}",
                        enforce_precision=False,
                    )
                except PlanValidationError as exc:
                    raise ValueError(str(exc)) from exc
            result[env] = normalized
    return result


def materialize(
    plan_path: Path,
    round_id_override: str | None,
    out_root: Path,
    conda_env: str,
    gpus: list[str],
    dry_run: bool,
    base_configs_path: Path | None = None,
) -> Path:
    plan = read_json(plan_path)
    validate_materialization_input(plan)
    targets = read_json(PAPER_TARGETS_PATH)
    base_tunables = load_base_tunables(base_configs_path)
    round_id = safe_name(round_id_override or plan.get("round_id") or "")
    if not round_id:
        raise ValueError("round_id is required")
    round_dir = out_root / round_id
    manifest_path = round_dir / "manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("source_plan_sha256") == sha256_payload(plan):
            print(f"[materialize] already materialized: {round_dir}")
            return round_dir
        raise FileExistsError(f"round directory already contains a different manifest: {round_dir}")

    if dry_run:
        print(f"[dry-run] round={round_id} jobs={len(plan['jobs'])} out={round_dir}")
        for index, job in enumerate(plan["jobs"], 1):
            print(f"{index:03d} {job['env']} {job['variant']} seed={job['seed']} {job.get('overrides', {})}")
        return round_dir

    if round_dir.exists():
        raise FileExistsError(f"round directory exists without a reusable manifest: {round_dir}")
    for raw_job in plan["jobs"]:
        base_path = CONFIG_DIR / f"config_{raw_job['env']}.json"
        if not base_path.is_file():
            raise FileNotFoundError(f"missing standalone base config: {base_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{round_id}.materializing.", dir=str(out_root)))
    try:
        (staging_dir / "runs").mkdir()
        materialized: list[Dict[str, Any]] = []
        for index, raw_job in enumerate(plan["jobs"], 1):
            env = str(raw_job["env"])
            seed = int(raw_job["seed"])
            variant = safe_name(str(raw_job["variant"]))
            base_path = CONFIG_DIR / f"config_{env}.json"
            config = copy.deepcopy(read_json(base_path))
            config.update(copy.deepcopy(base_tunables.get(env, {})))
            config.update(copy.deepcopy(raw_job.get("overrides", {})))
            tunables = {
                key: config[key]
                for key in ALL_TUNABLE_PARAMETERS
                if key in config
            }
            try:
                validate_tunable_config_precision(
                    tunables,
                    f"{env}/{variant}.resolved_tunables",
                )
            except PlanValidationError as exc:
                raise ValueError(str(exc)) from exc
            if str(plan.get("purpose", "tuning")) == "tuning":
                try:
                    validate_resolved_override_activity(
                        raw_job.get("overrides", {}),
                        config,
                        f"{env}/{variant}.overrides",
                    )
                except PlanValidationError as exc:
                    raise ValueError(str(exc)) from exc
            config.update(copy.deepcopy(FIXED_PARAMETERS))
            config["env_id"] = env
            config["seed"] = seed
            config["total_timesteps"] = int(targets["environments"][env]["budget"])
            config["track"] = False
            config["capture_video"] = False
            config["save_reward_log"] = True
            config["log_tensorboard"] = False
            config["sync_reward_log"] = True
            config["metrics_interval"] = int(config.get("metrics_interval", 1))

            job_id = safe_name(raw_job.get("job_id") or f"{index:03d}_{env}_{variant}_seed{seed}")
            final_job_dir = round_dir / "runs" / job_id
            staging_job_dir = staging_dir / "runs" / job_id
            staging_job_dir.mkdir()
            trainer_run_dir = ROOT / "runs" / round_id / job_id
            run_name = f"{round_id}/{job_id}"
            config_path = final_job_dir / "config.resolved.json"
            metrics_path = final_job_dir / "metrics.jsonl"
            status_path = final_job_dir / "status.json"
            summary_path = final_job_dir / "summary.json"
            rewards_path = final_job_dir / "rewards.txt"
            config.update(
                {
                    "run_name": run_name,
                    "run_root": str(ROOT / "runs"),
                    "reward_log_path": str(rewards_path),
                    "research_tag": f"{round_id}/{job_id}",
                    "metrics_jsonl": str(metrics_path),
                    "status_json": str(status_path),
                    "summary_json": str(summary_path),
                    "early_stop_file": str(final_job_dir / "early_stop.disabled.json"),
                }
            )
            write_json_atomic(staging_job_dir / "config.resolved.json", config)
            materialized.append(
                {
                    "job_id": job_id,
                    "env": env,
                    "seed": seed,
                    "variant": variant,
                    "overrides": copy.deepcopy(raw_job.get("overrides", {})),
                    "hypothesis": raw_job.get("hypothesis", ""),
                    "evidence": raw_job.get("evidence", ""),
                    "cohort_signature": cohort_signature(env, config),
                    "config_sha256": sha256_payload(config),
                    "config_path": str(config_path),
                    "job_run_dir": str(final_job_dir),
                    "trainer_run_dir": str(trainer_run_dir),
                    "run_name": run_name,
                    "metrics_jsonl": str(metrics_path),
                    "status_json": str(status_path),
                    "summary_json": str(summary_path),
                    "rewards_path": str(rewards_path),
                    "stdout_log": str(final_job_dir / "stdout.log"),
                    "expected_final_step": expected_final_step(config["total_timesteps"]),
                    "budget": config["total_timesteps"],
                }
            )

        manifest = {
            "schema_version": 1,
            "round_id": round_id,
            "round_index": int(round_id.split("_")[1]) if round_id.startswith("round_") else None,
            "stage": int(plan.get("stage", 1)),
            "purpose": str(plan.get("purpose", "tuning")),
            "created_at": iso_now(),
            "source_plan": str(plan_path.resolve()),
            "source_plan_sha256": sha256_payload(plan),
            "base_configs": str(base_configs_path.resolve()) if base_configs_path else None,
            "script": str(ROOT / "src" / "flow_ppo_single2.py"),
            "source_sha256": {
                "flow_ppo_single2.py": sha256_file(ROOT / "src" / "flow_ppo_single2.py"),
                "flow_matching.py": sha256_file(ROOT / "src" / "flow_matching.py"),
            },
            "orchestration_sha256": {
                name: sha256_file(ROOT / "scripts" / name)
                for name in (
                    "common.py",
                    "materialize_round.py",
                    "launch_round.py",
                    "job_process.py",
                    "score_plot_protocol.py",
                    "validate_plan.py",
                    "call_gpt_planner.py",
                    "continuous_tune.py",
                    "approved_task_plan.py",
                    "process_identity.py",
                    "unification.py",
                )
            },
            "contract_sha256": {
                "parameter_contract.json": sha256_file(CONFIG_DIR / "parameter_contract.json"),
                "paper_targets.json": sha256_file(PAPER_TARGETS_PATH),
            },
            "conda_env": conda_env,
            "gpus": gpus,
            "slots_per_gpu": 6,
            "fixed_parameters": FIXED_PARAMETERS,
            "jobs": materialized,
        }
        write_json_atomic(staging_dir / "manifest.json", manifest)
        write_json_atomic(staging_dir / "next_plan.json", plan)
        write_json_atomic(
            staging_dir / "round_state.json",
            {
                "round_id": round_id,
                "state": "materialized",
                "created_at": iso_now(),
                "jobs_total": len(materialized),
                "jobs": {job["job_id"]: {"state": "pending", "attempt": 0} for job in materialized},
            },
        )
        lines = [
            f"# {round_id}",
            "",
            f"- stage: `{manifest['stage']}`",
            f"- jobs: `{len(materialized)}`",
            f"- GPUs: `{','.join(gpus)}` with `6` slots per GPU",
            f"- plan SHA256: `{manifest['source_plan_sha256']}`",
            "",
            "| job | environment | variant | seed | expected final step |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        for job in materialized:
            lines.append(
                f"| `{job['job_id']}` | `{job['env']}` | `{job['variant']}` | {job['seed']} | {job['expected_final_step']} |"
            )
        write_text_atomic(staging_dir / "ROUND.md", "\n".join(lines) + "\n")
        os.replace(staging_dir, round_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    print(f"[materialize] round={round_id} jobs={len(materialized)} path={round_dir}")
    return round_dir


def main() -> int:
    args = parse_args()
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if gpus != ["0", "1", "2", "3"]:
        raise SystemExit("this paper loop is contracted to physical GPUs 0,1,2,3")
    materialize(
        Path(args.plan).expanduser().resolve(),
        args.round_id,
        Path(args.out_root).expanduser().resolve(),
        args.conda_env,
        gpus,
        args.dry_run,
        Path(args.base_configs).expanduser().resolve() if args.base_configs else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
