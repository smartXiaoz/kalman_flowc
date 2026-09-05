#!/usr/bin/env python3
"""Resumable local scheduler for one materialized paper-tuning round."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from common import FIXED_PARAMETERS, iso_now, read_json, sha256_file, sha256_payload, write_json_atomic
from process_identity import capture_process_identity


STOP_REQUESTED = False
OOM_MARKERS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "cuda runtime error (2)",
)

# These states mean that a previous attempt produced an incomplete or
# otherwise transient result.  They may be explicitly recycled by the
# continuous supervisor after the attempt's artifacts have been archived.
# Integrity failures are deliberately excluded: retrying a changed/malformed
# manifest would hide a contract violation rather than repair it.
RECOVERABLE_RETRY_STATES = frozenset(
    {
        "failed_nonzero",
        "failed_short",
        "failed_oom",
        "failed_exit_unknown",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_dir")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=6)
    parser.add_argument("--poll-sec", type=float, default=10.0)
    parser.add_argument("--max-oom-retries", type=int, default=3)
    parser.add_argument("--oom-retry-delay-sec", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Recycle recoverable failed jobs on resume. Existing artifacts are "
            "archived as a new attempt; integrity failures are never retried."
        ),
    )
    return parser.parse_args()


def handle_stop(signum: int, frame: object) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def process_alive(pid: int, config_path: str | None = None) -> bool:
    if pid <= 0:
        return False
    proc_dir = Path("/proc") / str(pid)
    if not proc_dir.exists():
        return False
    try:
        stat_fields = (proc_dir / "stat").read_text(encoding="utf-8").split()
        if len(stat_fields) > 2 and stat_fields[2] == "Z":
            return False
        if config_path:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            if config_path not in cmdline:
                return False
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return False


def read_optional_json(path: str | Path) -> Dict[str, Any]:
    try:
        payload = read_json(path)
        return payload if isinstance(payload, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def completed_budget(job: Dict[str, Any]) -> bool:
    summary = read_optional_json(job["summary_json"])
    status = read_optional_json(job["status_json"]) if job.get("status_json") else {}
    steps = [int(value) for value in (summary.get("global_step"), status.get("global_step")) if isinstance(value, (int, float))]
    metrics_step = last_metrics_step(Path(job["metrics_jsonl"])) if job.get("metrics_jsonl") else None
    if metrics_step is not None:
        steps.append(metrics_step)
    global_step = max(steps, default=0)
    return global_step >= int(job["expected_final_step"])


def last_metrics_step(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(-min(size, 1_000_000), os.SEEK_END)
            lines = handle.read().decode(errors="replace").splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(row, dict):
                continue
            value = row.get("global_step", row.get("run/global_step"))
            if isinstance(value, (int, float)):
                return int(value)
    except OSError:
        return None
    return None


def process_record_path(job: Dict[str, Any]) -> Path:
    return Path(job["job_run_dir"]) / "process_exit.json"


def recorded_exit_code(job: Dict[str, Any]) -> Optional[int]:
    record = read_optional_json(process_record_path(job))
    value = record.get("exit_code")
    return int(value) if isinstance(value, (int, float)) else None


def find_process_for_config(config_path: str) -> tuple[Optional[int], Optional[str]]:
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if config_path not in cmdline or "job_process.py" not in cmdline:
            continue
        fields = cmdline.split()
        gpu = None
        if "--gpu-id" in fields:
            index = fields.index("--gpu-id")
            if index + 1 < len(fields):
                gpu = fields[index + 1]
        return int(proc_dir.name), gpu
    return None, None


def verify_integrity(job: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    expected_config = job.get("config_sha256")
    if expected_config:
        config = read_json(job["config_path"])
        actual_config = sha256_payload(config)
        if actual_config != expected_config:
            raise RuntimeError(f"resolved config hash mismatch for {job['job_id']}")
        for key, expected in FIXED_PARAMETERS.items():
            if config.get(key) != expected:
                raise RuntimeError(f"fixed parameter mismatch for {job['job_id']}: {key}")
        if config.get("env_id") != job.get("env") or int(config.get("seed", -1)) != int(job.get("seed", -2)):
            raise RuntimeError(f"job identity mismatch for {job['job_id']}")
        if "budget" in job and int(config.get("total_timesteps", -1)) != int(job["budget"]):
            raise RuntimeError(f"budget mismatch for {job['job_id']}")
    source_hashes = manifest.get("source_sha256") or {}
    script_path = Path(manifest["script"])
    source_paths = {
        "flow_ppo_single2.py": script_path,
        "flow_matching.py": script_path.with_name("flow_matching.py"),
    }
    for name, expected in source_hashes.items():
        path = source_paths.get(name)
        if path is None or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"source hash mismatch for {name}")


def log_contains_oom(path: str | Path) -> bool:
    log_path = Path(path)
    if not log_path.exists():
        return False
    try:
        tail_size = min(log_path.stat().st_size, 2_000_000)
        with log_path.open("rb") as handle:
            handle.seek(-tail_size, os.SEEK_END)
            tail = handle.read().decode(errors="replace")
        return any(marker in tail for marker in OOM_MARKERS)
    except OSError:
        return False


def gpu_preflight(gpus: list[str]) -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    seen: Dict[str, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3:
            seen[fields[0]] = (int(fields[1]), int(fields[2]))
    missing = [gpu for gpu in gpus if gpu not in seen]
    busy = [gpu for gpu in gpus if gpu in seen and seen[gpu][0] >= 500]
    if missing:
        raise RuntimeError(f"requested GPUs not visible: {missing}")
    if busy:
        detail = {gpu: seen[gpu] for gpu in busy}
        raise RuntimeError(f"GPU preflight requires initially idle cards (<500 MiB): {detail}")


def build_command(job: Dict[str, Any], manifest: Dict[str, Any], gpu: str) -> list[str]:
    conda_exe = os.environ.get("FLOWCRITIC_CONDA_EXE") or shutil.which("conda") or "conda"
    return [
        conda_exe,
        "run",
        "--no-capture-output",
        "-n",
        manifest.get("conda_env", "RLS"),
        "python",
        "-u",
        manifest["script"],
        "--config-path",
        job["config_path"],
        "--run-name",
        job["run_name"],
        "--gpu-id",
        str(gpu),
        "--research-tag",
        f"{manifest['round_id']}/{job['job_id']}",
        "--metrics-jsonl",
        job["metrics_jsonl"],
        "--status-json",
        job["status_json"],
        "--summary-json",
        job["summary_json"],
        "--reward-log-path",
        job["rewards_path"],
        "--early-stop-file",
        str(Path(job["job_run_dir"]) / "early_stop.disabled.json"),
        "--save-minimal-logs",
    ]


def archive_partial_attempt(job: Dict[str, Any], attempt: int) -> None:
    outputs = [
        Path(job["stdout_log"]),
        Path(job["metrics_jsonl"]),
        Path(job["status_json"]),
        Path(job["summary_json"]),
        Path(job["rewards_path"]),
        Path(job["job_run_dir"]) / "launcher_status.json",
        process_record_path(job),
    ]
    existing = [path for path in outputs if path.exists() or path.is_symlink()]
    if not existing:
        return
    archive_dir = Path(job["job_run_dir"]) / "attempts" / f"attempt_{attempt:03d}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for source in existing:
        destination = archive_dir / source.name
        if destination.exists():
            destination = archive_dir / f"{source.stem}_{int(time.time())}{source.suffix}"
        shutil.move(str(source), str(destination))


@dataclass
class RunningJob:
    job: Dict[str, Any]
    gpu: str
    pid: int
    started_at_epoch: float
    proc: Optional[subprocess.Popen] = None
    log_handle: Optional[Any] = None


def state_template(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "round_id": manifest["round_id"],
        "state": "materialized",
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "jobs_total": len(manifest["jobs"]),
        "jobs": {job["job_id"]: {"state": "pending", "attempt": 0} for job in manifest["jobs"]},
    }


def persist_state(round_dir: Path, state: Dict[str, Any]) -> None:
    state["updated_at"] = iso_now()
    counts: Dict[str, int] = {}
    for job_state in state["jobs"].values():
        name = str(job_state.get("state", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    state["counts"] = counts
    write_json_atomic(round_dir / "round_state.json", state)


def launch_job(
    job: Dict[str, Any],
    manifest: Dict[str, Any],
    gpu: str,
    state_entry: Dict[str, Any],
    round_dir: Path,
    state: Dict[str, Any],
) -> RunningJob:
    verify_integrity(job, manifest)
    previous_attempt = int(state_entry.get("attempt", 0) or 0)
    archive_partial_attempt(job, previous_attempt)
    command = build_command(job, manifest, gpu)
    attempt = previous_attempt + 1
    state_entry.update(
        {
            "state": "starting",
            "attempt": attempt,
            "gpu": gpu,
            "pid": None,
            "command": command,
            "started_at": iso_now(),
            "started_at_epoch": time.time(),
            "message": "launch reserved; waiting for durable wrapper pid",
        }
    )
    persist_state(round_dir, state)
    Path(job["job_run_dir"]).mkdir(parents=True, exist_ok=True)
    log_handle = Path(job["stdout_log"]).open("w", encoding="utf-8", buffering=1)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("CUDA_VISIBLE_DEVICES", None)
    wrapper_command = [
        sys.executable,
        str(Path(__file__).with_name("job_process.py")),
        "--record",
        str(process_record_path(job)),
        "--",
        *command,
    ]
    proc = subprocess.Popen(
        wrapper_command,
        cwd=str(Path(manifest["script"]).resolve().parents[1]),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    state_entry.update(
        {
            "state": "running",
            "attempt": attempt,
            "gpu": gpu,
            "pid": proc.pid,
            "command": command,
            "wrapper_command": wrapper_command,
            "started_at": iso_now(),
            "started_at_epoch": time.time(),
            "message": "",
        }
    )
    write_json_atomic(
        Path(job["job_run_dir"]) / "launcher_status.json",
        {
            "state": "running",
            "attempt": attempt,
            "gpu": gpu,
            "pid": proc.pid,
            "command": command,
            "wrapper_command": wrapper_command,
        },
    )
    persist_state(round_dir, state)
    print(f"[launch] gpu={gpu} attempt={attempt} job={job['job_id']}", flush=True)
    return RunningJob(job, gpu, proc.pid, time.time(), proc, log_handle)


def finish_running_handle(running: RunningJob) -> None:
    if running.log_handle is not None:
        try:
            running.log_handle.close()
        except OSError:
            pass
        running.log_handle = None


def terminal_state(job: Dict[str, Any], return_code: Optional[int]) -> tuple[str, str, bool]:
    full = completed_budget(job)
    if full and job["env"] == "FrankaCubeStack" and return_code in (-11, 139):
        return "complete_with_teardown_warning", f"fixed budget reached; known teardown exit {return_code}", False
    if full and return_code == 0:
        return "complete", "fixed budget reached", False
    if full and return_code is None:
        return "failed_exit_unknown", "fixed budget reached but durable exit code is missing", False
    if full:
        return "failed_nonzero", f"fixed budget reached but process exited {return_code}", False
    if log_contains_oom(job["stdout_log"]):
        return "failed_oom", f"CUDA OOM; process exited {return_code}", True
    if return_code == 0:
        return "failed_short", "process exited cleanly before fixed budget", False
    return "failed_nonzero", f"process exited {return_code} before fixed budget", False


def write_round_summary(round_dir: Path, manifest: Dict[str, Any], state: Dict[str, Any]) -> None:
    rows: list[Dict[str, Any]] = []
    for job in manifest["jobs"]:
        entry = state["jobs"][job["job_id"]]
        summary = read_optional_json(job["summary_json"])
        rows.append(
            {
                "job_id": job["job_id"],
                "env": job["env"],
                "variant": job["variant"],
                "seed": job["seed"],
                "state": entry.get("state"),
                "attempt": entry.get("attempt", 0),
                "gpu": entry.get("gpu"),
                "exit_code": entry.get("exit_code"),
                "global_step": summary.get("global_step"),
                "expected_final_step": job["expected_final_step"],
                "message": entry.get("message", ""),
            }
        )
    write_json_atomic(
        round_dir / "summary.json",
        {
            "round_id": manifest["round_id"],
            "state": state["state"],
            "finished_at": iso_now(),
            "counts": state.get("counts", {}),
            "jobs": rows,
        },
    )
    with (round_dir / "launcher_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    round_dir = Path(args.round_dir).expanduser().resolve()
    policy_abort_path = round_dir / "policy_abort.json"
    if policy_abort_path.is_file():
        marker = read_json(policy_abort_path)
        raise SystemExit(
            f"round is policy-aborted and cannot be launched: {marker.get('reason', round_dir.name)}"
        )
    manifest = read_json(round_dir / "manifest.json")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if gpus != ["0", "1", "2", "3"]:
        raise SystemExit("GPU contract requires --gpus 0,1,2,3")
    if args.slots_per_gpu != 6:
        raise SystemExit("paper baseline contract requires --slots-per-gpu 6")

    lock_path = round_dir / ".launcher.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another launcher already owns {round_dir}")

    commands = [
        {"job_id": job["job_id"], "gpu": gpus[index % len(gpus)], "command": build_command(job, manifest, gpus[index % len(gpus)])}
        for index, job in enumerate(manifest["jobs"])
    ]
    if args.dry_run:
        write_json_atomic(round_dir / "dry_run_commands.json", commands)
        for item in commands:
            print(f"[{item['gpu']}] {' '.join(item['command'])}")
        print(f"[dry-run] jobs={len(commands)} max_concurrent={len(gpus) * args.slots_per_gpu}")
        return 0

    state_path = round_dir / "round_state.json"
    state = read_optional_json(state_path) or state_template(manifest)
    if state.get("round_id") != manifest["round_id"]:
        raise RuntimeError("round_state.json belongs to a different manifest")
    state.setdefault("jobs", {})
    for job in manifest["jobs"]:
        state["jobs"].setdefault(job["job_id"], {"state": "pending", "attempt": 0})

    adopted: Dict[str, RunningJob] = {}
    retry_after: Dict[str, float] = {}
    for job in manifest["jobs"]:
        entry = state["jobs"][job["job_id"]]
        pid = int(entry.get("pid", 0) or 0)
        gpu = str(entry.get("gpu")) if entry.get("gpu") is not None else None
        if not process_alive(pid, job["config_path"]):
            discovered_pid, discovered_gpu = find_process_for_config(job["config_path"])
            if discovered_pid is not None:
                pid = discovered_pid
                gpu = discovered_gpu or gpu
        if entry.get("state") in {"starting", "running", "pending"} and process_alive(pid, job["config_path"]):
            if gpu not in gpus:
                entry.update(
                    {
                        "state": "failed_integrity",
                        "message": f"could not recover physical GPU for live pid {pid}",
                    }
                )
                continue
            entry.update({"state": "running", "pid": pid, "gpu": gpu, "message": "live wrapper adopted"})
            adopted[job["job_id"]] = RunningJob(
                job=job,
                gpu=gpu,
                pid=pid,
                started_at_epoch=float(entry.get("started_at_epoch", time.time())),
            )
            print(f"[adopt] gpu={gpu} pid={pid} job={job['job_id']}", flush=True)
            continue

        # A failed attempt normally remains terminal so an accidental launcher
        # restart cannot silently rerun it.  The supervisor opts into recovery
        # explicitly after recording a round retry event.  Keep the old exit
        # code in audit metadata, clear the live classification, and skip the
        # output-based terminal check below; launch_job() archives those files
        # before starting the next attempt.
        if args.retry_failed and entry.get("state") in RECOVERABLE_RETRY_STATES:
            previous_state = entry.get("state")
            previous_exit = entry.get("exit_code")
            entry.update(
                {
                    "state": "pending",
                    "pid": None,
                    "message": (
                        f"explicit recovery retry requested from {previous_state}"
                    ),
                    "retry_requested_at": iso_now(),
                    "retry_previous_state": previous_state,
                    "retry_previous_exit_code": previous_exit,
                }
            )
            entry.pop("exit_code", None)
            continue

        has_outputs = any(
            Path(path).exists()
            for path in (job["summary_json"], job["status_json"], job["metrics_jsonl"], job["stdout_log"])
        )
        trusted_complete = entry.get("state") in {"complete", "complete_with_teardown_warning"}
        exit_code = recorded_exit_code(job)
        if exit_code is None and isinstance(entry.get("exit_code"), (int, float)):
            exit_code = int(entry["exit_code"])
        if exit_code is None and trusted_complete:
            exit_code = 139 if entry.get("state") == "complete_with_teardown_warning" else 0
        if completed_budget(job) or (has_outputs and exit_code is not None):
            outcome, message, retryable_oom = terminal_state(job, exit_code)
            if retryable_oom and int(entry.get("attempt", 0)) <= args.max_oom_retries:
                outcome = "pending"
                retry_after[job["job_id"]] = time.time() + args.oom_retry_delay_sec
                message += f"; retry scheduled after {args.oom_retry_delay_sec}s"
            entry.update(
                {
                    "state": outcome,
                    "exit_code": 139 if outcome == "complete_with_teardown_warning" and exit_code is None else exit_code,
                    "pid": None,
                    "message": f"resume classification: {message}",
                }
            )
            write_json_atomic(Path(job["job_run_dir"]) / "launcher_status.json", entry)
            continue
        if entry.get("state") in {"starting", "running"}:
            entry.update({"state": "pending", "pid": None, "message": "interrupted attempt recovered for restart"})

    if not adopted and not args.skip_gpu_preflight:
        gpu_preflight(gpus)

    write_json_atomic(
        round_dir / "launch_config.json",
        {
            "started_at": iso_now(),
            "gpus": gpus,
            "slots_per_gpu": args.slots_per_gpu,
            "max_concurrent_jobs": len(gpus) * args.slots_per_gpu,
            "poll_sec": args.poll_sec,
            "max_oom_retries": args.max_oom_retries,
            "oom_retry_delay_sec": args.oom_retry_delay_sec,
            "retry_failed": bool(args.retry_failed),
            "early_stop": False,
            "cuda_visible_devices": False,
            "launcher_sha256": sha256_file(Path(__file__).resolve()),
            "job_process_sha256": sha256_file(Path(__file__).with_name("job_process.py")),
            "manifest_sha256": sha256_file(round_dir / "manifest.json"),
        },
    )
    state["state"] = "running"
    state["launcher_pid"] = os.getpid()
    state["launcher_identity"] = capture_process_identity(os.getpid())
    state.setdefault("started_at", iso_now())
    persist_state(round_dir, state)

    jobs_by_id = {job["job_id"]: job for job in manifest["jobs"]}
    running = dict(adopted)
    terminal_failures = {
        "failed_nonzero",
        "failed_short",
        "failed_oom",
        "failed_exit_unknown",
        "failed_integrity",
    }

    while not STOP_REQUESTED:
        now = time.time()
        for job_id, active in list(running.items()):
            if active.proc is not None:
                return_code = active.proc.poll()
                alive = return_code is None
            else:
                alive = process_alive(active.pid, active.job["config_path"])
                return_code = None if alive else recorded_exit_code(active.job)
            if alive:
                continue
            finish_running_handle(active)
            entry = state["jobs"][job_id]
            outcome, message, retryable_oom = terminal_state(active.job, return_code)
            entry.update(
                {
                    "state": outcome,
                    "exit_code": 139 if outcome == "complete_with_teardown_warning" and return_code is None else return_code,
                    "finished_at": iso_now(),
                    "message": message,
                    "pid": None,
                }
            )
            if retryable_oom and int(entry.get("attempt", 0)) <= args.max_oom_retries:
                entry["state"] = "pending"
                retry_after[job_id] = now + args.oom_retry_delay_sec
                entry["message"] += f"; retry scheduled after {args.oom_retry_delay_sec}s"
            write_json_atomic(Path(active.job["job_run_dir"]) / "launcher_status.json", entry)
            print(f"[settled] state={entry['state']} job={job_id} {entry['message']}", flush=True)
            running.pop(job_id)

        used = {gpu: sum(1 for item in running.values() if item.gpu == gpu) for gpu in gpus}
        free_slots: list[str] = []
        for _ in range(args.slots_per_gpu):
            for gpu in gpus:
                if used[gpu] < args.slots_per_gpu:
                    free_slots.append(gpu)
                    used[gpu] += 1

        pending_ids = [
            job["job_id"]
            for job in manifest["jobs"]
            if state["jobs"][job["job_id"]].get("state") == "pending"
            and retry_after.get(job["job_id"], 0.0) <= now
        ]
        while pending_ids and free_slots:
            job_id = pending_ids.pop(0)
            gpu = free_slots.pop(0)
            try:
                active = launch_job(
                    jobs_by_id[job_id],
                    manifest,
                    gpu,
                    state["jobs"][job_id],
                    round_dir,
                    state,
                )
                running[job_id] = active
            except Exception as exc:
                entry = state["jobs"][job_id]
                entry.update(
                    {
                        "state": "failed_integrity",
                        "message": str(exc),
                        "finished_at": iso_now(),
                    }
                )
                write_json_atomic(Path(jobs_by_id[job_id]["job_run_dir"]) / "launcher_status.json", entry)
                print(f"[integrity-failure] job={job_id} error={exc}", flush=True)

        persist_state(round_dir, state)
        nonterminal = [
            entry
            for entry in state["jobs"].values()
            if entry.get("state") in {"pending", "starting", "running"}
        ]
        if not nonterminal and not running:
            break
        time.sleep(max(0.1, args.poll_sec))

    if STOP_REQUESTED:
        state["state"] = "launcher_interrupted_jobs_left_running"
        state["last_launcher_identity"] = state.get("launcher_identity")
        state["launcher_pid"] = None
        state["launcher_identity"] = None
        state["message"] = "launcher stopped; detached training jobs were intentionally left running"
        persist_state(round_dir, state)
        print("[interrupt] launcher exited without terminating training jobs", flush=True)
        return 130

    for job in manifest["jobs"]:
        try:
            verify_integrity(job, manifest)
        except Exception as exc:
            entry = state["jobs"][job["job_id"]]
            entry.update(
                {
                    "state": "failed_integrity",
                    "message": str(exc),
                    "finished_at": iso_now(),
                }
            )
            write_json_atomic(Path(job["job_run_dir"]) / "launcher_status.json", entry)
    failed = [entry for entry in state["jobs"].values() if entry.get("state") in terminal_failures]
    state["state"] = "failed" if failed else "complete"
    state["last_launcher_identity"] = state.get("launcher_identity")
    state["launcher_pid"] = None
    state["launcher_identity"] = None
    state["finished_at"] = iso_now()
    persist_state(round_dir, state)
    write_round_summary(round_dir, manifest, state)
    print(f"[round-settled] state={state['state']} jobs={len(manifest['jobs'])}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    raise SystemExit(main())
