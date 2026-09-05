#!/usr/bin/env python3
"""Persistent train -> score -> GPT-plan loop for the twelve paper tasks."""

from __future__ import annotations

import argparse
import atexit
import copy
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from common import (
    ALL_TUNABLE_PARAMETERS,
    CONFIG_DIR,
    ENVIRONMENTS,
    EXPERIMENTS_DIR,
    FIXED_PARAMETERS,
    PAPER_TARGETS_PATH,
    ROOT,
    SEEDS,
    STATE_DIR,
    iso_now,
    read_json,
    sha256_payload,
    write_json_compact_atomic,
    write_json_atomic,
    write_text_atomic,
)
from unification import (
    DEFAULT_UNIFIED_PARAMETERS,
    REWARD_SCALE_PARAMETERS,
    evaluate_unification_candidate,
)
from process_identity import verify_process_identity
from validate_plan import (
    PARAMETER_FAMILIES,
    PARAMETER_TO_FAMILY,
    SEARCH_GRID,
    SEARCH_MAX_DECIMAL_PLACES,
    SEARCH_PRECISION_EXEMPT_PARAMETERS,
    canonicalize_behavioral_tunables,
    tunable_precision_violations,
)
from approved_task_plan import ApprovedTaskError, apply_pending_task


SCRIPT_DIR = Path(__file__).resolve().parent
LOOP_STATE_PATH = STATE_DIR / "loop_state.json"
CURRENT_CONFIGS_PATH = STATE_DIR / "current_configs.json"
HISTORY_PATH = STATE_DIR / "historical_unique_cohorts.json"
PLANNER_HISTORY_PATH = STATE_DIR / "planner_history_index.json"
FINAL_REPORT_PATH = EXPERIMENTS_DIR / "final_target_report.md"
FINAL_CONFIGS_PATH = STATE_DIR / "final_configs.json"
FINAL_UNIFICATION_REPORT_PATH = EXPERIMENTS_DIR / "final_unification_report.md"
CONTINUOUS_PID_PATH = STATE_DIR / "continuous_tune.pid"
_SUPERVISOR_OWNS_PID = False
_SUPERVISOR_STOP_REASON: Optional[str] = None
_LOOP_STATE_LOCK = threading.RLock()
_HEARTBEAT_STATE_KEYS = frozenset(
    {
        "heartbeat_at",
        "heartbeat_round_id",
        "heartbeat_launcher_pid",
        "phase",
        "phase_pid",
        "phase_heartbeat_at",
        "last_phase",
    }
)

# Round failures are recoverable orchestration events by default.  The
# explicit --stop-on-incomplete switch retains the old terminal-blocking
# behavior for operators who want a fail-fast run.
ROUND_RETRY_DEFAULT_DELAY_SECONDS = 120.0
ROUND_RETRY_HISTORY_LIMIT = 100


def _cleanup_supervisor_state() -> None:
    """Remove stale PID state and leave an explicit lifecycle marker."""

    global _SUPERVISOR_OWNS_PID
    if not _SUPERVISOR_OWNS_PID:
        return
    try:
        state = read_json(LOOP_STATE_PATH) if LOOP_STATE_PATH.exists() else {}
        terminal = {
            "complete",
            "blocked",
            "paused_manual_user_abort",
            "stopped_supervisor",
        }
        if (
            isinstance(state, dict)
            and state.get("supervisor_pid") == os.getpid()
            and state.get("state") not in terminal
        ):
            state["state"] = "stopped_supervisor"
            state["stop_reason"] = _SUPERVISOR_STOP_REASON or "supervisor exited"
            state["stopped_at"] = iso_now()
            write_json_atomic(LOOP_STATE_PATH, state)
    except Exception:
        # Cleanup must never mask the original exception or exit status.
        pass
    try:
        if CONTINUOUS_PID_PATH.is_file():
            contents = CONTINUOUS_PID_PATH.read_text(encoding="utf-8").strip()
            if contents == str(os.getpid()):
                CONTINUOUS_PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    _SUPERVISOR_OWNS_PID = False


def _handle_supervisor_signal(signum: int, _frame: Any) -> None:
    global _SUPERVISOR_STOP_REASON
    _SUPERVISOR_STOP_REASON = f"received signal {signum}"
    raise SystemExit(128 + signum)


atexit.register(_cleanup_supervisor_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=6)
    parser.add_argument("--continue-after-stagnation", action="store_true")
    parser.add_argument(
        "--stop-on-incomplete",
        action="store_true",
        help="Opt in to the old behavior that blocks at soft limits or planner failure.",
    )
    parser.add_argument(
        "--planner-retry-delay-sec",
        type=float,
        default=float(os.environ.get("FLOWCRITIC_PLANNER_RETRY_DELAY_SEC", "300")),
        help="Delay between failed four-attempt GPT planner cycles (default: 300s).",
    )
    parser.add_argument(
        "--round-retry-delay-sec",
        type=float,
        default=float(
            os.environ.get(
                "FLOWCRITIC_ROUND_RETRY_DELAY_SEC",
                str(ROUND_RETRY_DEFAULT_DELAY_SECONDS),
            )
        ),
        help="Delay before retrying a failed launcher/score/diagnostic phase (default: 120s).",
    )
    parser.add_argument("--once", action="store_true", help="Settle at most one round, then exit cleanly.")
    parser.add_argument("--skip-gpu-preflight", action="store_true")
    return parser.parse_args()


def run_checked(command: list[str], *, log_path: Optional[Path] = None) -> int:
    if log_path is None:
        completed = subprocess.run(command, cwd=ROOT, check=False)
        return completed.returncode
    log_path.parent.mkdir(parents=True, exist_ok=True)
    round_id = phase_round_id(log_path)
    phase = phase_name(log_path)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(f"\n[{iso_now()}] command: {json.dumps(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        stop_event = threading.Event()
        heartbeat_thread: Optional[threading.Thread] = None
        if round_id is not None:
            persist_phase_heartbeat(round_id, phase, process.pid)
            heartbeat_thread = threading.Thread(
                target=phase_heartbeat_worker,
                args=(round_id, phase, process.pid, stop_event),
                name="continuous-tune-phase-heartbeat",
                daemon=True,
            )
            heartbeat_thread.start()
        return_code: Optional[int] = None
        try:
            return_code = int(process.wait())
        finally:
            stop_event.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=2.0)
            if round_id is not None:
                clear_phase_heartbeat(round_id, phase, process.pid, return_code)
        handle.write(f"[{iso_now()}] exit_code={return_code}\n")
        return return_code


def phase_round_id(log_path: Path | str) -> Optional[str]:
    """Extract an experiment round ID from a phase log path."""

    try:
        relative = Path(log_path).resolve().relative_to(EXPERIMENTS_DIR.resolve())
    except (OSError, ValueError):
        return None
    if not relative.parts or not relative.parts[0].startswith("round_"):
        return None
    return relative.parts[0]


def phase_name(log_path: Path | str) -> str:
    name = Path(log_path).name.lower()
    if "gpt" in name:
        return "gpt_planner"
    if "scor" in name:
        return "scoring"
    if "material" in name:
        return "materialization"
    if "launcher" in name:
        return "launcher"
    return "orchestration"


def append_supervisor_log(message: str) -> None:
    """Append a compact lifecycle/progress line to the durable supervisor log."""

    try:
        path = Path(LOOP_STATE_PATH).with_name("continuous_tune.supervisor.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(f"[{iso_now()}] {message}\n")
    except OSError:
        return


def heartbeat_log_line(round_id: str, launcher_pid: Any, phase: Optional[str] = None) -> str:
    counts: Mapping[str, Any] = {}
    round_state_path = EXPERIMENTS_DIR / round_id / "round_state.json"
    try:
        round_state = read_json(round_state_path)
        raw_counts = round_state.get("counts", {})
        if isinstance(raw_counts, Mapping):
            counts = raw_counts
    except (OSError, json.JSONDecodeError):
        pass
    phase_text = f" phase={phase}" if phase else ""
    return f"heartbeat round={round_id} launcher={launcher_pid}{phase_text} counts={dict(counts)}"


def persist_phase_heartbeat(round_id: str, phase: str, phase_pid: Any) -> None:
    """Publish a heartbeat while scoring/planning/materialization is blocked."""

    try:
        with _LOOP_STATE_LOCK:
            state = read_json(LOOP_STATE_PATH) if LOOP_STATE_PATH.exists() else {}
            if state.get("supervisor_pid") != os.getpid():
                return
            now = iso_now()
            state.update(
                {
                    "heartbeat_at": now,
                    "heartbeat_round_id": round_id,
                    "phase": phase,
                    "phase_pid": phase_pid,
                    "phase_heartbeat_at": now,
                }
            )
            persist_loop_state(state)
            append_supervisor_log(heartbeat_log_line(round_id, phase_pid, phase))
    except Exception:
        return


def phase_heartbeat_worker(
    round_id: str, phase: str, phase_pid: Any, stop_event: threading.Event
) -> None:
    while not stop_event.wait(30.0):
        persist_phase_heartbeat(round_id, phase, phase_pid)


def clear_phase_heartbeat(
    round_id: str, phase: str, phase_pid: Any, return_code: Optional[int]
) -> None:
    try:
        with _LOOP_STATE_LOCK:
            state = read_json(LOOP_STATE_PATH) if LOOP_STATE_PATH.exists() else {}
            if state.get("supervisor_pid") != os.getpid():
                return
            if state.get("phase") != phase or state.get("phase_pid") != phase_pid:
                return
            state["last_phase"] = {
                "round_id": round_id,
                "phase": phase,
                "phase_pid": phase_pid,
                "finished_at": iso_now(),
                "return_code": return_code,
            }
            for key in ("phase", "phase_pid", "phase_heartbeat_at"):
                state.pop(key, None)
            persist_loop_state(state, preserve_heartbeat=False)
    except Exception:
        return


def launcher_adoption_status(
    round_dir: Path,
    round_state: Mapping[str, Any],
    expected_identity: Optional[Mapping[str, Any]] = None,
) -> tuple[bool, str, Optional[Dict[str, Any]]]:
    """Return whether a persisted launcher record still names this round."""

    if round_state.get("state") != "running":
        return False, f"round state is {round_state.get('state')!r}, not running", None
    identity = expected_identity
    if identity is None:
        stored = round_state.get("launcher_identity")
        identity = stored if isinstance(stored, Mapping) else None
    return verify_process_identity(
        round_state.get("launcher_pid"),
        identity,
        required_paths=(SCRIPT_DIR / "launch_round.py", round_dir.resolve()),
    )


def persist_supervisor_heartbeat(round_id: str, launcher_pid: Any) -> None:
    """Publish liveness while the detached launcher owns the long wait.

    The launcher owns ``round_state.json`` while jobs are running, so writing
    that file from the supervisor would race with its scheduler updates.  The
    loop state is supervisor-owned and is therefore the safe heartbeat target.
    """

    try:
        with _LOOP_STATE_LOCK:
            state = read_json(LOOP_STATE_PATH) if LOOP_STATE_PATH.exists() else {}
            if state.get("supervisor_pid") != os.getpid():
                return
            state["heartbeat_at"] = iso_now()
            state["heartbeat_round_id"] = round_id
            state["heartbeat_launcher_pid"] = launcher_pid
            if state.get("last_round_error") and not state.get("round_retry_pending"):
                state["active_round_retry"] = {
                    "round_id": round_id,
                    "retry_count": state.get("round_retry_count"),
                    "recovered_error": state.pop("last_round_error"),
                    "recovered_at": iso_now(),
                }
            persist_loop_state(state)
            append_supervisor_log(heartbeat_log_line(round_id, launcher_pid))
    except Exception:
        # A heartbeat must never interrupt a healthy training round.  The next
        # scheduler/status write will make the transient filesystem error
        # visible without converting it into a false terminal failure.
        return


def launch_or_wait_for_round(
    round_dir: Path,
    args: argparse.Namespace,
    *,
    retry_failed: bool = False,
) -> int:
    """Resume an existing detached launcher, or start a new one."""

    round_state_path = round_dir / "round_state.json"
    round_state = read_json(round_state_path) if round_state_path.exists() else {}
    launcher_pid = round_state.get("launcher_pid")
    adoptable, adoption_reason, live_identity = launcher_adoption_status(round_dir, round_state)
    if adoptable:
        log_path = round_dir / "launcher.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(
                f"[{iso_now()}] supervisor adopted launcher pid={launcher_pid}; "
                f"{adoption_reason}\n"
            )
        last_heartbeat = 0.0
        while True:
            current_state = dict(round_state)
            current_state["state"] = "running"
            still_owned, exit_reason, _ = launcher_adoption_status(
                round_dir,
                current_state,
                live_identity,
            )
            if not still_owned:
                with log_path.open("a", encoding="utf-8", buffering=1) as handle:
                    handle.write(
                        f"[{iso_now()}] adopted launcher pid={launcher_pid} no longer owned; "
                        f"{exit_reason}\n"
                    )
                break
            now = time.time()
            if now - last_heartbeat >= 30.0:
                persist_supervisor_heartbeat(round_dir.name, launcher_pid)
                last_heartbeat = now
            time.sleep(10.0)
        persist_supervisor_heartbeat(round_dir.name, launcher_pid)
        settled = read_json(round_state_path)
        return 0 if settled.get("state") == "complete" else 1

    if round_state.get("state") == "running" and launcher_pid:
        log_path = round_dir / "launcher.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(
                f"[{iso_now()}] stale launcher record rejected pid={launcher_pid}; "
                f"{adoption_reason}\n"
            )

    launch_command = [
        sys.executable,
        str(SCRIPT_DIR / "launch_round.py"),
        str(round_dir),
        "--gpus",
        args.gpus,
        "--slots-per-gpu",
        str(args.slots_per_gpu),
    ]
    if args.skip_gpu_preflight:
        launch_command.append("--skip-gpu-preflight")
    if retry_failed:
        launch_command.append("--retry-failed")
    return run_launcher_with_heartbeat(launch_command, round_dir)


def run_launcher_with_heartbeat(command: list[str], round_dir: Path) -> int:
    """Run a launcher without coupling its lifetime to the supervisor.

    ``subprocess.run`` kills its child when the waiting Python frame is
    interrupted.  That made a supervisor-only restart also kill the launcher,
    even though the detached training jobs survived.  Keep a plain ``Popen``
    handle instead: the supervisor can exit, the launcher remains adoptable,
    and normal waits publish the same heartbeat as the adoption path.
    """

    log_path = round_dir / "launcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    process: Optional[subprocess.Popen[str]] = None
    try:
        handle.write(f"\n[{iso_now()}] command: {json.dumps(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_heartbeat = 0.0
        while True:
            return_code = process.poll()
            if return_code is not None:
                persist_supervisor_heartbeat(round_dir.name, process.pid)
                handle.write(f"[{iso_now()}] exit_code={return_code}\n")
                return int(return_code)
            now = time.time()
            if now - last_heartbeat >= 30.0:
                persist_supervisor_heartbeat(round_dir.name, process.pid)
                last_heartbeat = now
            time.sleep(10.0)
    finally:
        # The child owns a duplicate of the descriptor.  Closing the parent's
        # copy does not terminate or detach the launcher after a supervisor
        # signal; a replacement supervisor will adopt it from round_state.json.
        handle.close()


def initial_state(max_rounds: int) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "initializing",
        "started_at": iso_now(),
        "updated_at": iso_now(),
        "supervisor_pid": os.getpid(),
        "current_round_id": None,
        "rounds_completed": 0,
        "next_round_index": 1,
        "max_rounds": max_rounds,
        "stage": 1,
        "global_stagnation_count": 0,
        "env_stagnation_counts": {},
        "incumbents": {},
        "unification": {"state": "not_started", "attempts": []},
        "round_history": [],
    }


def persist_loop_state(
    state: Dict[str, Any], *, preserve_heartbeat: bool = True
) -> None:
    # Heartbeat writers update the durable file while the main loop is waiting
    # for a launcher/planner.  The main loop intentionally keeps a richer
    # in-memory snapshot, so blindly writing it here used to erase a fresh
    # phase/last_phase marker.  Preserve a newer durable heartbeat even when
    # the caller's snapshot contains an older value (the common case after a
    # phase subprocess returns).  ``preserve_heartbeat=False`` is reserved for
    # explicit lifecycle operations such as ``clear_phase_heartbeat``.
    with _LOOP_STATE_LOCK:
        payload = dict(state)
        try:
            on_disk = read_json(LOOP_STATE_PATH) if LOOP_STATE_PATH.exists() else {}
            if preserve_heartbeat and (
                isinstance(on_disk, dict)
                and on_disk.get("supervisor_pid") == os.getpid()
            ):
                def _timestamp(value: Any) -> Optional[float]:
                    if not isinstance(value, str) or not value:
                        return None
                    try:
                        # ``fromisoformat`` on Python 3.8 does not accept the
                        # RFC-3339 ``Z`` suffix, while tests and old state
                        # snapshots may contain it.
                        from datetime import datetime

                        return datetime.fromisoformat(
                            value.replace("Z", "+00:00")
                        ).timestamp()
                    except (TypeError, ValueError, OverflowError):
                        return None

                def _disk_is_newer(disk_value: Any, state_value: Any) -> bool:
                    disk_time = _timestamp(disk_value)
                    state_time = _timestamp(state_value)
                    return disk_time is not None and (
                        state_time is None or disk_time > state_time
                    )

                # Supervisor/phase heartbeats share ``heartbeat_at`` but the
                # phase marker has its own timestamp.  Merge each event as a
                # group so its associated round/pid fields stay consistent.
                if _disk_is_newer(
                    on_disk.get("heartbeat_at"), state.get("heartbeat_at")
                ):
                    for key in (
                        "heartbeat_at",
                        "heartbeat_round_id",
                        "heartbeat_launcher_pid",
                    ):
                        if key in on_disk:
                            payload[key] = on_disk[key]
                        else:
                            payload.pop(key, None)
                if _disk_is_newer(
                    on_disk.get("phase_heartbeat_at"),
                    state.get("phase_heartbeat_at"),
                ):
                    for key in ("phase", "phase_pid", "phase_heartbeat_at"):
                        if key in on_disk:
                            payload[key] = on_disk[key]
                        else:
                            payload.pop(key, None)
                disk_last_phase = on_disk.get("last_phase")
                state_last_phase = state.get("last_phase")
                disk_finished = (
                    disk_last_phase.get("finished_at")
                    if isinstance(disk_last_phase, Mapping)
                    else None
                )
                state_finished = (
                    state_last_phase.get("finished_at")
                    if isinstance(state_last_phase, Mapping)
                    else None
                )
                if _disk_is_newer(disk_finished, state_finished):
                    if "last_phase" in on_disk:
                        payload["last_phase"] = disk_last_phase
                    else:
                        payload.pop("last_phase", None)

                # ``clear_phase_heartbeat`` deliberately removes the phase
                # keys after recording ``last_phase``.  A stale main-loop
                # snapshot may still contain the old phase and would
                # otherwise resurrect it on its next state write.  Treat the
                # durable absence of a phase, together with a lifecycle
                # timestamp at least as new as the caller's phase timestamp,
                # as an explicit removal.  A genuinely newer phase heartbeat
                # in the caller is retained.
                if "phase" not in on_disk and "phase" in state:
                    durable_phase_times = [
                        _timestamp(on_disk.get("heartbeat_at")),
                        _timestamp(on_disk.get("phase_heartbeat_at")),
                        _timestamp(disk_finished),
                    ]
                    durable_phase_time = max(
                        (value for value in durable_phase_times if value is not None),
                        default=None,
                    )
                    caller_phase_time = _timestamp(state.get("phase_heartbeat_at"))
                    # With no durable lifecycle timestamp at all, the caller
                    # may be publishing the very first phase heartbeat; keep
                    # it.  Once a durable timestamp exists, an absent phase
                    # is authoritative whenever it is newer than (or equal
                    # to) the caller's stale phase timestamp.
                    if durable_phase_time is not None and (
                        caller_phase_time is None
                        or durable_phase_time >= caller_phase_time
                    ):
                        for key in ("phase", "phase_pid", "phase_heartbeat_at"):
                            payload.pop(key, None)
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        payload["updated_at"] = iso_now()
        payload["supervisor_pid"] = os.getpid()
        payload["supervisor_sha256"] = file_sha256(Path(__file__).resolve())
        write_json_atomic(LOOP_STATE_PATH, payload)


def tunable_snapshot(config: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: config[key] for key in sorted(ALL_TUNABLE_PARAMETERS) if key in config}


def config_precision_compliant(config: Mapping[str, Any]) -> bool:
    return not tunable_precision_violations(config)


def incumbent_precision_compliant(record: Mapping[str, Any]) -> bool:
    config = record.get("tunable_config", {})
    return isinstance(config, Mapping) and config_precision_compliant(config)


def precision_migration_parameters(record: Mapping[str, Any]) -> list[str]:
    config = record.get("tunable_config", {})
    if not isinstance(config, Mapping):
        return []
    return [
        key
        for key in sorted(set(config) & ALL_TUNABLE_PARAMETERS)
        if tunable_precision_violations({key: config[key]}, f"incumbent.{key}")
    ]


def round_search_families(round_id: str, env: str) -> set[str]:
    manifest_path = EXPERIMENTS_DIR / round_id / "manifest.json"
    if not manifest_path.is_file():
        return set()
    manifest = read_json(manifest_path)
    families: set[str] = set()
    for job in manifest.get("jobs", []):
        if not isinstance(job, Mapping) or str(job.get("env")) != env:
            continue
        overrides = job.get("overrides", {})
        if not isinstance(overrides, Mapping):
            continue
        families.update(
            PARAMETER_TO_FAMILY[key]
            for key in overrides
            if key in PARAMETER_TO_FAMILY
        )
    return families


def diversification_constraints(
    state: Mapping[str, Any], target_envs: Iterable[str], lookback: int = 3
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    history = [
        record
        for record in state.get("round_history", [])
        if isinstance(record, Mapping) and record.get("purpose") == "tuning"
    ]
    stagnation = state.get("env_stagnation_counts", {})
    for env in target_envs:
        recent: list[Dict[str, Any]] = []
        family_counts: Dict[str, int] = defaultdict(int)
        for record in reversed(history):
            round_id = str(record.get("round_id", ""))
            families = round_search_families(round_id, env)
            if not families:
                continue
            recent.append({"round_id": round_id, "families": sorted(families)})
            for family in families:
                family_counts[family] += 1
            if len(recent) >= lookback:
                break
        cooldown = sorted(
            family for family, count in family_counts.items() if count >= 2
        )
        cooldown_before_reset = list(cooldown)
        released_families: list[str] = []
        cooldown_exhausted_reset = False
        # If every searchable family is cooled, requiring a change outside
        # the cooled set is unsatisfiable: there is no legal proposal the GPT
        # planner can emit.  Release the least-used family (or all families
        # when their recent counts are tied) so the loop can continue while
        # still recording the policy relaxation explicitly.  The normal
        # no-duplicate and precision validators remain authoritative.
        all_families = set(PARAMETER_FAMILIES)
        if all_families and all_families.issubset(cooldown):
            minimum_count = min(family_counts[family] for family in cooldown)
            released_families = sorted(
                family
                for family in cooldown
                if family_counts[family] == minimum_count
            )
            cooldown = [
                family for family in cooldown if family not in released_families
            ]
            cooldown_exhausted_reset = True
        required = int(stagnation.get(env, 0) or 0) >= 2 and bool(cooldown)
        if cooldown_exhausted_reset and not cooldown:
            rule = (
                "All parameter families were cooled with equal recent usage; "
                "the cooldown was fully reset. No family cooldown is currently required."
            )
        elif required and cooldown_exhausted_reset:
            rule = (
                "All parameter families were cooled; the least-used families "
                f"{released_families} were released. Every variant must make a real "
                "change in another family outside the remaining cooldown_families."
            )
        elif required:
            rule = (
                "Every variant must make a real change in at least one parameter family "
                "outside cooldown_families. Adding an unchanged key does not count."
            )
        else:
            rule = "No family cooldown is currently required."
        result[env] = {
            "required": required,
            "stagnation_count": int(stagnation.get(env, 0) or 0),
            "cooldown_families": cooldown if required else [],
            "cooldown_exhausted_reset": cooldown_exhausted_reset,
            "released_families": released_families,
            "cooldown_families_before_reset": cooldown_before_reset
            if cooldown_exhausted_reset
            else [],
            "family_search_counts": {
                family: int(family_counts.get(family, 0))
                for family in sorted(PARAMETER_FAMILIES)
            },
            "recent_searches": recent,
            "rule": rule,
        }
    return result


def history_config_signature(
    tunables: Mapping[str, Any], fixed_contract: Mapping[str, Any] = FIXED_PARAMETERS
) -> str:
    return sha256_payload(
        {
            "fixed_contract": {key: fixed_contract[key] for key in sorted(fixed_contract)},
            "tunables": canonicalize_behavioral_tunables(tunables),
        }
    )


def load_manifest_configs(round_dir: Path) -> Dict[tuple[str, str], Dict[str, Any]]:
    manifest = read_json(round_dir / "manifest.json")
    result: Dict[tuple[str, str], Dict[str, Any]] = {}
    for job in manifest["jobs"]:
        key = (str(job["env"]), str(job["variant"]))
        if key not in result:
            result[key] = tunable_snapshot(read_json(job["config_path"]))
    return result


def file_sha256(path: Path) -> Optional[str]:
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


def append_history(round_dir: Path, score_payload: Mapping[str, Any]) -> None:
    history = read_json(HISTORY_PATH) if HISTORY_PATH.exists() else {"schema_version": 1, "cohorts": []}
    cohorts = history.setdefault("cohorts", [])
    manifest = read_json(round_dir / "manifest.json")
    jobs = manifest["jobs"]
    config_by_key = load_manifest_configs(round_dir)
    config_index: Dict[tuple[str, str], Dict[str, Any]] = {}
    reward_index: Dict[tuple[str, tuple[tuple[str, Any], ...]], Dict[str, Any]] = {}
    observation_keys: set[tuple[str, str, str, str]] = set()
    for record in cohorts:
        if not isinstance(record, dict):
            continue
        env = str(record.get("env"))
        config_signature = str(
            record.get("config_signature")
            or history_config_signature(
                record.get("resolved_tunables", {}),
                record.get("fixed_contract", FIXED_PARAMETERS),
            )
        )
        config_index[(env, config_signature)] = record
        reward_hashes = record.get("reward_sha256")
        if isinstance(reward_hashes, dict):
            reward_index[(env, tuple(sorted(reward_hashes.items())))] = record
        observation_keys.add(
            (str(record.get("round_id")), env, str(record.get("variant")), config_signature)
        )
        for duplicate in record.get("duplicate_observations", []):
            if isinstance(duplicate, dict):
                observation_keys.add(
                    (
                        str(duplicate.get("round_id")),
                        env,
                        str(duplicate.get("variant")),
                        str(duplicate.get("config_signature") or config_signature),
                    )
                )
    for result in score_payload.get("results", []):
        if not result.get("complete"):
            continue
        env = str(result["env"])
        variant = str(result["variant"])
        tunables = config_by_key[(env, variant)]
        config_signature = history_config_signature(tunables)
        matching_jobs = [job for job in jobs if job["env"] == env and job["variant"] == variant]
        reward_hashes = {
            str(job["seed"]): file_sha256(Path(job["rewards_path"])) for job in matching_jobs
        }
        reward_key = (env, tuple(sorted(reward_hashes.items())))
        observation_key = (manifest["round_id"], env, variant, config_signature)
        if observation_key in observation_keys:
            continue
        record = {
            "env": env,
            "variant": variant,
            "round_id": manifest["round_id"],
            "resolved_tunables": tunables,
            "fixed_contract": dict(FIXED_PARAMETERS),
            "config_signature": config_signature,
            "top5_mean": result.get("top5_mean"),
            "top5_std": result.get("top5_std"),
            "all_bin_mean": result.get("all_bin_mean"),
            "paper_ratio": result.get("ratio"),
            "seed_scores": {
                str(seed.get("seed")): seed.get("top5_tail_mean") for seed in result.get("seeds", [])
            },
            "reward_sha256": reward_hashes,
            "reward_fingerprint": sha256_payload(reward_hashes),
            "recorded_at": iso_now(),
            "duplicate_observations": [],
        }
        prior = config_index.get((env, config_signature)) or reward_index.get(reward_key)
        if prior is not None:
            reasons: list[str] = []
            if (env, config_signature) in config_index:
                reasons.append("same_resolved_tunables_and_fixed_contract")
            if reward_key in reward_index:
                reasons.append("repeated_reward_sha256")
            prior.setdefault("duplicate_observations", []).append(
                {
                    "round_id": manifest["round_id"],
                    "variant": variant,
                    "config_signature": config_signature,
                    "resolved_tunables": tunables,
                    "reward_sha256": reward_hashes,
                    "reward_fingerprint": record["reward_fingerprint"],
                    "duplicate_reasons": reasons,
                    "top5_mean": result.get("top5_mean"),
                    "top5_std": result.get("top5_std"),
                    "paper_ratio": result.get("ratio"),
                }
            )
        else:
            cohorts.append(record)
            config_index[(env, config_signature)] = record
            reward_index[reward_key] = record
        observation_keys.add(observation_key)
    history["updated_at"] = iso_now()
    history["unique_config_count"] = len(cohorts)
    history.setdefault("stats", {})["independent_unique_cohorts"] = len(cohorts)
    history["stats"]["duplicate_observations"] = sum(
        len(record.get("duplicate_observations", [])) for record in cohorts if isinstance(record, dict)
    )
    write_json_compact_atomic(HISTORY_PATH, history)
    write_planner_history_index(history)


def write_planner_history_index(history: Optional[Mapping[str, Any]] = None) -> None:
    if history is None:
        history = read_json(HISTORY_PATH) if HISTORY_PATH.exists() else {"cohorts": []}
    records = []
    for raw in history.get("cohorts", []):
        if not isinstance(raw, dict):
            continue
        records.append(
            {
                key: raw.get(key)
                for key in (
                    "env",
                    "variant",
                    "round_id",
                    "resolved_tunables",
                    "fixed_contract",
                    "config_signature",
                    "top5_mean",
                    "top5_std",
                    "paper_ratio",
                    "evidence_status",
                )
            }
        )
    write_json_compact_atomic(
        PLANNER_HISTORY_PATH,
        {
            "schema_version": 1,
            "generated_at": iso_now(),
            "source": str(HISTORY_PATH),
            "unique_config_count": len(records),
            "cohorts": records,
        },
    )


def build_historical_comparison(
    round_dir: Path, score_payload: Mapping[str, Any]
) -> Dict[tuple[str, str], Dict[str, Any]]:
    history = read_json(HISTORY_PATH) if HISTORY_PATH.exists() else {"cohorts": []}
    best_by_env: Dict[str, Dict[str, Any]] = {}
    for raw in history.get("cohorts", []):
        if not isinstance(raw, dict) or raw.get("env") not in ENVIRONMENTS:
            continue
        try:
            score = float(raw.get("top5_mean"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        env = str(raw["env"])
        previous = best_by_env.get(env)
        if previous is None or score > float(previous["top5_mean"]):
            best_by_env[env] = dict(raw)

    comparison: Dict[tuple[str, str], Dict[str, Any]] = {}
    rows: list[Dict[str, Any]] = []
    for raw in score_payload.get("results", []):
        if not isinstance(raw, dict):
            continue
        env = str(raw.get("env", ""))
        variant = str(raw.get("variant", ""))
        candidate = raw.get("top5_mean")
        try:
            candidate_score = float(candidate) if candidate is not None else None
        except (TypeError, ValueError):
            candidate_score = None
        if candidate_score is not None and not math.isfinite(candidate_score):
            candidate_score = None
        historical = best_by_env.get(env)
        historical_score = float(historical["top5_mean"]) if historical else None
        delta = (
            candidate_score - historical_score
            if candidate_score is not None and historical_score is not None
            else None
        )
        ratio = (
            candidate_score / historical_score
            if candidate_score is not None and historical_score not in (None, 0.0)
            else None
        )
        record = {
            "env": env,
            "variant": variant,
            "candidate_top5_mean": candidate_score,
            "candidate_complete": bool(raw.get("complete")),
            "historical_best_top5_mean": historical_score,
            "historical_best_top5_std": historical.get("top5_std") if historical else None,
            "historical_best_round_id": historical.get("round_id") if historical else None,
            "historical_best_variant": historical.get("variant") if historical else None,
            "historical_best_config_signature": historical.get("config_signature") if historical else None,
            "delta_vs_historical_best": delta,
            "ratio_vs_historical_best": ratio,
            "better_than_historical_best": (
                candidate_score > historical_score
                if candidate_score is not None and historical_score is not None
                else None
            ),
        }
        comparison[(env, variant)] = record
        rows.append(record)
    payload = {
        "schema_version": 1,
        "generated_at": iso_now(),
        "round_id": round_dir.name,
        "comparison_basis": "independent unique historical cohorts before this round was appended",
        "results": rows,
    }
    write_json_atomic(round_dir / "historical_unique_comparison.json", payload)
    lines = [
        "# Historical unique-cohort comparison",
        "",
        "| Environment | Variant | Candidate | Historical best | Delta | Better | Historical source |",
        "| --- | --- | ---: | ---: | ---: | :---: | --- |",
    ]
    for row in rows:
        better = row.get("better_than_historical_best")
        lines.append(
            f"| {row['env']} | {row['variant']} | {format_number(row.get('candidate_top5_mean'))} | "
            f"{format_number(row.get('historical_best_top5_mean'))} | "
            f"{format_number(row.get('delta_vs_historical_best'))} | "
            f"{'yes' if better is True else 'no' if better is False else '-'} | "
            f"{row.get('historical_best_round_id') or '-'} / {row.get('historical_best_variant') or '-'} |"
        )
    write_text_atomic(
        round_dir / "historical_unique_comparison.md",
        "\n".join(lines) + "\n",
    )
    return comparison


def attach_historical_comparison(
    state: Dict[str, Any],
    comparison: Mapping[tuple[str, str], Mapping[str, Any]],
    accepted_keys: Optional[set[tuple[str, str]]] = None,
) -> None:
    for env, incumbent in state.get("incumbents", {}).items():
        key = (str(env), str(incumbent.get("variant", "")))
        if accepted_keys is not None and key not in accepted_keys:
            continue
        row = comparison.get(key)
        if not row:
            continue
        incumbent["historical_best_before"] = row.get("historical_best_top5_mean")
        incumbent["delta_vs_historical_best"] = row.get("delta_vs_historical_best")
        incumbent["better_than_historical_best"] = row.get("better_than_historical_best")
        incumbent["historical_best_round_id"] = row.get("historical_best_round_id")
        incumbent["historical_best_variant"] = row.get("historical_best_variant")


def update_incumbents(
    state: Dict[str, Any], round_dir: Path, score_payload: Mapping[str, Any]
) -> tuple[Dict[str, Optional[float]], list[Dict[str, Any]]]:
    configs = load_manifest_configs(round_dir)
    candidates_by_env: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for raw in score_payload.get("results", []):
        if raw.get("complete") and raw.get("top5_mean") is not None:
            candidates_by_env[str(raw["env"])].append(dict(raw))

    ratio_improvements: Dict[str, Optional[float]] = {}
    decisions: list[Dict[str, Any]] = []
    incumbents = state.setdefault("incumbents", {})
    for env, candidates in candidates_by_env.items():
        best = max(candidates, key=lambda row: float(row["top5_mean"]))
        previous = incumbents.get(env)
        previous_ratio = float(previous.get("ratio", 0.0)) if previous else 0.0
        new_ratio = float(best.get("ratio", 0.0) or 0.0)
        improvement = new_ratio - previous_ratio
        ratio_improvements[env] = improvement if previous else None
        regression = (
            ((float(best["top5_mean"]) / float(previous["top5_mean"])) - 1.0)
            if previous and float(previous["top5_mean"]) != 0.0
            else None
        )
        candidate_config = configs[(env, str(best["variant"]))]
        candidate_precision_compliant = config_precision_compliant(candidate_config)
        previous_precision_compliant = (
            incumbent_precision_compliant(previous) if previous else False
        )
        precision_migration = bool(
            previous
            and not previous_precision_compliant
            and candidate_precision_compliant
            and regression is not None
            and regression >= -0.05
            and (
                float(previous.get("ratio", 0.0)) < 0.9
                or new_ratio >= 0.9
            )
        )
        accept = (
            previous is None
            or float(best["top5_mean"]) > float(previous["top5_mean"])
            or precision_migration
        )
        reason = "initial_complete_cohort" if previous is None else "higher_top5_bin_score"
        if precision_migration and previous is not None and float(best["top5_mean"]) <= float(previous["top5_mean"]):
            reason = "precision_policy_migration_within_5pct_retention"
        if not accept:
            reason = "incumbent_retained"
        if accept:
            seed_scores = {
                str(seed.get("seed")): seed.get("top5_tail_mean") for seed in best.get("seeds", [])
            }
            incumbents[env] = {
                "env": env,
                "round_id": round_dir.name,
                "variant": best["variant"],
                "top5_mean": best["top5_mean"],
                "top5_std": best["top5_std"],
                "all_bin_mean": best.get("all_bin_mean"),
                "all_bin_std": best.get("all_bin_std"),
                "paper_mean": best["paper_mean"],
                "ratio": best["ratio"],
                "passed_90": best["passed_90"],
                "seed_scores": seed_scores,
                "tunable_config": candidate_config,
                "precision_compliant": candidate_precision_compliant,
                "accepted_at": iso_now(),
            }
        decisions.append(
            {
                "env": env,
                "candidate_variant": best["variant"],
                "candidate_ratio": new_ratio,
                "previous_ratio": previous_ratio if previous else None,
                "ratio_improvement": improvement if previous else None,
                "candidate_regression_from_incumbent": regression,
                "candidate_precision_compliant": candidate_precision_compliant,
                "previous_precision_compliant": previous_precision_compliant if previous else None,
                "accepted": accept,
                "reason": reason,
            }
        )
    return ratio_improvements, decisions


def write_current_configs(state: Mapping[str, Any]) -> None:
    configs = {
        env: record["tunable_config"]
        for env, record in state.get("incumbents", {}).items()
    }
    metadata = {
        env: {
            "round_id": record["round_id"],
            "variant": record["variant"],
            "top5_mean": record["top5_mean"],
            "ratio": record["ratio"],
            "precision_compliant": incumbent_precision_compliant(record),
            "precision_migration_parameters": precision_migration_parameters(record),
        }
        for env, record in state.get("incumbents", {}).items()
    }
    write_json_atomic(
        CURRENT_CONFIGS_PATH,
        {
            "schema_version": 1,
            "updated_at": iso_now(),
            "configs": configs,
            "metadata": metadata,
        },
    )


def all_targets_met(state: Mapping[str, Any]) -> bool:
    incumbents = state.get("incumbents", {})
    return all(
        env in incumbents
        and float(incumbents[env].get("ratio", 0.0)) >= 0.9
        and incumbent_precision_compliant(incumbents[env])
        for env in ENVIRONMENTS
    )


def underperforming_envs(state: Mapping[str, Any]) -> list[str]:
    incumbents = state.get("incumbents", {})
    return [
        env
        for env in ENVIRONMENTS
        if float(incumbents.get(env, {}).get("ratio", 0.0)) < 0.9
        or not incumbent_precision_compliant(incumbents.get(env, {}))
    ]


def write_final_report(state: Mapping[str, Any], status: str, reason: str = "") -> None:
    incumbents = state.get("incumbents", {})
    lines = [
        "# FlowCritic paper target report",
        "",
        f"- status: `{status}`",
        f"- updated: `{iso_now()}`",
        f"- completed rounds: `{state.get('rounds_completed', 0)}`",
        f"- target: four-seed top-5/bin mean >= `90%` of the paper mean",
    ]
    final_strategy = state.get("final_strategy")
    if final_strategy:
        lines.append(f"- final parameter strategy: `{final_strategy}`")
    if reason:
        lines.append(f"- reason: {reason}")
    lines.extend(
        [
            "",
            "| Environment | Top-5/bin | Population std | All-bin | Paper | Ratio | >=90% | Precision | Prior historical best | Delta | Better | Source |",
            "| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: | :---: | --- |",
        ]
    )
    for env in ENVIRONMENTS:
        row = incumbents.get(env, {})
        ratio = row.get("ratio")
        ratio_text = f"{float(ratio) * 100:.2f}%" if ratio is not None else "-"
        better = row.get("better_than_historical_best")
        precision_ok = incumbent_precision_compliant(row)
        lines.append(
            f"| {env} | {format_number(row.get('top5_mean'))} | {format_number(row.get('top5_std'))} | "
            f"{format_number(row.get('all_bin_mean'))} | {format_number(row.get('paper_mean'))} | "
            f"{ratio_text} | {'yes' if ratio is not None and float(ratio) >= 0.9 else 'no'} | "
            f"{'yes' if precision_ok else 'migration required'} | "
            f"{format_number(row.get('historical_best_before'))} | "
            f"{format_number(row.get('delta_vs_historical_best'))} | "
            f"{'yes' if better is True else 'no' if better is False else '-'} | "
            f"{row.get('round_id', '-')} / {row.get('variant', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed target ratios",
            "",
            "| Environment | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seeds >=90% |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for env in ENVIRONMENTS:
        row = incumbents.get(env, {})
        paper_mean = row.get("paper_mean")
        seed_scores = row.get("seed_scores", {})
        ratios: list[Optional[float]] = []
        for seed in SEEDS:
            value = seed_scores.get(str(seed))
            ratios.append(float(value) / float(paper_mean) if value is not None and paper_mean else None)
        cells = [f"{value * 100:.2f}%" if value is not None else "-" for value in ratios]
        passing = sum(value is not None and value >= 0.9 for value in ratios)
        lines.append(f"| {env} | {' | '.join(cells)} | {passing}/4 |")
    write_text_atomic(FINAL_REPORT_PATH, "\n".join(lines) + "\n")


def ensure_unification_state(state: Dict[str, Any]) -> Dict[str, Any]:
    value = state.get("unification")
    if not isinstance(value, dict):
        value = {}
        state["unification"] = value
    value.setdefault("state", "not_started")
    value.setdefault("attempts", [])
    return value


def round_purpose(round_dir: Path) -> str:
    manifest = read_json(round_dir / "manifest.json")
    return str(manifest.get("purpose", "tuning"))


def write_unification_evaluation(round_dir: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(round_dir / "unification_evaluation.json", payload)
    lines = [
        "# Parameter unification evaluation",
        "",
        f"- mode: `{payload.get('mode')}`",
        f"- accepted: `{'yes' if payload.get('accepted') else 'no'}`",
        "- gates: complete cohort, >=90% paper target, and >=95% incumbent retention",
        "",
        "| Environment | Candidate | Paper ratio | Incumbent retention | Complete | Target | Regression | Accepted |",
        "| --- | ---: | ---: | ---: | :---: | :---: | :---: | :---: |",
    ]
    for env in ENVIRONMENTS:
        row = payload.get("per_env", {}).get(env, {})
        paper_ratio = row.get("target_achievement_ratio")
        retention = row.get("incumbent_retention_ratio")
        lines.append(
            f"| {env} | {format_number(row.get('candidate_score'))} | "
            f"{format_number(float(paper_ratio) * 100 if paper_ratio is not None else None)}% | "
            f"{format_number(float(retention) * 100 if retention is not None else None)}% | "
            f"{'yes' if row.get('complete_gate') else 'no'} | "
            f"{'yes' if row.get('target_gate') else 'no'} | "
            f"{'yes' if row.get('regression_gate') else 'no'} | "
            f"{'yes' if row.get('accepted') else 'no'} |"
        )
    write_text_atomic(round_dir / "unification_evaluation.md", "\n".join(lines) + "\n")


def apply_unification_candidate(
    state: Dict[str, Any],
    round_dir: Path,
    score_payload: Mapping[str, Any],
    mode: str,
    historical_comparison: Optional[Mapping[tuple[str, str], Mapping[str, Any]]] = None,
) -> None:
    configs = load_manifest_configs(round_dir)
    by_env: Dict[str, Dict[str, Any]] = {}
    for raw in score_payload.get("results", []):
        env = str(raw.get("env", ""))
        if env in ENVIRONMENTS:
            if env in by_env:
                raise RuntimeError(f"duplicate unification result for {env}")
            by_env[env] = dict(raw)
    if set(by_env) != set(ENVIRONMENTS):
        raise RuntimeError("unification result does not cover all twelve environments")

    incumbents = state.setdefault("incumbents", {})
    for env in ENVIRONMENTS:
        result = by_env[env]
        variant = str(result["variant"])
        seed_scores = {
            str(seed.get("seed")): seed.get("top5_tail_mean")
            for seed in result.get("seeds", [])
        }
        incumbents[env] = {
            "env": env,
            "round_id": round_dir.name,
            "variant": variant,
            "top5_mean": result["top5_mean"],
            "top5_std": result["top5_std"],
            "all_bin_mean": result.get("all_bin_mean"),
            "all_bin_std": result.get("all_bin_std"),
            "paper_mean": result["paper_mean"],
            "ratio": result["ratio"],
            "passed_90": result["passed_90"],
            "seed_scores": seed_scores,
            "tunable_config": configs[(env, variant)],
            "accepted_at": iso_now(),
            "unification_mode": mode,
        }
    if historical_comparison:
        attach_historical_comparison(state, historical_comparison)
    write_current_configs(state)


def write_final_configs(state: Mapping[str, Any], strategy: str) -> None:
    incumbents = state.get("incumbents", {})
    write_json_atomic(
        FINAL_CONFIGS_PATH,
        {
            "schema_version": 1,
            "generated_at": iso_now(),
            "strategy": strategy,
            "fixed_parameters": FIXED_PARAMETERS,
            "model_L": "current five-hidden-layer implementation preserved",
            "unified_parameters_attempted": list(DEFAULT_UNIFIED_PARAMETERS),
            "environment_specific_parameters": sorted(REWARD_SCALE_PARAMETERS),
            "configs": {
                env: incumbents[env]["tunable_config"]
                for env in ENVIRONMENTS
                if env in incumbents
            },
            "evidence": {
                env: {
                    "round_id": incumbents[env].get("round_id"),
                    "variant": incumbents[env].get("variant"),
                    "top5_mean": incumbents[env].get("top5_mean"),
                    "top5_std": incumbents[env].get("top5_std"),
                    "paper_ratio": incumbents[env].get("ratio"),
                    "historical_best_before": incumbents[env].get("historical_best_before"),
                    "delta_vs_historical_best": incumbents[env].get("delta_vs_historical_best"),
                    "better_than_historical_best": incumbents[env].get("better_than_historical_best"),
                }
                for env in ENVIRONMENTS
                if env in incumbents
            },
        },
    )


def write_final_unification_report(state: Mapping[str, Any]) -> None:
    unification = state.get("unification", {})
    lines = [
        "# FlowCritic parameter unification report",
        "",
        f"- final strategy: `{state.get('final_strategy', 'unknown')}`",
        f"- status: `{unification.get('state', 'not_started')}`",
        "- shared candidates exclude reward_scaler, acf_coef, vf_coef, and flow_sigma",
    ]
    reason = unification.get("fallback_reason")
    if reason:
        lines.append(f"- fallback reason: {reason}")
    lines.extend(
        [
            "",
            "| Mode | Round | Accepted | Failed environments | Evaluation |",
            "| --- | --- | :---: | --- | --- |",
        ]
    )
    for attempt in unification.get("attempts", []):
        failed = ", ".join(attempt.get("failed_envs", [])) or "-"
        evaluation = attempt.get("evaluation_path", "-")
        lines.append(
            f"| {attempt.get('mode', '-')} | {attempt.get('round_id', '-')} | "
            f"{'yes' if attempt.get('accepted') else 'no'} | {failed} | `{evaluation}` |"
        )
    write_text_atomic(FINAL_UNIFICATION_REPORT_PATH, "\n".join(lines) + "\n")


def finalize_complete(state: Dict[str, Any], strategy: str, reason: str) -> int:
    state["state"] = "complete"
    state["completed_at"] = iso_now()
    state["final_strategy"] = strategy
    write_current_configs(state)
    write_final_configs(state, strategy)
    write_final_report(state, "complete", reason)
    write_final_unification_report(state)
    persist_loop_state(state)
    print(f"[continuous-tune] complete strategy={strategy}: {reason}", flush=True)
    return 0


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.3f}" if math.isfinite(number) else "-"


def tail_diagnostics(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.is_file():
        return {"missing": True}
    rows: list[Dict[str, float]] = []
    malformed = 0
    nonfinite_values: list[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                raw = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                malformed += 1
                continue
            selected: Dict[str, float] = {}
            for key, value in raw.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)) and len(nonfinite_values) < 100:
                        nonfinite_values.append({"key": str(key), "value": repr(value)})
                lowered = str(key).lower()
                if not any(marker in lowered for marker in ("kl", "ess", "critic", "weight", "loss")):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    selected[str(key)] = float(value)
            if selected:
                rows.append(selected)
    tail_count = max(1, math.ceil(len(rows) * 0.1)) if rows else 0
    tail = rows[-tail_count:] if tail_count else []
    values: Dict[str, list[float]] = defaultdict(list)
    for row in tail:
        for key, value in row.items():
            values[key].append(value)
    return {
        "missing": False,
        "row_count": len(rows),
        "tail_row_count": len(tail),
        "malformed_rows": malformed,
        "nonfinite_values": nonfinite_values,
        "metrics": {
            key: {
                "mean": sum(items) / len(items),
                "min": min(items),
                "max": max(items),
                "last": items[-1],
            }
            for key, items in sorted(values.items())
        },
    }


def build_diagnostics(round_dir: Path) -> Dict[str, Any]:
    manifest = read_json(round_dir / "manifest.json")
    jobs = []
    fatal_issues: list[Dict[str, Any]] = []
    anomaly_terms = (
        "out of memory",
        "traceback",
        "nan",
        "segmentation fault",
        "段错误",
        "core dumped",
        "cuda error",
    )
    for job in manifest["jobs"]:
        stdout_path = Path(job["stdout_log"])
        anomalies: list[str] = []
        if stdout_path.is_file():
            with stdout_path.open("rb") as handle:
                size = stdout_path.stat().st_size
                handle.seek(-min(size, 1_000_000), os.SEEK_END)
                tail = handle.read().decode(errors="replace")
            anomalies = [term for term in anomaly_terms if term in tail.lower()]
        metrics = tail_diagnostics(Path(job["metrics_jsonl"]))
        launcher_status = read_json(Path(job["job_run_dir"]) / "launcher_status.json")
        fcs_teardown = (
            job["env"] == "FrankaCubeStack"
            and launcher_status.get("state") == "complete_with_teardown_warning"
            and launcher_status.get("exit_code") in (-11, 139)
        )
        fatal_anomalies = [
            anomaly
            for anomaly in anomalies
            if not (
                fcs_teardown
                and anomaly in {"segmentation fault", "段错误", "core dumped"}
            )
        ]
        job_issues: list[str] = []
        if metrics.get("missing"):
            job_issues.append("missing_metrics")
        if metrics.get("malformed_rows"):
            job_issues.append("malformed_metrics")
        if metrics.get("nonfinite_values"):
            job_issues.append("nonfinite_metrics")
        if fatal_anomalies:
            job_issues.extend(f"stdout:{value}" for value in fatal_anomalies)
        if launcher_status.get("state") not in {"complete", "complete_with_teardown_warning"}:
            job_issues.append(f"launcher_state:{launcher_status.get('state')}")
        if job_issues:
            fatal_issues.append({"job_id": job["job_id"], "issues": job_issues})
        jobs.append(
            {
                "job_id": job["job_id"],
                "env": job["env"],
                "variant": job["variant"],
                "seed": job["seed"],
                "tail_metrics": metrics,
                "stdout_anomalies": anomalies,
                "fcs_teardown_allowed": fcs_teardown,
                "fatal_issues": job_issues,
            }
        )
    payload = {
        "generated_at": iso_now(),
        "integrity_ok": not fatal_issues,
        "fatal_issues": fatal_issues,
        "jobs": jobs,
    }
    write_json_atomic(round_dir / "diagnostics.json", payload)
    return payload


def block(state: Dict[str, Any], round_dir: Optional[Path], reason: str) -> int:
    state["state"] = "blocked"
    state["blocked_reason"] = reason
    state["blocked_at"] = iso_now()
    persist_loop_state(state)
    destination = (round_dir / "blocked_report.md") if round_dir else (STATE_DIR / "blocked_report.md")
    write_text_atomic(
        destination,
        "# Tuning loop blocked\n\n"
        f"- time: `{iso_now()}`\n"
        f"- reason: {reason}\n"
        f"- current round: `{state.get('current_round_id')}`\n"
        f"- completed rounds: `{state.get('rounds_completed', 0)}`\n",
    )
    write_final_report(state, "blocked", reason)
    return 3


def wait_for_planner_retry(
    state: Dict[str, Any], round_dir: Path, reason: str, delay_seconds: float
) -> None:
    """Persist a planner failure and keep the supervisor alive before retrying."""

    retry_count = int(state.get("planner_retry_count", 0)) + 1
    event = {
        "retry_count": retry_count,
        "failed_at": iso_now(),
        "round_id": round_dir.name,
        "reason": reason,
        "retry_delay_sec": delay_seconds,
    }
    history = state.setdefault("planner_retry_history", [])
    history.append(event)
    if len(history) > 100:
        del history[:-100]
    state["planner_retry_count"] = retry_count
    state["last_planner_error"] = reason
    state["state"] = "waiting_to_retry_gpt_planner"
    state["next_planner_retry_at_epoch"] = time.time() + delay_seconds
    persist_loop_state(state)
    write_json_atomic(
        round_dir / "planner_retry_status.json",
        {
            "status": "waiting_to_retry",
            **event,
            "supervisor_pid": os.getpid(),
        },
    )
    print(
        f"[continuous-tune] planner cycle failed; retrying in {delay_seconds:.1f}s: {reason}",
        flush=True,
    )
    time.sleep(delay_seconds)


def wait_for_round_retry(
    state: Dict[str, Any], round_dir: Path, reason: str, delay_seconds: float
) -> None:
    """Persist a recoverable round failure and retry the same cohort later."""

    retry_count = int(state.get("round_retry_count", 0)) + 1
    event = {
        "retry_count": retry_count,
        "failed_at": iso_now(),
        "round_id": round_dir.name,
        "reason": reason,
        "retry_delay_sec": delay_seconds,
    }
    history = state.setdefault("round_retry_history", [])
    history.append(event)
    if len(history) > ROUND_RETRY_HISTORY_LIMIT:
        del history[:-ROUND_RETRY_HISTORY_LIMIT]
    state["round_retry_count"] = retry_count
    state["round_retry_pending"] = True
    state["last_round_error"] = reason
    state["state"] = "waiting_to_retry_round"
    state["next_round_retry_at_epoch"] = time.time() + delay_seconds
    persist_loop_state(state)
    write_json_atomic(
        round_dir / "round_retry_status.json",
        {"status": "waiting_to_retry", **event, "supervisor_pid": os.getpid()},
    )
    print(
        f"[continuous-tune] round phase failed; retrying in {delay_seconds:.1f}s: {reason}",
        flush=True,
    )
    time.sleep(delay_seconds)


def evaluate_unification_round(
    state: Dict[str, Any],
    round_dir: Path,
    score_payload: Mapping[str, Any],
    mode: str,
    historical_comparison: Optional[Mapping[tuple[str, str], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    unification = ensure_unification_state(state)
    reference = unification.get("reference_incumbents")
    if not isinstance(reference, dict) or set(reference) != set(ENVIRONMENTS):
        raise RuntimeError("missing complete pre-unification incumbent reference")
    report = evaluate_unification_candidate(
        score_payload,
        read_json(PAPER_TARGETS_PATH),
        {"incumbents": reference},
    )
    report["mode"] = mode
    report["round_id"] = round_dir.name
    report["generated_at"] = iso_now()
    write_unification_evaluation(round_dir, report)
    failed_envs = [
        env for env in ENVIRONMENTS if not report.get("per_env", {}).get(env, {}).get("accepted")
    ]
    attempt = {
        "mode": mode,
        "round_id": round_dir.name,
        "accepted": bool(report.get("accepted")),
        "failed_envs": failed_envs,
        "evaluation_path": str(round_dir / "unification_evaluation.json"),
        "completed_at": iso_now(),
    }
    attempts = unification.setdefault("attempts", [])
    if not any(item.get("round_id") == round_dir.name for item in attempts if isinstance(item, dict)):
        attempts.append(attempt)
    if report.get("accepted"):
        apply_unification_candidate(
            state,
            round_dir,
            score_payload,
            mode,
            historical_comparison,
        )
        unification["state"] = f"{mode}_accepted"
        unification["accepted_round_id"] = round_dir.name
    else:
        unification["state"] = f"{mode}_rejected"
    return report


def finalize_failed_unification_attempt(
    state: Dict[str, Any], round_dir: Path, purpose: str, reason: str
) -> int:
    mode = purpose[len("unification_") :]
    unification = ensure_unification_state(state)
    payload = {
        "schema_version": 1,
        "mode": mode,
        "round_id": round_dir.name,
        "generated_at": iso_now(),
        "accepted": False,
        "complete_gate": False,
        "target_gate": False,
        "regression_gate": False,
        "failure_reason": reason,
        "per_env": {},
    }
    write_unification_evaluation(round_dir, payload)
    attempts = unification.setdefault("attempts", [])
    if not any(item.get("round_id") == round_dir.name for item in attempts if isinstance(item, dict)):
        attempts.append(
            {
                "mode": mode,
                "round_id": round_dir.name,
                "accepted": False,
                "failed_envs": list(ENVIRONMENTS),
                "evaluation_path": str(round_dir / "unification_evaluation.json"),
                "failure_reason": reason,
                "completed_at": iso_now(),
            }
        )
    if not any(
        item.get("round_id") == round_dir.name
        for item in state.get("round_history", [])
        if isinstance(item, dict)
    ):
        state["rounds_completed"] = int(state.get("rounds_completed", 0)) + 1
        state.setdefault("round_history", []).append(
            {
                "round_id": round_dir.name,
                "purpose": purpose,
                "completed_at": iso_now(),
                "stage": 3,
                "unification_mode": mode,
                "unification_accepted": False,
                "failure_reason": reason,
                "failed_environments": list(ENVIRONMENTS),
            }
        )
    unification["state"] = "per_environment_retained"
    unification["fallback_reason"] = reason
    return finalize_complete(state, "per_environment", reason)


def schedule_unification_round(
    state: Dict[str, Any], source_round_dir: Path, mode: str
) -> tuple[Optional[Path], Optional[str]]:
    if mode not in {"global", "cluster"}:
        raise ValueError(f"unknown unification mode: {mode}")
    unification = ensure_unification_state(state)
    if "reference_incumbents" not in unification:
        unification["reference_incumbents"] = copy.deepcopy(state.get("incumbents", {}))
    constraints_path = source_round_dir / "unification_constraints.json"
    write_json_atomic(
        constraints_path,
        {
            "phase": "stage_4_parameter_unification",
            "mode": mode,
            "must_target_exactly": list(ENVIRONMENTS),
            "required_seeds": list(SEEDS),
            "one_variant_per_environment": True,
            "required_shared_parameter_keys": list(DEFAULT_UNIFIED_PARAMETERS),
            "environment_specific_parameters": sorted(REWARD_SCALE_PARAMETERS),
            "required_distinct_shared_configs": 1 if mode == "global" else 2,
            "acceptance_gates": {
                "paper_ratio_minimum": 0.9,
                "incumbent_retention_minimum": 0.95,
                "all_jobs_complete": True,
            },
            "reference_incumbents": {
                env: {
                    "top5_mean": record.get("top5_mean"),
                    "top5_std": record.get("top5_std"),
                    "paper_ratio": record.get("ratio"),
                    "tunable_config": record.get("tunable_config"),
                }
                for env, record in unification["reference_incumbents"].items()
            },
        },
    )
    planner_command = [
        sys.executable,
        str(SCRIPT_DIR / "call_gpt_planner.py"),
        str(source_round_dir),
        "--stage",
        "3",
        "--history",
        str(PLANNER_HISTORY_PATH),
        "--base-configs",
        str(CURRENT_CONFIGS_PATH),
        "--context",
        str(constraints_path),
        "--unification-mode",
        mode,
        "--force",
    ]
    state["state"] = f"waiting_for_{mode}_unification_planner"
    unification["state"] = f"planning_{mode}"
    persist_loop_state(state)
    planner_code = run_checked(
        planner_command,
        log_path=source_round_dir / f"gpt_unification_{mode}.log",
    )
    if planner_code != 0:
        reason = f"{mode} unification GPT planner failed after all retries (exit={planner_code})"
        unification["state"] = f"{mode}_planner_failed"
        unification["fallback_reason"] = reason
        persist_loop_state(state)
        return None, reason

    approved = read_json(source_round_dir / "next_plan.json")
    if approved.get("status") != "continue":
        reason = f"{mode} unification planner returned status={approved.get('status')}"
        unification["state"] = f"{mode}_planner_failed"
        unification["fallback_reason"] = reason
        persist_loop_state(state)
        return None, reason

    next_index = int(state["next_round_index"])
    next_round_id = f"round_{next_index:03d}_unify_{mode}"
    materialization_plan = dict(approved)
    materialization_plan.update(
        {
            "round_id": next_round_id,
            "stage": 3,
            "purpose": f"unification_{mode}",
        }
    )
    plan_path = source_round_dir / "materialization_plan.json"
    write_json_atomic(plan_path, materialization_plan)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "materialize_round.py"),
        "--plan",
        str(plan_path),
        "--base-configs",
        str(CURRENT_CONFIGS_PATH),
    ]
    if run_checked(command, log_path=source_round_dir / "materialize_next.log") != 0:
        reason = f"validated {mode} unification plan could not be materialized"
        unification["state"] = f"{mode}_materialization_failed"
        unification["fallback_reason"] = reason
        persist_loop_state(state)
        return None, reason

    next_round_dir = EXPERIMENTS_DIR / next_round_id
    state["current_round_id"] = next_round_id
    state["next_round_index"] = next_index + 1
    state["state"] = "round_materialized"
    unification["state"] = f"{mode}_materialized"
    persist_loop_state(state)
    return next_round_dir, None


def materialize_initial(state: Dict[str, Any]) -> Path:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "materialize_round.py"),
        "--plan",
        str(CONFIG_DIR / "round_000_plan.json"),
    ]
    if run_checked(command) != 0:
        raise RuntimeError("failed to materialize round_000_baseline")
    round_dir = EXPERIMENTS_DIR / "round_000_baseline"
    state["current_round_id"] = round_dir.name
    state["state"] = "round_materialized"
    persist_loop_state(state)
    return round_dir


def choose_next_stage(state: Dict[str, Any]) -> int:
    stage = int(state.get("stage", 1))
    completed = int(state.get("rounds_completed", 0))
    maximum_stagnation = max(state.get("env_stagnation_counts", {}).values(), default=0)
    if stage == 1 and (completed >= 3 or (completed >= 2 and maximum_stagnation >= 1)):
        state["stage"] = 2
    elif stage == 2 and maximum_stagnation >= 2:
        state["stage"] = 3
    return int(state["stage"])


def is_schedulable_plan(plan: Any) -> bool:
    """Return whether a planner proposal is safe to pass to materialization.

    ``call_gpt_planner.py`` marks every schema-valid response as ``approved``;
    that status means *locally validated*, not necessarily *ready to run*.
    In particular, a valid ``blocked``/``stop`` proposal has no jobs and must
    never enter ``materialize_round.py``.  Keep this predicate deliberately
    small and independent of the full validator so it can protect recovery
    after a supervisor restart as well as the normal planner path.
    """

    return (
        isinstance(plan, Mapping)
        and plan.get("status") == "continue"
        and isinstance(plan.get("jobs"), list)
        and bool(plan.get("jobs"))
    )


def quarantine_materialization_plan(path: Path, reason: str) -> Optional[Path]:
    """Atomically move a non-schedulable recovery plan aside for audit."""

    if not path.exists():
        return None
    destination = path.with_name(
        f".{path.name}.rejected.{time.time_ns()}.json"
    )
    try:
        # ``replace`` is atomic when source and destination share the round
        # directory filesystem, and preserves the rejected bytes verbatim.
        path.replace(destination)
    except FileNotFoundError:
        # Another recovery attempt may have quarantined it already.
        return None
    append_supervisor_log(
        f"quarantined non-schedulable materialization plan path={path} "
        f"archive={destination} reason={reason}"
    )
    return destination


def recover_materialization_plan(
    round_dir: Path, state: Dict[str, Any]
) -> tuple[Optional[Path], bool]:
    """Recover only a schedulable plan after a supervisor restart.

    Returns ``(path, created_from_planner_output)``.  A pre-existing valid
    materialization plan is returned unchanged.  A blocked/stop/empty plan is
    quarantined (if it was already materialized) or ignored (if it is merely
    the planner output), allowing the main loop to call GPT again instead of
    repeatedly invoking the materializer with an impossible input.
    """

    materialization_path = round_dir / "materialization_plan.json"
    if materialization_path.exists():
        try:
            candidate = read_json(materialization_path)
        except (OSError, json.JSONDecodeError) as exc:
            quarantine_materialization_plan(
                materialization_path, f"unreadable JSON: {exc}"
            )
            return None, False
        if not is_schedulable_plan(candidate):
            status = candidate.get("status") if isinstance(candidate, Mapping) else None
            jobs = candidate.get("jobs") if isinstance(candidate, Mapping) else None
            quarantine_materialization_plan(
                materialization_path,
                f"status={status!r}, jobs={len(jobs) if isinstance(jobs, list) else 'invalid'}",
            )
            return None, False
        return materialization_path, False

    planner_status_path = round_dir / "gpt_planner_status.json"
    next_plan_path = round_dir / "next_plan.json"
    if not planner_status_path.exists() or not next_plan_path.exists():
        return None, False
    try:
        planner_status = read_json(planner_status_path)
        candidate = read_json(next_plan_path)
    except (OSError, json.JSONDecodeError):
        # The normal planner retry path will regenerate the proposal.  Do not
        # manufacture a recovery plan from a partially written file.
        return None, False
    if planner_status.get("status") != "approved":
        return None, False
    # If the supervisor was restarted after GPT wrote its proposal but before
    # the normal planner branch applied the user-approved task, apply that
    # task here before checking schedulability.  This closes the power-loss
    # window in which recovery could otherwise launch the stale GPT proposal.
    try:
        pending_recovery = apply_pending_task(
            round_dir,
            current_configs_path=CURRENT_CONFIGS_PATH,
            history_path=HISTORY_PATH,
            gpt_plan=candidate,
        )
    except ApprovedTaskError:
        raise
    if pending_recovery is not None:
        candidate = pending_recovery
    if not is_schedulable_plan(candidate):
        status = candidate.get("status") if isinstance(candidate, Mapping) else None
        jobs = candidate.get("jobs") if isinstance(candidate, Mapping) else None
        append_supervisor_log(
            f"ignored non-schedulable planner output round={round_dir.name} "
            f"status={status!r} jobs={len(jobs) if isinstance(jobs, list) else 'invalid'}"
        )
        return None, False

    recovered = dict(candidate)
    # The planner has completed successfully; clear only the retry markers
    # associated with that planner cycle before handing the plan to the next
    # round.  This is intentionally after the schedulability check.
    state["planner_retry_count"] = 0
    state.pop("last_planner_error", None)
    state.pop("next_planner_retry_at_epoch", None)
    state["last_planner_success_at"] = planner_status.get("finished_at") or iso_now()
    unification_mode = planner_status.get("unification_mode")
    if unification_mode in {"global", "cluster"}:
        recovered["round_id"] = (
            f"round_{int(state['next_round_index']):03d}_unify_{unification_mode}"
        )
        recovered["stage"] = 3
        recovered["purpose"] = f"unification_{unification_mode}"
    else:
        recovered["round_id"] = (
            f"round_{int(state['next_round_index']):03d}_gpt_stage{int(state['stage'])}"
        )
        recovered["stage"] = int(state["stage"])
        recovered["purpose"] = "tuning"
    write_json_atomic(materialization_path, recovered)
    return materialization_path, True


def main() -> int:
    args = parse_args()
    if args.gpus != "0,1,2,3" or args.slots_per_gpu != 6:
        raise SystemExit("continuous paper tuning is fixed to GPUs 0,1,2,3 and 6 slots/GPU")
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be positive")
    if not math.isfinite(args.planner_retry_delay_sec) or args.planner_retry_delay_sec < 0:
        raise SystemExit("--planner-retry-delay-sec must be finite and non-negative")
    if not math.isfinite(args.round_retry_delay_sec) or args.round_retry_delay_sec < 0:
        raise SystemExit("--round-retry-delay-sec must be finite and non-negative")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = (STATE_DIR / ".continuous_tune.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another continuous_tune supervisor is already running")

    global _SUPERVISOR_OWNS_PID
    signal.signal(signal.SIGTERM, _handle_supervisor_signal)
    signal.signal(signal.SIGINT, _handle_supervisor_signal)
    write_text_atomic(CONTINUOUS_PID_PATH, f"{os.getpid()}\n")
    _SUPERVISOR_OWNS_PID = True
    state = read_json(LOOP_STATE_PATH) if LOOP_STATE_PATH.exists() else initial_state(args.max_rounds)
    if state.get("state") == "complete" and all_targets_met(state):
        print(
            f"[continuous-tune] already complete strategy={state.get('final_strategy', 'unknown')}",
            flush=True,
        )
        return 0
    if state.get("stop_reason") or state.get("stopped_at"):
        resume_event = {
            "resumed_at": iso_now(),
            "previous_state": state.get("state"),
            "previous_stop_reason": state.pop("stop_reason", None),
            "previous_stopped_at": state.pop("stopped_at", None),
            "previous_supervisor_pid": state.get("supervisor_pid"),
            "round_id": state.get("current_round_id"),
        }
        history = state.setdefault("supervisor_resume_history", [])
        history.append(resume_event)
        if len(history) > 100:
            del history[:-100]
        state["last_supervisor_resume"] = resume_event
    state["max_rounds"] = args.max_rounds
    for stale_key in ("blocked_at", "blocked_reason"):
        state.pop(stale_key, None)
    state["last_resumed_at"] = iso_now()
    state["state"] = "running"
    persist_loop_state(state)
    if not HISTORY_PATH.exists():
        write_json_atomic(HISTORY_PATH, {"schema_version": 1, "cohorts": [], "unique_config_count": 0})
    write_planner_history_index()

    try:
        round_dir = (
            EXPERIMENTS_DIR / state["current_round_id"]
            if state.get("current_round_id")
            else materialize_initial(state)
        )
        # A manually adopted/materialized round supersedes planner retry
        # markers from the source round.  Clear those stale markers only when
        # the active round has a scheduler state; leave a live planner status
        # untouched so a genuine in-flight request remains observable.
        active_round_state_path = round_dir / "round_state.json"
        active_round_state = (
            read_json(active_round_state_path)
            if active_round_state_path.is_file()
            else {}
        )
        if (
            state.get("last_planner_error")
            and active_round_state.get("state") in {"materialized", "running", "complete"}
            and not (round_dir / "gpt_planner_status.json").is_file()
        ):
            for stale_key in (
                "planner_retry_count",
                "last_planner_error",
                "next_planner_retry_at_epoch",
            ):
                state.pop(stale_key, None)
            persist_loop_state(state)
        settled_this_invocation = 0
        while True:
            purpose = round_purpose(round_dir)
            optional_unification = purpose in {"unification_global", "unification_cluster"}
            already_scored = any(
                record.get("round_id") == round_dir.name
                for record in state.get("round_history", [])
                if isinstance(record, dict)
            )
            if not already_scored:
                state["state"] = "launching_or_resuming_round"
                persist_loop_state(state)
                retry_failed = bool(state.pop("round_retry_pending", False))
                if retry_failed:
                    state.pop("next_round_retry_at_epoch", None)
                    persist_loop_state(state)
                launcher_code = launch_or_wait_for_round(
                    round_dir,
                    args,
                    retry_failed=retry_failed,
                )
                round_state = read_json(round_dir / "round_state.json")
                if launcher_code != 0 or round_state.get("state") != "complete":
                    reason = (
                        f"round launcher did not settle cleanly (exit={launcher_code}, "
                        f"state={round_state.get('state')})"
                    )
                    if optional_unification:
                        return finalize_failed_unification_attempt(
                            state, round_dir, purpose, reason
                        )
                    if args.stop_on_incomplete:
                        return block(state, round_dir, reason)
                    wait_for_round_retry(
                        state,
                        round_dir,
                        reason,
                        args.round_retry_delay_sec,
                    )
                    continue

                score_command = [
                    sys.executable,
                    str(SCRIPT_DIR / "score_plot_protocol.py"),
                    str(round_dir),
                    "--targets",
                    str(PAPER_TARGETS_PATH),
                    "--require-complete",
                ]
                score_code = run_checked(score_command, log_path=round_dir / "scoring.log")
                if score_code != 0:
                    reason = f"fixed-budget scoring rejected the round (exit={score_code})"
                    if optional_unification:
                        return finalize_failed_unification_attempt(
                            state, round_dir, purpose, reason
                        )
                    if args.stop_on_incomplete:
                        return block(state, round_dir, reason)
                    wait_for_round_retry(
                        state,
                        round_dir,
                        reason,
                        args.round_retry_delay_sec,
                    )
                    continue
                score_payload = read_json(round_dir / "score_plot_protocol.json")
                diagnostics = build_diagnostics(round_dir)
                if not diagnostics.get("integrity_ok"):
                    reason = (
                        f"diagnostic integrity gate failed: {diagnostics.get('fatal_issues', [])}"
                    )
                    if optional_unification:
                        return finalize_failed_unification_attempt(
                            state, round_dir, purpose, reason
                        )
                    if args.stop_on_incomplete:
                        return block(state, round_dir, reason)
                    wait_for_round_retry(
                        state,
                        round_dir,
                        reason,
                        args.round_retry_delay_sec,
                    )
                    continue
                state.pop("round_retry_pending", None)
                state.pop("next_round_retry_at_epoch", None)
                state.pop("last_round_error", None)
                state.pop("round_retry_count", None)
                state.pop("active_round_retry", None)
                historical_comparison = build_historical_comparison(round_dir, score_payload)
                append_history(round_dir, score_payload)
                if purpose == "tuning":
                    had_incumbents = bool(state.get("incumbents"))
                    ratio_improvements, decisions = update_incumbents(state, round_dir, score_payload)
                    attach_historical_comparison(
                        state,
                        historical_comparison,
                        {
                            (str(decision["env"]), str(decision["candidate_variant"]))
                            for decision in decisions
                            if decision.get("accepted")
                        },
                    )
                    write_json_atomic(round_dir / "acceptance_decisions.json", decisions)
                    write_current_configs(state)
                    stagnation = state.setdefault("env_stagnation_counts", {})
                    for env in ENVIRONMENTS:
                        incumbent = state.get("incumbents", {}).get(env, {})
                        if float(incumbent.get("ratio", 0.0)) >= 0.9:
                            stagnation.pop(env, None)
                            continue
                        improvement = ratio_improvements.get(env)
                        if not had_incumbents or improvement is None:
                            stagnation.setdefault(env, 0)
                        elif improvement < 0.02:
                            stagnation[env] = int(stagnation.get(env, 0)) + 1
                        else:
                            stagnation[env] = 0
                    state["global_stagnation_count"] = max(stagnation.values(), default=0)
                    history_record = {
                        "round_id": round_dir.name,
                        "purpose": purpose,
                        "completed_at": iso_now(),
                        "stage": read_json(round_dir / "manifest.json").get("stage", 1),
                        "ratio_improvements": ratio_improvements if had_incumbents else {},
                        "env_stagnation_counts": dict(stagnation),
                        "global_stagnation_count": state.get("global_stagnation_count", 0),
                        "underperforming_envs": underperforming_envs(state),
                        "decisions": decisions,
                    }
                elif purpose in {"unification_global", "unification_cluster"}:
                    mode = purpose[len("unification_") :]
                    evaluation = evaluate_unification_round(
                        state,
                        round_dir,
                        score_payload,
                        mode,
                        historical_comparison,
                    )
                    history_record = {
                        "round_id": round_dir.name,
                        "purpose": purpose,
                        "completed_at": iso_now(),
                        "stage": 3,
                        "unification_mode": mode,
                        "unification_accepted": bool(evaluation.get("accepted")),
                        "failed_environments": [
                            env
                            for env in ENVIRONMENTS
                            if not evaluation.get("per_env", {}).get(env, {}).get("accepted")
                        ],
                    }
                else:
                    raise RuntimeError(f"unsupported round purpose: {purpose}")
                state["rounds_completed"] = int(state.get("rounds_completed", 0)) + 1
                state.setdefault("round_history", []).append(history_record)
                state["state"] = "round_scored"
                persist_loop_state(state)
                write_final_report(state, "running")
                settled_this_invocation += 1
            else:
                state["state"] = "resuming_after_scored_round"
                persist_loop_state(state)

            # Resume without paying for another GPT call if the previous
            # supervisor stopped after a *schedulable* plan was approved or
            # materialized.  A schema-valid ``blocked``/``stop`` proposal is
            # reported as planner status=approved, but it is not executable;
            # recovery must leave it for the normal planner-retry path.
            recovered_plan_path, _recovered_from_planner = recover_materialization_plan(
                round_dir, state
            )
            if recovered_plan_path is not None:
                recovered_candidate = read_json(recovered_plan_path)
                recovered_candidate_id = str(recovered_candidate.get("round_id", ""))
                recovered_candidate_dir = EXPERIMENTS_DIR / recovered_candidate_id
                if (recovered_candidate_dir / "policy_abort.json").is_file():
                    state.setdefault("superseded_materialization_plans", []).append(
                        {
                            "source_round_id": round_dir.name,
                            "target_round_id": recovered_candidate_id,
                            "reason": "target round was superseded by parameter precision policy",
                            "recorded_at": iso_now(),
                        }
                    )
                    persist_loop_state(state)
                    recovered_plan_path = None
            if recovered_plan_path is not None:
                recovered = read_json(recovered_plan_path)
                recovered_round_id = str(recovered["round_id"])
                recovered_round_dir = EXPERIMENTS_DIR / recovered_round_id
                if not (recovered_round_dir / "manifest.json").exists():
                    recovered_command = [
                        sys.executable,
                        str(SCRIPT_DIR / "materialize_round.py"),
                        "--plan",
                        str(recovered_plan_path),
                        "--base-configs",
                        str(CURRENT_CONFIGS_PATH),
                    ]
                    if run_checked(recovered_command, log_path=round_dir / "materialize_next.log") != 0:
                        reason = "approved plan could not be recovered/materialized"
                        if args.stop_on_incomplete:
                            return block(state, round_dir, reason)
                        wait_for_planner_retry(
                            state,
                            round_dir,
                            reason,
                            args.planner_retry_delay_sec,
                        )
                        continue
                round_dir = recovered_round_dir
                state["current_round_id"] = recovered_round_id
                state["next_round_index"] = max(
                    int(state["next_round_index"]),
                    int(recovered_round_id.split("_")[1]) + 1,
                )
                state["state"] = "round_materialized"
                persist_loop_state(state)
                continue

            purpose = round_purpose(round_dir)
            if purpose == "unification_global":
                unification = ensure_unification_state(state)
                if unification.get("state") == "global_accepted":
                    return finalize_complete(
                        state,
                        "global_shared_stability",
                        "all twelve targets passed with one shared non-reward-scale stability configuration",
                    )
                if int(state["rounds_completed"]) >= int(state["max_rounds"]):
                    unification["state"] = "per_environment_retained"
                    unification["fallback_reason"] = "global candidate failed and the 12-round safety budget was exhausted"
                    return finalize_complete(
                        state,
                        "per_environment",
                        unification["fallback_reason"],
                    )
                next_round, error = schedule_unification_round(state, round_dir, "cluster")
                if next_round is None:
                    unification["state"] = "per_environment_retained"
                    unification["fallback_reason"] = error or "cluster unification could not be scheduled"
                    return finalize_complete(state, "per_environment", unification["fallback_reason"])
                round_dir = next_round
                continue

            if purpose == "unification_cluster":
                unification = ensure_unification_state(state)
                if unification.get("state") == "cluster_accepted":
                    return finalize_complete(
                        state,
                        "two_shared_stability_clusters",
                        "all twelve targets passed with two shared non-reward-scale stability configurations",
                    )
                unification["state"] = "per_environment_retained"
                unification["fallback_reason"] = (
                    "two-cluster candidate failed the completeness, paper-target, or 5% regression gate"
                )
                return finalize_complete(state, "per_environment", unification["fallback_reason"])

            if all_targets_met(state):
                if args.once and settled_this_invocation >= 1:
                    state["state"] = "paused_after_one_round"
                    persist_loop_state(state)
                    return 0
                if int(state["rounds_completed"]) >= int(state["max_rounds"]):
                    unification = ensure_unification_state(state)
                    unification["state"] = "per_environment_retained"
                    unification["fallback_reason"] = "all paper targets passed at the 12-round safety limit; no unification budget remained"
                    return finalize_complete(state, "per_environment", unification["fallback_reason"])
                next_round, error = schedule_unification_round(state, round_dir, "global")
                if next_round is None:
                    unification = ensure_unification_state(state)
                    unification["state"] = "per_environment_retained"
                    unification["fallback_reason"] = error or "global unification could not be scheduled"
                    return finalize_complete(state, "per_environment", unification["fallback_reason"])
                round_dir = next_round
                continue

            if int(state["rounds_completed"]) >= int(state["max_rounds"]):
                reason = f"soft safety limit of {state['max_rounds']} rounds reached before all targets"
                if args.stop_on_incomplete:
                    return block(state, round_dir, reason)
                state["soft_round_limit_exceeded"] = True
                state["soft_round_limit_reason"] = reason
                persist_loop_state(state)
            if args.once and settled_this_invocation >= 1:
                state["state"] = "paused_after_one_round"
                persist_loop_state(state)
                return 0

            stage = choose_next_stage(state)
            if (
                stage == 3
                and int(state.get("global_stagnation_count", 0)) >= 3
                and not args.continue_after_stagnation
            ):
                reason = (
                    "at least one remaining environment improved by less than 2 percentage "
                    "points for three consecutive rounds, including a Stage-3 attempt"
                )
                if args.stop_on_incomplete:
                    return block(state, round_dir, reason)
                state["stagnation_warning"] = reason
                state["stagnation_warning_at"] = iso_now()
                persist_loop_state(state)
            failed_envs = underperforming_envs(state)
            incumbents = state.get("incumbents", {})
            paper_underperforming = [
                env
                for env in ENVIRONMENTS
                if float(incumbents.get(env, {}).get("ratio", 0.0)) < 0.9
            ]
            precision_migrations = {
                env: precision_migration_parameters(incumbents.get(env, {}))
                for env in failed_envs
                if precision_migration_parameters(incumbents.get(env, {}))
            }
            diversification = diversification_constraints(state, failed_envs)
            write_json_atomic(
                round_dir / "loop_constraints.json",
                {
                    "stage": stage,
                    "underperforming_envs": failed_envs,
                    "paper_underperforming_envs": paper_underperforming,
                    "precision_migration_envs": precision_migrations,
                    "must_target_exactly": failed_envs,
                    "all_other_environments_are_frozen": True,
                    "required_seeds": list(SEEDS),
                    "max_variants_per_environment": 2,
                    "max_jobs": 48,
                    "next_round_index": state["next_round_index"],
                    "search_precision_policy": {
                        "max_decimal_places": SEARCH_MAX_DECIMAL_PLACES,
                        "required_increment": float(SEARCH_GRID),
                        "preferred_increments": [0.01, float(SEARCH_GRID)],
                        "exempt_parameters": sorted(SEARCH_PRECISION_EXEMPT_PARAMETERS),
                        "fixed_non_search_exceptions": ["learning_rate", "adam_eps"],
                        "binary_subdivision_below_grid_forbidden": True,
                    },
                    "diversification": diversification,
                },
            )
            planner_command = [
                sys.executable,
                str(SCRIPT_DIR / "call_gpt_planner.py"),
                str(round_dir),
                "--stage",
                str(stage),
                "--history",
                str(PLANNER_HISTORY_PATH),
                "--base-configs",
                str(CURRENT_CONFIGS_PATH),
                "--context",
                str(round_dir / "loop_constraints.json"),
                "--force",
            ]
            state["state"] = "waiting_for_gpt_planner"
            state["planner_started_at"] = iso_now()
            state["planner_round_id"] = round_dir.name
            persist_loop_state(state)
            planner_code = run_checked(planner_command, log_path=round_dir / "gpt_planner.log")
            state.pop("planner_started_at", None)
            state.pop("planner_round_id", None)
            persist_loop_state(state)
            if planner_code != 0:
                reason = f"GPT planner failed after all retries (exit={planner_code})"
                if planner_code == 4:
                    # An input-budget failure is an orchestration failure, not
                    # evidence that the experiment met (or cannot meet) its
                    # target.  Keep the loop alive and retry after recording
                    # the deterministic cause; never turn an unfinished
                    # tuning run into a terminal blocked state here.
                    reason += "; planner input remained above the Codex endpoint limit after local compaction"
                if args.stop_on_incomplete:
                    return block(state, round_dir, reason)
                wait_for_planner_retry(
                    state,
                    round_dir,
                    reason,
                    args.planner_retry_delay_sec,
                )
                continue
            approved = read_json(round_dir / "next_plan.json")
            # The user approved a bounded post-round search before this
            # supervisor started.  Keep the GPT proposal and status for audit,
            # but execute the locally validated approved task when this source
            # round matches.  The helper is idempotent across supervisor or
            # host restarts and writes the executable plan atomically.
            try:
                pending_approved = apply_pending_task(
                    round_dir,
                    current_configs_path=CURRENT_CONFIGS_PATH,
                    history_path=HISTORY_PATH,
                    gpt_plan=approved,
                )
            except ApprovedTaskError as exc:
                reason = f"user-approved post-round plan failed validation: {exc}"
                append_supervisor_log(reason)
                if args.stop_on_incomplete:
                    return block(state, round_dir, reason)
                wait_for_planner_retry(
                    state,
                    round_dir,
                    reason,
                    args.planner_retry_delay_sec,
                )
                continue
            pending_task_active = pending_approved is not None
            if pending_task_active:
                approved = pending_approved
            if approved.get("status") != "continue":
                reason = (
                    f"GPT returned status={approved.get('status')} before all target gates were met"
                )
                if args.stop_on_incomplete:
                    return block(state, round_dir, reason)
                wait_for_planner_retry(
                    state,
                    round_dir,
                    reason,
                    args.planner_retry_delay_sec,
                )
                continue
            proposed_envs = set(approved.get("target_envs", []))
            required_envs = set(failed_envs)
            # A user-approved task may deliberately include a precision
            # revalidation environment that the incumbent currently counts as
            # paper-passing.  Its scope is validated locally in
            # approved_task_plan.py, so do not reject it as a GPT coverage
            # mismatch.  Ordinary GPT proposals remain exact-match only.
            if not pending_task_active and proposed_envs != required_envs:
                reason = (
                    "GPT target coverage mismatch: "
                    f"missing={sorted(required_envs - proposed_envs)}, "
                    f"frozen_extra={sorted(proposed_envs - required_envs)}"
                )
                if args.stop_on_incomplete:
                    return block(state, round_dir, reason)
                wait_for_planner_retry(
                    state,
                    round_dir,
                    reason,
                    args.planner_retry_delay_sec,
                )
                continue

            state["planner_retry_count"] = 0
            state.pop("last_planner_error", None)
            state.pop("next_planner_retry_at_epoch", None)
            state["last_planner_success_at"] = iso_now()
            next_index = int(state["next_round_index"])
            next_round_id = f"round_{next_index:03d}_gpt_stage{stage}"
            materialization_plan = dict(approved)
            materialization_plan["round_id"] = next_round_id
            materialization_plan["stage"] = stage
            materialization_plan["purpose"] = "tuning"
            plan_path = round_dir / "materialization_plan.json"
            write_json_atomic(plan_path, materialization_plan)
            materialize_command = [
                sys.executable,
                str(SCRIPT_DIR / "materialize_round.py"),
                "--plan",
                str(plan_path),
                "--base-configs",
                str(CURRENT_CONFIGS_PATH),
            ]
            if run_checked(materialize_command, log_path=round_dir / "materialize_next.log") != 0:
                reason = "validated GPT plan could not be materialized"
                if args.stop_on_incomplete:
                    return block(state, round_dir, reason)
                wait_for_planner_retry(
                    state,
                    round_dir,
                    reason,
                    args.planner_retry_delay_sec,
                )
                continue
            round_dir = EXPERIMENTS_DIR / next_round_id
            state["current_round_id"] = next_round_id
            state["next_round_index"] = next_index + 1
            state["state"] = "round_materialized"
            persist_loop_state(state)
    except Exception as exc:
        round_dir = locals().get("round_dir")
        reason = f"unhandled supervisor error: {exc}"
        if args.stop_on_incomplete:
            return block(state, round_dir, reason)
        # Do not convert a transient filesystem/JSON/race error into a
        # terminal blocked experiment.  Persist the failure, leave a durable
        # retry marker, and re-exec the supervisor after the configured delay;
        # the process lock is released by exec and the active launcher is
        # adopted on the next invocation.
        if isinstance(round_dir, Path):
            wait_for_round_retry(
                state,
                round_dir,
                reason,
                args.round_retry_delay_sec,
            )
        else:
            state["state"] = "waiting_to_retry_supervisor"
            state["last_round_error"] = reason
            state["next_round_retry_at_epoch"] = time.time() + args.round_retry_delay_sec
            persist_loop_state(state)
            time.sleep(args.round_retry_delay_sec)
        os.execv(sys.executable, [sys.executable, *sys.argv])
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
