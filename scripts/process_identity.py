#!/usr/bin/env python3
"""Linux process identity helpers for reboot-safe supervisor adoption."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def current_boot_id() -> Optional[str]:
    try:
        value = BOOT_ID_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _proc_stat(pid: int) -> tuple[Optional[str], Optional[int]]:
    """Return process state and start-time ticks from ``/proc/<pid>/stat``."""

    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None, None
    close_paren = raw.rfind(")")
    if close_paren < 0:
        return None, None
    fields = raw[close_paren + 1 :].split()
    if len(fields) <= 19:
        return None, None
    try:
        return fields[0], int(fields[19])
    except (TypeError, ValueError):
        return None, None


def capture_process_identity(pid: int) -> Optional[dict[str, Any]]:
    """Capture fields that distinguish a process from a reused numeric PID."""

    if pid <= 0:
        return None
    proc_dir = Path("/proc") / str(pid)
    try:
        cmdline_bytes = (proc_dir / "cmdline").read_bytes()
    except OSError:
        return None
    state, start_time_ticks = _proc_stat(pid)
    boot_id = current_boot_id()
    if state in {None, "Z"} or start_time_ticks is None or boot_id is None or not cmdline_bytes:
        return None
    argv = [
        item.decode(errors="replace")
        for item in cmdline_bytes.split(b"\x00")
        if item
    ]
    if not argv:
        return None
    return {
        "pid": pid,
        "boot_id": boot_id,
        "start_time_ticks": start_time_ticks,
        "cmdline_sha256": hashlib.sha256(cmdline_bytes).hexdigest(),
        "argv": argv,
    }


def _argv_contains_path(argv: Sequence[str], expected: Path) -> bool:
    expected = expected.expanduser().resolve()
    for token in argv:
        try:
            if Path(token).expanduser().resolve() == expected:
                return True
        except (OSError, RuntimeError, ValueError):
            continue
    return False


def verify_process_identity(
    pid: Any,
    expected_identity: Optional[Mapping[str, Any]],
    *,
    required_paths: Sequence[Path] = (),
) -> tuple[bool, str, Optional[dict[str, Any]]]:
    """Verify liveness, command ownership, boot identity, and PID generation.

    Older round states have no stored identity. They may be adopted only when
    their live argv contains every required path exactly. The returned current
    identity should then be used for subsequent polling so PID reuse cannot
    occur after adoption.
    """

    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False, "launcher pid is missing or invalid", None
    current = capture_process_identity(value)
    if current is None:
        return False, f"pid {value} is not a live non-zombie process", None
    argv = current["argv"]
    missing = [str(path) for path in required_paths if not _argv_contains_path(argv, path)]
    if missing:
        return False, f"pid {value} cmdline does not own required paths: {missing}", current

    if expected_identity is None:
        return True, "legacy launcher accepted by exact cmdline paths", current
    required_fields = ("pid", "boot_id", "start_time_ticks", "cmdline_sha256")
    absent = [field for field in required_fields if expected_identity.get(field) is None]
    if absent:
        return False, f"stored launcher identity is incomplete: {absent}", current
    for field in required_fields:
        if current[field] != expected_identity[field]:
            return (
                False,
                f"stored launcher identity mismatch for {field}",
                current,
            )
    return True, "launcher identity verified", current
