#!/usr/bin/env python3
"""Durable process wrapper that records the real trainer exit code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def main() -> int:
    args = parse_args()
    record = {
        "state": "running",
        "wrapper_pid": os.getpid(),
        "started_at": iso_now(),
        "command": args.command,
    }
    write_json_atomic(args.record, record)
    try:
        completed = subprocess.run(args.command, check=False)
        return_code = completed.returncode
        record.update({"state": "finished", "exit_code": return_code, "finished_at": iso_now()})
        write_json_atomic(args.record, record)
        return return_code
    except BaseException as exc:
        record.update(
            {
                "state": "wrapper_failed",
                "exit_code": None,
                "finished_at": iso_now(),
                "error": repr(exc),
            }
        )
        write_json_atomic(args.record, record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

