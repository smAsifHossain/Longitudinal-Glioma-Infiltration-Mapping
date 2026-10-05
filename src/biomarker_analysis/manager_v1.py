#!/usr/bin/env python3
"""Run preflight and analysis sequentially with resumable stage markers."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import traceback
from pathlib import Path

from biomarker_common_v1 import OUTPUT_ROOT, atomic_json


SCRIPT_DIR = Path(__file__).resolve().parent


def run_stage(name: str, script: str, marker: Path, log: Path) -> None:
    if marker.is_file():
        return
    with log.open("a") as handle:
        handle.write(f"START {name}: {sys.executable} -u {SCRIPT_DIR / script}\n")
        handle.flush()
        code = subprocess.call(
            [sys.executable, "-u", str(SCRIPT_DIR / script)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    if code != 0:
        raise RuntimeError(f"{name} exited with code {code}")
    if not marker.is_file():
        raise RuntimeError(f"{name} exited successfully but did not create {marker}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (OUTPUT_ROOT / "manager.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another manager is already running.", flush=True)
        return
    progress = OUTPUT_ROOT / "manager_progress.json"
    try:
        if (OUTPUT_ROOT / "COMPLETE").is_file():
            atomic_json(progress, {"status": "PASS", "phase": "complete"})
            return
        atomic_json(progress, {"status": "RUNNING", "phase": "preflight"})
        run_stage(
            "preflight",
            "preflight_v1.py",
            OUTPUT_ROOT / "preflight" / "PASS",
            OUTPUT_ROOT / "preflight.log",
        )
        atomic_json(progress, {"status": "RUNNING", "phase": "analysis"})
        run_stage(
            "analysis",
            "run_analysis_v1.py",
            OUTPUT_ROOT / "COMPLETE",
            OUTPUT_ROOT / "analysis.log",
        )
        atomic_json(progress, {"status": "PASS", "phase": "complete"})
    except Exception as error:
        atomic_json(
            progress,
            {"status": "FAILED", "phase": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
