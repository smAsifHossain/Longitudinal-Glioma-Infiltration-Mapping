#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from path_config import CODE_ROOT, CONTROL_ROOT, OUTPUT_ROOT, PROJECT_ROOT


V5 = PROJECT_ROOT
CODE = CODE_ROOT
CONTROL = CONTROL_ROOT
OUTPUT = OUTPUT_ROOT
STATUS = CONTROL / "manager_status.json"
STOP = CONTROL / "STOP_REQUESTED"
COMPLETE = OUTPUT / "EXACT_CACHE_COMPLETE"
ACTIVE: subprocess.Popen | None = None
SIGNALLED = False


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def handle_signal(signum: int, _frame: Any) -> None:
    global SIGNALLED
    SIGNALLED = True
    STOP.touch()
    if ACTIVE is not None and ACTIVE.poll() is None:
        os.killpg(ACTIVE.pid, signum)


def stage(name: str, script: str) -> None:
    global ACTIVE
    if STOP.is_file():
        raise KeyboardInterrupt("stop requested")
    log_path = CONTROL / f"{name}.log"
    atomic_json(STATUS, {"status": "RUNNING", "phase": name, "log": str(log_path)})
    with log_path.open("a", buffering=1) as log:
        ACTIVE = subprocess.Popen(
            [sys.executable, "-u", str(CODE / script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        (CONTROL / f"{name}.pid").write_text(f"{ACTIVE.pid}\n", encoding="utf-8")
        code = ACTIVE.wait()
    ACTIVE = None
    if SIGNALLED or STOP.is_file():
        raise KeyboardInterrupt("stop requested")
    if code != 0:
        raise RuntimeError(f"{name} exited with code {code}; log={log_path}")


def main() -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    lock = (CONTROL / "manager.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    STOP.unlink(missing_ok=True)
    if COMPLETE.is_file():
        atomic_json(STATUS, {"status": "PASS", "phase": "complete"})
        return
    stage("preflight", "preflight_mu_exact_v2.py")
    stage("generation", "run_mu_exact_391_v2.py")
    stage("exact_output_audit", "audit_mu_exact_outputs_v2.py")
    if not COMPLETE.is_file():
        raise RuntimeError("Generation returned without EXACT_CACHE_COMPLETE")
    atomic_json(
        STATUS,
        {
            "status": "PASS",
            "phase": "complete",
            "output": str(OUTPUT),
            "cases": 391,
        },
    )


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        main()
    except KeyboardInterrupt as error:
        atomic_json(
            STATUS,
            {"status": "STOPPED", "phase": "stopped", "message": str(error)},
        )
        raise SystemExit(130)
    except Exception as error:
        atomic_json(
            STATUS,
            {
                "status": "FAILED",
                "phase": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        traceback.print_exc()
        raise
