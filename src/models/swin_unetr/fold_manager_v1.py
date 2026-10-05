#!/usr/bin/env python3
"""Run training and held-out evaluation for one CV fold."""
from __future__ import annotations

import fcntl
import json
import signal
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from common_v1 import CV_FOLD, EXPERIMENT_ROOT, SCRIPT_DIR, STOP_FILE, atomic_json


ACTIVE: subprocess.Popen | None = None
SIGNALLED = False


def handle_signal(signum: int, _frame: Any) -> None:
    global SIGNALLED
    SIGNALLED = True
    STOP_FILE.touch()
    if ACTIVE is not None and ACTIVE.poll() is None:
        ACTIVE.send_signal(signum)


def run_stage(name: str, command: list[str], log_path: Path) -> int:
    global ACTIVE
    print(f"FOLD {CV_FOLD} START {name}: {' '.join(command)}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        ACTIVE = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (EXPERIMENT_ROOT / f"{name}.pid").write_text(
            f"{ACTIVE.pid}\n", encoding="utf-8"
        )
        code = ACTIVE.wait()
    ACTIVE = None
    return int(code)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    if CV_FOLD is None:
        raise RuntimeError("SWINUNETR_CV_FOLD must be set for the fold manager")
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = (EXPERIMENT_ROOT / "manager.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Fold {CV_FOLD} already has a running manager")
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    progress = EXPERIMENT_ROOT / "manager_progress.json"
    try:
        STOP_FILE.unlink(missing_ok=True)
        if (EXPERIMENT_ROOT / "COMPLETE").is_file():
            atomic_json(progress, {"status": "PASS", "phase": "complete", "fold": CV_FOLD})
            return
        if not (EXPERIMENT_ROOT / "preflight/PASS").is_file():
            raise RuntimeError(f"Global preflight did not prepare fold {CV_FOLD}")
        if not (EXPERIMENT_ROOT / "training/TRAINING_COMPLETE").is_file():
            atomic_json(progress, {"status": "RUNNING", "phase": "training", "fold": CV_FOLD})
            code = run_stage(
                "training",
                [
                    sys.executable,
                    "-u",
                    str(SCRIPT_DIR / "train_v1.py"),
                    "--epochs",
                    "30",
                    "--samples-per-epoch",
                    "480",
                    "--minimum-epochs",
                    "8",
                    "--patience",
                    "6",
                    "--pair-batch-size",
                    "0",
                    "--effective-pair-batch-size",
                    "8",
                    "--sw-batch-size",
                    "1",
                    "--checkpoint-every",
                    "40",
                    "--batch-workers",
                    "8",
                    "--preload-workers",
                    "1",
                ],
                EXPERIMENT_ROOT / "training.log",
            )
            if code == 75 or SIGNALLED:
                atomic_json(
                    progress,
                    {
                        "status": "STOPPED",
                        "phase": "checkpointed",
                        "fold": CV_FOLD,
                        "message": "Relaunch the two-GPU scheduler to resume",
                    },
                )
                raise SystemExit(75)
            if code != 0:
                raise RuntimeError(f"Fold {CV_FOLD} training exited with code {code}")
        if not (EXPERIMENT_ROOT / "test/EVALUATION_COMPLETE").is_file():
            atomic_json(progress, {"status": "RUNNING", "phase": "outer_test", "fold": CV_FOLD})
            code = run_stage(
                "test",
                [
                    sys.executable,
                    "-u",
                    str(SCRIPT_DIR / "evaluate_test_v1.py"),
                    "--sw-batch-size",
                    "1",
                ],
                EXPERIMENT_ROOT / "test.log",
            )
            if SIGNALLED:
                atomic_json(
                    progress,
                    {
                        "status": "STOPPED",
                        "phase": "outer_test_partial",
                        "fold": CV_FOLD,
                        "message": "Relaunch the two-GPU scheduler to resume",
                    },
                )
                raise SystemExit(75)
            if code != 0:
                raise RuntimeError(f"Fold {CV_FOLD} outer test exited with code {code}")
        training = read_json(EXPERIMENT_ROOT / "training/summary.json")
        test = read_json(EXPERIMENT_ROOT / "test/summary.json")
        summary = {
            "status": "PASS",
            "fold": CV_FOLD,
            "scientific_role": "patient-disjoint outer five-fold test; inner validation selected the checkpoint",
            "training": training,
            "outer_test": test,
        }
        atomic_json(EXPERIMENT_ROOT / "stage_summary.json", summary)
        (EXPERIMENT_ROOT / "COMPLETE").touch()
        atomic_json(progress, {"status": "PASS", "phase": "complete", "fold": CV_FOLD})
        print(
            f"FOLD {CV_FOLD} COMPLETE | seg={test.get('segmentation_mean_dice_positive')} | "
            f"infiltration Dice/AUPRC={test.get('infiltration_dice_nonempty')}/"
            f"{test.get('infiltration_auprc_nonempty')}",
            flush=True,
        )
    except SystemExit:
        raise
    except Exception as error:
        atomic_json(
            progress,
            {
                "status": "FAILED",
                "phase": "failed",
                "fold": CV_FOLD,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
