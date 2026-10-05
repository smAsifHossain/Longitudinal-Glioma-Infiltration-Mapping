#!/usr/bin/env python3
"""Two-GPU scheduler for resumable MU MedNeXt-S five-fold CV."""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from common_v1 import CV_FOLDS, CV_ROOT, SCRIPT_DIR, atomic_json


STOP_REQUESTED = False
ACTIVE: dict[int, dict[str, Any]] = {}


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    (CV_ROOT / "STOP_REQUESTED").touch()
    print(f"Scheduler received signal {signum}; stopping active folds safely.", flush=True)
    for item in ACTIVE.values():
        process: subprocess.Popen = item["process"]
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)


def _base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("MEDNEXT_CV_FOLD", None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    environment["MALLOC_ARENA_MAX"] = "4"
    return environment


def _fold_environment(fold: int, gpu: str) -> dict[str, str]:
    environment = _base_environment()
    environment["MEDNEXT_CV_FOLD"] = str(fold)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["MEDNEXT_CPU_THREADS"] = "8"
    environment["OMP_NUM_THREADS"] = "4"
    environment["MKL_NUM_THREADS"] = "4"
    temporary = Path(os.environ.get("GLIOMA_TMP_ROOT", "/tmp/glioma_infiltration")) / "mednext_mu_5fold" / f"fold_{fold}"
    for name in ("cuda", "cache", "tmp", "mpl"):
        (temporary / name).mkdir(parents=True, exist_ok=True)
    environment["CUDA_CACHE_PATH"] = str(temporary / "cuda")
    environment["XDG_CACHE_HOME"] = str(temporary / "cache")
    environment["TMPDIR"] = str(temporary / "tmp")
    environment["MPLCONFIGDIR"] = str(temporary / "mpl")
    return environment


def _run_preflight(gpu: str) -> None:
    if (CV_ROOT / "preflight/PASS").is_file():
        return
    atomic_json(
        CV_ROOT / "manager_progress.json",
        {"status": "RUNNING", "phase": "preflight", "gpu": gpu},
    )
    environment = _fold_environment(0, gpu)
    environment.pop("MEDNEXT_CV_FOLD", None)
    log_path = CV_ROOT / "preflight.log"
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(SCRIPT_DIR / "preflight_cv_v1.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        (CV_ROOT / "preflight.pid").write_text(f"{process.pid}\n", encoding="utf-8")
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Global five-fold preflight exited with code {code}")


def _start_fold(fold: int, gpu: str) -> None:
    fold_root = CV_ROOT / f"fold_{fold}"
    fold_root.mkdir(parents=True, exist_ok=True)
    log_handle = (fold_root / "manager.log").open("a", buffering=1)
    process = subprocess.Popen(
        [sys.executable, "-u", str(SCRIPT_DIR / "fold_manager_v1.py")],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=_fold_environment(fold, gpu),
    )
    (fold_root / "manager.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    ACTIVE[fold] = {"process": process, "gpu": gpu, "log": log_handle}
    print(f"SCHEDULED fold {fold} on physical GPU {gpu} as PID {process.pid}", flush=True)


def _status_payload(pending: list[int], completed: list[int]) -> dict[str, Any]:
    return {
        "status": "RUNNING",
        "phase": "cross_validation",
        "active_folds": {
            str(fold): {"gpu": item["gpu"], "pid": item["process"].pid}
            for fold, item in ACTIVE.items()
        },
        "pending_folds": pending,
        "completed_folds": sorted(completed),
        "folds_total": CV_FOLDS,
    }


def _run_folds(gpus: list[str]) -> None:
    completed = [fold for fold in range(CV_FOLDS) if (CV_ROOT / f"fold_{fold}/COMPLETE").is_file()]
    pending = [fold for fold in range(CV_FOLDS) if fold not in completed]
    free_gpus = list(gpus)
    while pending or ACTIVE:
        if STOP_REQUESTED or (CV_ROOT / "STOP_REQUESTED").is_file():
            for item in ACTIVE.values():
                process: subprocess.Popen = item["process"]
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            for item in ACTIVE.values():
                item["process"].wait()
                item["log"].close()
            ACTIVE.clear()
            raise SystemExit(75)
        while pending and free_gpus:
            _start_fold(pending.pop(0), free_gpus.pop(0))
        atomic_json(CV_ROOT / "manager_progress.json", _status_payload(pending, completed))
        time.sleep(5)
        for fold in list(ACTIVE):
            item = ACTIVE[fold]
            process: subprocess.Popen = item["process"]
            code = process.poll()
            if code is None:
                continue
            item["log"].close()
            gpu = str(item["gpu"])
            del ACTIVE[fold]
            free_gpus.append(gpu)
            free_gpus.sort(key=gpus.index)
            if code == 75:
                raise SystemExit(75)
            if code != 0:
                raise RuntimeError(f"Fold {fold} manager exited with code {code}")
            if not (CV_ROOT / f"fold_{fold}/COMPLETE").is_file():
                raise RuntimeError(f"Fold {fold} manager exited without COMPLETE")
            completed.append(fold)
            print(f"SCHEDULER recorded fold {fold} complete", flush=True)


def _run_aggregate() -> None:
    if (CV_ROOT / "aggregate/AGGREGATION_COMPLETE").is_file():
        return
    atomic_json(
        CV_ROOT / "manager_progress.json",
        {"status": "RUNNING", "phase": "aggregation", "completed_folds": list(range(CV_FOLDS))},
    )
    environment = _base_environment()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    log_path = CV_ROOT / "aggregate.log"
    with log_path.open("a", buffering=1) as log:
        code = subprocess.call(
            [sys.executable, "-u", str(SCRIPT_DIR / "aggregate_cv_v1.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    if code != 0:
        raise RuntimeError(f"Five-fold aggregation exited with code {code}")


def main() -> None:
    CV_ROOT.mkdir(parents=True, exist_ok=True)
    lock = (CV_ROOT / "manager.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another five-fold scheduler already holds the lock")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    progress = CV_ROOT / "manager_progress.json"
    try:
        (CV_ROOT / "STOP_REQUESTED").unlink(missing_ok=True)
        visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
        if len(visible) != 2:
            raise RuntimeError(
                f"Expose exactly two GPUs for the scheduler; found {visible or 'none'}"
            )
        if (CV_ROOT / "COMPLETE").is_file():
            atomic_json(progress, {"status": "PASS", "phase": "complete"})
            return
        _run_preflight(visible[0])
        _run_folds(visible)
        _run_aggregate()
        (CV_ROOT / "COMPLETE").touch()
        atomic_json(
            progress,
            {"status": "PASS", "phase": "complete", "completed_folds": list(range(CV_FOLDS))},
        )
        print("MedNeXt-S MU five-fold cross-validation complete.", flush=True)
    except SystemExit as error:
        atomic_json(
            progress,
            {
                "status": "STOPPED",
                "phase": "checkpointed",
                "message": "Relaunch the same command to resume all incomplete folds",
            },
        )
        raise error
    except Exception as error:
        for item in ACTIVE.values():
            process: subprocess.Popen = item["process"]
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        atomic_json(
            progress,
            {"status": "FAILED", "phase": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
