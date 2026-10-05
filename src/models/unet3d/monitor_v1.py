#!/usr/bin/env python3
"""Compact one-shot dashboard for all five folds."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from common_v1 import CV_FOLDS, CV_ROOT


def read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def number(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "waiting"


def fold_line(fold: int) -> str:
    root = CV_ROOT / f"fold_{fold}"
    manager = read(root / "manager_progress.json")
    training = read(root / "training/progress.json")
    test = read(root / "test/progress.json")
    status = manager.get("status", "queued")
    phase = manager.get("phase", "waiting")
    if status == "PASS":
        summary = read(root / "test/summary.json")
        return (
            f"Fold {fold}: PASS | best={summary.get('selected_validation_epoch', 'waiting')} | "
            f"seg={number(summary.get('segmentation_mean_dice_positive'))} | "
            f"infiltration Dice/AUPRC={number(summary.get('infiltration_dice_nonempty'))}/"
            f"{number(summary.get('infiltration_auprc_nonempty'))} | "
            f"test={summary.get('test_cases', 'waiting')} pairs"
        )
    if phase == "training" and training:
        train_phase = training.get("phase", "starting")
        if train_phase == "ram_preload":
            return (
                f"Fold {fold}: RUNNING | RAM preload {training.get('preload_stage')} "
                f"{training.get('preload_completed')}/{training.get('preload_total')} | "
                f"cache={number(training.get('ram_cache_gb'), 1)}GB"
            )
        if train_phase == "training":
            return (
                f"Fold {fold}: RUNNING | epoch {training.get('epoch')}/{training.get('epochs')} "
                f"step {training.get('step')}/{training.get('steps')} | "
                f"loss={number(training.get('loss'))} | ETA={number(training.get('eta_minutes'), 1)}m"
            )
        if train_phase == "validation":
            return (
                f"Fold {fold}: RUNNING | validation epoch {training.get('epoch')} "
                f"{training.get('validation_case')}/{training.get('validation_total')} | "
                f"seg={number(training.get('validation_segmentation'))} | "
                f"Dice/AUPRC={number(training.get('validation_infiltration_dice'))}/"
                f"{number(training.get('validation_infiltration_auprc'))}"
            )
        return (
            f"Fold {fold}: {training.get('status', status)} | {train_phase} | "
            f"best={number(training.get('best_score'))}@{training.get('best_epoch', 'waiting')}"
        )
    if phase == "outer_test" and test:
        return (
            f"Fold {fold}: RUNNING | outer test {test.get('completed', 0)}/"
            f"{test.get('total', 'waiting')} | seg={number(test.get('segmentation'))} | "
            f"Dice/AUPRC={number(test.get('infiltration_dice'))}/"
            f"{number(test.get('infiltration_auprc'))}"
        )
    error = manager.get("error")
    return f"Fold {fold}: {status} | {phase}" + (f" | {error}" if error else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.parse_args()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    manager = read(CV_ROOT / "manager_progress.json")
    print(f"Scheduler: {manager.get('status', 'waiting')} | {manager.get('phase', 'waiting')}")
    preflight = read(CV_ROOT / "preflight/progress.json")
    if preflight:
        print(
            f"Preflight: {preflight.get('status')} | {preflight.get('phase')} | "
            f"{preflight.get('completed', preflight.get('eligible_pairs', ''))}/"
            f"{preflight.get('total', preflight.get('eligible_pairs', ''))} | "
            f"GT-rule Dice={number(preflight.get('gt_rule_dice_nonempty', preflight.get('running_gt_rule_dice_nonempty')))}"
        )
    else:
        print("Preflight: waiting")
    for fold in range(CV_FOLDS):
        print(fold_line(fold))
    aggregate = read(CV_ROOT / "aggregate/progress.json")
    if aggregate:
        print(f"Aggregate: {aggregate.get('status')} | {aggregate.get('phase')}")
    else:
        print("Aggregate: waiting")
    if manager.get("error"):
        print(f"Scheduler error: {manager['error']}")


if __name__ == "__main__":
    main()
