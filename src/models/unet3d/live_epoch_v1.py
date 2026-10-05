#!/usr/bin/env python3
"""Interactive single-line epoch display; completed epochs remain on separate lines."""
from __future__ import annotations

import json
import math
import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from common_v1 import CV_FOLDS, CV_ROOT


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def value(item: Any, digits: int = 3) -> str:
    try:
        result = float(item)
        return f"{result:.{digits}f}" if math.isfinite(result) else "nan"
    except Exception:
        return "--"


def wt_value(items: Any) -> str:
    if not isinstance(items, list) or len(items) != 1:
        return "--"
    return value(items[0])


def completed_line(row: pd.Series, fold: int) -> str:
    return (
        f"Fold {fold} | Epoch {int(row.epoch):02d} COMPLETE | loss={value(row.train_loss, 4)} | "
        f"E seg={value(row.val_early_mean_dice_positive, 4)} | "
        f"R seg={value(row.val_recurrent_mean_dice_positive, 4)} | "
        f"Infiltration Dice/AUPRC={value(row.val_infiltration_dice_nonempty, 4)}/"
        f"{value(row.val_infiltration_auprc_nonempty, 4)} | "
        f"empty={value(row.val_empty_accuracy, 4)} | "
        f"best={value(row.best_score, 4)}@{int(row.best_epoch)} | {value(row.epoch_minutes, 1)}m"
    )


def dynamic_line(root: Path) -> str:
    manager = read_json(root / "manager_progress.json")
    phase = manager.get("phase", "waiting")
    if phase == "preflight":
        state = read_json(root / "preflight/progress.json")
        return (
            f"Preflight | {state.get('phase', 'starting')} | "
            f"{state.get('completed', 0)}/{state.get('total', 391)} | "
            f"GT-rule Dice={value(state.get('running_gt_rule_dice_nonempty'), 4)}"
        )
    if phase == "training":
        state = read_json(root / "training/progress.json")
        stage = state.get("phase", "starting")
        config = read_json(root / "training/config.json")
        epochs = state.get("epochs") or config.get("epochs", 30)
        if stage == "ram_preload":
            return (
                f"RAM preload {state.get('preload_stage', '--')} | "
                f"{state.get('preload_completed', 0)}/{state.get('preload_total', 0)} | "
                f"cache={value(state.get('ram_cache_gb'), 1)}GB | {state.get('case_id', '--')}"
            )
        if stage == "training":
            return (
                f"Epoch {int(state.get('epoch', 0)):02d}/{epochs} TRAIN "
                f"{state.get('step', 0):03d}/{state.get('steps', 120)} | "
                f"loss={value(state.get('loss'), 4)} | soft E-WT={wt_value(state.get('early_soft'))} | "
                f"R-WT={wt_value(state.get('recurrent_soft'))} | cases={state.get('distinct_cases', 0)} | "
                f"wait={value(100 * float(state.get('data_wait_fraction', 0)), 0)}% | "
                f"ETA={value(state.get('eta_minutes'), 1)}m"
            )
        if stage == "validation":
            return (
                f"Epoch {int(state.get('epoch', 0)):02d}/{epochs} VALID "
                f"{state.get('validation_case', 0):02d}/{state.get('validation_total', 0)} | "
                f"seg={value(state.get('validation_segmentation'), 4)} | "
                f"Infiltration Dice/AUPRC={value(state.get('validation_infiltration_dice'), 4)}/"
                f"{value(state.get('validation_infiltration_auprc'), 4)} | "
                f"empty={value(state.get('validation_empty_accuracy'), 4)} | "
                f"ETA={value(state.get('eta_minutes'), 1)}m"
            )
        return f"Training | {stage} | preparing next update"
    if phase == "outer_test":
        state = read_json(root / "test/progress.json")
        return (
            f"MU OUTER TEST {state.get('completed', 0)}/{state.get('total', '--')} | "
            f"seg={value(state.get('segmentation'), 4)} | "
            f"Infiltration Dice/AUPRC={value(state.get('infiltration_dice'), 4)}/"
            f"{value(state.get('infiltration_auprc'), 4)} | "
            f"empty={value(state.get('empty_accuracy'), 4)}"
        )
    return f"Manager | {manager.get('status', 'waiting')} | {phase}"


def clear_line() -> None:
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(CV_FOLDS))
    args = parser.parse_args()
    root = CV_ROOT / f"fold_{args.fold}"
    history_path = root / "training/history.csv"
    printed_epochs: set[int] = set()
    last_dynamic = ""
    interactive = sys.stdout.isatty()
    while True:
        if history_path.is_file():
            try:
                history = pd.read_csv(history_path).sort_values("epoch")
                for _, row in history.iterrows():
                    epoch = int(row.epoch)
                    if epoch not in printed_epochs:
                        if interactive:
                            clear_line()
                        print(completed_line(row, args.fold), flush=True)
                        printed_epochs.add(epoch)
                        last_dynamic = ""
            except Exception:
                pass
        manager = read_json(root / "manager_progress.json")
        line = dynamic_line(root)
        if interactive:
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
        elif line != last_dynamic:
            print(line, flush=True)
        last_dynamic = line
        status = manager.get("status")
        if status in ("PASS", "FAILED", "STOPPED"):
            if interactive:
                clear_line()
            if status == "PASS":
                summary = read_json(root / "test/summary.json")
                print(
                    f"FOLD {args.fold} COMPLETE | Held-out MU outer test | "
                    f"seg={value(summary.get('segmentation_mean_dice_positive'), 4)} | "
                    f"Infiltration Dice/AUPRC={value(summary.get('infiltration_dice_nonempty'), 4)}/"
                    f"{value(summary.get('infiltration_auprc_nonempty'), 4)} | "
                    f"empty={value(summary.get('empty_accuracy'), 4)}",
                    flush=True,
                )
            else:
                print(f"Manager {status}: {manager.get('error', manager.get('message', 'see logs'))}", flush=True)
            return
        time.sleep(2.0)


if __name__ == "__main__":
    main()
