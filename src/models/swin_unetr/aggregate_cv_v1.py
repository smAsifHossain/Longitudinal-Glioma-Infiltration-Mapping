#!/usr/bin/env python3
"""Aggregate all held-out predictions after five-fold CV completes."""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_v1 import (
    CV_FOLDS,
    CV_ROOT,
    atomic_csv,
    atomic_json,
    finite_mean,
    sha256,
    summarize_case_metrics,
)
from evaluate_test_v1 import (
    make_volume_bin_table,
    patient_cluster_bootstrap,
    save_volume_plot,
)


PRIMARY_METRICS = (
    "early_wt_dice_positive",
    "recurrent_wt_dice_positive",
    "segmentation_mean_dice_positive",
    "infiltration_dice_nonempty",
    "infiltration_iou_nonempty",
    "infiltration_precision_nonempty",
    "infiltration_recall_nonempty",
    "infiltration_auprc_nonempty",
    "infiltration_lesion_f1_nonempty",
    "infiltration_hd95_mm_detected",
    "infiltration_assd_mm_detected",
    "infiltration_surface_dice_1mm_detected",
    "empty_accuracy",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_fold_metrics(frame: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("segmentation_mean_dice_positive", "Segmentation Dice"),
        ("infiltration_dice_nonempty", "Infiltration Dice"),
        ("infiltration_auprc_nonempty", "Infiltration AUPRC"),
    ]
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    x = np.arange(CV_FOLDS)
    width = 0.24
    for index, (column, label) in enumerate(metrics):
        axis.bar(x + (index - 1) * width, frame[column], width=width, label=label)
    axis.set_xticks(x, [f"Fold {fold}" for fold in frame.fold])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Swin-UNETR held-out five-fold performance")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    atomic_json(progress, {"status": "RUNNING", "phase": "collecting_folds"})
    fold_rows: list[dict[str, Any]] = []
    case_frames: list[pd.DataFrame] = []
    qc_frames: list[pd.DataFrame] = []
    for fold in range(CV_FOLDS):
        fold_root = CV_ROOT / f"fold_{fold}"
        if not (fold_root / "COMPLETE").is_file():
            raise RuntimeError(f"Fold {fold} is incomplete")
        summary = _read_json(fold_root / "test/summary.json")
        training = _read_json(fold_root / "training/summary.json")
        cases = pd.read_csv(fold_root / "test/case_metrics.csv", low_memory=False)
        cases["cv_fold"] = fold
        for column in (
            "label_nonempty",
            "early_wt_target_nonempty",
            "recurrent_wt_target_nonempty",
        ):
            if column in cases:
                cases[column] = cases[column].astype(str).str.lower().isin(
                    {"true", "1", "yes"}
                )
        case_frames.append(cases)
        row: dict[str, Any] = {
            "fold": fold,
            "test_pairs": int(len(cases)),
            "test_patients": int(cases.patient_id.nunique()),
            "best_epoch": int(training["best_epoch"]),
            "best_validation_score": float(training["best_score"]),
        }
        for metric in PRIMARY_METRICS:
            row[metric] = float(summary.get(metric, float("nan")))
        fold_rows.append(row)
        qc_path = fold_root / "test/qc_3x2_all_test_cases/qc_manifest.csv"
        if not qc_path.is_file():
            raise FileNotFoundError(qc_path)
        qc = pd.read_csv(qc_path, low_memory=False)
        qc["cv_fold"] = fold
        qc["qc_directory"] = str(qc_path.parent.resolve())
        qc_frames.append(qc)
    fold_frame = pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)
    atomic_csv(fold_frame, output / "fold_metrics.csv")
    _plot_fold_metrics(fold_frame, output / "fold_performance.png")

    cases = pd.concat(case_frames, ignore_index=True, sort=False)
    if len(cases) != 353 or cases.case_id.astype(str).nunique() != 353:
        raise RuntimeError(
            f"Expected 353 unique out-of-fold scan pairs; found {len(cases)}/"
            f"{cases.case_id.astype(str).nunique()}"
        )
    if cases.patient_id.nunique() != 146:
        raise RuntimeError(
            f"Expected 146 out-of-fold patients; found {cases.patient_id.nunique()}"
        )
    registry_path = CV_ROOT / "preflight/outer_fold_registry.csv"
    lock = _read_json(CV_ROOT / "preflight/outer_fold_registry_lock.json")
    if sha256(registry_path) != str(lock["manifest_sha256"]):
        raise RuntimeError("Outer-fold registry changed after preflight")
    registry = pd.read_csv(registry_path, usecols=["case_id", "patient_id", "outer_fold"])
    observed = cases[["case_id", "patient_id", "cv_fold"]].copy()
    observed["case_id"] = observed.case_id.astype(str)
    observed["patient_id"] = observed.patient_id.astype(str)
    registry["case_id"] = registry.case_id.astype(str)
    registry["patient_id"] = registry.patient_id.astype(str)
    merged = registry.merge(observed, on=["case_id", "patient_id"], how="outer", indicator=True)
    if not bool((merged._merge == "both").all()):
        raise RuntimeError("Out-of-fold predictions do not match the locked registry")
    if not bool((merged.outer_fold.astype(int) == merged.cv_fold.astype(int)).all()):
        raise RuntimeError("At least one case was evaluated in the wrong outer fold")
    atomic_csv(cases.sort_values(["cv_fold", "patient_id", "case_id"]), output / "oof_case_metrics.csv")

    pooled = summarize_case_metrics(cases)
    pooled.update(
        {
            "status": "PASS",
            "model": "Swin-UNETR",
            "role": "pooled out-of-fold predictions from patient-grouped five-fold cross-validation",
            "folds": CV_FOLDS,
            "oof_pairs": int(len(cases)),
            "oof_patients": int(cases.patient_id.nunique()),
            "outer_fold_registry_sha256": str(lock["manifest_sha256"]),
            "one_sided_oracle": {
                "gt_early_gt_recurrent_dice": finite_mean(
                    cases.oracle_gt_early_gt_recurrent_dice.astype(float)
                ),
                "gt_early_pred_recurrent_dice": finite_mean(
                    cases.oracle_gt_early_pred_recurrent_dice.astype(float)
                ),
                "pred_early_gt_recurrent_dice": finite_mean(
                    cases.oracle_pred_early_gt_recurrent_dice.astype(float)
                ),
                "pred_early_pred_recurrent_dice": float(
                    pooled["infiltration_dice_nonempty"]
                ),
            },
        }
    )
    atomic_json(output / "pooled_oof_summary.json", pooled)

    macro_rows = []
    for metric in PRIMARY_METRICS:
        values = pd.to_numeric(fold_frame[metric], errors="coerce").to_numpy(float)
        finite = values[np.isfinite(values)]
        macro_rows.append(
            {
                "metric": metric,
                "fold_mean": float(finite.mean()) if len(finite) else float("nan"),
                "fold_sd": float(finite.std(ddof=1)) if len(finite) > 1 else float("nan"),
                "fold_minimum": float(finite.min()) if len(finite) else float("nan"),
                "fold_maximum": float(finite.max()) if len(finite) else float("nan"),
                "pooled_oof_estimate": float(pooled.get(metric, float("nan"))),
                "finite_folds": int(len(finite)),
            }
        )
    macro = pd.DataFrame(macro_rows)
    atomic_csv(macro, output / "macro_mean_sd_and_pooled.csv")

    volume = make_volume_bin_table(cases)
    atomic_csv(volume, output / "volume_bin_metrics.csv")
    save_volume_plot(volume, output / "volume_bin_performance.png")
    bootstrap = patient_cluster_bootstrap(cases, repetitions=2000, seed=20260823)
    atomic_csv(bootstrap, output / "patient_bootstrap_95ci.csv")
    qc = pd.concat(qc_frames, ignore_index=True, sort=False)
    if len(qc) != 353 or qc.case_id.astype(str).nunique() != 353:
        raise RuntimeError("Expected QC for all 353 out-of-fold scan pairs")
    atomic_csv(qc.sort_values(["cv_fold", "case_id"]), output / "oof_qc_manifest.csv")
    summary = {
        "status": "PASS",
        "model": "Swin-UNETR",
        "folds": CV_FOLDS,
        "eligible_pairs": 353,
        "eligible_patients": 146,
        "all_pairs_tested_exactly_once": True,
        "patient_leakage": False,
        "fold_metrics_file": str((output / "fold_metrics.csv").resolve()),
        "macro_metrics_file": str((output / "macro_mean_sd_and_pooled.csv").resolve()),
        "pooled_oof_summary_file": str((output / "pooled_oof_summary.json").resolve()),
        "volume_bin_metrics_file": str((output / "volume_bin_metrics.csv").resolve()),
        "patient_bootstrap_95ci_file": str((output / "patient_bootstrap_95ci.csv").resolve()),
        "oof_case_metrics_file": str((output / "oof_case_metrics.csv").resolve()),
        "oof_qc_images": int(len(qc)),
        "oof_qc_manifest": str((output / "oof_qc_manifest.csv").resolve()),
        "primary_reporting": "fold mean +/- SD plus pooled out-of-fold estimate and patient-bootstrap 95% CI",
    }
    atomic_json(output / "summary.json", summary)
    (output / "AGGREGATION_COMPLETE").touch()
    atomic_json(progress, {"status": "PASS", "phase": "complete", **summary})
    print(
        "AGGREGATION COMPLETE | "
        f"seg={pooled['segmentation_mean_dice_positive']:.4f} | "
        f"infiltration Dice/AUPRC={pooled['infiltration_dice_nonempty']:.4f}/"
        f"{pooled['infiltration_auprc_nonempty']:.4f}",
        flush=True,
    )
    return summary


def main() -> None:
    output = CV_ROOT / "aggregate"
    try:
        run(output)
    except Exception as error:
        atomic_json(
            output / "progress.json",
            {"status": "FAILED", "phase": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
