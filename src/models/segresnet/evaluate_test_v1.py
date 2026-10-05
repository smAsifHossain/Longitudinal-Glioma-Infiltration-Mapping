#!/usr/bin/env python3
"""One-time evaluation of one held-out MU outer fold."""
from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from common_v1 import (
    CV_FOLD,
    EXPERIMENT_ROOT,
    atomic_csv,
    atomic_json,
    binary_metrics,
    build_model,
    derive_infiltration_rule,
    evaluate_case_arrays,
    finite_mean,
    inference_specialist_pair,
    load_pair,
    save_nifti,
    seed_everything,
    sha256,
    summarize_case_metrics,
    infiltration_volume_bin,
    VOLUME_STAGE_LABELS,
)


def prediction_labelmap(regions: np.ndarray) -> np.ndarray:
    regions = np.asarray(regions, bool)
    return regions[0].astype(np.uint8)


def normalize_display(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, np.float32)
    selected = np.isfinite(array) & (array != 0)
    output = np.zeros_like(array)
    if selected.any():
        low, high = np.percentile(array[selected], [1, 99])
        output[selected] = np.clip((array[selected] - low) / max(high - low, 1e-6), 0, 1)
    return output


def save_qc(
    row: Any,
    pair: dict[str, Any],
    early_prediction: np.ndarray,
    recurrent_prediction: np.ndarray,
    expansion_prediction: np.ndarray,
    output: Path,
    metrics: dict[str, Any] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_early = pair["early_target"][0] > 0.5
    gt_recurrent = pair["recurrent_target"][0] > 0.5
    gt_infiltration = pair["label"][0] > 0.5
    pred_early = early_prediction[0] > 0.5
    pred_recurrent = recurrent_prediction[0] > 0.5
    pred_infiltration = np.asarray(expansion_prediction, bool)
    score = (
        8 * gt_infiltration.sum(axis=(0, 1))
        + 6 * pred_infiltration.sum(axis=(0, 1))
        + 2 * gt_recurrent.sum(axis=(0, 1))
        + 2 * pred_recurrent.sum(axis=(0, 1))
        + 0.25 * gt_early.sum(axis=(0, 1))
        + 0.25 * pred_early.sum(axis=(0, 1))
    )
    z = int(np.argmax(score))
    early_display = normalize_display(pair["early_display"])
    recurrent_display = normalize_display(pair["recurrent_display"])
    figure, axes = plt.subplots(3, 2, figsize=(12, 15), facecolor="black")
    for axis in axes.ravel():
        axis.set_facecolor("black")
        axis.axis("off")

    # Row 1: earlier whole-tumor segmentation.
    axes[0, 0].imshow(early_display[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    if gt_early[:, :, z].any():
        axes[0, 0].contour(gt_early[:, :, z].T, [0.5], colors="red", linewidths=1.5)
    axes[0, 0].set_title("Ground Truth: Earlier Whole Tumor", color="white", fontsize=12)
    axes[0, 1].imshow(early_display[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    if pred_early[:, :, z].any():
        axes[0, 1].contour(pred_early[:, :, z].T, [0.5], colors="red", linewidths=1.5)
    axes[0, 1].set_title("Model Prediction: Earlier Whole Tumor", color="white", fontsize=12)

    # Row 2: registered recurrent whole-tumor segmentation.
    axes[1, 0].imshow(
        recurrent_display[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1
    )
    if gt_recurrent[:, :, z].any():
        axes[1, 0].contour(gt_recurrent[:, :, z].T, [0.5], colors="yellow", linewidths=1.5)
    axes[1, 0].set_title(
        "Ground Truth: Registered Recurrent Whole Tumor", color="white", fontsize=12
    )
    axes[1, 1].imshow(
        recurrent_display[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1
    )
    if pred_recurrent[:, :, z].any():
        axes[1, 1].contour(
            pred_recurrent[:, :, z].T, [0.5], colors="yellow", linewidths=1.5
        )
    axes[1, 1].set_title(
        "Model Prediction: Registered Recurrent Whole Tumor", color="white", fontsize=12
    )

    # Row 3: finalized target versus the rule-derived model prediction.
    axes[2, 0].imshow(
        recurrent_display[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1
    )
    if gt_infiltration[:, :, z].any():
        overlay = np.ma.masked_where(
            ~gt_infiltration[:, :, z].T, gt_infiltration[:, :, z].T
        )
        axes[2, 0].imshow(overlay, cmap="Blues", origin="lower", alpha=0.78, vmin=0, vmax=1)
    axes[2, 0].set_title("Ground Truth: Infiltration Label", color="white", fontsize=12)
    axes[2, 1].imshow(
        recurrent_display[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=1
    )
    if pred_infiltration[:, :, z].any():
        overlay = np.ma.masked_where(
            ~pred_infiltration[:, :, z].T, pred_infiltration[:, :, z].T
        )
        axes[2, 1].imshow(overlay, cmap="Blues", origin="lower", alpha=0.78, vmin=0, vmax=1)
    axes[2, 1].set_title("Model Prediction: Infiltration Label", color="white", fontsize=12)

    metric_text = ""
    if metrics is not None:
        early_dice = float(metrics.get("early_wt_dice", metrics.get("early_WT_dice", np.nan)))
        recurrent_dice = float(
            metrics.get("recurrent_wt_dice", metrics.get("recurrent_WT_dice", np.nan))
        )
        metric_text = (
            f" | Dice: earlier WT {early_dice:.3f}, "
            f"recurrent WT {recurrent_dice:.3f}, "
            f"infiltration {float(metrics['expansion_dice']):.3f}"
        )
    figure.suptitle(
        f"{row.case_id} | axial slice {z}{metric_text}",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.025,
        "Red = earlier whole tumor | Yellow = registered recurrent whole tumor | "
        "Blue = infiltration | T1CE is display background only",
        color="white",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.01, 0.05, 0.99, 0.95))
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{row.case_id}__QC.png", dpi=170, facecolor="black")
    plt.close(figure)


def select_qc_case_ids(case_frame: pd.DataFrame, count: int = 20) -> list[str]:
    """Select deterministic volume- and performance-spanning locked-test cases."""
    if count < 1:
        raise ValueError("QC case count must be positive")
    required = {"case_id", "label_nonempty", "expansion_dice"}
    missing = required - set(case_frame.columns)
    if missing:
        raise RuntimeError(f"Case metrics missing QC selection columns: {sorted(missing)}")
    candidates = case_frame.copy()
    if candidates.empty:
        raise RuntimeError("No evaluated cases are available for QC")
    candidates["volume_bin"] = candidates.label_volume_ml.astype(float).map(
        infiltration_volume_bin
    )
    selected: list[str] = []
    volume_order = ("empty", "<=0.1mL", "0.1-1mL", "1-10mL", ">10mL")
    per_bin = max(1, count // len(volume_order))
    for volume_bin in volume_order:
        group = candidates[candidates.volume_bin == volume_bin].sort_values(
            ["expansion_dice", "case_id"], kind="mergesort"
        )
        if group.empty:
            continue
        positions = np.unique(
            np.rint(
                np.linspace(0, len(group) - 1, num=min(per_bin, len(group)))
            ).astype(int)
        )
        selected.extend(group.iloc[positions].case_id.astype(str).tolist())
    if len(selected) < min(count, len(case_frame)):
        remaining = candidates[~candidates.case_id.astype(str).isin(selected)].sort_values(
            ["expansion_dice", "case_id"], kind="mergesort"
        )
        positions = np.unique(
            np.rint(
                np.linspace(
                    0,
                    max(0, len(remaining) - 1),
                    num=min(count - len(selected), len(remaining)),
                )
            ).astype(int)
        )
        for case_id in remaining.iloc[positions].case_id.astype(str):
            if case_id not in selected:
                selected.append(case_id)
            if len(selected) == min(count, len(case_frame)):
                break
    return selected[:count]


def make_volume_bin_table(case_frame: pd.DataFrame) -> pd.DataFrame:
    frame = case_frame.copy()
    frame["volume_bin"] = frame.label_volume_ml.astype(float).map(infiltration_volume_bin)
    rows: list[dict[str, Any]] = []
    for volume_bin in ("empty", "<=0.1mL", "0.1-1mL", "1-10mL", ">10mL"):
        selected = frame[frame.volume_bin == volume_bin].copy()
        detected = selected[selected.expansion_predicted_voxels.astype(int) > 0]
        row: dict[str, Any] = {
            "volume_bin": volume_bin,
            "volume_stage": VOLUME_STAGE_LABELS[volume_bin],
            "cases": int(len(selected)),
            "patients": int(selected.patient_id.nunique()),
            "volume_ml_mean": finite_mean(selected.label_volume_ml.astype(float)),
            "volume_ml_median": float(selected.label_volume_ml.astype(float).median())
            if len(selected)
            else float("nan"),
            "predicted_positive_cases": int(len(detected)),
        }
        if volume_bin == "empty":
            row.update(
                {
                    "dice": float("nan"),
                    "iou": float("nan"),
                    "auprc": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "lesion_f1": float("nan"),
                    "hd95_mm_detected": float("nan"),
                    "assd_mm_detected": float("nan"),
                    "surface_dice_1mm_detected": float("nan"),
                    "empty_accuracy": finite_mean(
                        (selected.expansion_predicted_voxels.astype(int) == 0).astype(float)
                    ),
                    "empty_false_positive_cases": int(len(detected)),
                }
            )
        else:
            row.update(
                {
                    "dice": finite_mean(selected.expansion_dice.astype(float)),
                    "iou": finite_mean(selected.expansion_iou.astype(float)),
                    "auprc": finite_mean(selected.expansion_auprc.astype(float)),
                    "precision": finite_mean(selected.expansion_precision.astype(float)),
                    "recall": finite_mean(selected.expansion_recall.astype(float)),
                    "lesion_f1": finite_mean(selected.expansion_lesion_f1.astype(float)),
                    "hd95_mm_detected": finite_mean(detected.expansion_hd95_mm.astype(float)),
                    "assd_mm_detected": finite_mean(detected.expansion_assd_mm.astype(float)),
                    "surface_dice_1mm_detected": finite_mean(
                        detected.expansion_surface_dice_1mm.astype(float)
                    ),
                    "empty_accuracy": float("nan"),
                    "empty_false_positive_cases": 0,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def patient_cluster_bootstrap(
    case_frame: pd.DataFrame, repetitions: int = 2000, seed: int = 20260819
) -> pd.DataFrame:
    """Patient-level bootstrap CIs; repeated pairs never become independent units."""
    patients = sorted(case_frame.patient_id.astype(str).unique())
    if len(patients) < 2:
        raise RuntimeError("At least two test patients are required for bootstrap CIs")
    groups = {
        patient: case_frame[case_frame.patient_id.astype(str) == patient].copy()
        for patient in patients
    }
    metric_keys = (
        "early_wt_dice_positive",
        "recurrent_wt_dice_positive",
        "segmentation_mean_dice_positive",
        "infiltration_dice_nonempty",
        "infiltration_iou_nonempty",
        "infiltration_auprc_nonempty",
        "infiltration_precision_nonempty",
        "infiltration_recall_nonempty",
        "infiltration_lesion_f1_nonempty",
        "empty_accuracy",
    )
    estimates = summarize_case_metrics(case_frame)
    samples = {key: [] for key in metric_keys}
    rng = np.random.default_rng(seed)
    for _ in range(int(repetitions)):
        selected_patients = rng.choice(patients, size=len(patients), replace=True)
        resampled = pd.concat(
            [groups[str(patient)] for patient in selected_patients], ignore_index=True
        )
        summary = summarize_case_metrics(resampled)
        for key in metric_keys:
            value = float(summary.get(key, float("nan")))
            if np.isfinite(value):
                samples[key].append(value)
    rows = []
    for key in metric_keys:
        values = np.asarray(samples[key], dtype=np.float64)
        rows.append(
            {
                "metric": key,
                "estimate": float(estimates.get(key, float("nan"))),
                "ci95_low": float(np.percentile(values, 2.5)) if len(values) else float("nan"),
                "ci95_high": float(np.percentile(values, 97.5)) if len(values) else float("nan"),
                "finite_bootstrap_repetitions": int(len(values)),
                "requested_bootstrap_repetitions": int(repetitions),
                "resampling_unit": "patient",
            }
        )
    return pd.DataFrame(rows)


def save_volume_plot(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nonempty = table[table.volume_bin != "empty"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    positions = np.arange(len(nonempty))
    axes[0].bar(positions - 0.18, nonempty.dice, width=0.36, label="Dice")
    axes[0].bar(positions + 0.18, nonempty.auprc, width=0.36, label="AUPRC")
    axes[0].set_xticks(positions, nonempty.volume_bin, rotation=20)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("MU test performance by infiltration volume")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(table.volume_bin, table.cases, color="#4c72b0")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("Cases")
    axes[1].set_title("MU test case counts")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=170, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def save_result_plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Early seg", "Recurrent seg", "Infiltration Dice", "Infiltration AUPRC", "Empty accuracy"]
    values = [
        summary["early_mean_dice_positive"],
        summary["recurrent_mean_dice_positive"],
        summary["infiltration_dice_nonempty"],
        summary["infiltration_auprc_nonempty"],
        summary["empty_accuracy"],
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(names, values, color=["#c44e52", "#ddc000", "#4c72b0", "#55a868", "#8172b3"])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("MU locked-test performance")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    figure.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=170, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def run(output: Path, sw_batch_size: int) -> dict[str, Any]:
    if CV_FOLD is None:
        raise RuntimeError("SEGRESNET_CV_FOLD must be set for fold evaluation")
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    training = EXPERIMENT_ROOT / "training"
    if not (training / "TRAINING_COMPLETE").is_file():
        raise RuntimeError("Training has not completed")
    best_path = training / "checkpoints/best.pt"
    test_path = EXPERIMENT_ROOT / "preflight/mu_test_locked.csv"
    test_lock_path = EXPERIMENT_ROOT / "preflight/test_lock.json"
    if not best_path.is_file() or not test_path.is_file() or not test_lock_path.is_file():
        raise FileNotFoundError("Missing selected checkpoint or locked-test manifest")
    test_lock = json.loads(test_lock_path.read_text(encoding="utf-8"))
    if int(test_lock.get("fold", -1)) != CV_FOLD:
        raise RuntimeError("Held-out test lock belongs to a different CV fold")
    if sha256(test_path) != str(test_lock["manifest_sha256"]):
        raise RuntimeError("Locked MU test manifest changed after preflight")
    test = pd.read_csv(test_path, low_memory=False)
    seed_everything(20260823 + 10_000 * CV_FOLD)
    torch.set_num_threads(
        min(
            max(1, int(os.environ.get("SEGRESNET_CPU_THREADS", "8"))),
            max(1, os.cpu_count() or 1),
        )
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This pilot requires exactly one visible GPU; found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    early_model = build_model()
    recurrent_model = build_model()
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("cv_fold", -1)) != CV_FOLD:
        raise RuntimeError("Selected checkpoint belongs to a different CV fold")
    early_model.load_state_dict(checkpoint["early_model"], strict=True)
    recurrent_model.load_state_dict(checkpoint["recurrent_model"], strict=True)
    early_model.to(device).eval()
    recurrent_model.to(device).eval()
    prediction_root = output / "predictions"
    partial_path = output / "case_metrics.partial.csv"
    existing_records: dict[str, dict[str, Any]] = {}
    if partial_path.is_file():
        partial = pd.read_csv(partial_path, low_memory=False)
        for column in (
            "label_nonempty", "early_wt_target_nonempty",
            "recurrent_wt_target_nonempty",
        ):
            if column in partial.columns:
                partial[column] = partial[column].astype(str).str.lower().isin(
                    {"true", "1", "yes"}
                )
        if partial.case_id.astype(str).duplicated().any():
            raise RuntimeError("Duplicate cases in resumable MU-test metrics")
        existing_records = {
            str(record["case_id"]): record for record in partial.to_dict("records")
        }
    records: list[dict[str, Any]] = []
    atomic_json(
        progress,
        {"status": "RUNNING", "phase": "test", "completed": 0, "total": len(test)},
    )
    for index, row in enumerate(test.itertuples(index=False), start=1):
        case_output = prediction_root / str(row.case_id)
        expected_predictions = (
            case_output / "earlier_segmentation_pred.nii.gz",
            case_output / "recurrent_segmentation_pred.nii.gz",
            case_output / "infiltration_pred.nii.gz",
        )
        if str(row.case_id) in existing_records and all(
            path.is_file() for path in expected_predictions
        ):
            records.append(existing_records[str(row.case_id)])
            running = summarize_case_metrics(pd.DataFrame(records))
            atomic_json(
                progress,
                {
                    "status": "RUNNING", "phase": "test", "completed": index,
                    "total": len(test), "case_id": str(row.case_id),
                    "segmentation": running["segmentation_mean_dice_positive"],
                    "infiltration_dice": running["infiltration_dice_nonempty"],
                    "infiltration_auprc": running["infiltration_auprc_nonempty"],
                    "empty_accuracy": running["empty_accuracy"], "resumed_case": True,
                },
            )
            print(f"TEST {index:2d}/{len(test)} | {row.case_id} | restored", flush=True)
            continue
        pair = load_pair(row)
        early_probability, recurrent_probability = inference_specialist_pair(
            early_model,
            recurrent_model,
            pair,
            device,
            sw_batch_size=sw_batch_size,
        )
        metrics, arrays = evaluate_case_arrays(
            early_probability, recurrent_probability, pair, extended=True
        )
        gt_early = pair["early_target"][0] > 0.5
        gt_recurrent = pair["recurrent_target"][0] > 0.5
        label = pair["label"][0] > 0.5
        pred_early = arrays["early_prediction"][0]
        pred_recurrent = arrays["recurrent_prediction"][0]
        metrics["oracle_gt_early_gt_recurrent_dice"] = binary_metrics(
            derive_infiltration_rule(gt_early, gt_recurrent, pair["spacing"]), label
        )["dice"]
        metrics["oracle_gt_early_pred_recurrent_dice"] = binary_metrics(
            derive_infiltration_rule(gt_early, pred_recurrent, pair["spacing"]), label
        )["dice"]
        metrics["oracle_pred_early_gt_recurrent_dice"] = binary_metrics(
            derive_infiltration_rule(pred_early, gt_recurrent, pair["spacing"]), label
        )["dice"]
        record = {
            "dataset": "MU-Glioma-Post",
            "case_id": str(row.case_id),
            "patient_id": str(row.patient_id),
            "split": f"cv_outer_test_fold_{CV_FOLD}",
            "cv_fold": CV_FOLD,
            **metrics,
        }
        records.append(record)
        save_nifti(
            prediction_labelmap(arrays["early_prediction"]),
            pair["reference"],
            case_output / "earlier_segmentation_pred.nii.gz",
        )
        save_nifti(
            prediction_labelmap(arrays["recurrent_prediction"]),
            pair["reference"],
            case_output / "recurrent_segmentation_pred.nii.gz",
        )
        save_nifti(
            arrays["expansion_prediction"],
            pair["reference"],
            case_output / "infiltration_pred.nii.gz",
        )
        atomic_csv(pd.DataFrame(records), partial_path)
        running = summarize_case_metrics(pd.DataFrame(records))
        atomic_json(
            progress,
            {
                "status": "RUNNING",
                "phase": "test",
                "completed": index,
                "total": len(test),
                "case_id": str(row.case_id),
                "segmentation": running["segmentation_mean_dice_positive"],
                "infiltration_dice": running["infiltration_dice_nonempty"],
                "infiltration_auprc": running["infiltration_auprc_nonempty"],
                "empty_accuracy": running["empty_accuracy"],
            },
        )
        print(
            f"TEST {index:2d}/{len(test)} | {row.case_id} | "
            f"seg={running['segmentation_mean_dice_positive']:.4f} | "
            f"infiltration Dice/AUPRC={running['infiltration_dice_nonempty']:.4f}/"
            f"{running['infiltration_auprc_nonempty']:.4f}",
            flush=True,
        )
        del pair, early_probability, recurrent_probability, arrays
        gc.collect()
    case_frame = pd.DataFrame(records)
    atomic_csv(case_frame, output / "case_metrics.csv")
    summary = summarize_case_metrics(case_frame)
    summary.update(
        {
            "status": "PASS",
            "model": "SegResNet",
            "role": "held-out patient-disjoint MU outer-test fold; evaluated once after inner validation selected the checkpoint",
            "cv_fold": CV_FOLD,
            "selected_validation_epoch": int(checkpoint["epoch"]),
            "selected_validation_score": float(checkpoint["best_score"]),
            "threshold": 0.5,
            "one_sided_oracle": {
                "gt_early_gt_recurrent_dice": finite_mean(
                    case_frame.oracle_gt_early_gt_recurrent_dice.astype(float)
                ),
                "gt_early_pred_recurrent_dice": finite_mean(
                    case_frame.oracle_gt_early_pred_recurrent_dice.astype(float)
                ),
                "pred_early_gt_recurrent_dice": finite_mean(
                    case_frame.oracle_pred_early_gt_recurrent_dice.astype(float)
                ),
                "pred_early_pred_recurrent_dice": summary["infiltration_dice_nonempty"],
            },
        }
    )
    volume_table = make_volume_bin_table(case_frame)
    atomic_csv(volume_table, output / "volume_bin_metrics.csv")
    save_volume_plot(volume_table, output / "volume_bin_performance.png")
    bootstrap = patient_cluster_bootstrap(case_frame)
    atomic_csv(bootstrap, output / "patient_bootstrap_95ci.csv")
    summary["volume_bin_metrics_file"] = str(
        (output / "volume_bin_metrics.csv").resolve()
    )
    summary["patient_bootstrap_95ci_file"] = str(
        (output / "patient_bootstrap_95ci.csv").resolve()
    )
    summary["test_dataset"] = "MU-Glioma-Post"
    summary["test_cases"] = int(len(case_frame))
    summary["test_patients"] = int(case_frame.patient_id.nunique())
    summary["test_manifest_sha256"] = str(test_lock["manifest_sha256"])
    atomic_json(output / "summary.json", summary)
    save_result_plot(summary, output / "test_metrics.png")

    selected_ids = sorted(case_frame.case_id.astype(str).tolist())
    if len(selected_ids) != len(test) or len(set(selected_ids)) != len(test):
        raise RuntimeError(
            f"Expected QC for all {len(test)} MU test pairs; found {len(selected_ids)}"
        )
    qc_output = output / "qc_3x2_all_test_cases"
    qc_output.mkdir(parents=True, exist_ok=True)
    for old_image in qc_output.glob("*__QC.png"):
        old_image.unlink()
    row_lookup = {str(row.case_id): row for row in test.itertuples(index=False)}
    metric_lookup = case_frame.set_index(case_frame.case_id.astype(str), drop=False)
    for case_id in selected_ids:
        row = row_lookup[case_id]
        pair = load_pair(row)
        case_output = prediction_root / case_id
        early_map = np.asarray(nib.load(case_output / "earlier_segmentation_pred.nii.gz").dataobj)
        recurrent_map = np.asarray(nib.load(case_output / "recurrent_segmentation_pred.nii.gz").dataobj)
        early_regions = np.stack([early_map > 0])
        recurrent_regions = np.stack([recurrent_map > 0])
        expansion = np.asarray(nib.load(case_output / "infiltration_pred.nii.gz").dataobj) > 0.5
        save_qc(
            row,
            pair,
            early_regions,
            recurrent_regions,
            expansion,
            qc_output,
            metrics=metric_lookup.loc[case_id].to_dict(),
        )
        del pair
        gc.collect()
    qc_manifest = case_frame[case_frame.case_id.astype(str).isin(selected_ids)].copy()
    qc_manifest["qc_order"] = qc_manifest.case_id.astype(str).map(
        {case_id: index + 1 for index, case_id in enumerate(selected_ids)}
    )
    qc_manifest["volume_bin"] = qc_manifest.label_volume_ml.astype(float).map(
        infiltration_volume_bin
    )
    atomic_csv(qc_manifest.sort_values("qc_order"), qc_output / "qc_manifest.csv")
    atomic_json(
        qc_output / "summary.json",
        {
            "status": "PASS",
            "requested": len(test),
            "generated": len(selected_ids),
            "selection": "all locked MU test cases",
            "case_ids": selected_ids,
        },
    )
    summary["qc_images"] = len(selected_ids)
    summary["qc_directory"] = str(qc_output.resolve())
    summary["qc_manifest"] = str((qc_output / "qc_manifest.csv").resolve())
    atomic_json(output / "summary.json", summary)
    (output / "EVALUATION_COMPLETE").touch()
    partial_path.unlink(missing_ok=True)
    atomic_json(
        progress,
        {
            "status": "PASS",
            "phase": "complete",
            "cases": len(case_frame),
            "segmentation": summary["segmentation_mean_dice_positive"],
            "infiltration_dice": summary["infiltration_dice_nonempty"],
            "infiltration_auprc": summary["infiltration_auprc_nonempty"],
            "empty_accuracy": summary["empty_accuracy"],
            "qc_images": len(selected_ids),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "test")
    parser.add_argument("--sw-batch-size", type=int, default=1)
    args = parser.parse_args()
    try:
        run(args.output, args.sw_batch_size)
    except Exception as error:
        atomic_json(
            args.output / "progress.json",
            {"status": "FAILED", "phase": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
