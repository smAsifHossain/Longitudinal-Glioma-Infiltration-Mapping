#!/usr/bin/env python3
"""Shared paths, checks, and statistics for the MU five-fold OOF volume analysis."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = Path(
    os.environ.get("GLIOMA_PROJECT_ROOT", str(REPOSITORY_ROOT))
).expanduser().resolve()
FINETUNING_ROOT = V5_ROOT / "finetuning" / "experiments"
OUTPUT_ROOT = Path(
    os.environ.get(
        "BIOMARKER_OUTPUT_ROOT",
        str(
            V5_ROOT
            / "biomarker_analysis"
            / "experiments"
            / "MU_5fold_OOF_volume_agreement_v1"
        ),
    )
)

MODEL_SPECS = OrderedDict(
    [
        ("Swin-UNETR", "SwinUNETR_MU_5fold_CV_v1"),
        ("nnU-Net ResEnc-L", "nnUNetResEncL_MU_5fold_CV_v1"),
        ("MedNeXt-S", "MedNeXtS_MU_5fold_CV_v1"),
        ("SegResNet", "SegResNet_MU_5fold_CV_v1"),
        ("3D U-Net", "UNet3D_MU_5fold_CV_v1"),
    ]
)
PRIMARY_MODEL = "Swin-UNETR"
EXPECTED_PAIRS = 353
EXPECTED_PATIENTS = 146
EXPECTED_NONEMPTY = 346
EXPECTED_EMPTY = 7
EXPECTED_FOLDS = {0, 1, 2, 3, 4}
EXPECTED_REGISTRY_SHA256 = (
    "3be56ada8ae69a62d3abedb7af282fe3e4ecef3d21811ffe4b28269dc227f677"
)
BOOTSTRAP_REPETITIONS = int(os.environ.get("BIOMARKER_BOOTSTRAPS", "2000"))
BOOTSTRAP_SEED = 20260825

REQUIRED_COLUMNS = {
    "case_id",
    "patient_id",
    "cv_fold",
    "label_nonempty",
    "label_volume_ml",
    "expansion_target_voxels",
    "expansion_predicted_voxels",
}

GROUP_ORDER = ["empty", "<=1mL", "1-10mL", ">10mL"]
GROUP_LABELS = {
    "empty": "Empty",
    "<=1mL": "≤1 mL",
    "1-10mL": "1–10 mL",
    ">10mL": ">10 mL",
}


def model_root(model: str) -> Path:
    return FINETUNING_ROOT / MODEL_SPECS[model]


def model_case_path(model: str) -> Path:
    return model_root(model) / "aggregate" / "oof_case_metrics.csv"


def model_summary_path(model: str) -> Path:
    return model_root(model) / "aggregate" / "pooled_oof_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    return value


def atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_progress(status: str, phase: str, message: str, **extra) -> None:
    atomic_json(
        OUTPUT_ROOT / "progress.json",
        {"status": status, "phase": phase, "message": message, **extra},
    )


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_model_cases(model: str) -> pd.DataFrame:
    frame = pd.read_csv(model_case_path(model))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{model}: missing OOF columns: {sorted(missing)}")
    frame["case_id"] = frame.case_id.astype(str)
    frame["patient_id"] = frame.patient_id.astype(str)
    frame["cv_fold"] = pd.to_numeric(frame.cv_fold, errors="raise").astype(int)
    for column in [
        "label_volume_ml",
        "expansion_target_voxels",
        "expansion_predicted_voxels",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    frame["label_nonempty"] = frame.label_nonempty.astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    return frame.sort_values(["patient_id", "case_id"]).reset_index(drop=True)


def infer_voxel_volume_ml(frame: pd.DataFrame) -> float:
    positive = frame.expansion_target_voxels.to_numpy(float) > 0
    ratios = (
        frame.loc[positive, "label_volume_ml"].to_numpy(float)
        / frame.loc[positive, "expansion_target_voxels"].to_numpy(float)
    )
    if len(ratios) == 0 or not np.isfinite(ratios).all():
        raise RuntimeError("Cannot infer voxel volume from OOF targets")
    voxel_ml = float(np.median(ratios))
    if float(np.max(np.abs(ratios - voxel_ml))) > 1e-9:
        raise RuntimeError("Inconsistent voxel-to-mL conversion in OOF targets")
    if not (0.0001 <= voxel_ml <= 0.01):
        raise RuntimeError(f"Implausible voxel volume: {voxel_ml} mL")
    return voxel_ml


def add_volumes(frame: pd.DataFrame, voxel_volume_ml: float) -> pd.DataFrame:
    output = frame.copy()
    output["reference_volume_ml"] = output.label_volume_ml.astype(float)
    output["predicted_volume_ml"] = (
        output.expansion_predicted_voxels.astype(float) * voxel_volume_ml
    )
    output["volume_error_ml"] = output.predicted_volume_ml - output.reference_volume_ml
    output["absolute_error_ml"] = output.volume_error_ml.abs()
    output["volume_group"] = output.reference_volume_ml.map(volume_group)
    return output


def volume_group(volume_ml: float) -> str:
    value = float(volume_ml)
    if value < 0 or not math.isfinite(value):
        raise ValueError(f"Invalid infiltration volume: {volume_ml}")
    if value == 0:
        return "empty"
    if value <= 1.0:
        return "<=1mL"
    if value <= 10.0:
        return "1-10mL"
    return ">10mL"


def finite_xy(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    return x_array[keep], y_array[keep]


def lin_ccc(x: Iterable[float], y: Iterable[float]) -> float:
    x_array, y_array = finite_xy(x, y)
    if len(x_array) < 2:
        return float("nan")
    mean_x, mean_y = float(x_array.mean()), float(y_array.mean())
    denominator = (
        float(np.var(x_array))
        + float(np.var(y_array))
        + (mean_x - mean_y) ** 2
    )
    if denominator <= 0:
        return float("nan")
    covariance = float(np.mean((x_array - mean_x) * (y_array - mean_y)))
    return float(2.0 * covariance / denominator)


def agreement_metrics(x: Iterable[float], y: Iterable[float]) -> dict:
    reference, prediction = finite_xy(x, y)
    if len(reference) == 0:
        raise RuntimeError("No finite reference/prediction pairs")
    difference = prediction - reference
    sd = float(np.std(difference, ddof=1)) if len(difference) > 1 else float("nan")
    rho = (
        float(spearmanr(reference, prediction).statistic)
        if len(reference) >= 3
        and float(np.ptp(reference)) > 0
        and float(np.ptp(prediction)) > 0
        else float("nan")
    )
    return {
        "pairs": int(len(reference)),
        "reference_mean_ml": float(reference.mean()),
        "reference_median_ml": float(np.median(reference)),
        "predicted_mean_ml": float(prediction.mean()),
        "predicted_median_ml": float(np.median(prediction)),
        "spearman_rho": rho,
        "lin_ccc": lin_ccc(reference, prediction),
        "mae_ml": float(np.mean(np.abs(difference))),
        "median_absolute_error_ml": float(np.median(np.abs(difference))),
        "bias_ml": float(np.mean(difference)),
        "bland_altman_lower_ml": float(np.mean(difference) - 1.96 * sd),
        "bland_altman_upper_ml": float(np.mean(difference) + 1.96 * sd),
    }


BOOTSTRAP_METRICS = [
    "spearman_rho",
    "lin_ccc",
    "mae_ml",
    "bias_ml",
    "bland_altman_lower_ml",
    "bland_altman_upper_ml",
]


def patient_bootstrap(
    frame: pd.DataFrame,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    metrics: Iterable[str] = BOOTSTRAP_METRICS,
) -> pd.DataFrame:
    requested_metrics = list(metrics)
    unknown = set(requested_metrics).difference(BOOTSTRAP_METRICS)
    if unknown:
        raise ValueError(f"Unsupported bootstrap metrics: {sorted(unknown)}")
    work = frame[["patient_id", "reference_volume_ml", "predicted_volume_ml"]].copy()
    patients = work.patient_id.unique()
    if len(patients) < 2:
        raise RuntimeError("Patient bootstrap requires at least two patients")
    groups = {patient: part for patient, part in work.groupby("patient_id", sort=False)}
    point = agreement_metrics(work.reference_volume_ml, work.predicted_volume_ml)
    draws = {metric: [] for metric in requested_metrics}
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        selected = rng.choice(patients, size=len(patients), replace=True)
        sample = pd.concat([groups[patient] for patient in selected], ignore_index=True)
        result = agreement_metrics(sample.reference_volume_ml, sample.predicted_volume_ml)
        for metric in requested_metrics:
            draws[metric].append(result[metric])
    rows = []
    for metric in requested_metrics:
        values = np.asarray(draws[metric], dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            low, high = np.quantile(values, [0.025, 0.975])
        else:
            low, high = float("nan"), float("nan")
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci95_low": float(low),
                "ci95_high": float(high),
                "finite_bootstrap_repetitions": int(len(values)),
                "requested_bootstrap_repetitions": int(repetitions),
                "resampling_unit": "patient",
            }
        )
    return pd.DataFrame(rows)
