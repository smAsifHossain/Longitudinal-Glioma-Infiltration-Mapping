#!/usr/bin/env python3
"""Shared definitions for MU-only training, validation, and locked testing."""
from __future__ import annotations

import hashlib
import faulthandler
import json
import math
import os
import random
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import average_precision_score


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
V5 = Path(os.environ.get("GLIOMA_PROJECT_ROOT", str(REPOSITORY_ROOT))).expanduser().resolve()
V4 = Path(
    os.environ.get("GLIOMA_PRETRAINING_ROOT", str(V5.parent / "Brain_Tumor_Summer26_V4"))
).expanduser().resolve()
PACKAGE_NAME = "mednext_mu_5fold_cv_v1"
EXPERIMENT_NAME = "MedNeXtS_MU_5fold_CV_v1"
SCRIPT_DIR = Path(__file__).resolve().parent
CV_ROOT = V5 / "finetuning/experiments" / EXPERIMENT_NAME
REFERENCE_UNET_CV_ROOT = (
    V5 / "finetuning/experiments/UNet3D_MU_5fold_CV_v1"
)
_FOLD_TEXT = os.environ.get("MEDNEXT_CV_FOLD", "").strip()
CV_FOLD = int(_FOLD_TEXT) if _FOLD_TEXT else None
if CV_FOLD is not None and CV_FOLD not in range(5):
    raise RuntimeError(f"MEDNEXT_CV_FOLD must be 0-4, found {CV_FOLD}")
EXPERIMENT_ROOT = CV_ROOT if CV_FOLD is None else CV_ROOT / f"fold_{CV_FOLD}"
CACHE_MANIFEST = (
    V5
    / "label/MU-Glioma-Post/summary"
    / "MU_Glioma_Post__WT_Full391__metrics_summary.csv"
)
PRETRAINED_CHECKPOINT = (
    V4
    / "pretraining_data/experiments/MedNeXt_S_full_pretraining_4GPU/checkpoints"
    / "best_model_weights.pt"
)
V5_MU_METRICS = (
    V5
    / "label/MU-Glioma-Post/summary"
    / "MU_Glioma_Post__WT_Full391__metrics_summary.csv"
)
V5_MU_SUMMARY = (
    V5
    / "label/MU-Glioma-Post/summary"
    / "MU_Glioma_Post__WT_Full391__summary.json"
)
STOP_FILE = EXPERIMENT_ROOT / "STOP_REQUESTED"

DATASET = "MU-Glioma-Post"
MODALITIES = ("t1", "t1ce", "t2", "flair")
REGIONS = ("wt",)
PATCH_SIZE = (128, 128, 128)
DIRECT_SPLIT_SEED = 20260823
CV_SPLIT_SEED = 20260823
CV_FOLDS = 5
TRAINING_SEED = 20260823 + (0 if CV_FOLD is None else 10_000 * CV_FOLD)
RUNTIME_VERSION = "2026-08-24-mednexts-mu-5fold-cv-r1"
EXPECTED_CASES = 391
EXPECTED_PATIENTS = 155
MU_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
VOLUME_BIN_ORDER = ("empty", "<=0.1mL", "0.1-1mL", "1-10mL", ">10mL")
VOLUME_STAGE_LABELS = {
    "empty": "Empty",
    "<=0.1mL": "<=0.1 mL",
    "0.1-1mL": "Small",
    "1-10mL": "Medium",
    ">10mL": "Large",
}
QC_EXCLUSIONS: dict[str, str] = {}

CACHE_PATH_COLUMNS = [
    "earlier_t1",
    "earlier_t1ce",
    "earlier_t2",
    "earlier_flair",
    "recurrent_t1_registered",
    "recurrent_t1ce_registered",
    "recurrent_t2_registered",
    "recurrent_flair_registered",
    "earlier_seg",
    "recurrent_seg_registered",
]
MASK_PATH_COLUMNS: list[str] = []


class GracefulStop(RuntimeError):
    """Raised after a safe checkpoint when a stop was requested."""


def parse_bool_series(values: pd.Series) -> pd.Series:
    """Normalize persisted CSV boolean values without relying on truthy strings."""
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    selected = np.isfinite(array)
    return float(array[selected].mean()) if selected.any() else float("nan")


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, include_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        result["sha256"] = sha256(path)
    return result


def build_model() -> torch.nn.Module:
    from mednext_v1 import create_mednext_v1

    # The audited pretrained checkpoint contains the primary output plus four
    # deep-supervision heads.  Keep every module for an exact strict load, then
    # return only the primary full-resolution output during this experiment.
    model = create_mednext_v1(
        num_input_channels=4,
        num_classes=3,
        model_id="S",
        kernel_size=3,
        deep_supervision=True,
    )
    model.do_ds = False
    for head_name in ("out_1", "out_2", "out_3", "out_4"):
        for parameter in getattr(model, head_name).parameters():
            parameter.requires_grad_(False)
    return model


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict", "network"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise RuntimeError("No model state dictionary found in pretrained checkpoint")


def strict_load_pretrained(
    model: torch.nn.Module, checkpoint_path: Path = PRETRAINED_CHECKPOINT
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = extract_model_state(checkpoint)
    target = model.state_dict()
    if set(state) != set(target):
        missing = sorted(set(target) - set(state))
        unexpected = sorted(set(state) - set(target))
        raise RuntimeError(
            "MedNeXt checkpoint key mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    mismatched = [
        (key, tuple(state[key].shape), tuple(target[key].shape))
        for key in sorted(target)
        if tuple(state[key].shape) != tuple(target[key].shape)
    ]
    if mismatched:
        raise RuntimeError(f"MedNeXt checkpoint shape mismatch: {mismatched[:3]}")
    if len(state) != 237:
        raise RuntimeError(f"Expected 237 tensors in audited MedNeXt-S checkpoint; found {len(state)}")
    nonfinite = sum(
        int((~torch.isfinite(value)).sum().item())
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    )
    if nonfinite:
        raise RuntimeError(f"Pretrained checkpoint contains {nonfinite} non-finite weights")
    model.load_state_dict(state, strict=True)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if parameter_count != 5_553_936:
        raise RuntimeError(
            f"Unexpected MedNeXt-S parameter count: {parameter_count} != 5553936"
        )
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256(checkpoint_path),
        "checkpoint_format": "raw_state_dict",
        "architecture": "MedNeXt-v1 Small kernel-3",
        "modalities": ["T1", "T1CE", "T2", "FLAIR"],
        "targets": ["WT", "TC", "ET"],
        "tensor_count": int(len(state)),
        "parameters": parameter_count,
    }


def infiltration_volume_bin(volume_ml: float) -> str:
    """Fixed, publication-facing infiltration-volume categories."""
    value = float(volume_ml)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid infiltration volume: {volume_ml}")
    if value == 0:
        return "empty"
    if value <= 0.1:
        return "<=0.1mL"
    if value <= 1.0:
        return "0.1-1mL"
    if value <= 10.0:
        return "1-10mL"
    return ">10mL"


def split_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    volumes = pd.to_numeric(frame.infiltration_volume_ml, errors="raise").astype(float)
    bins = volumes.map(infiltration_volume_bin)
    return {
        "pairs": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "nonempty_pairs": int((volumes > 0).sum()),
        "empty_pairs": int((volumes == 0).sum()),
        "volume_ml": {
            "mean": finite_mean(volumes),
            "median": float(volumes.median()) if len(volumes) else float("nan"),
            "minimum": float(volumes.min()) if len(volumes) else float("nan"),
            "maximum": float(volumes.max()) if len(volumes) else float("nan"),
        },
        "volume_bin_pairs": {
            name: int((bins == name).sum()) for name in VOLUME_BIN_ORDER
        },
        "volume_bin_patients": {
            name: int(frame.loc[bins == name, "patient_id"].nunique())
            for name in VOLUME_BIN_ORDER
        },
    }


def _mu_patient_assignment(frame: pd.DataFrame) -> dict[str, str]:
    """Balance MU patients, pair counts, and volume bins across 70/15/15."""
    working = frame.copy()
    working["volume_bin"] = pd.to_numeric(
        working.infiltration_volume_ml, errors="raise"
    ).map(infiltration_volume_bin)
    feature_names = ("pairs", *VOLUME_BIN_ORDER)
    patients: list[dict[str, Any]] = []
    for patient_id, group in working.groupby("patient_id", sort=True):
        counts = group.volume_bin.value_counts()
        features = np.asarray(
            [len(group), *[int(counts.get(name, 0)) for name in VOLUME_BIN_ORDER]],
            dtype=np.float64,
        )
        patients.append(
            {
                "patient_id": str(patient_id),
                "features": features,
                "rank": hashlib.sha256(
                    f"direct|{DIRECT_SPLIT_SEED}|{DATASET}|{patient_id}".encode()
                ).hexdigest(),
            }
        )
    if len(patients) < 30:
        raise RuntimeError(
            f"Too few MU patients for train/validation/test split: {len(patients)}"
        )
    total_features = np.sum([item["features"] for item in patients], axis=0)
    rarity = np.maximum(total_features, 1.0)
    patients.sort(
        key=lambda item: (
            -float(np.sum((item["features"][1:] > 0) / rarity[1:])),
            -float(item["features"][0]),
            item["rank"],
        )
    )
    names = tuple(MU_SPLIT_RATIOS)
    patient_total = len(patients)
    capacities = {
        "validation": max(
            1, int(round(patient_total * MU_SPLIT_RATIOS["validation"]))
        ),
        "test": max(1, int(round(patient_total * MU_SPLIT_RATIOS["test"]))),
    }
    capacities["train"] = (
        patient_total - capacities["validation"] - capacities["test"]
    )
    targets = {
        name: total_features * MU_SPLIT_RATIOS[name] for name in names
    }
    assigned_features = {name: np.zeros_like(total_features) for name in names}
    assigned_patients = {name: 0 for name in names}
    assignment: dict[str, str] = {}
    for item in patients:
        candidates = [
            name for name in names if assigned_patients[name] < capacities[name]
        ]
        if not candidates:
            raise RuntimeError("Direct split exhausted all patient capacities")
        scored: list[tuple[float, str, str]] = []
        for name in candidates:
            score = 0.0
            for candidate_name in names:
                proposed = assigned_features[candidate_name] + (
                    item["features"] if candidate_name == name else 0.0
                )
                scale = np.sqrt(targets[candidate_name] + 1.0)
                score += float(np.sum(((proposed - targets[candidate_name]) / scale) ** 2))
            patient_fill = (assigned_patients[name] + 1) / capacities[name]
            target_fill = (len(assignment) + 1) / patient_total
            score += 2.0 * (patient_fill - target_fill) ** 2
            tie = hashlib.sha256(
                f"tie|{DIRECT_SPLIT_SEED}|{item['patient_id']}|{name}".encode()
            ).hexdigest()
            scored.append((score, tie, name))
        selected = min(scored)[2]
        assignment[item["patient_id"]] = selected
        assigned_features[selected] += item["features"]
        assigned_patients[selected] += 1
    if assigned_patients != capacities:
        raise RuntimeError(
            f"Direct split patient targets failed: {assigned_patients} != {capacities}"
        )
    return assignment


def split_mu_train_validation_test(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment = _mu_patient_assignment(frame)
    result = frame.copy()
    result["split"] = result.patient_id.astype(str).map(assignment)
    if result.split.isna().any():
        raise RuntimeError(
            "At least one MU patient was not assigned to train/validation/test"
        )
    parts = {
        name: result[result.split == name].copy().reset_index(drop=True)
        for name in MU_SPLIT_RATIOS
    }
    patient_sets = {
        name: set(part.patient_id.astype(str)) for name, part in parts.items()
    }
    for first_index, first in enumerate(MU_SPLIT_RATIOS):
        for second in tuple(MU_SPLIT_RATIOS)[first_index + 1 :]:
            if patient_sets[first] & patient_sets[second]:
                raise RuntimeError(f"Patient leakage between {first} and {second}")
    if sum(len(part) for part in parts.values()) != len(frame):
        raise RuntimeError("Direct split lost one or more pairs")
    for name, part in parts.items():
        if part.empty or part.patient_id.nunique() < 10:
            raise RuntimeError(f"Direct {name} split is unexpectedly small")
    all_bins = pd.to_numeric(frame.infiltration_volume_ml, errors="raise").map(infiltration_volume_bin)
    for volume_bin in VOLUME_BIN_ORDER:
        available_patients = int(frame.loc[all_bins == volume_bin, "patient_id"].nunique())
        if available_patients >= len(MU_SPLIT_RATIOS):
            for name, part in parts.items():
                part_bins = pd.to_numeric(
                    part.infiltration_volume_ml, errors="raise"
                ).map(infiltration_volume_bin)
                if not bool((part_bins == volume_bin).any()):
                    raise RuntimeError(
                        f"MU split failed to represent {volume_bin} in {name}"
                    )
    return parts["train"], parts["validation"], parts["test"]


def _patient_feature_records(
    frame: pd.DataFrame, namespace: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Summarize each patient by pair count and fixed infiltration-volume bins."""
    working = frame.copy()
    working["volume_bin"] = pd.to_numeric(
        working.infiltration_volume_ml, errors="raise"
    ).map(infiltration_volume_bin)
    records: list[dict[str, Any]] = []
    for patient_id, group in working.groupby("patient_id", sort=True):
        counts = group.volume_bin.value_counts()
        features = np.asarray(
            [len(group), *[int(counts.get(name, 0)) for name in VOLUME_BIN_ORDER]],
            dtype=np.float64,
        )
        records.append(
            {
                "patient_id": str(patient_id),
                "features": features,
                "rank": hashlib.sha256(
                    f"{namespace}|{CV_SPLIT_SEED}|{DATASET}|{patient_id}".encode()
                ).hexdigest(),
            }
        )
    if not records:
        raise RuntimeError("Cannot split an empty patient cohort")
    total = np.sum([item["features"] for item in records], axis=0)
    rarity = np.maximum(total, 1.0)
    records.sort(
        key=lambda item: (
            -float(np.sum((item["features"][1:] > 0) / rarity[1:])),
            -float(item["features"][0]),
            item["rank"],
        )
    )
    return records, total


def _balanced_patient_groups(
    frame: pd.DataFrame,
    capacities: dict[str, int],
    ratios: dict[str, float],
    namespace: str,
) -> dict[str, str]:
    """Deterministically balance patients, pairs, and volume bins across groups."""
    records, total_features = _patient_feature_records(frame, namespace)
    if sum(capacities.values()) != len(records):
        raise RuntimeError(
            f"Patient capacities do not cover the cohort: {capacities} vs {len(records)}"
        )
    names = tuple(capacities)
    targets = {name: total_features * float(ratios[name]) for name in names}
    assigned_features = {name: np.zeros_like(total_features) for name in names}
    assigned_patients = {name: 0 for name in names}
    assignment: dict[str, str] = {}
    for item in records:
        candidates = [
            name for name in names if assigned_patients[name] < capacities[name]
        ]
        if not candidates:
            raise RuntimeError("Balanced split exhausted all patient capacities")
        scores: list[tuple[float, str, str]] = []
        for candidate in candidates:
            score = 0.0
            for name in names:
                proposed = assigned_features[name] + (
                    item["features"] if name == candidate else 0.0
                )
                scale = np.sqrt(targets[name] + 1.0)
                score += float(np.sum(((proposed - targets[name]) / scale) ** 2))
            patient_fill = (assigned_patients[candidate] + 1) / capacities[candidate]
            target_fill = (len(assignment) + 1) / len(records)
            score += 2.0 * (patient_fill - target_fill) ** 2
            tie = hashlib.sha256(
                f"{namespace}|tie|{CV_SPLIT_SEED}|{item['patient_id']}|{candidate}".encode()
            ).hexdigest()
            scores.append((score, tie, candidate))
        selected = min(scores)[2]
        assignment[item["patient_id"]] = selected
        assigned_features[selected] += item["features"]
        assigned_patients[selected] += 1
    if assigned_patients != capacities:
        raise RuntimeError(
            f"Balanced split failed patient capacities: {assigned_patients} != {capacities}"
        )
    return assignment


def assign_outer_folds(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one immutable patient-grouped five-fold outer-test registry."""
    patient_total = int(frame.patient_id.nunique())
    if patient_total < CV_FOLDS * 10:
        raise RuntimeError(f"Too few patients for {CV_FOLDS}-fold CV: {patient_total}")
    base, remainder = divmod(patient_total, CV_FOLDS)
    capacities = {
        f"fold_{fold}": base + (1 if fold < remainder else 0)
        for fold in range(CV_FOLDS)
    }
    ratios = {name: capacity / patient_total for name, capacity in capacities.items()}
    assignment = _balanced_patient_groups(
        frame, capacities, ratios, namespace="outer-fivefold"
    )
    result = frame.copy()
    result["outer_fold"] = result.patient_id.astype(str).map(
        {patient: int(group.split("_")[1]) for patient, group in assignment.items()}
    )
    if result.outer_fold.isna().any():
        raise RuntimeError("At least one patient is absent from the outer-fold registry")
    result["outer_fold"] = result.outer_fold.astype(int)
    patient_fold_counts = result.groupby("patient_id").outer_fold.nunique()
    if not bool((patient_fold_counts == 1).all()):
        raise RuntimeError("A patient was assigned to more than one outer fold")
    if set(result.outer_fold.unique()) != set(range(CV_FOLDS)):
        raise RuntimeError("The outer-fold registry does not contain folds 0-4")
    return result.sort_values(["outer_fold", "patient_id", "case_id"]).reset_index(
        drop=True
    )


def split_mu_cv_fold(
    registry: pd.DataFrame, fold: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return approximately 70/10/20 train/validation/outer-test partitions."""
    fold = int(fold)
    if fold not in range(CV_FOLDS):
        raise ValueError(f"Fold must be 0-{CV_FOLDS - 1}, found {fold}")
    test = registry[registry.outer_fold.astype(int) == fold].copy()
    development = registry[registry.outer_fold.astype(int) != fold].copy()
    development_patients = int(development.patient_id.nunique())
    validation_patients = max(10, int(round(development_patients * 0.125)))
    capacities = {
        "train": development_patients - validation_patients,
        "validation": validation_patients,
    }
    ratios = {
        name: capacity / development_patients for name, capacity in capacities.items()
    }
    assignment = _balanced_patient_groups(
        development,
        capacities,
        ratios,
        namespace=f"fold-{fold}-inner-validation",
    )
    development["split"] = development.patient_id.astype(str).map(assignment)
    test["split"] = "test"
    train = development[development.split == "train"].copy()
    validation = development[development.split == "validation"].copy()
    parts = (train, validation, test)
    patient_sets = [set(part.patient_id.astype(str)) for part in parts]
    if any(
        patient_sets[first] & patient_sets[second]
        for first in range(len(parts))
        for second in range(first + 1, len(parts))
    ):
        raise RuntimeError(f"Patient leakage in CV fold {fold}")
    if sum(len(part) for part in parts) != len(registry):
        raise RuntimeError(f"CV fold {fold} lost one or more scan pairs")
    if test.patient_id.nunique() not in {
        registry.patient_id.nunique() // CV_FOLDS,
        math.ceil(registry.patient_id.nunique() / CV_FOLDS),
    }:
        raise RuntimeError(f"Unexpected outer-test patient count in fold {fold}")
    return tuple(
        part.sort_values(["patient_id", "case_id"]).reset_index(drop=True)
        for part in parts
    )


def load_mu_frame(check_paths: bool = False) -> pd.DataFrame:
    if not V5_MU_SUMMARY.is_file():
        raise FileNotFoundError(V5_MU_SUMMARY)
    label_summary = json.loads(V5_MU_SUMMARY.read_text(encoding="utf-8"))
    expected_summary = {
        "status": "PASS",
        "method": "earlier_WT123_to_recurrent_WT123_T1N_rigid_local_gt2_le20mm",
        "runtime_version": "2026-08-15-exact-v2",
        "registration_and_cache_generated_together": True,
        "cases_requested": 391,
        "completed_cases": 391,
        "failed_cases": 0,
    }
    for key, expected in expected_summary.items():
        if label_summary.get(key) != expected:
            raise RuntimeError(
                f"V5 MU label generation is not finalized: {key}={label_summary.get(key)!r}"
            )
    frame = pd.read_csv(CACHE_MANIFEST, low_memory=False)
    required = {
        "dataset",
        "case_id",
        "patient",
        "earlier_timepoint",
        "recurrent_timepoint",
        *CACHE_PATH_COLUMNS,
        "infiltration_label_path",
        "infiltration_volume_ml",
        "earlier_reference_volume_ml",
        "registered_recurrent_tumor_volume_ml",
        "earlier_reference_type",
        "registration_transform_path",
        "registration_transform_sha256",
        "registration_interpolator_mri",
        "registration_interpolator_segmentation",
        "exact_registration_cache_complete",
        "review_required",
        "review_reason",
        "ncc_after",
        "ncc_gain",
    }
    if required - set(frame.columns):
        raise RuntimeError(f"Exact V5 MU manifest missing {sorted(required - set(frame.columns))}")
    frame = frame[frame.dataset.astype(str) == DATASET].copy()
    if len(frame) != EXPECTED_CASES or frame.case_id.astype(str).duplicated().any():
        raise RuntimeError("Exact V5 MU manifest is not the unique completed 391-case cohort")
    frame["patient_id"] = frame.patient.astype(str)
    frame["earlier_session"] = frame.earlier_timepoint.astype(str)
    frame["later_session"] = frame.recurrent_timepoint.astype(str)
    frame["label_path"] = frame.infiltration_label_path.astype(str)
    frame["label_nonempty_verified"] = (
        pd.to_numeric(frame.infiltration_volume_ml, errors="raise") > 0
    )
    frame["volume_bin"] = pd.to_numeric(
        frame.infiltration_volume_ml, errors="raise"
    ).map(infiltration_volume_bin)
    frame["v5_review_required"] = parse_bool_series(frame["review_required"])
    if len(frame) != EXPECTED_CASES or frame.patient_id.nunique() != EXPECTED_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES}/{EXPECTED_PATIENTS} MU pairs/patients; "
            f"found {len(frame)}/{frame.patient_id.nunique()}"
        )
    if frame.duplicated(["dataset", "case_id"]).any():
        raise RuntimeError("Duplicate MU case")
    if set(frame.registration_interpolator_mri.astype(str)) != {"linear"}:
        raise RuntimeError("Exact V5 MRI cache does not use linear interpolation")
    if set(frame.registration_interpolator_segmentation.astype(str)) != {"genericLabel"}:
        raise RuntimeError("Exact V5 segmentation cache does not use genericLabel")
    exact_flags = frame.exact_registration_cache_complete.astype(str).str.lower()
    if not exact_flags.isin({"true", "1", "yes"}).all():
        raise RuntimeError("One or more exact registration-cache rows are incomplete")
    allowed_early = {"visible_labels_1_2_3", "resection_cavity_label_4"}
    observed_early = set(frame.earlier_reference_type.astype(str))
    if not observed_early or observed_early - allowed_early:
        raise RuntimeError(f"Finalized earlier-reference semantics changed: {observed_early}")
    if check_paths:
        missing = [
            str(path)
            for column in [*CACHE_PATH_COLUMNS, "label_path", "registration_transform_path"]
            for path in frame[column].astype(str)
            if not Path(path).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest paths are missing; first={missing[0]}")
    return frame.sort_values("case_id").reset_index(drop=True)


def apply_qc_exclusions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = frame.case_id.astype(str).isin(QC_EXCLUSIONS)
    review = frame.v5_review_required.astype(bool)
    excluded = frame[known | review].copy()
    modeling = frame[~(known | review)].copy()
    excluded["exclusion_reason"] = ""
    excluded.loc[excluded.v5_review_required.astype(bool), "exclusion_reason"] = (
        "V5 automatic registration-QC review required: "
        + excluded.loc[excluded.v5_review_required.astype(bool), "review_reason"].fillna("").astype(str)
    )
    for case_id, reason in QC_EXCLUSIONS.items():
        selected = excluded.case_id.astype(str) == case_id
        if not selected.any():
            raise RuntimeError(f"Known QC exclusion is absent: {case_id}")
        excluded.loc[selected, "exclusion_reason"] = reason
    if len(modeling) < 300 or modeling.patient_id.nunique() < 125:
        raise RuntimeError(
            f"Too few review-cleared MU pairs/patients: {len(modeling)}/{modeling.patient_id.nunique()}"
        )
    if modeling.case_id.astype(str).duplicated().any():
        raise RuntimeError("Duplicate cases after V5 QC filtering")
    return modeling.reset_index(drop=True), excluded.reset_index(drop=True)


def normalize_nonzero(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    mask = np.isfinite(array) & (array != 0)
    if int(mask.sum()) < 100:
        raise ValueError("MRI has fewer than 100 finite nonzero voxels")
    values = array[mask]
    mean, standard = float(values.mean()), float(values.std())
    if not math.isfinite(mean) or not math.isfinite(standard) or standard < 1e-6:
        raise ValueError(f"Invalid intensity statistics: {mean}/{standard}")
    output = np.zeros_like(array, dtype=np.float32)
    output[mask] = (values - mean) / standard
    output[~np.isfinite(output)] = 0
    return output


def region_targets(raw: np.ndarray) -> np.ndarray:
    raw = np.rint(raw).astype(np.int16, copy=False)
    observed = set(int(value) for value in np.unique(raw))
    if observed - {0, 1, 2, 3, 4}:
        raise ValueError(f"Unexpected MU segmentation labels: {sorted(observed)}")
    return np.stack(
        [np.isin(raw, [1, 2, 3]), np.isin(raw, [1, 3]), raw == 3], axis=0
    ).astype(np.float32)


def geometry_equal(first: nib.Nifti1Image, second: nib.Nifti1Image) -> bool:
    return tuple(first.shape[:3]) == tuple(second.shape[:3]) and np.allclose(
        first.affine, second.affine, atol=1e-4
    )


def paired_paths(row: pd.Series | Any) -> tuple[list[str], list[str]]:
    early = [str(getattr(row, f"earlier_{modality}")) for modality in MODALITIES]
    recurrent = [str(getattr(row, f"recurrent_{modality}_registered")) for modality in MODALITIES]
    return early, recurrent


def load_pair(row: pd.Series | Any) -> dict[str, Any]:
    early_paths, recurrent_paths = paired_paths(row)
    early_images = [nib.load(path) for path in early_paths]
    recurrent_images = [nib.load(path) for path in recurrent_paths]
    early_seg = nib.load(str(row.earlier_seg))
    recurrent_seg = nib.load(str(row.recurrent_seg_registered))
    label_image = nib.load(str(row.label_path))
    reference = early_images[0]
    all_images = [
        *early_images,
        *recurrent_images,
        early_seg,
        recurrent_seg,
        label_image,
    ]
    if any(not geometry_equal(reference, image) for image in all_images[1:]):
        raise RuntimeError(f"{row.case_id}: paired geometry mismatch")
    raw_early = [np.asarray(image.dataobj, dtype=np.float32) for image in early_images]
    raw_recurrent = [np.asarray(image.dataobj, dtype=np.float32) for image in recurrent_images]
    early_raw = np.rint(np.asarray(early_seg.dataobj, dtype=np.float32)).astype(np.int16)
    early_target = region_targets(early_raw)
    recurrent_target = region_targets(np.asarray(recurrent_seg.dataobj, dtype=np.float32))
    if str(row.earlier_reference_type) == "visible_labels_1_2_3":
        early_target[0] = np.isin(early_raw, [1, 2, 3])
    elif str(row.earlier_reference_type) == "resection_cavity_label_4":
        early_target[0] = early_raw == 4
    else:
        raise RuntimeError(
            f"{row.case_id}: unexpected earlier reference {row.earlier_reference_type}"
        )
    # Only the WT channel is optimized. Keep the remaining channels solely so the
    # pretrained three-channel output head can be loaded without risky adaptation.
    early_target[1] = np.maximum(early_target[1], early_target[2])
    early_target[0] = np.maximum(early_target[0], early_target[1])
    recurrent_target[1] = np.maximum(recurrent_target[1], recurrent_target[2])
    recurrent_target[0] = np.maximum(recurrent_target[0], recurrent_target[1])
    early_brain = np.any(
        np.stack([np.isfinite(value) & (value != 0) for value in raw_early]), axis=0
    )
    recurrent_brain = np.any(
        np.stack([np.isfinite(value) & (value != 0) for value in raw_recurrent]), axis=0
    )
    valid = early_brain | recurrent_brain | (early_target[0] > 0.5) | (recurrent_target[0] > 0.5)
    if int(valid.sum()) < 1000:
        raise RuntimeError(f"{row.case_id}: invalid paired brain mask")
    label = np.asarray(label_image.dataobj) > 0.5
    return {
        "early_image": np.stack([normalize_nonzero(value) for value in raw_early]).astype(np.float32),
        "recurrent_image": np.stack([normalize_nonzero(value) for value in raw_recurrent]).astype(np.float32),
        "early_target": early_target.astype(np.float32),
        "recurrent_target": recurrent_target.astype(np.float32),
        "valid": valid.astype(np.float32)[None],
        "label": label.astype(np.float32)[None],
        "spacing": tuple(float(value) for value in reference.header.get_zooms()[:3]),
        "reference": reference,
        "early_display": np.asarray(early_images[1].dataobj, dtype=np.float32),
        "recurrent_display": np.asarray(recurrent_images[1].dataobj, dtype=np.float32),
    }


class LRUPairCache:
    def __init__(self, maximum: int = 4) -> None:
        self.maximum = max(1, int(maximum))
        self.data: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, row: pd.Series | Any) -> dict[str, Any]:
        key = str(row.case_id)
        if key in self.data:
            value = self.data.pop(key)
            self.data[key] = value
            return value
        value = load_pair(row)
        self.data[key] = value
        while len(self.data) > self.maximum:
            self.data.popitem(last=False)
        return value


def _sampling_indices(pair: dict[str, Any]) -> dict[str, np.ndarray]:
    """Compute patch-sampling locations once instead of once per training sample."""
    early = pair["early_target"] > 0.5
    recurrent = pair["recurrent_target"] > 0.5
    valid = pair["valid"][0] > 0.5
    structure = ndimage.generate_binary_structure(3, 1)
    early_boundary = ndimage.binary_dilation(
        early[0], structure, iterations=3
    ) ^ ndimage.binary_erosion(early[0], structure, iterations=3)
    sources = {
        "recurrent_wt": recurrent[0],
        "earlier_wt_boundary": early_boundary,
        "paired_tumor": early[0] | recurrent[0],
        "brain": valid,
    }
    return {
        name: np.ascontiguousarray(np.flatnonzero(source), dtype=np.int32)
        for name, source in sources.items()
    }


def compact_cached_pair(pair: dict[str, Any], prepare_sampling: bool) -> dict[str, Any]:
    """Keep exact FP32 MRI data but compact binary arrays for the RAM cache."""
    result = {
        "early_image": np.ascontiguousarray(pair["early_image"], dtype=np.float32),
        "recurrent_image": np.ascontiguousarray(pair["recurrent_image"], dtype=np.float32),
        "early_target": np.ascontiguousarray(pair["early_target"] > 0.5, dtype=np.uint8),
        "recurrent_target": np.ascontiguousarray(
            pair["recurrent_target"] > 0.5, dtype=np.uint8
        ),
        "valid": np.ascontiguousarray(pair["valid"] > 0.5, dtype=np.uint8),
        "label": np.ascontiguousarray(pair["label"] > 0.5, dtype=np.uint8),
        "spacing": pair["spacing"],
    }
    if prepare_sampling:
        result["sampling_indices"] = _sampling_indices(result)
    return result


def _array_bytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(_array_bytes(item) for item in value.values())
    return 0


class RAMPairStore:
    """Read-only pair store shared by all CPU patch-building threads."""

    def __init__(self, data: dict[str, dict[str, Any]]) -> None:
        self.data = data
        self.bytes = sum(_array_bytes(value) for value in data.values())

    @classmethod
    def preload(
        cls,
        frame: pd.DataFrame,
        workers: int,
        prepare_sampling: bool,
        progress: Callable[[int, int, str, float], None] | None = None,
    ) -> "RAMPairStore":
        rows = list(frame.itertuples(index=False))
        if len({str(row.case_id) for row in rows}) != len(rows):
            raise RuntimeError("RAM preload received duplicate case IDs")

        def load_one(row: Any) -> tuple[str, dict[str, Any]]:
            pair = compact_cached_pair(load_pair(row), prepare_sampling=prepare_sampling)
            return str(row.case_id), pair

        data: dict[str, dict[str, Any]] = {}
        loaded_bytes = 0
        if int(workers) == 1:
            for completed, row in enumerate(rows, start=1):
                faulthandler.dump_traceback_later(300, repeat=False)
                try:
                    case_id, pair = load_one(row)
                finally:
                    faulthandler.cancel_dump_traceback_later()
                data[case_id] = pair
                loaded_bytes += _array_bytes(pair)
                if progress is not None:
                    progress(completed, len(rows), case_id, loaded_bytes / 1024**3)
            return cls(data)
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {executor.submit(load_one, row): str(row.case_id) for row in rows}
            for completed, future in enumerate(as_completed(futures), start=1):
                case_id, pair = future.result()
                data[case_id] = pair
                loaded_bytes += _array_bytes(pair)
                if progress is not None:
                    progress(completed, len(rows), case_id, loaded_bytes / 1024**3)
        return cls(data)

    def get(self, row: pd.Series | Any) -> dict[str, Any]:
        case_id = str(row.case_id)
        if case_id not in self.data:
            raise KeyError(f"Pair {case_id} is absent from the RAM cache")
        return self.data[case_id]


def crop_or_pad(array: np.ndarray, center: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    squeeze = array.ndim == 3
    source = array[None] if squeeze else array
    spatial = np.asarray(source.shape[-3:], dtype=int)
    wanted = np.asarray(size, dtype=int)
    start = np.asarray(center, dtype=int) - wanted // 2
    end = start + wanted
    source_start, source_end = np.maximum(start, 0), np.minimum(end, spatial)
    destination_start = source_start - start
    destination_end = destination_start + source_end - source_start
    output = np.zeros((source.shape[0], *wanted.tolist()), dtype=source.dtype)
    ds = tuple(slice(int(a), int(b)) for a, b in zip(destination_start, destination_end))
    ss = tuple(slice(int(a), int(b)) for a, b in zip(source_start, source_end))
    output[(...,) + ds] = source[(...,) + ss]
    return output[0] if squeeze else output


def choose_pair_center(pair: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, str]:
    sources = pair.get("sampling_indices")
    if sources is None:
        sources = _sampling_indices(pair)
    value = float(rng.random())
    # Sampling uses only segmentation targets. The infiltration label is never an
    # input, a loss target, or a patch-selection signal for the model.
    if value < 0.30 and len(sources["recurrent_wt"]):
        source, category = sources["recurrent_wt"], "recurrent_wt"
    elif value < 0.55 and len(sources["earlier_wt_boundary"]):
        source, category = sources["earlier_wt_boundary"], "earlier_wt_boundary"
    elif value < 0.85 and len(sources["paired_tumor"]):
        source, category = sources["paired_tumor"], "paired_tumor"
    else:
        source, category = sources["brain"], "brain"
    if not len(source):
        source, category = sources["brain"], "brain_fallback"
    if not len(source):
        raise RuntimeError("Pair has no valid patch-sampling coordinates")
    flat_index = int(source[int(rng.integers(0, len(source)))])
    center = np.asarray(np.unravel_index(flat_index, pair["valid"].shape[-3:]), dtype=int)
    return center + rng.integers(-12, 13, size=3), category


def augment_pair(
    early_image: np.ndarray,
    recurrent_image: np.ndarray,
    early_target: np.ndarray,
    recurrent_target: np.ndarray,
    valid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ...]:
    values = [early_image, recurrent_image, early_target, recurrent_target, valid]
    for spatial_axis in range(3):
        if rng.random() < 0.5:
            values = [np.flip(value, axis=spatial_axis + 1) for value in values]
    early_image, recurrent_image, early_target, recurrent_target, valid = values
    for image in (early_image, recurrent_image):
        scale = rng.uniform(0.90, 1.10, size=(4, 1, 1, 1)).astype(np.float32)
        shift = rng.normal(0.0, 0.05, size=(4, 1, 1, 1)).astype(np.float32)
        image[...] = np.where(image != 0, image * scale + shift, 0.0)
    return tuple(
        np.ascontiguousarray(value)
        for value in (early_image, recurrent_image, early_target, recurrent_target, valid)
    )


class PairedPatchDataset(torch.utils.data.Dataset):
    """Deterministic same-coordinate paired patches with patient-diverse batches."""

    def __init__(
        self,
        frame: pd.DataFrame,
        epoch: int,
        steps: int,
        batch_size: int,
        start_step: int = 0,
        cache_pairs: int = 4,
        pair_store: RAMPairStore | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.epoch = int(epoch)
        self.steps = int(steps)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.cache = pair_store if pair_store is not None else LRUPairCache(cache_pairs)
        self.plan = self._make_plan()

    def _make_plan(self) -> list[int]:
        rng = np.random.default_rng(TRAINING_SEED + self.epoch * 1009)
        by_patient = {
            str(patient): np.asarray(indexes, dtype=int)
            for patient, indexes in self.frame.groupby("patient_id").groups.items()
        }
        patients = np.asarray(sorted(by_patient), dtype=object)
        if self.batch_size > len(patients):
            raise RuntimeError("Batch size exceeds number of distinct training patients")
        plan: list[int] = []
        for _ in range(self.steps):
            chosen = rng.choice(patients, size=self.batch_size, replace=False)
            for patient in chosen:
                indexes = by_patient[str(patient)]
                plan.append(int(indexes[int(rng.integers(0, len(indexes)))]))
        return plan[self.start_step * self.batch_size :]

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict[str, Any]:
        absolute_sample = self.start_step * self.batch_size + index
        rng = np.random.default_rng(
            TRAINING_SEED + self.epoch * 1_000_003 + absolute_sample * 97
        )
        row = self.frame.iloc[self.plan[index]]
        pair = self.cache.get(row)
        center, category = choose_pair_center(pair, rng)
        values = [
            crop_or_pad(pair["early_image"], center, PATCH_SIZE),
            crop_or_pad(pair["recurrent_image"], center, PATCH_SIZE),
            crop_or_pad(pair["early_target"], center, PATCH_SIZE),
            crop_or_pad(pair["recurrent_target"], center, PATCH_SIZE),
            crop_or_pad(pair["valid"], center, PATCH_SIZE),
        ]
        early_image, recurrent_image, early_target, recurrent_target, valid = augment_pair(
            *values, rng
        )
        return {
            "early_image": torch.from_numpy(early_image),
            "recurrent_image": torch.from_numpy(recurrent_image),
            "early_target": torch.from_numpy(early_target),
            "recurrent_target": torch.from_numpy(recurrent_target),
            "valid": torch.from_numpy(valid),
            "case_id": str(row.case_id),
            "category": category,
        }


def segmentation_loss(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # The pretrained MedNeXt retains its compatible WT/TC/ET head, but this run
    # optimizes only channel 0 (WT) because both V5 rule inputs are whole tumor.
    logits = logits[:, :1].float()
    target = target[:, :1].float()
    valid = valid.float()
    probability = torch.sigmoid(logits)
    mask = valid.expand_as(target)
    dimensions = tuple(range(2, logits.ndim))
    intersection = (probability * target * mask).sum(dim=dimensions)
    denominator = ((probability + target) * mask).sum(dim=dimensions)
    soft_dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    dice_loss = 1.0 - soft_dice.mean()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_loss = (bce * mask).sum() / mask.sum().clamp_min(1.0)
    loss = 0.70 * dice_loss + 0.30 * bce_loss
    return loss, {"soft_dice": soft_dice.mean(dim=0), "dice_loss": dice_loss, "bce": bce_loss}


def paired_segmentation_loss(
    early_logits: torch.Tensor,
    recurrent_logits: torch.Tensor,
    early_target: torch.Tensor,
    recurrent_target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    early_loss, early_live = segmentation_loss(early_logits, early_target, valid)
    recurrent_loss, recurrent_live = segmentation_loss(recurrent_logits, recurrent_target, valid)
    total = 0.5 * early_loss + 0.5 * recurrent_loss
    return total, {
        "early_loss": early_loss,
        "recurrent_loss": recurrent_loss,
        "early_soft": early_live["soft_dice"],
        "recurrent_soft": recurrent_live["soft_dice"],
    }


def hard_regions(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probability)[:1] >= float(threshold)).astype(bool)


def dice_coefficient(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else float(2 * (prediction & target).sum() / denominator)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    return {
        "dice": dice_coefficient(prediction, target),
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else (1.0 if not target.any() else 0.0),
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "predicted_voxels": int(prediction.sum()),
        "target_voxels": int(target.sum()),
    }


def lesion_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    pred_components, pred_count = ndimage.label(prediction, structure=structure)
    target_components, target_count = ndimage.label(target, structure=structure)
    matched_pred = sum(
        np.any(target & (pred_components == value)) for value in range(1, pred_count + 1)
    )
    detected_target = sum(
        np.any(prediction & (target_components == value))
        for value in range(1, target_count + 1)
    )
    precision = matched_pred / pred_count if pred_count else (1.0 if target_count == 0 else 0.0)
    recall = detected_target / target_count if target_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "lesion_precision": float(precision),
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
        "predicted_lesions": int(pred_count),
        "target_lesions": int(target_count),
    }


def surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float = 1.0,
) -> dict[str, float]:
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if not prediction.any() or not target.any():
        return {
            "hd95_mm": float("nan"),
            "assd_mm": float("nan"),
            "surface_dice_1mm": float("nan"),
        }
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_prediction = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    first = distance_to_target[pred_surface]
    second = distance_to_prediction[target_surface]
    combined = np.concatenate([first, second])
    surface_dice = (
        float((first <= tolerance_mm).sum() + (second <= tolerance_mm).sum())
        / float(len(first) + len(second))
    )
    return {
        "hd95_mm": float(np.percentile(combined, 95)),
        "assd_mm": float(combined.mean()),
        "surface_dice_1mm": surface_dice,
    }


def derive_infiltration_rule(
    earlier_wt: np.ndarray,
    recurrent_wt: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Derive the local longitudinal whole-tumor infiltration region."""
    earlier = np.asarray(earlier_wt, bool)
    recurrent = np.asarray(recurrent_wt, bool)
    if not earlier.any() or not recurrent.any():
        return np.zeros_like(earlier, dtype=bool)
    distance = ndimage.distance_transform_edt(~earlier, sampling=spacing)
    eligible = (distance > 2.0) & (distance <= 20.0)
    components, number = ndimage.label(
        recurrent, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    retained = np.zeros_like(recurrent, dtype=bool)
    for component_id in range(1, number + 1):
        component = components == component_id
        if component.any() and float(distance[component].min()) <= 2.0 + 1e-6:
            retained |= component
    return retained & eligible


def infiltration_score(
    earlier_wt: np.ndarray,
    recurrent_wt: np.ndarray,
    recurrent_wt_probability: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    prediction = derive_infiltration_rule(earlier_wt, recurrent_wt, spacing)
    return prediction, np.where(prediction, recurrent_wt_probability, 0.0).astype(np.float32)


def safe_auprc(score: np.ndarray, target: np.ndarray, valid: np.ndarray) -> float:
    selected = np.asarray(valid, bool)
    y = np.asarray(target, bool)[selected]
    p = np.asarray(score, np.float32)[selected]
    if y.size == 0 or not y.any():
        return float("nan")
    return float(average_precision_score(y.astype(np.uint8), p))


def evaluate_case_arrays(
    early_probability: np.ndarray,
    recurrent_probability: np.ndarray,
    pair: dict[str, Any],
    threshold: float = 0.5,
    extended: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    early_prediction = hard_regions(early_probability, threshold)
    recurrent_prediction = hard_regions(recurrent_probability, threshold)
    early_target = pair["early_target"] > 0.5
    recurrent_target = pair["recurrent_target"] > 0.5
    valid = pair["valid"][0] > 0.5
    label = pair["label"][0] > 0.5
    expansion, expansion_score = infiltration_score(
        early_prediction[0], recurrent_prediction[0], recurrent_probability[0], pair["spacing"]
    )
    output: dict[str, Any] = {
        "label_nonempty": bool(label.any()),
        "expansion_auprc": safe_auprc(expansion_score, label, valid),
    }
    for phase, prediction, target in (
        ("early", early_prediction, early_target),
        ("recurrent", recurrent_prediction, recurrent_target),
    ):
        for index, region in enumerate(REGIONS):
            metrics = binary_metrics(prediction[index], target[index])
            for name, value in metrics.items():
                output[f"{phase}_{region}_{name}"] = value
            output[f"{phase}_{region}_target_nonempty"] = bool(target[index].any())
    for name, value in binary_metrics(expansion, label).items():
        output[f"expansion_{name}"] = value
    if extended:
        for name, value in lesion_metrics(expansion, label).items():
            output[f"expansion_{name}"] = value
        for name, value in surface_metrics(expansion, label, pair["spacing"]).items():
            output[f"expansion_{name}"] = value
    else:
        output.update(
            {
                "expansion_lesion_precision": float("nan"),
                "expansion_lesion_recall": float("nan"),
                "expansion_lesion_f1": float("nan"),
                "expansion_predicted_lesions": float("nan"),
                "expansion_target_lesions": float("nan"),
                "expansion_hd95_mm": float("nan"),
                "expansion_assd_mm": float("nan"),
                "expansion_surface_dice_1mm": float("nan"),
            }
        )
    output["label_volume_ml"] = float(label.sum() * np.prod(pair["spacing"]) / 1000.0)
    return output, {
        "early_prediction": early_prediction,
        "recurrent_prediction": recurrent_prediction,
        "expansion_prediction": expansion,
        "expansion_score": expansion_score,
    }


def summarize_case_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"cases": int(len(frame))}
    segmentation_positive: list[float] = []
    for phase in ("early", "recurrent"):
        region_values = []
        for region in REGIONS:
            selected = frame[frame[f"{phase}_{region}_target_nonempty"].astype(bool)]
            value = finite_mean(selected[f"{phase}_{region}_dice"].astype(float))
            summary[f"{phase}_{region}_dice_positive"] = value
            region_values.append(value)
            segmentation_positive.extend(selected[f"{phase}_{region}_dice"].astype(float).tolist())
        summary[f"{phase}_mean_dice_positive"] = finite_mean(region_values)
    summary["segmentation_mean_dice_positive"] = finite_mean(segmentation_positive)
    nonempty = frame[frame.label_nonempty.astype(bool)]
    empty = frame[~frame.label_nonempty.astype(bool)]
    summary["infiltration_nonempty_cases"] = int(len(nonempty))
    summary["infiltration_empty_cases"] = int(len(empty))
    for metric in ("dice", "iou", "precision", "recall", "auprc"):
        summary[f"infiltration_{metric}_nonempty"] = finite_mean(
            nonempty[f"expansion_{metric}"].astype(float)
        )
    summary["empty_accuracy"] = finite_mean(
        (empty.expansion_predicted_voxels.astype(int) == 0).astype(float)
    )
    summary["empty_false_positive_cases"] = int(
        (empty.expansion_predicted_voxels.astype(int) > 0).sum()
    )
    for metric in ("lesion_precision", "lesion_recall", "lesion_f1"):
        summary[f"infiltration_{metric}_nonempty"] = finite_mean(
            nonempty[f"expansion_{metric}"].astype(float)
        )
    detected = nonempty[nonempty.expansion_predicted_voxels.astype(int) > 0]
    for metric in ("hd95_mm", "assd_mm", "surface_dice_1mm"):
        summary[f"infiltration_{metric}_detected"] = finite_mean(
            detected[f"expansion_{metric}"].astype(float)
        )
    bins = {
        "<=0.1mL": (0.0, 0.1),
        "0.1-1mL": (0.1, 1.0),
        "1-10mL": (1.0, 10.0),
        ">10mL": (10.0, float("inf")),
    }
    summary["size_bins"] = {}
    for name, (lower, upper) in bins.items():
        selected = nonempty[
            (nonempty.label_volume_ml.astype(float) > lower)
            & (nonempty.label_volume_ml.astype(float) <= upper)
        ]
        summary["size_bins"][name] = {
            "n": int(len(selected)),
            "patients": int(selected.patient_id.nunique())
            if "patient_id" in selected.columns
            else None,
            "volume_ml_mean": finite_mean(selected.label_volume_ml.astype(float)),
            "volume_ml_median": float(selected.label_volume_ml.astype(float).median())
            if len(selected)
            else float("nan"),
            "dice": finite_mean(selected.expansion_dice.astype(float)),
            "iou": finite_mean(selected.expansion_iou.astype(float)),
            "auprc": finite_mean(selected.expansion_auprc.astype(float)),
            "precision": finite_mean(selected.expansion_precision.astype(float)),
            "recall": finite_mean(selected.expansion_recall.astype(float)),
            "lesion_f1": finite_mean(selected.expansion_lesion_f1.astype(float)),
            "hd95_mm_detected": finite_mean(
                selected.loc[
                    selected.expansion_predicted_voxels.astype(int) > 0,
                    "expansion_hd95_mm",
                ].astype(float)
            ),
            "assd_mm_detected": finite_mean(
                selected.loc[
                    selected.expansion_predicted_voxels.astype(int) > 0,
                    "expansion_assd_mm",
                ].astype(float)
            ),
            "surface_dice_1mm_detected": finite_mean(
                selected.loc[
                    selected.expansion_predicted_voxels.astype(int) > 0,
                    "expansion_surface_dice_1mm",
                ].astype(float)
            ),
        }
    summary["selection_score"] = 0.5 * summary["segmentation_mean_dice_positive"] + 0.5 * summary[
        "infiltration_dice_nonempty"
    ]
    return summary


def inference_specialist_pair(
    early_model: torch.nn.Module,
    recurrent_model: torch.nn.Module,
    pair: dict[str, Any],
    device: torch.device,
    sw_batch_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    from monai.inferers import sliding_window_inference

    early_image = torch.from_numpy(pair["early_image"][None]).to(
        device=device, dtype=torch.float32
    )
    recurrent_image = torch.from_numpy(pair["recurrent_image"][None]).to(
        device=device, dtype=torch.float32
    )
    with torch.no_grad():
        amp_enabled = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled
        ):
            early_logits = sliding_window_inference(
                early_image,
                roi_size=PATCH_SIZE,
                sw_batch_size=int(sw_batch_size),
                predictor=early_model,
                overlap=0.5,
                mode="gaussian",
            )
            recurrent_logits = sliding_window_inference(
                recurrent_image,
                roi_size=PATCH_SIZE,
                sw_batch_size=int(sw_batch_size),
                predictor=recurrent_model,
                overlap=0.5,
                mode="gaussian",
            )
    early_probability = torch.sigmoid(early_logits.float()).cpu().numpy()[0]
    recurrent_probability = torch.sigmoid(recurrent_logits.float()).cpu().numpy()[0]
    return early_probability, recurrent_probability


def save_nifti(mask: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), reference.affine, header), path)


def source_provenance() -> dict[str, Any]:
    return {
        "cache_manifest": file_identity(CACHE_MANIFEST, include_hash=True),
        "v5_mu_metrics": file_identity(V5_MU_METRICS, include_hash=True),
        "v5_mu_summary": file_identity(V5_MU_SUMMARY, include_hash=True),
        "pretrained_checkpoint": file_identity(PRETRAINED_CHECKPOINT, include_hash=True),
        "source": {
            path.name: file_identity(path, include_hash=True)
            for path in sorted(SCRIPT_DIR.glob("*.py"))
        },
    }
