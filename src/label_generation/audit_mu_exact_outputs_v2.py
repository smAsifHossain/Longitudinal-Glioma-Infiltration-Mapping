#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label

from path_config import OUTPUT_ROOT


ROOT = OUTPUT_ROOT
METRICS = ROOT / "summary/MU_Glioma_Post__WT_Full391__metrics_summary.csv"
PROGRESS = ROOT / "exact_output_audit_progress.json"
CASE_CSV = ROOT / "summary/MU_Glioma_Post__WT_Full391__exact_output_audit.csv"
SUMMARY = ROOT / "summary/MU_Glioma_Post__WT_Full391__exact_output_audit.json"
COMPLETE = ROOT / "EXACT_CACHE_COMPLETE"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def geometry_equal(first: nib.Nifti1Image, second: nib.Nifti1Image) -> bool:
    return tuple(first.shape[:3]) == tuple(second.shape[:3]) and np.allclose(
        first.affine, second.affine, atol=1e-4
    )


def derive(earlier: np.ndarray, recurrent: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    distance = distance_transform_edt(~earlier, sampling=spacing)
    components, count = label(recurrent, structure=np.ones((3, 3, 3), np.uint8))
    anchored = np.zeros_like(recurrent)
    for component_id in range(1, count + 1):
        component = components == component_id
        if component.any() and float(distance[component].min()) <= 2.0 + 1e-6:
            anchored |= component
    return anchored & (distance > 2.0) & (distance <= 20.0)


def run() -> None:
    COMPLETE.unlink(missing_ok=True)
    if not METRICS.is_file():
        raise FileNotFoundError(METRICS)
    frame = pd.read_csv(METRICS, low_memory=False)
    if len(frame) != 391 or frame.case_id.astype(str).duplicated().any():
        raise RuntimeError(f"Expected 391 unique exact MU rows; found {len(frame)}")
    path_columns = [
        "earlier_t1",
        "earlier_t1ce",
        "earlier_t2",
        "earlier_flair",
        "earlier_seg",
        "recurrent_t1_registered",
        "recurrent_t1ce_registered",
        "recurrent_t2_registered",
        "recurrent_flair_registered",
        "recurrent_seg_registered",
        "infiltration_label_path",
    ]
    records: list[dict] = []
    atomic_json(
        PROGRESS,
        {"status": "RUNNING", "phase": "geometry_rule_hash", "completed": 0, "total": 391},
    )
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        paths = [Path(str(getattr(row, column))) for column in path_columns]
        transform = Path(str(row.registration_transform_path))
        for path in [*paths, transform, Path(str(row.qc_image)), Path(str(row.excluded_region_path))]:
            if not path.is_file():
                raise FileNotFoundError(path)
        digest = hashlib.sha256(transform.read_bytes()).hexdigest()
        if digest != str(row.registration_transform_sha256):
            raise RuntimeError(f"{row.case_id}: transform SHA-256 mismatch")
        images = [nib.load(str(path)) for path in paths]
        reference = images[0]
        if any(not geometry_equal(reference, image) for image in images[1:]):
            raise RuntimeError(f"{row.case_id}: exact cache geometry mismatch")
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        early_labels = np.rint(np.asarray(images[4].dataobj)).astype(np.int16)
        recurrent_labels = np.rint(np.asarray(images[9].dataobj)).astype(np.int16)
        if set(np.unique(early_labels).astype(int)) - {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{row.case_id}: invalid earlier labels")
        if set(np.unique(recurrent_labels).astype(int)) - {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{row.case_id}: invalid recurrent labels")
        if str(row.earlier_reference_type) == "visible_labels_1_2_3":
            early = np.isin(early_labels, [1, 2, 3])
        elif str(row.earlier_reference_type) == "resection_cavity_label_4":
            early = early_labels == 4
        else:
            raise RuntimeError(f"{row.case_id}: invalid earlier reference type")
        recurrent = np.isin(recurrent_labels, [1, 2, 3])
        target = np.asarray(images[10].dataobj) > 0.5
        reconstructed = derive(early, recurrent, spacing)
        if not np.array_equal(reconstructed, target):
            difference = int(np.logical_xor(reconstructed, target).sum())
            raise RuntimeError(
                f"{row.case_id}: exact rule differs from saved label by {difference} voxels"
            )
        voxel_ml = float(np.prod(spacing) / 1000.0)
        earlier_ml = float(early.sum() * voxel_ml)
        recurrent_ml = float(recurrent.sum() * voxel_ml)
        infiltration_ml = float(target.sum() * voxel_ml)
        expected = (
            float(row.earlier_reference_volume_ml),
            float(row.registered_recurrent_tumor_volume_ml),
            float(row.infiltration_volume_ml),
        )
        observed = (earlier_ml, recurrent_ml, infiltration_ml)
        if any(abs(first - second) > 0.001 for first, second in zip(observed, expected)):
            raise RuntimeError(
                f"{row.case_id}: exact output volumes changed; observed={observed}, expected={expected}"
            )
        records.append(
            {
                "case_id": str(row.case_id),
                "patient": str(row.patient),
                "shape": "x".join(str(value) for value in reference.shape[:3]),
                "spacing": "x".join(f"{value:.5f}" for value in spacing),
                "transform_sha256_verified": True,
                "geometry_verified": True,
                "rule_exact": True,
                "earlier_reference_volume_ml": earlier_ml,
                "registered_recurrent_tumor_volume_ml": recurrent_ml,
                "infiltration_volume_ml": infiltration_ml,
                "review_required": str(row.review_required).lower() in {"true", "1", "yes"},
            }
        )
        print(f"AUDIT {index:3d}/391 | {row.case_id} | exact rule PASS", flush=True)
        if index == 1 or index % 10 == 0 or index == 391:
            atomic_json(
                PROGRESS,
                {
                    "status": "RUNNING",
                    "phase": "geometry_rule_hash",
                    "completed": index,
                    "total": 391,
                    "case_id": str(row.case_id),
                },
            )
    audit = pd.DataFrame(records)
    temporary_csv = CASE_CSV.with_suffix(".csv.tmp")
    audit.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, CASE_CSV)
    summary = {
        "status": "PASS",
        "cases": 391,
        "transform_sha256_verified": 391,
        "geometry_verified": 391,
        "rule_exact": 391,
        "maximum_allowed_volume_difference_ml": 0.001,
        "read_only_raw_data": True,
    }
    atomic_json(SUMMARY, summary)
    COMPLETE.touch()
    atomic_json(PROGRESS, {**summary, "phase": "complete", "completed": 391, "total": 391})


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        atomic_json(
            PROGRESS,
            {
                "status": "FAILED",
                "phase": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
