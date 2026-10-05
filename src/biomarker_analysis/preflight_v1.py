#!/usr/bin/env python3
"""Fail-fast audit of the five completed MU out-of-fold result sets."""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from biomarker_common_v1 import (
    EXPECTED_EMPTY,
    EXPECTED_FOLDS,
    EXPECTED_NONEMPTY,
    EXPECTED_PAIRS,
    EXPECTED_PATIENTS,
    EXPECTED_REGISTRY_SHA256,
    MODEL_SPECS,
    OUTPUT_ROOT,
    PRIMARY_MODEL,
    atomic_csv,
    atomic_json,
    infer_voxel_volume_ml,
    load_json,
    load_model_cases,
    model_case_path,
    model_root,
    model_summary_path,
    sha256_file,
    volume_group,
    write_progress,
)


EXPECTED_GROUP_COUNTS = {
    "empty": 7,
    "<=1mL": 54,
    "1-10mL": 153,
    ">10mL": 139,
}


def run() -> None:
    destination = OUTPUT_ROOT / "preflight"
    destination.mkdir(parents=True, exist_ok=True)
    write_progress("RUNNING", "preflight", "Auditing five OOF result sets")
    source_rows = []
    frames = {}
    summaries = {}
    for index, model in enumerate(MODEL_SPECS, start=1):
        root = model_root(model)
        complete = root / "aggregate" / "AGGREGATION_COMPLETE"
        case_path = model_case_path(model)
        summary_path = model_summary_path(model)
        for required in [complete, case_path, summary_path]:
            if not required.is_file():
                raise RuntimeError(f"{model}: missing required file {required}")
        frame = load_model_cases(model)
        summary = load_json(summary_path)
        if summary.get("status") != "PASS":
            raise RuntimeError(f"{model}: pooled OOF summary is not PASS")
        if summary.get("role") != "pooled out-of-fold predictions from patient-grouped five-fold cross-validation":
            raise RuntimeError(f"{model}: unexpected analysis role")
        if str(summary.get("outer_fold_registry_sha256")) != EXPECTED_REGISTRY_SHA256:
            raise RuntimeError(f"{model}: outer-fold registry hash mismatch")
        if len(frame) != EXPECTED_PAIRS or frame.case_id.nunique() != EXPECTED_PAIRS:
            raise RuntimeError(f"{model}: expected {EXPECTED_PAIRS} unique OOF pairs")
        if frame.patient_id.nunique() != EXPECTED_PATIENTS:
            raise RuntimeError(f"{model}: expected {EXPECTED_PATIENTS} OOF patients")
        if set(frame.cv_fold.unique()) != EXPECTED_FOLDS:
            raise RuntimeError(f"{model}: invalid fold coverage {sorted(frame.cv_fold.unique())}")
        if frame.duplicated(["case_id"]).any():
            raise RuntimeError(f"{model}: duplicate OOF case_id")
        if (frame[["label_volume_ml", "expansion_target_voxels", "expansion_predicted_voxels"]] < 0).any().any():
            raise RuntimeError(f"{model}: negative target or prediction values")
        frames[model] = frame
        summaries[model] = summary
        source_rows.append(
            {
                "model": model,
                "experiment": MODEL_SPECS[model],
                "oof_pairs": len(frame),
                "oof_patients": frame.patient_id.nunique(),
                "oof_case_metrics": str(case_path),
                "oof_case_metrics_sha256": sha256_file(case_path),
                "pooled_summary": str(summary_path),
                "pooled_summary_sha256": sha256_file(summary_path),
                "outer_fold_registry_sha256": summary["outer_fold_registry_sha256"],
            }
        )
        write_progress(
            "RUNNING",
            "preflight",
            f"Audited {model}",
            completed_models=index,
            total_models=len(MODEL_SPECS),
        )

    primary = frames[PRIMARY_MODEL]
    identity = primary[["case_id", "patient_id", "cv_fold"]].copy()
    target_columns = ["label_volume_ml", "expansion_target_voxels", "label_nonempty"]
    for model, frame in frames.items():
        candidate_identity = frame[["case_id", "patient_id", "cv_fold"]]
        if not identity.equals(candidate_identity):
            raise RuntimeError(f"{model}: OOF case/patient/fold registry differs from primary")
        for column in target_columns:
            left = primary[column].to_numpy()
            right = frame[column].to_numpy()
            identical = np.array_equal(left, right) if column == "label_nonempty" else np.allclose(left, right, atol=1e-12, rtol=0)
            if not identical:
                raise RuntimeError(f"{model}: ground-truth column differs: {column}")

    nonempty = int(primary.label_nonempty.sum())
    empty = int((~primary.label_nonempty).sum())
    if (nonempty, empty) != (EXPECTED_NONEMPTY, EXPECTED_EMPTY):
        raise RuntimeError(f"Expected nonempty/empty {EXPECTED_NONEMPTY}/{EXPECTED_EMPTY}; found {nonempty}/{empty}")
    if not np.array_equal(primary.label_nonempty.to_numpy(), primary.label_volume_ml.to_numpy() > 0):
        raise RuntimeError("label_nonempty disagrees with label_volume_ml")
    group_counts = primary.label_volume_ml.map(volume_group).value_counts().to_dict()
    if group_counts != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(f"Unexpected reference-volume counts: {group_counts}")
    voxel_volume_ml = infer_voxel_volume_ml(primary)
    atomic_csv(pd.DataFrame(source_rows), destination / "source_inventory.csv")
    atomic_csv(identity, destination / "locked_oof_registry.csv")
    payload = {
        "status": "PASS",
        "phase": "complete",
        "dataset": "MU-Glioma-Post",
        "oof_pairs": EXPECTED_PAIRS,
        "oof_patients": EXPECTED_PATIENTS,
        "nonempty_pairs": nonempty,
        "empty_pairs": empty,
        "folds": 5,
        "fold_pair_counts": {
            str(key): int(value)
            for key, value in primary.cv_fold.value_counts().sort_index().items()
        },
        "reference_volume_group_counts": group_counts,
        "models": list(MODEL_SPECS),
        "primary_model": PRIMARY_MODEL,
        "outer_fold_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "voxel_volume_ml": voxel_volume_ml,
        "all_models_identical_oof_registry": True,
        "all_models_identical_ground_truth": True,
    }
    atomic_json(destination / "summary.json", payload)
    (destination / "PASS").touch()
    write_progress("PASS", "preflight_complete", "OOF inputs verified", oof_pairs=EXPECTED_PAIRS)


def main() -> None:
    try:
        run()
    except Exception as error:
        atomic_json(
            OUTPUT_ROOT / "preflight" / "summary.json",
            {"status": "FAILED", "phase": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        write_progress("FAILED", "preflight_failed", f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
