#!/usr/bin/env python3
from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label

from path_config import CODE_ROOT, CONTROL_ROOT, DATA_ROOT, MANIFEST_PATH, PROJECT_ROOT


V5 = PROJECT_ROOT
RAW = DATA_ROOT
CODE = CODE_ROOT
MANIFEST = MANIFEST_PATH
CONTROL = CONTROL_ROOT
PROGRESS = CONTROL / "preflight_progress.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def derive(earlier: np.ndarray, recurrent: np.ndarray) -> np.ndarray:
    distance = distance_transform_edt(~earlier, sampling=(1.0, 1.0, 1.0))
    components, count = label(recurrent, structure=np.ones((3, 3, 3), np.uint8))
    anchored = np.zeros_like(recurrent)
    for component_id in range(1, count + 1):
        component = components == component_id
        if component.any() and float(distance[component].min()) <= 2.0 + 1e-6:
            anchored |= component
    return anchored & (distance > 2.0) & (distance <= 20.0)


def run() -> None:
    atomic_json(PROGRESS, {"status": "RUNNING", "phase": "source_audit"})
    required_code = [
        CODE / "generate_mu_case_exact_v2.py",
        CODE / "run_mu_exact_391_v2.py",
        MANIFEST,
    ]
    for path in required_code:
        if not path.is_file():
            raise FileNotFoundError(path)
    frame = pd.read_csv(MANIFEST)
    required_columns = {
        "case_id",
        "patient",
        "earlier_timepoint",
        "recurrent_timepoint",
        "recurrent_target",
    }
    if required_columns - set(frame.columns):
        raise RuntimeError(
            f"MU manifest missing {sorted(required_columns - set(frame.columns))}"
        )
    if len(frame) != 391 or frame.case_id.astype(str).duplicated().any():
        raise RuntimeError(f"Expected 391 unique MU pairs; found {len(frame)}")
    if frame.patient.astype(str).nunique() != 155:
        raise RuntimeError(
            f"Expected 155 MU patients; found {frame.patient.astype(str).nunique()}"
        )
    if set(frame.recurrent_target.astype(str)) != {"WT123_labels_1_2_3"}:
        raise RuntimeError(
            f"MU manifest recurrent target is not WT123: {set(frame.recurrent_target.astype(str))}"
        )

    modality_suffixes = ("t1n", "t1c", "t2w", "t2f")
    missing: list[str] = []
    for row in frame.itertuples(index=False):
        for timepoint in (str(row.earlier_timepoint), str(row.recurrent_timepoint)):
            directory = RAW / str(row.patient) / timepoint
            for suffix in modality_suffixes:
                path = directory / f"{row.patient}_{timepoint}_brain_{suffix}.nii.gz"
                if not path.is_file():
                    missing.append(str(path))
            segmentation = directory / f"{row.patient}_{timepoint}_tumorMask.nii.gz"
            if not segmentation.is_file():
                missing.append(str(segmentation))
    if missing:
        raise FileNotFoundError(f"{len(missing)} required raw inputs missing; first={missing[0]}")

    early = np.zeros((32, 32, 32), dtype=bool)
    early[12:16, 12:16, 12:16] = True
    recurrent = np.zeros_like(early)
    recurrent[14:22, 12:16, 12:16] = True
    recurrent[28:30, 28:30, 28:30] = True
    result = derive(early, recurrent)
    if int(result.sum()) != 64 or result[28:30, 28:30, 28:30].any():
        raise RuntimeError("WT-to-WT 2-20 mm rule self-test failed")

    generator_text = (CODE / "generate_mu_case_exact_v2.py").read_text(encoding="utf-8")
    required_tokens = [
        'RECURRENT_TARGET_LABELS = (1, 2, 3)',
        'interpolator="genericLabel"',
        'interpolator="linear"',
        'registration_transform_path',
        'registered_recurrent_4mod',
        'exact_registration_cache_complete',
    ]
    missing_tokens = [token for token in required_tokens if token not in generator_text]
    if missing_tokens:
        raise RuntimeError(f"Exact generator is missing controls: {missing_tokens}")
    atomic_json(
        PROGRESS,
        {
            "status": "PASS",
            "phase": "complete",
            "cases": 391,
            "patients": 155,
            "raw_policy": "read_only",
            "definition": "earlier WT123 to recurrent WT123; >2 and <=20 mm",
            "registration_cache": "generated and saved in the same case process",
        },
    )


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
