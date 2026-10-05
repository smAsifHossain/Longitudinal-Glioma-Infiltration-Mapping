#!/usr/bin/env python3
"""Tests for five-fold splitting, Swin-UNETR construction, and scheduler wiring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common_v1 import (
    CV_FOLDS,
    EXPERIMENT_NAME,
    PACKAGE_NAME,
    VOLUME_BIN_ORDER,
    assign_outer_folds,
    binary_metrics,
    build_model,
    derive_infiltration_rule,
    infiltration_volume_bin,
    parse_bool_series,
    split_mu_cv_fold,
)


def synthetic_cohort() -> pd.DataFrame:
    volumes = [0.0, 0.05, 0.5, 5.0, 20.0]
    rows = []
    case_index = 0
    for patient_index in range(146):
        pairs = 2 + int(patient_index < 61)
        for pair_index in range(pairs):
            case_index += 1
            rows.append(
                {
                    "case_id": f"case_{case_index:04d}",
                    "patient_id": f"patient_{patient_index:03d}",
                    "infiltration_volume_ml": volumes[
                        (patient_index + pair_index) % len(volumes)
                    ],
                }
            )
    frame = pd.DataFrame(rows)
    assert len(frame) == 353
    return frame


def test_volume_bins() -> None:
    expected = {
        0.0: "empty",
        0.1: "<=0.1mL",
        0.10001: "0.1-1mL",
        1.0: "0.1-1mL",
        1.0001: "1-10mL",
        10.0: "1-10mL",
        10.0001: ">10mL",
    }
    for value, name in expected.items():
        assert infiltration_volume_bin(value) == name


def test_boolean_normalization() -> None:
    values = pd.Series([True, "TRUE", "1", "yes", False, "FALSE", "0", "no"])
    assert parse_bool_series(values).tolist() == [
        True, True, True, True, False, False, False, False
    ]


def test_infiltration_rule_and_metrics() -> None:
    earlier = np.zeros((48, 48, 48), dtype=bool)
    earlier[12:16, 20:24, 20:24] = True
    recurrent = np.zeros_like(earlier)
    recurrent[14:25, 20:24, 20:24] = True
    recurrent[40:44, 40:44, 40:44] = True
    result = derive_infiltration_rule(earlier, recurrent, (1.0, 1.0, 1.0))
    assert result.any()
    assert not result[40:44, 40:44, 40:44].any()
    assert not (result & earlier).any()
    metrics = binary_metrics(
        np.asarray([1, 1, 0, 0], bool), np.asarray([1, 0, 1, 0], bool)
    )
    assert abs(metrics["dice"] - 0.5) < 1e-12
    assert abs(metrics["iou"] - 1 / 3) < 1e-12


def test_fivefold_registry() -> None:
    frame = synthetic_cohort()
    registry = assign_outer_folds(frame)
    registry2 = assign_outer_folds(frame)
    assert registry[["case_id", "outer_fold"]].equals(
        registry2[["case_id", "outer_fold"]]
    )
    assert set(registry.outer_fold) == set(range(CV_FOLDS))
    assert (registry.groupby("patient_id").outer_fold.nunique() == 1).all()
    all_tests = []
    for fold in range(CV_FOLDS):
        train, validation, test = split_mu_cv_fold(registry, fold)
        assert len(train) + len(validation) + len(test) == len(registry)
        assert not (set(train.patient_id) & set(validation.patient_id))
        assert not (set(train.patient_id) & set(test.patient_id))
        assert not (set(validation.patient_id) & set(test.patient_id))
        assert train.patient_id.nunique() in {101, 102, 103}
        assert validation.patient_id.nunique() in {14, 15}
        assert test.patient_id.nunique() in {29, 30}
        assert set(test.infiltration_volume_ml.map(infiltration_volume_bin)) == set(
            VOLUME_BIN_ORDER
        )
        all_tests.extend(test.case_id.astype(str))
    assert len(all_tests) == len(frame)
    assert len(set(all_tests)) == len(frame)


def test_wiring() -> None:
    assert PACKAGE_NAME == "swinunetr_mu_5fold_cv_v1"
    assert EXPERIMENT_NAME == "SwinUNETR_MU_5fold_CV_v1"
    package = Path(__file__).resolve().parent
    for filename in (
        "preflight_cv_v1.py",
        "fold_manager_v1.py",
        "train_v1.py",
        "evaluate_test_v1.py",
        "aggregate_cv_v1.py",
        "manager_v1.py",
        "monitor_v1.py",
        "live_epoch_v1.py",
    ):
        assert (package / filename).is_file(), filename
    training = (package / "train_v1.py").read_text(encoding="utf-8")
    assert "mu_test_locked.csv" in training
    assert 'usecols=["case_id", "patient_id"]' in training
    assert "effective_pair_batch_size" in training
    assert "gradient_accumulation_steps" in training
    common = (package / "common_v1.py").read_text(encoding="utf-8")
    for expected in (
        "from monai.networks.nets import SwinUNETR",
        "inspect.signature(SwinUNETR)",
        '"feature_size": 48',
        '"use_checkpoint": True',
        "SwinUNETR_supervised_pretraining_4mod_ddp_v1",
        "PRETRAINING_CONFIGURATION",
        '"patch_size": [96, 96, 96]',
        'configuration.get("trainable_parameters", -1)',
    ):
        assert expected in common, expected
    model = build_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 50_000_000 < parameter_count < 100_000_000
    assert "swinViT.patch_embed.proj.weight" in model.state_dict()
    assert '"--pair-batch-size", type=int, default=0' in training
    assert '"--effective-pair-batch-size", type=int, default=8' in training
    preflight = (package / "preflight_cv_v1.py").read_text(encoding="utf-8")
    assert "identical_to_completed_unet_folds" in preflight
    assert "REFERENCE_UNET_CV_ROOT" in preflight
    manager = (package / "manager_v1.py").read_text(encoding="utf-8")
    assert "len(visible) != 2" in manager
    assert "fold_manager_v1.py" in manager


def main() -> None:
    test_volume_bins()
    test_boolean_normalization()
    test_infiltration_rule_and_metrics()
    test_fivefold_registry()
    test_wiring()
    print(
        "SELF-TEST PASS: patient-grouped five-fold registry, inner validation, "
        "infiltration rule, metrics, two-GPU scheduler, and resume wiring"
    )


if __name__ == "__main__":
    main()
