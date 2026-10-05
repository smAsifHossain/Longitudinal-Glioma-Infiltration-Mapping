#!/usr/bin/env python3
"""Small deterministic tests for volume conversion and agreement statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biomarker_common_v1 import (
    GROUP_ORDER,
    add_volumes,
    agreement_metrics,
    infer_voxel_volume_ml,
    lin_ccc,
    patient_bootstrap,
    volume_group,
)


def main() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"patient_{index:02d}" for index in range(12)],
            "label_volume_ml": [0.0, 0.05, 0.5, 1.0, 2.0, 5.0, 10.0, 11.0, 20.0, 3.0, 7.0, 15.0],
            "expansion_target_voxels": [0, 50, 500, 1000, 2000, 5000, 10000, 11000, 20000, 3000, 7000, 15000],
            "expansion_predicted_voxels": [0, 60, 450, 900, 2200, 4800, 10500, 12000, 19000, 3200, 6800, 14500],
        }
    )
    voxel_volume_ml = infer_voxel_volume_ml(frame)
    assert abs(voxel_volume_ml - 0.001) < 1e-12
    enriched = add_volumes(frame, voxel_volume_ml)
    assert enriched.volume_group.tolist() == [volume_group(value) for value in frame.label_volume_ml]
    assert set(enriched.volume_group) == set(GROUP_ORDER)
    identity = np.arange(10, dtype=float)
    assert abs(lin_ccc(identity, identity) - 1.0) < 1e-12
    metrics = agreement_metrics(identity, identity + 0.5)
    assert abs(metrics["spearman_rho"] - 1.0) < 1e-12
    assert abs(metrics["mae_ml"] - 0.5) < 1e-12
    bootstrap = patient_bootstrap(enriched, repetitions=100, seed=7)
    assert set(bootstrap.metric) == {
        "spearman_rho",
        "lin_ccc",
        "mae_ml",
        "bias_ml",
        "bland_altman_lower_ml",
        "bland_altman_upper_ml",
    }
    assert bootstrap.finite_bootstrap_repetitions.min() == 100
    focused = patient_bootstrap(
        enriched,
        repetitions=100,
        seed=8,
        metrics=["mae_ml", "bias_ml"],
    )
    assert set(focused.metric) == {"mae_ml", "bias_ml"}
    print(
        "SELF-TEST PASS: volume conversion, groups, agreement, bias, "
        "Bland-Altman limits, and patient bootstrap"
    )


if __name__ == "__main__":
    main()
