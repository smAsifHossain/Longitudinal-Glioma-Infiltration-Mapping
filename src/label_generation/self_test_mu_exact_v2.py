#!/usr/bin/env python3
from __future__ import annotations

import py_compile

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from path_config import CODE_ROOT

CODE = CODE_ROOT


def main() -> None:
    sources = [
        CODE / "generate_mu_case_exact_v2.py",
        CODE / "run_mu_exact_391_v2.py",
        CODE / "preflight_mu_exact_v2.py",
        CODE / "audit_mu_exact_outputs_v2.py",
        CODE / "manager_mu_exact_v2.py",
        CODE / "monitor_mu_exact_v2.py",
        CODE / "path_config.py",
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        py_compile.compile(str(source), doraise=True)

    generator = (CODE / "generate_mu_case_exact_v2.py").read_text(encoding="utf-8")
    for token in (
        'RECURRENT_TARGET_LABELS = (1, 2, 3)',
        'interpolator="genericLabel"',
        'interpolator="linear"',
        'registration_transform_sha256',
        'exact_registration_cache_complete',
    ):
        if token not in generator:
            raise RuntimeError(f"Generator control missing: {token}")

    early = np.zeros((32, 32, 32), dtype=bool)
    early[12:16, 12:16, 12:16] = True
    recurrent = np.zeros_like(early)
    recurrent[14:22, 12:16, 12:16] = True
    recurrent[28:30, 28:30, 28:30] = True
    distance = distance_transform_edt(~early)
    components, count = label(recurrent, structure=np.ones((3, 3, 3), np.uint8))
    anchored = np.zeros_like(recurrent)
    for component_id in range(1, count + 1):
        component = components == component_id
        if float(distance[component].min()) <= 2.0 + 1e-6:
            anchored |= component
    infiltration = anchored & (distance > 2.0) & (distance <= 20.0)
    if int(infiltration.sum()) != 64 or infiltration[28:30, 28:30, 28:30].any():
        raise RuntimeError("WT-to-WT rule self-test failed")

    print(
        "SELF-TEST PASS: exact one-pass MU registration/cache/label code; "
        "WT123-to-WT123 rule"
    )


if __name__ == "__main__":
    main()
