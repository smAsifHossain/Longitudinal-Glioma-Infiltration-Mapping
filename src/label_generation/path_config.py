"""Environment-based paths shared by label-generation utilities."""
from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_ROOT.parents[1]
PROJECT_ROOT = Path(
    os.environ.get("GLIOMA_PROJECT_ROOT", str(REPOSITORY_ROOT))
).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get("GLIOMA_DATA_ROOT", str(PROJECT_ROOT / "data" / "MU-Glioma-Post"))
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("GLIOMA_LABEL_ROOT", str(PROJECT_ROOT / "label" / "MU-Glioma-Post"))
).expanduser().resolve()
CONTROL_ROOT = Path(
    os.environ.get(
        "GLIOMA_LABEL_CONTROL_ROOT",
        str(PROJECT_ROOT / "label" / "control_mu_exact_v2"),
    )
).expanduser().resolve()
MANIFEST_PATH = Path(
    os.environ.get(
        "GLIOMA_MANIFEST", str(CODE_ROOT / "manifests" / "mu_full391_manifest.csv")
    )
).expanduser().resolve()
