#!/usr/bin/env python3
"""Remove current figure artifacts and reset the completed-stage marker."""

from biomarker_common_v1 import OUTPUT_ROOT, atomic_json


def main() -> None:
    preflight = OUTPUT_ROOT / "preflight" / "PASS"
    if not preflight.is_file():
        raise RuntimeError(f"Verified preflight marker is missing: {preflight}")
    figures = OUTPUT_ROOT / "figures"
    removed = 0
    if figures.is_dir():
        for pattern in ("*.png", "*.pdf"):
            for figure in figures.glob(pattern):
                figure.unlink()
                removed += 1
    (OUTPUT_ROOT / "COMPLETE").unlink(missing_ok=True)
    atomic_json(
        OUTPUT_ROOT / "manager_progress.json",
        {"status": "waiting", "phase": "figure_regeneration_prepared"},
    )
    atomic_json(
        OUTPUT_ROOT / "progress.json",
        {
            "status": "waiting",
            "phase": "figure_regeneration_prepared",
            "message": f"Removed {removed} previous figure files",
        },
    )
    print(f"Removed {removed} previous PNG/PDF figures from: {figures}")
    print("Existing verified tables and preflight inputs were preserved.")


if __name__ == "__main__":
    main()
