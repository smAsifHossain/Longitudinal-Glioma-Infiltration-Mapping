#!/usr/bin/env python3
"""One-shot status display for the OOF biomarker agreement analysis."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from biomarker_common_v1 import OUTPUT_ROOT


def read_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def state(path, default="waiting"):
    payload = read_json(path)
    return payload.get("status", default), payload.get("phase", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.parse_args()
    manager_status, manager_phase = state(OUTPUT_ROOT / "manager_progress.json")
    preflight_status, preflight_phase = state(OUTPUT_ROOT / "preflight" / "summary.json")
    analysis = read_json(OUTPUT_ROOT / "progress.json")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Manager: {manager_status} | {manager_phase or 'waiting'}")
    print(f"Preflight: {preflight_status} | {preflight_phase or 'waiting'}")
    if analysis.get("status") == "PASS" and analysis.get("phase") == "complete":
        print(
            "Analysis: PASS | complete | "
            f"Swin rho/CCC/MAE={analysis.get('primary_spearman_rho', float('nan')):.3f}/"
            f"{analysis.get('primary_lin_ccc', float('nan')):.3f}/"
            f"{analysis.get('primary_mae_ml', float('nan')):.3f}mL | "
            f"figures={analysis.get('figures', 0)}"
        )
    else:
        print(
            f"Analysis: {analysis.get('status', 'waiting')} | "
            f"{analysis.get('phase', 'waiting')} | {analysis.get('message', 'waiting')}"
        )


if __name__ == "__main__":
    main()
