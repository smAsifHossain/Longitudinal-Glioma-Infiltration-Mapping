#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from path_config import CONTROL_ROOT, OUTPUT_ROOT


CONTROL = CONTROL_ROOT
OUTPUT = OUTPUT_ROOT


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def count(pattern: str) -> int:
    return sum(1 for _ in OUTPUT.glob(pattern)) if OUTPUT.is_dir() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.parse_args()
    manager = read(CONTROL / "manager_status.json")
    preflight = read(CONTROL / "preflight_progress.json")
    generation = read(OUTPUT / "progress.json")
    audit = read(OUTPUT / "exact_output_audit_progress.json")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Manager: {manager.get('status', 'waiting')} | {manager.get('phase', 'waiting')}")
    print(f"Preflight: {preflight.get('status', 'waiting')} | {preflight.get('phase', 'waiting')}")
    if generation:
        print(
            f"MU exact generation: {generation.get('status')} | "
            f"{generation.get('completed', 0)}/{generation.get('total', 391)} | "
            f"{generation.get('case_id', generation.get('phase', 'waiting'))} | "
            f"ETA={generation.get('eta_minutes', 'waiting')}m"
        )
    else:
        print("MU exact generation: waiting")
    if audit:
        print(
            f"Exact output audit: {audit.get('status')} | "
            f"{audit.get('completed', 0)}/{audit.get('total', 391)} | "
            f"{audit.get('case_id', audit.get('phase', 'waiting'))}"
        )
    else:
        print("Exact output audit: waiting")
    print(
        "Outputs: "
        f"labels={count('infiltration_labels_nifti/*.nii.gz')} | "
        f"QC={count('qc_images/*.png')} | "
        f"transforms={count('registration_transforms/*/*.mat')} | "
        f"registered-4mod={count('registered_recurrent_4mod/*/*.nii.gz')} | "
        f"registered-seg={count('registered_recurrent_segmentation/*/*.nii.gz')}"
    )
    if manager.get("error"):
        print(f"Error: {manager['error']}")


if __name__ == "__main__":
    main()
