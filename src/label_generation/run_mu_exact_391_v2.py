#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from path_config import CODE_ROOT, DATA_ROOT, MANIFEST_PATH, OUTPUT_ROOT, PROJECT_ROOT


V5_ROOT = PROJECT_ROOT
MANIFEST_INPUT = MANIFEST_PATH
CASE_SCRIPT = CODE_ROOT / "generate_mu_case_exact_v2.py"
RUNTIME_VERSION = "2026-08-15-exact-v2"

SUMMARY_DIR = OUTPUT_ROOT / "summary"
LOG_DIR = OUTPUT_ROOT / "logs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
METRICS_DIR = OUTPUT_ROOT / "metrics_json"
PROGRESS_PATH = OUTPUT_ROOT / "progress.json"
RUN_STATUS_CSV = SUMMARY_DIR / "MU_Glioma_Post__WT_Full391__run_status.csv"
METRICS_CSV = SUMMARY_DIR / "MU_Glioma_Post__WT_Full391__metrics_summary.csv"
REVIEW_CSV = SUMMARY_DIR / "MU_Glioma_Post__WT_Full391__review_required.csv"
FAILURES_CSV = SUMMARY_DIR / "MU_Glioma_Post__WT_Full391__failures.csv"
SUMMARY_JSON = SUMMARY_DIR / "MU_Glioma_Post__WT_Full391__summary.json"
SUMMARY_TXT = SUMMARY_DIR / "MU_Glioma_Post__WT_Full391__summary.txt"
EXACT_COMPLETE = OUTPUT_ROOT / "EXACT_CACHE_COMPLETE"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tp_tag(value: str) -> str:
    return f"TP{int(str(value).split('_')[-1]):02d}"


def stem(row: pd.Series) -> str:
    return (
        f"{row['case_id']}__{row['patient']}__"
        f"{tp_tag(row['earlier_timepoint'])}-to-{tp_tag(row['recurrent_timepoint'])}"
        "__T1N__Rigid__WT123__Local2to20mm"
    )


def review_reasons(metrics: dict) -> list[str]:
    reasons: list[str] = []
    before = metrics.get("ncc_before")
    after = metrics.get("ncc_after")
    gain = metrics.get("ncc_gain")
    if isinstance(after, (int, float)) and after < 0.65:
        reasons.append("ncc_after_below_0.65")
    if isinstance(gain, (int, float)) and gain <= 0:
        reasons.append("registration_did_not_improve")
    if before is None or after is None:
        reasons.append("ncc_unavailable")
    return reasons


def exact_outputs_valid(metrics: dict) -> bool:
    if metrics.get("runtime_version") != RUNTIME_VERSION:
        return False
    if metrics.get("exact_registration_cache_complete") is not True:
        return False
    if metrics.get("registration_interpolator_mri") != "linear":
        return False
    if metrics.get("registration_interpolator_segmentation") != "genericLabel":
        return False
    path_fields = [
        "registration_transform_path",
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
        "excluded_region_path",
        "qc_image",
    ]
    if not all(Path(str(metrics.get(field, ""))).is_file() for field in path_fields):
        return False
    transform = Path(str(metrics["registration_transform_path"]))
    digest = hashlib.sha256(transform.read_bytes()).hexdigest()
    return digest == str(metrics.get("registration_transform_sha256", ""))


def main(limit: int | None) -> None:
    for directory in (SUMMARY_DIR, LOG_DIR, MANIFEST_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_INPUT)
    if len(manifest) != 391:
        raise RuntimeError(f"Expected 391 MU cases, found {len(manifest)}")
    if limit is not None:
        manifest = manifest.iloc[:limit].copy()
    manifest["recurrent_target"] = "WT123_labels_1_2_3"
    manifest["runtime_version"] = RUNTIME_VERSION
    manifest.to_csv(MANIFEST_DIR / "MU_Glioma_Post__WT_Full391__manifest.csv", index=False)

    total = len(manifest)
    if limit is None:
        EXACT_COMPLETE.unlink(missing_ok=True)
    run_rows: list[dict] = []
    metrics_rows: list[dict] = []
    failures: list[dict] = []
    started = time.time()
    threads = os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "8"))
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = threads
    environment["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = threads

    for index, row in manifest.iterrows():
        case_number = len(run_rows) + 1
        case_stem = stem(row)
        metrics_path = METRICS_DIR / f"{case_stem}__metrics.json"
        log_path = LOG_DIR / f"{case_stem}.log"
        status = "existing"
        return_code = 0
        error = ""

        elapsed = time.time() - started
        rate = case_number / elapsed if elapsed > 0 else 0.0
        eta_minutes = (total - case_number + 1) / rate / 60.0 if rate > 0 else None
        atomic_json(PROGRESS_PATH, {
            "status": "RUNNING",
            "phase": "MU-Glioma-Post",
            "completed": case_number - 1,
            "total": total,
            "case_id": str(row["case_id"]),
            "patient": str(row["patient"]),
            "eta_minutes": eta_minutes,
        })

        print(f"MU {case_number:03d}/{total:03d} | {row['case_id']} | {row['patient']}", flush=True)

        if metrics_path.is_file():
            try:
                existing_metrics = json.loads(metrics_path.read_text())
                if not exact_outputs_valid(existing_metrics):
                    metrics_path.unlink()
            except Exception:
                metrics_path.unlink(missing_ok=True)

        if not metrics_path.is_file():
            status = "completed"
            command = [
                sys.executable,
                "-u",
                str(CASE_SCRIPT),
                "--case-id", str(row["case_id"]),
                "--patient", str(row["patient"]),
                "--earlier-timepoint", str(row["earlier_timepoint"]),
                "--recurrent-timepoint", str(row["recurrent_timepoint"]),
            ]
            with log_path.open("w") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    check=False,
                )
            return_code = result.returncode

        if return_code == 0 and metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text())
            if not exact_outputs_valid(metrics):
                raise RuntimeError(
                    f"{row['case_id']}: generator returned without a complete exact cache"
                )
            reasons = review_reasons(metrics)
            metrics["review_required"] = bool(reasons)
            metrics["review_reason"] = ";".join(reasons)
            metrics_rows.append(metrics)
        else:
            status = "failed"
            error = f"case generator exited with code {return_code}"
            failures.append({
                "case_id": str(row["case_id"]),
                "patient": str(row["patient"]),
                "error": error,
                "log_path": str(log_path),
            })

        run_rows.append({
            "case_id": str(row["case_id"]),
            "patient": str(row["patient"]),
            "earlier_timepoint": str(row["earlier_timepoint"]),
            "recurrent_timepoint": str(row["recurrent_timepoint"]),
            "status": status,
            "return_code": return_code,
            "metrics_path": str(metrics_path),
            "log_path": str(log_path),
            "error": error,
        })
        write_csv(RUN_STATUS_CSV, run_rows)
        write_csv(FAILURES_CSV, failures)
        if return_code != 0:
            raise RuntimeError(
                f"{row['case_id']}: exact case generation failed; log={log_path}"
            )

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(METRICS_CSV, index=False)
    review_frame = (
        metrics_frame[metrics_frame["review_required"] == True].copy()
        if len(metrics_frame)
        else pd.DataFrame()
    )
    review_frame.to_csv(REVIEW_CSV, index=False)

    summary = {
        "status": "PASS" if not failures and len(metrics_rows) == total else "FAILED",
        "method": "earlier_WT123_to_recurrent_WT123_T1N_rigid_local_gt2_le20mm",
        "runtime_version": RUNTIME_VERSION,
        "registration_and_cache_generated_together": True,
        "cases_requested": total,
        "completed_cases": len(metrics_rows),
        "failed_cases": len(failures),
        "review_required_cases": int(len(review_frame)),
        "empty_infiltration_labels": int(
            (metrics_frame["infiltration_volume_ml"] == 0).sum()
        ) if len(metrics_frame) else 0,
        "median_infiltration_volume_ml": float(
            metrics_frame["infiltration_volume_ml"].median()
        ) if len(metrics_frame) else None,
        "output_root": str(OUTPUT_ROOT),
    }
    atomic_json(SUMMARY_JSON, summary)
    SUMMARY_TXT.write_text("\n".join([
        "MU-Glioma-Post Whole-Tumor Label Generation",
        "Method: one-pass T1N rigid; saved transform + registered 4-modal MRI + genericLabel segmentation; earlier/recurrent WT labels 1+2+3; local distance >2 mm and <=20 mm",
        f"Completed: {summary['completed_cases']}/{summary['cases_requested']}",
        f"Failed: {summary['failed_cases']}",
        f"Review required: {summary['review_required_cases']}",
        f"Empty labels: {summary['empty_infiltration_labels']}",
        f"Output root: {OUTPUT_ROOT}",
    ]) + "\n")
    atomic_json(PROGRESS_PATH, {
        "status": summary["status"],
        "phase": "complete",
        "completed": len(metrics_rows),
        "total": total,
        "failed": len(failures),
    })
    if summary["status"] != "PASS":
        raise RuntimeError(f"MU generation incomplete: {summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    arguments = parser.parse_args()
    try:
        main(arguments.limit)
    except Exception:
        atomic_json(PROGRESS_PATH, {
            "status": "FAILED",
            "phase": "failed",
            "error": traceback.format_exc(),
        })
        raise
