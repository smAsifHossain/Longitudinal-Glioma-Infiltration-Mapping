#!/usr/bin/env python3
"""Global fail-fast audit and immutable patient-level five-fold registry."""
from __future__ import annotations

import json
import math
import os
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from common_v1 import (
    CACHE_PATH_COLUMNS,
    CV_FOLDS,
    CV_ROOT,
    CV_SPLIT_SEED,
    MASK_PATH_COLUMNS,
    PATCH_SIZE,
    PRETRAINED_CHECKPOINT,
    QC_EXCLUSIONS,
    REFERENCE_UNET_CV_ROOT,
    VOLUME_BIN_ORDER,
    VOLUME_STAGE_LABELS,
    PairedPatchDataset,
    RAMPairStore,
    apply_qc_exclusions,
    assign_outer_folds,
    atomic_csv,
    atomic_json,
    binary_metrics,
    build_model,
    derive_infiltration_rule,
    geometry_equal,
    infiltration_volume_bin,
    load_mu_frame,
    segmentation_loss,
    seed_everything,
    sha256,
    source_provenance,
    split_distribution,
    split_mu_cv_fold,
    strict_load_pretrained,
)


def _runtime_resources() -> dict:
    if not PRETRAINED_CHECKPOINT.is_file():
        raise FileNotFoundError(PRETRAINED_CHECKPOINT)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Global preflight must be isolated to exactly one visible CUDA GPU"
        )
    available_cpus = len(os.sched_getaffinity(0))
    if available_cpus < 32:
        raise RuntimeError(f"32 allocated CPU cores are required; found {available_cpus}")
    cgroup_relative = next(
        (
            line.split(":", 2)[2]
            for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
            if line.startswith("0::")
        ),
        "",
    )
    memory_path = Path(f"/sys/fs/cgroup{cgroup_relative}/memory.max")
    memory_text = memory_path.read_text(encoding="utf-8").strip()
    memory_gib = float("inf") if memory_text == "max" else int(memory_text) / 1024**3
    if memory_gib < 400.0:
        raise RuntimeError(
            f"Request 512 GB RAM (>=400 GiB usable); found {memory_gib:.1f} GiB"
        )
    gpu_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if gpu_gib < 80.0:
        raise RuntimeError(f"Expected an >=80 GB GPU; found {gpu_gib:.1f} GB")
    return {
        "cpu_cores": available_cpus,
        "memory_limit_gib": memory_gib,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": gpu_gib,
    }


def _write_fold_manifests(registry: pd.DataFrame, output: Path) -> dict[str, dict]:
    fold_summaries: dict[str, dict] = {}
    all_test_ids: list[str] = []
    all_test_patients: list[str] = []
    for fold in range(CV_FOLDS):
        train, validation, test = split_mu_cv_fold(registry, fold)
        fold_root = CV_ROOT / f"fold_{fold}"
        fold_preflight = fold_root / "preflight"
        fold_preflight.mkdir(parents=True, exist_ok=True)
        train_path = fold_preflight / "mu_train.csv"
        validation_path = fold_preflight / "mu_validation.csv"
        test_path = fold_preflight / "mu_test_locked.csv"
        atomic_csv(train, train_path)
        atomic_csv(validation, validation_path)
        atomic_csv(test, test_path)
        patient_sets = {
            "train": set(train.patient_id.astype(str)),
            "validation": set(validation.patient_id.astype(str)),
            "test": set(test.patient_id.astype(str)),
        }
        if any(
            patient_sets[first] & patient_sets[second]
            for first, second in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            )
        ):
            raise RuntimeError(f"Patient leakage in fold {fold}")
        all_test_ids.extend(test.case_id.astype(str))
        all_test_patients.extend(test.patient_id.astype(str).unique())
        distributions = {
            "train": split_distribution(train),
            "validation": split_distribution(validation),
            "outer_test": split_distribution(test),
        }
        available_bins = pd.to_numeric(
            registry.infiltration_volume_ml, errors="raise"
        ).map(infiltration_volume_bin)
        for volume_bin in VOLUME_BIN_ORDER:
            available_patients = int(
                registry.loc[available_bins == volume_bin, "patient_id"].nunique()
            )
            if available_patients >= CV_FOLDS and not bool(
                pd.to_numeric(test.infiltration_volume_ml, errors="raise")
                .map(infiltration_volume_bin)
                .eq(volume_bin)
                .any()
            ):
                raise RuntimeError(
                    f"Outer fold {fold} lacks representable volume bin {volume_bin}"
                )
        test_lock = {
            "status": "LOCKED",
            "dataset": "MU-Glioma-Post",
            "fold": fold,
            "manifest": str(test_path.resolve()),
            "manifest_sha256": sha256(test_path),
            "pairs": int(len(test)),
            "patients": int(test.patient_id.nunique()),
            "outer_registry_seed": CV_SPLIT_SEED,
            "access_policy": "metrics are computed once after inner validation selects the checkpoint",
        }
        atomic_json(fold_preflight / "test_lock.json", test_lock)
        atomic_json(fold_preflight / "split_volume_distribution.json", distributions)
        fold_summary = {
            "status": "PASS",
            "fold": fold,
            "train": distributions["train"],
            "validation": distributions["validation"],
            "outer_test": distributions["outer_test"],
            "patient_leakage": False,
            "test_manifest_sha256": test_lock["manifest_sha256"],
        }
        atomic_json(fold_preflight / "summary.json", fold_summary)
        atomic_json(
            fold_preflight / "progress.json",
            {"status": "PASS", "phase": "complete", "fold": fold},
        )
        (fold_preflight / "PASS").touch()
        fold_summaries[str(fold)] = fold_summary
    if len(all_test_ids) != len(registry) or len(set(all_test_ids)) != len(registry):
        raise RuntimeError("Outer tests do not cover every eligible scan pair exactly once")
    if set(all_test_patients) != set(registry.patient_id.astype(str)):
        raise RuntimeError("Outer tests do not cover every eligible patient")
    atomic_json(output / "fold_summaries.json", fold_summaries)
    return fold_summaries


def _audit_exact_labels(mu_frame: pd.DataFrame, output: Path, progress: Path) -> dict:
    records = []
    nonempty_dice: list[float] = []
    empty_exact: list[float] = []
    for index, row in enumerate(mu_frame.itertuples(index=False), start=1):
        paths = [
            str(getattr(row, column))
            for column in [*CACHE_PATH_COLUMNS, *MASK_PATH_COLUMNS, "label_path"]
        ]
        images = [nib.load(path) for path in paths]
        reference = images[0]
        if any(not geometry_equal(reference, image) for image in images[1:]):
            raise RuntimeError(f"{row.case_id}: cross-timepoint geometry mismatch")
        spacing = tuple(float(value) for value in reference.header.get_zooms()[:3])
        early_labels = np.rint(
            np.asarray(nib.load(str(row.earlier_seg)).dataobj)
        ).astype(np.int16)
        recurrent_labels = np.rint(
            np.asarray(nib.load(str(row.recurrent_seg_registered)).dataobj)
        ).astype(np.int16)
        if set(np.unique(early_labels).astype(int)) - {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{row.case_id}: unexpected earlier labels")
        if set(np.unique(recurrent_labels).astype(int)) - {0, 1, 2, 3, 4}:
            raise RuntimeError(f"{row.case_id}: unexpected recurrent labels")
        if str(row.earlier_reference_type) == "visible_labels_1_2_3":
            early = np.isin(early_labels, [1, 2, 3])
        elif str(row.earlier_reference_type) == "resection_cavity_label_4":
            early = early_labels == 4
        else:
            raise RuntimeError(
                f"{row.case_id}: unexpected earlier reference {row.earlier_reference_type}"
            )
        recurrent = np.isin(recurrent_labels, [1, 2, 3])
        label = np.asarray(nib.load(str(row.label_path)).dataobj) > 0.5
        transform = Path(str(row.registration_transform_path))
        if sha256(transform) != str(row.registration_transform_sha256):
            raise RuntimeError(f"{row.case_id}: rigid-transform hash changed")
        reconstructed = derive_infiltration_rule(early, recurrent, spacing)
        metrics = binary_metrics(reconstructed, label)
        voxel_ml = float(np.prod(spacing) / 1000.0)
        observed_volumes = {
            "earlier": float(early.sum() * voxel_ml),
            "recurrent": float(recurrent.sum() * voxel_ml),
            "label": float(label.sum() * voxel_ml),
        }
        expected_volumes = {
            "earlier": float(row.earlier_reference_volume_ml),
            "recurrent": float(row.registered_recurrent_tumor_volume_ml),
            "label": float(row.infiltration_volume_ml),
        }
        for name in observed_volumes:
            if abs(observed_volumes[name] - expected_volumes[name]) > 0.001:
                raise RuntimeError(
                    f"{row.case_id}: {name} volume changed "
                    f"({observed_volumes[name]:.6f} != {expected_volumes[name]:.6f} mL)"
                )
        if label.any():
            nonempty_dice.append(float(metrics["dice"]))
        else:
            empty_exact.append(float(not reconstructed.any()))
        records.append(
            {
                "case_id": str(row.case_id),
                "patient_id": str(row.patient_id),
                "label_volume_ml": observed_volumes["label"],
                "gt_rule_dice": float(metrics["dice"]),
                "v5_review_required": bool(row.v5_review_required),
            }
        )
        running = float(np.mean(nonempty_dice)) if nonempty_dice else float("nan")
        if index == 1 or index % 10 == 0 or index == len(mu_frame):
            atomic_json(
                progress,
                {
                    "status": "RUNNING",
                    "phase": "geometry_labels_rule",
                    "completed": index,
                    "total": len(mu_frame),
                    "case_id": str(row.case_id),
                    "running_gt_rule_dice_nonempty": running,
                },
            )
            print(
                f"PREFLIGHT {index:3d}/{len(mu_frame)} | {row.case_id} | "
                f"GT-rule Dice={running:.4f}",
                flush=True,
            )
    frame = pd.DataFrame(records)
    atomic_csv(frame, output / "case_checks.csv")
    mean_dice = float(np.mean(nonempty_dice))
    minimum_dice = float(np.min(nonempty_dice))
    empty_accuracy = float(np.mean(empty_exact)) if empty_exact else float("nan")
    if mean_dice < 0.999999 or minimum_dice < 0.999999:
        raise RuntimeError(
            f"Final labels differ from saved registration cache: {mean_dice}/{minimum_dice}"
        )
    if empty_exact and empty_accuracy < 0.999999:
        raise RuntimeError(f"Empty-label reconstruction failed: {empty_accuracy}")
    return {
        "nonempty_mean_dice": mean_dice,
        "nonempty_minimum_dice": minimum_dice,
        "empty_accuracy": empty_accuracy,
    }


def _gpu_smoke(train: pd.DataFrame, output: Path, progress: Path) -> dict:
    from train_v1 import ThreadedBatchLoader

    seed_everything(CV_SPLIT_SEED)
    device = torch.device("cuda:0")
    early_model = build_model()
    initialization = strict_load_pretrained(early_model)
    recurrent_model = build_model()
    recurrent_model.load_state_dict(early_model.state_dict(), strict=True)
    early_model.to(device).train()
    recurrent_model.to(device).train()
    smoke_frame = train.drop_duplicates("patient_id").iloc[:2].reset_index(drop=True)
    if len(smoke_frame) != 2:
        raise RuntimeError("Could not select two patients for the real-data smoke test")
    atomic_json(progress, {"status": "RUNNING", "phase": "gpu_smoke"})
    store = RAMPairStore.preload(smoke_frame, workers=1, prepare_sampling=True)
    dataset = PairedPatchDataset(
        smoke_frame,
        epoch=1,
        steps=1,
        batch_size=1,
        pair_store=store,
    )
    batch = next(iter(ThreadedBatchLoader(dataset, 1, workers=1, prefetch_batches=1)))
    if "label" in batch or "infiltration" in batch:
        raise RuntimeError("Infiltration label leaked into the training batch")
    early_image = batch["early_image"].to(device)
    recurrent_image = batch["recurrent_image"].to(device)
    early_target = batch["early_target"].to(device)
    recurrent_target = batch["recurrent_target"].to(device)
    valid = batch["valid"].to(device)
    amp = bool(torch.cuda.is_bf16_supported())
    early_model.zero_grad(set_to_none=True)
    recurrent_model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(0)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        early_logits = early_model(early_image)
        early_loss, _ = segmentation_loss(early_logits, early_target, valid)
    (0.5 * early_loss).backward()
    del early_logits
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        recurrent_logits = recurrent_model(recurrent_image)
        recurrent_loss, _ = segmentation_loss(recurrent_logits, recurrent_target, valid)
    (0.5 * recurrent_loss).backward()
    if not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for model in (early_model, recurrent_model)
        for parameter in model.parameters()
    ):
        raise RuntimeError("Non-finite gradient in GPU smoke test")
    torch.cuda.synchronize()
    result = {
        "status": "PASS",
        "paired_loss": float(0.5 * early_loss + 0.5 * recurrent_loss),
        "peak_gpu_gib": float(torch.cuda.max_memory_allocated(0) / 1024**3),
        "bf16": amp,
        "pretrained_initialization": initialization,
    }
    atomic_json(output / "gpu_smoke.json", result)
    return result


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    atomic_json(progress, {"status": "RUNNING", "phase": "manifest"})
    resources = _runtime_resources()
    mu_frame = load_mu_frame(check_paths=True)
    modeling, excluded = apply_qc_exclusions(mu_frame)
    if len(modeling) != 353 or modeling.patient_id.nunique() != 146:
        raise RuntimeError(
            f"Expected 353 review-cleared pairs from 146 patients; "
            f"found {len(modeling)}/{modeling.patient_id.nunique()}"
        )
    registry = assign_outer_folds(modeling)
    reference_registry_path = REFERENCE_UNET_CV_ROOT / "preflight/outer_fold_registry.csv"
    reference_complete = REFERENCE_UNET_CV_ROOT / "aggregate/AGGREGATION_COMPLETE"
    if not reference_complete.is_file():
        raise FileNotFoundError(
            "The completed 3D U-Net five-fold aggregate is required to lock the "
            f"comparison folds: {reference_complete}"
        )
    if not reference_registry_path.is_file():
        raise FileNotFoundError(reference_registry_path)
    reference_registry = pd.read_csv(
        reference_registry_path,
        usecols=["case_id", "patient_id", "outer_fold"],
        low_memory=False,
    )
    generated_identity = (
        registry[["case_id", "patient_id", "outer_fold"]]
        .astype({"case_id": str, "patient_id": str, "outer_fold": int})
        .sort_values(["case_id", "patient_id"])
        .reset_index(drop=True)
    )
    reference_identity = (
        reference_registry.astype(
            {"case_id": str, "patient_id": str, "outer_fold": int}
        )
        .sort_values(["case_id", "patient_id"])
        .reset_index(drop=True)
    )
    if not generated_identity.equals(reference_identity):
        raise RuntimeError(
            "Swin-UNETR outer folds differ from the completed 3D U-Net registry"
        )
    atomic_csv(mu_frame, output / "mu_v5_391_exact_full_audit.csv")
    atomic_csv(modeling, output / "mu_v5_review_cleared_modeling.csv")
    atomic_csv(excluded, output / "qc_exclusions.csv")
    registry_path = output / "outer_fold_registry.csv"
    atomic_csv(registry, registry_path)
    if sha256(registry_path) != sha256(reference_registry_path):
        raise RuntimeError(
            "Swin-UNETR registry bytes differ from the completed 3D U-Net registry"
        )
    fold_summaries = _write_fold_manifests(registry, output)
    registry_lock = {
        "status": "LOCKED",
        "dataset": "MU-Glioma-Post",
        "folds": CV_FOLDS,
        "seed": CV_SPLIT_SEED,
        "pairs": int(len(registry)),
        "patients": int(registry.patient_id.nunique()),
        "manifest": str(registry_path.resolve()),
        "manifest_sha256": sha256(registry_path),
        "patient_grouped": True,
        "outer_test_coverage": "every eligible scan pair exactly once",
    }
    atomic_json(output / "outer_fold_registry_lock.json", registry_lock)
    label_audit = _audit_exact_labels(mu_frame, output, progress)
    fold0_train = pd.read_csv(CV_ROOT / "fold_0/preflight/mu_train.csv", low_memory=False)
    smoke = _gpu_smoke(fold0_train, output, progress)
    summary = {
        "status": "PASS",
        "design": {
            "architecture": "two independent pretrained MONAI Swin-UNETRs with feature size 48",
            "inputs": "four registered MRI modalities per timepoint",
            "outputs": "earlier WT and recurrent WT; deterministic finalized 2-20 mm infiltration rule",
            "loss": "0.5 earlier + 0.5 recurrent; each 0.70 Dice + 0.30 BCEWithLogits",
            "outer_cv": "five patient-grouped held-out folds",
            "inner_validation": "deterministic balanced subset of each outer-development cohort",
            "qc_policy": "exclude V5 registration-QC review-required pairs; labels are unchanged",
        },
        "counts": {
            "audited_pairs": int(len(mu_frame)),
            "audited_patients": int(mu_frame.patient_id.nunique()),
            "excluded_pairs": int(len(excluded)),
            "eligible_pairs": int(len(modeling)),
            "eligible_patients": int(modeling.patient_id.nunique()),
        },
        "registry_lock": registry_lock,
        "comparison_lock": {
            "identical_to_completed_unet_folds": True,
            "reference_registry": str(reference_registry_path.resolve()),
            "reference_registry_sha256": sha256(reference_registry_path),
        },
        "folds": fold_summaries,
        "exact_label_audit": label_audit,
        "resources": resources,
        "gpu_smoke": smoke,
        "pretrained_checkpoint_sha256": sha256(PRETRAINED_CHECKPOINT),
        "documented_qc_exclusions": QC_EXCLUSIONS,
        "source_provenance": source_provenance(),
    }
    atomic_json(output / "summary.json", summary)
    (output / "PASS").touch()
    atomic_json(
        progress,
        {
            "status": "PASS",
            "phase": "complete",
            "eligible_pairs": len(modeling),
            "eligible_patients": modeling.patient_id.nunique(),
            "gt_rule_dice_nonempty": label_audit["nonempty_mean_dice"],
        },
    )
    print(
        f"PREFLIGHT PASS | five folds | {len(modeling)} pairs / "
        f"{modeling.patient_id.nunique()} patients | GT-rule Dice="
        f"{label_audit['nonempty_mean_dice']:.4f}",
        flush=True,
    )
    return summary


def main() -> None:
    output = CV_ROOT / "preflight"
    try:
        run(output)
    except Exception as error:
        atomic_json(
            output / "progress.json",
            {"status": "FAILED", "phase": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
