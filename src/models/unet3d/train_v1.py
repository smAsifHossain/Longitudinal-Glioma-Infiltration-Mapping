#!/usr/bin/env python3
"""Resumable 3D U-Net finetuning for one MU five-fold-CV partition."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import os
import signal
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common_v1 import (
    CV_FOLD,
    EXPERIMENT_ROOT,
    PATCH_SIZE,
    REGIONS,
    RUNTIME_VERSION,
    STOP_FILE,
    TRAINING_SEED,
    GracefulStop,
    PairedPatchDataset,
    RAMPairStore,
    atomic_csv,
    atomic_json,
    atomic_torch_save,
    build_model,
    evaluate_case_arrays,
    inference_specialist_pair,
    segmentation_loss,
    seed_everything,
    sha256,
    strict_load_pretrained,
    summarize_case_metrics,
)


STOP_REQUESTED = False


def request_stop(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpointing after the current step.", flush=True)


def empty_accumulator() -> dict[str, Any]:
    return {
        "loss_sum": 0.0,
        "early_loss_sum": 0.0,
        "recurrent_loss_sum": 0.0,
        "early_soft_sum": [0.0],
        "recurrent_soft_sum": [0.0],
        "samples": 0,
        "distinct_cases": [],
        "elapsed_seconds": 0.0,
        "data_wait_seconds": 0.0,
        "gpu_step_seconds": 0.0,
    }


def update_accumulator(
    accumulator: dict[str, Any],
    batch_size: int,
    loss: torch.Tensor,
    live: dict[str, torch.Tensor],
    case_ids: list[str],
    seconds: float,
    data_wait_seconds: float,
    gpu_step_seconds: float,
) -> None:
    accumulator["loss_sum"] += float(loss.detach().cpu()) * batch_size
    accumulator["early_loss_sum"] += float(live["early_loss"].detach().cpu()) * batch_size
    accumulator["recurrent_loss_sum"] += float(live["recurrent_loss"].detach().cpu()) * batch_size
    early = live["early_soft"].detach().cpu().numpy()
    recurrent = live["recurrent_soft"].detach().cpu().numpy()
    accumulator["early_soft_sum"] = (
        np.asarray(accumulator["early_soft_sum"]) + early * batch_size
    ).tolist()
    accumulator["recurrent_soft_sum"] = (
        np.asarray(accumulator["recurrent_soft_sum"]) + recurrent * batch_size
    ).tolist()
    accumulator["samples"] += int(batch_size)
    accumulator["distinct_cases"] = sorted(
        set(accumulator.get("distinct_cases", [])) | set(str(value) for value in case_ids)
    )
    accumulator["elapsed_seconds"] += float(seconds)
    accumulator["data_wait_seconds"] = float(
        accumulator.get("data_wait_seconds", 0.0) + data_wait_seconds
    )
    accumulator["gpu_step_seconds"] = float(
        accumulator.get("gpu_step_seconds", 0.0) + gpu_step_seconds
    )


def accumulator_metrics(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = max(1, int(accumulator["samples"]))
    return {
        "train_loss": float(accumulator["loss_sum"] / count),
        "train_early_loss": float(accumulator["early_loss_sum"] / count),
        "train_recurrent_loss": float(accumulator["recurrent_loss_sum"] / count),
        "train_early_soft_wt": float(accumulator["early_soft_sum"][0] / count),
        "train_recurrent_soft_wt": float(accumulator["recurrent_soft_sum"][0] / count),
        "train_distinct_cases": int(len(accumulator.get("distinct_cases", []))),
        "train_elapsed_minutes": float(accumulator["elapsed_seconds"] / 60.0),
        "train_data_wait_minutes": float(accumulator.get("data_wait_seconds", 0.0) / 60.0),
        "train_gpu_step_minutes": float(accumulator.get("gpu_step_seconds", 0.0) / 60.0),
    }


class ThreadedBatchLoader:
    """Build and prefetch RAM-resident patches without multiprocessing copies."""

    def __init__(
        self,
        dataset: PairedPatchDataset,
        batch_size: int,
        workers: int,
        prefetch_batches: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.workers = max(1, int(workers))
        self.prefetch_batches = max(1, int(prefetch_batches))

    def __iter__(self):
        total_batches = len(self.dataset) // self.batch_size
        executor = ThreadPoolExecutor(max_workers=self.workers)
        queued: deque[list[Future]] = deque()

        def submit(batch_index: int) -> list[Future]:
            start = batch_index * self.batch_size
            return [
                executor.submit(self.dataset.__getitem__, start + offset)
                for offset in range(self.batch_size)
            ]

        next_batch = 0
        try:
            while next_batch < min(total_batches, self.prefetch_batches):
                queued.append(submit(next_batch))
                next_batch += 1
            while queued:
                futures = queued.popleft()
                samples = [future.result() for future in futures]
                if next_batch < total_batches:
                    queued.append(submit(next_batch))
                    next_batch += 1
                batch = torch.utils.data.default_collate(samples)
                for key in (
                    "early_image",
                    "recurrent_image",
                    "early_target",
                    "recurrent_target",
                    "valid",
                ):
                    batch[key] = batch[key].pin_memory()
                yield batch
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def configure_single_gpu(model: torch.nn.Module) -> torch.nn.Module:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This pilot requires exactly one visible CUDA GPU; found {torch.cuda.device_count()}"
        )
    return model


def autocast_context(enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)


def checkpoint_payload(
    early_model: torch.nn.Module,
    recurrent_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    next_step: int,
    batch_size: int,
    accumulator: dict[str, Any],
    best_score: float,
    best_epoch: int,
    bad_epochs: int,
    args: argparse.Namespace,
    amp_enabled: bool,
) -> dict[str, Any]:
    return {
        "version": RUNTIME_VERSION,
        "cv_fold": CV_FOLD,
        "early_model": early_model.state_dict(),
        "recurrent_model": recurrent_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "next_step": int(next_step),
        "pair_batch_size": int(batch_size),
        "accumulator": accumulator,
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "bad_epochs": int(bad_epochs),
        "epochs": int(args.epochs),
        "steps_per_epoch": int(args.steps_per_epoch),
        "samples_per_epoch": int(args.samples_per_epoch),
        "patch_size": list(PATCH_SIZE),
        "loss": "independent WT specialists; 0.5*early + 0.5*recurrent; each 0.70 Dice + 0.30 BCEWithLogits",
        "precision": "BF16 mixed precision" if amp_enabled else "FP32",
        "gpu_count": 1,
    }


def probe_batch_size(
    early_model: torch.nn.Module,
    recurrent_model: torch.nn.Module,
    fit: pd.DataFrame,
    device: torch.device,
    requested: int,
    pair_store: RAMPairStore,
    amp_enabled: bool,
) -> int:
    candidates = [requested] if requested > 0 else [8, 6, 4, 2, 1]
    failures = []

    def try_candidate(candidate: int) -> None:
        dataset = PairedPatchDataset(
            fit,
            epoch=1,
            steps=1,
            batch_size=candidate,
            start_step=0,
            cache_pairs=1,
            pair_store=pair_store,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=candidate, shuffle=False, num_workers=0, pin_memory=True
        )
        batch = next(iter(loader))
        early_image = batch["early_image"].to(device, non_blocking=True)
        recurrent_image = batch["recurrent_image"].to(device, non_blocking=True)
        early_target = batch["early_target"].to(device, non_blocking=True)
        recurrent_target = batch["recurrent_target"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        early_model.zero_grad(set_to_none=True)
        recurrent_model.zero_grad(set_to_none=True)
        with autocast_context(amp_enabled):
            early_logits = early_model(early_image)
            early_loss, _ = segmentation_loss(early_logits, early_target, valid)
        if not torch.isfinite(early_loss):
            raise RuntimeError("non-finite earlier-specialist probe loss")
        (0.5 * early_loss).backward()
        del early_logits
        with autocast_context(amp_enabled):
            recurrent_logits = recurrent_model(recurrent_image)
            recurrent_loss, _ = segmentation_loss(
                recurrent_logits, recurrent_target, valid
            )
        if not torch.isfinite(recurrent_loss):
            raise RuntimeError("non-finite recurrent-specialist probe loss")
        (0.5 * recurrent_loss).backward()
        torch.cuda.synchronize()
        early_model.zero_grad(set_to_none=True)
        recurrent_model.zero_grad(set_to_none=True)

    for candidate in candidates:
        try:
            try_candidate(candidate)
            torch.cuda.empty_cache()
            print(
                f"Single-GPU batch probe PASS: {candidate} registered pairs "
                f"({candidate} volumes/specialist forward; {2*candidate}/optimizer step).",
                flush=True,
            )
            return int(candidate)
        except torch.cuda.OutOfMemoryError as error:
            failures.append(f"{candidate}: CUDA OOM")
            early_model.zero_grad(set_to_none=True)
            recurrent_model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Batch probe fallback from {candidate}: CUDA OOM", flush=True)
            if requested > 0:
                raise
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or requested > 0:
                raise
            failures.append(f"{candidate}: {error}")
            early_model.zero_grad(set_to_none=True)
            recurrent_model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            gc.collect()
    raise RuntimeError(f"No batch size passed: {failures}")


def validate(
    early_model: torch.nn.Module,
    recurrent_model: torch.nn.Module,
    frame: pd.DataFrame,
    pair_store: RAMPairStore,
    device: torch.device,
    epoch: int,
    output: Path,
    sw_batch_size: int,
) -> dict[str, Any]:
    early_model.eval()
    recurrent_model.eval()
    records: list[dict[str, Any]] = []
    started = time.time()
    total = len(frame)
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        pair = pair_store.get(row)
        early_probability, recurrent_probability = inference_specialist_pair(
            early_model,
            recurrent_model,
            pair,
            device,
            sw_batch_size=sw_batch_size,
        )
        metrics, _ = evaluate_case_arrays(early_probability, recurrent_probability, pair)
        record = {
            "case_id": str(row.case_id),
            "patient_id": str(row.patient_id),
            "epoch": int(epoch),
            **metrics,
        }
        records.append(record)
        running = summarize_case_metrics(pd.DataFrame(records))
        elapsed = time.time() - started
        eta = elapsed / index * (total - index) / 60.0
        atomic_json(
            output / "progress.json",
            {
                "status": "RUNNING",
                "phase": "validation",
                "epoch": int(epoch),
                "epochs": None,
                "validation_case": index,
                "validation_total": total,
                "case_id": str(row.case_id),
                "validation_segmentation": running["segmentation_mean_dice_positive"],
                "validation_infiltration_dice": running["infiltration_dice_nonempty"],
                "validation_infiltration_auprc": running["infiltration_auprc_nonempty"],
                "validation_empty_accuracy": running["empty_accuracy"],
                "eta_minutes": eta,
            },
        )
        print(
            f"VALIDATE epoch={epoch:03d} {index:3d}/{total} | {row.case_id} | "
            f"seg={running['segmentation_mean_dice_positive']:.4f} | "
            f"infiltration Dice/AUPRC={running['infiltration_dice_nonempty']:.4f}/"
            f"{running['infiltration_auprc_nonempty']:.4f} | ETA={eta:.1f}m",
            flush=True,
        )
        del pair, early_probability, recurrent_probability
        if index % 10 == 0:
            gc.collect()
    case_frame = pd.DataFrame(records)
    atomic_csv(case_frame, output / "validation" / f"epoch_{epoch:03d}_cases.csv")
    summary = summarize_case_metrics(case_frame)
    summary["validation_minutes"] = float((time.time() - started) / 60.0)
    atomic_json(output / "validation" / f"epoch_{epoch:03d}_summary.json", summary)
    return summary


def plot_history(history: pd.DataFrame, path: Path) -> None:
    if history.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history.epoch, history.train_loss, marker="o", label="Train loss")
    axes[0].set_title("Paired segmentation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        history.epoch,
        history.val_segmentation_mean_dice_positive,
        marker="o",
        label="Segmentation Dice",
    )
    axes[1].plot(
        history.epoch,
        history.val_infiltration_dice_nonempty,
        marker="o",
        label="Infiltration Dice",
    )
    axes[1].plot(
        history.epoch,
        history.val_infiltration_auprc_nonempty,
        marker="o",
        label="Infiltration AUPRC",
    )
    axes[1].set_title("MU validation performance")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    if CV_FOLD is None:
        raise RuntimeError("UNET_CV_FOLD must be set for fold training")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    latest_path = output / "checkpoints/latest.pt"
    best_path = output / "checkpoints/best.pt"
    history_path = output / "history.csv"
    fit_path = EXPERIMENT_ROOT / "preflight/mu_train.csv"
    validation_path = EXPERIMENT_ROOT / "preflight/mu_validation.csv"
    test_path = EXPERIMENT_ROOT / "preflight/mu_test_locked.csv"
    test_lock_path = EXPERIMENT_ROOT / "preflight/test_lock.json"
    if not (EXPERIMENT_ROOT / "preflight/PASS").is_file():
        raise RuntimeError("Preflight has not passed")
    fit = pd.read_csv(fit_path, low_memory=False)
    validation = pd.read_csv(validation_path, low_memory=False)
    test_lock = json.loads(test_lock_path.read_text(encoding="utf-8"))
    if int(test_lock.get("fold", -1)) != CV_FOLD:
        raise RuntimeError("Held-out test lock belongs to a different CV fold")
    if sha256(test_path) != str(test_lock["manifest_sha256"]):
        raise RuntimeError("Locked MU test manifest changed after preflight")
    test_identity = pd.read_csv(
        test_path, usecols=["case_id", "patient_id"], low_memory=False
    )
    if set(fit.patient_id.astype(str)) & set(validation.patient_id.astype(str)):
        raise RuntimeError("Fit/validation leakage detected at training start")
    development_patients = set(fit.patient_id.astype(str)) | set(
        validation.patient_id.astype(str)
    )
    if development_patients & set(test_identity.patient_id.astype(str)):
        raise RuntimeError("Locked-test patient leakage detected at training start")
    checkpoint = None
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("version") != RUNTIME_VERSION:
            raise RuntimeError(
                "This checkpoint is incompatible with the V5 WT-to-WT runtime. "
                "Start this distinct experiment with prepare_fresh_start.sh first."
            )
        if int(checkpoint.get("cv_fold", -1)) != CV_FOLD:
            raise RuntimeError("Checkpoint belongs to a different CV fold")
        if int(checkpoint.get("epochs", -1)) != args.epochs:
            raise RuntimeError("Cannot resume after changing total epochs")
        if int(checkpoint.get("samples_per_epoch", -1)) != args.samples_per_epoch:
            raise RuntimeError("Cannot resume after changing samples per epoch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This pilot requires exactly one visible CUDA GPU; found {torch.cuda.device_count()}"
        )
    seed_everything(TRAINING_SEED)
    torch.set_num_threads(
        min(
            max(1, int(os.environ.get("UNET_CPU_THREADS", "8"))),
            max(1, os.cpu_count() or 1),
        )
    )
    device = torch.device("cuda:0")
    amp_enabled = bool(torch.cuda.is_bf16_supported())

    def preload_progress(stage: str):
        def update(completed: int, total: int, case_id: str, cache_gb: float) -> None:
            state = {
                "status": "RUNNING",
                "phase": "ram_preload",
                "preload_stage": stage,
                "preload_completed": completed,
                "preload_total": total,
                "case_id": case_id,
                "ram_cache_gb": cache_gb,
            }
            if completed == 1 or completed % 5 == 0 or completed == total:
                atomic_json(progress_path, state)
            if completed == 1 or completed % 20 == 0 or completed == total:
                print(
                    f"RAM PRELOAD {stage} {completed}/{total} | {case_id} | "
                    f"cache={cache_gb:.1f}GB",
                    flush=True,
                )
            if STOP_REQUESTED or STOP_FILE.exists():
                raise GracefulStop("Stopped safely during RAM preload; no optimizer step was lost")

        return update

    cache_started = time.time()
    fit_store = RAMPairStore.preload(
        fit,
        workers=args.preload_workers,
        prepare_sampling=True,
        progress=preload_progress("fit"),
    )
    validation_store = RAMPairStore.preload(
        validation,
        workers=args.preload_workers,
        prepare_sampling=False,
        progress=preload_progress("validation"),
    )
    cache_gb = (fit_store.bytes + validation_store.bytes) / 1024**3
    cache_minutes = (time.time() - cache_started) / 60.0
    print(
        f"RAM PRELOAD COMPLETE | pairs={len(fit) + len(validation)} | "
        f"cache={cache_gb:.1f}GB | time={cache_minutes:.1f}m",
        flush=True,
    )

    early_model = build_model()
    initialization = strict_load_pretrained(early_model)
    recurrent_model = build_model()
    recurrent_model.load_state_dict(early_model.state_dict(), strict=True)
    early_model.to(device)
    recurrent_model.to(device)
    if any(
        early_parameter.data_ptr() == recurrent_parameter.data_ptr()
        for early_parameter, recurrent_parameter in zip(
            early_model.parameters(), recurrent_model.parameters()
        )
    ):
        raise RuntimeError("Earlier and recurrent U-Nets unexpectedly share parameter storage")
    start_epoch, start_step = 1, 0
    best_score, best_epoch, bad_epochs = -math.inf, 0, 0
    accumulator = empty_accumulator()
    if checkpoint is not None:
        early_model.load_state_dict(checkpoint["early_model"], strict=True)
        recurrent_model.load_state_dict(checkpoint["recurrent_model"], strict=True)
        start_epoch = int(checkpoint["epoch"])
        start_step = int(checkpoint["next_step"])
        batch_size = int(checkpoint["pair_batch_size"])
        accumulator = checkpoint.get("accumulator", empty_accumulator())
        best_score = float(checkpoint.get("best_score", -math.inf))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        bad_epochs = int(checkpoint.get("bad_epochs", 0))
    early_model = configure_single_gpu(early_model)
    recurrent_model = configure_single_gpu(recurrent_model)
    if checkpoint is None:
        batch_size = probe_batch_size(
            early_model,
            recurrent_model,
            fit,
            device,
            args.pair_batch_size,
            fit_store,
            amp_enabled,
        )
    args.steps_per_epoch = int(math.ceil(args.samples_per_epoch / batch_size))
    if checkpoint is not None and int(checkpoint.get("steps_per_epoch", -1)) != args.steps_per_epoch:
        raise RuntimeError("Cannot resume after changing batch-derived steps per epoch")
    trainable_parameters = [
        *list(early_model.parameters()),
        *list(recurrent_model.parameters()),
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=5e-7
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(
            f"RESUME WT-to-WT epoch={start_epoch} step={start_step}/{args.steps_per_epoch}",
            flush=True,
        )
    config = {
        "runtime_version": RUNTIME_VERSION,
        "cv_fold": CV_FOLD,
        "architecture": "two independent MONAI 3D U-Nets: earlier-WT and recurrent-WT specialists",
        "initialization": initialization,
        "input": "registered earlier 4-modal MRI to earlier U-Net; registered recurrent 4-modal MRI to recurrent U-Net",
        "outputs": "WT for each timepoint; deterministic V5 WT-to-WT infiltration mask",
        "loss": "0.5*earlier-WT + 0.5*recurrent-WT; each 0.70 Dice + 0.30 BCEWithLogits",
        "precision": "BF16 mixed precision" if amp_enabled else "FP32 fallback",
        "gpu_execution": "two independent models trained sequentially on one visible CUDA GPU; no DataParallel",
        "gpu_count": 1,
        "pair_batch_size": batch_size,
        "volumes_per_specialist_forward": batch_size,
        "volumes_per_optimizer_step": 2 * batch_size,
        "patch_size": list(PATCH_SIZE),
        "epochs": args.epochs,
        "samples_per_epoch": args.samples_per_epoch,
        "steps_per_epoch": args.steps_per_epoch,
        "ram_cache": {
            "pairs": int(len(fit) + len(validation)),
            "gigabytes": cache_gb,
            "preload_minutes": cache_minutes,
            "preload_workers": args.preload_workers,
            "binary_arrays_compacted_to_uint8": True,
            "mri_arrays_preserved_as_float32": True,
        },
        "batch_preparation": {
            "threads": args.batch_workers,
            "prefetch_batches": args.prefetch_batches,
            "sampling_coordinates_precomputed_once": True,
        },
        "learning_rate": args.learning_rate,
        "threshold": 0.5,
        "selection": "best of pretrained epoch 0 and finetuned epochs by 0.5*positive-region segmentation Dice + 0.5*nonempty infiltration Dice",
        "data_split": "MU patient-grouped outer five-fold CV; approximately 70% train / 10% inner validation / 20% held-out outer test",
        "test_images_labels_or_model_metrics_accessed_during_training": False,
        "test_identity_columns_accessed_for_leakage_check": True,
        "locked_test_manifest_sha256": str(test_lock["manifest_sha256"]),
        "early_stopping": {"minimum_epochs": args.minimum_epochs, "patience": args.patience},
    }
    atomic_json(output / "config.json", config)

    # Evaluate the untouched pretrained initialization before any optimizer step.
    # This prevents mandatory finetuning from silently replacing a stronger model.
    if checkpoint is None:
        baseline_validation = validate(
            early_model,
            recurrent_model,
            validation,
            validation_store,
            device,
            0,
            output,
            sw_batch_size=args.sw_batch_size,
        )
        best_score = float(baseline_validation["selection_score"])
        best_epoch = 0
        bad_epochs = 0
        atomic_json(output / "baseline_epoch0_summary.json", baseline_validation)
        atomic_torch_save(
            {
                "version": RUNTIME_VERSION,
                "cv_fold": CV_FOLD,
                "early_model": early_model.state_dict(),
                "recurrent_model": recurrent_model.state_dict(),
                "epoch": 0,
                "best_score": best_score,
                "validation": baseline_validation,
                "pair_batch_size": batch_size,
                "threshold": 0.5,
                "initialization": initialization,
                "precision": "BF16 mixed precision" if amp_enabled else "FP32",
                "gpu_count": 1,
            },
            best_path,
        )
        atomic_torch_save(
            checkpoint_payload(
                early_model,
                recurrent_model,
                optimizer,
                scheduler,
                1,
                0,
                batch_size,
                empty_accumulator(),
                best_score,
                best_epoch,
                bad_epochs,
                args,
                amp_enabled,
            ),
            latest_path,
        )
        print(
            "EPOCH 0 PRETRAINED BASELINE | "
            f"seg={baseline_validation['segmentation_mean_dice_positive']:.4f} | "
            f"infiltration Dice/AUPRC={baseline_validation['infiltration_dice_nonempty']:.4f}/"
            f"{baseline_validation['infiltration_auprc_nonempty']:.4f} | "
            f"selection={best_score:.4f}",
            flush=True,
        )

    if start_epoch > args.epochs:
        (output / "TRAINING_COMPLETE").touch()
        return
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.time()
        early_model.train()
        recurrent_model.train()
        dataset = PairedPatchDataset(
            fit,
            epoch=epoch,
            steps=args.steps_per_epoch,
            batch_size=batch_size,
            start_step=start_step if epoch == start_epoch else 0,
            cache_pairs=args.cache_pairs,
            pair_store=fit_store,
        )
        loader = ThreadedBatchLoader(
            dataset,
            batch_size=batch_size,
            workers=args.batch_workers,
            prefetch_batches=args.prefetch_batches,
        )
        current_start = start_step if epoch == start_epoch else 0
        if epoch != start_epoch:
            accumulator = empty_accumulator()
        loader_iterator = iter(loader)
        for relative_step in range(1, len(dataset) // batch_size + 1):
            cycle_started = time.time()
            batch = next(loader_iterator)
            data_wait_seconds = time.time() - cycle_started
            step = current_start + relative_step
            step_started = time.time()
            early_image = batch["early_image"].to(device, non_blocking=True)
            recurrent_image = batch["recurrent_image"].to(device, non_blocking=True)
            early_target = batch["early_target"].to(device, non_blocking=True)
            recurrent_target = batch["recurrent_target"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled):
                early_logits = early_model(early_image)
                early_loss, early_live = segmentation_loss(
                    early_logits, early_target, valid
                )
            if not torch.isfinite(early_loss):
                raise RuntimeError(
                    f"Non-finite earlier-specialist loss at epoch {epoch} step {step}"
                )
            (0.5 * early_loss).backward()
            del early_logits
            with autocast_context(amp_enabled):
                recurrent_logits = recurrent_model(recurrent_image)
                recurrent_loss, recurrent_live = segmentation_loss(
                    recurrent_logits, recurrent_target, valid
                )
            if not torch.isfinite(recurrent_loss):
                raise RuntimeError(
                    f"Non-finite recurrent-specialist loss at epoch {epoch} step {step}"
                )
            (0.5 * recurrent_loss).backward()
            loss = 0.5 * early_loss.detach() + 0.5 * recurrent_loss.detach()
            live = {
                "early_loss": early_loss.detach(),
                "recurrent_loss": recurrent_loss.detach(),
                "early_soft": early_live["soft_dice"].detach(),
                "recurrent_soft": recurrent_live["soft_dice"].detach(),
            }
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=12.0)
            )
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch} step {step}")
            optimizer.step()
            torch.cuda.synchronize()
            gpu_step_seconds = time.time() - step_started
            elapsed_step = time.time() - cycle_started
            update_accumulator(
                accumulator,
                int(early_image.shape[0]),
                loss,
                live,
                list(batch["case_id"]),
                elapsed_step,
                data_wait_seconds,
                gpu_step_seconds,
            )
            running = accumulator_metrics(accumulator)
            completed_steps = max(1, step)
            eta = (
                accumulator["elapsed_seconds"]
                / completed_steps
                * (args.steps_per_epoch - step)
                / 60.0
            )
            data_wait_fraction = float(
                accumulator.get("data_wait_seconds", 0.0)
                / max(accumulator.get("elapsed_seconds", 0.0), 1e-6)
            )
            progress = {
                "status": "RUNNING",
                "phase": "training",
                "epoch": epoch,
                "epochs": args.epochs,
                "step": step,
                "steps": args.steps_per_epoch,
                "pair_batch_size": batch_size,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "loss": running["train_loss"],
                "early_soft": [running[f"train_early_soft_{region}"] for region in REGIONS],
                "recurrent_soft": [running[f"train_recurrent_soft_{region}"] for region in REGIONS],
                "distinct_cases": running["train_distinct_cases"],
                "grad_norm": grad_norm,
                "data_wait_fraction": data_wait_fraction,
                "gpu_step_seconds": gpu_step_seconds,
                "eta_minutes": eta,
            }
            if step == 1 or step % args.log_every == 0 or step == args.steps_per_epoch:
                atomic_json(progress_path, progress)
                print(
                    f"TRAIN epoch={epoch:03d} step={step:03d}/{args.steps_per_epoch} | "
                    f"loss={running['train_loss']:.4f} | "
                    f"E-WT={progress['early_soft'][0]:.3f} | "
                    f"R-WT={progress['recurrent_soft'][0]:.3f} | "
                    f"wait={100*data_wait_fraction:.0f}% | ETA={eta:.1f}m",
                    flush=True,
                )
            if step % args.checkpoint_every == 0 or STOP_REQUESTED or STOP_FILE.exists():
                atomic_torch_save(
                    checkpoint_payload(
                        early_model,
                        recurrent_model,
                        optimizer,
                        scheduler,
                        epoch,
                        step,
                        batch_size,
                        accumulator,
                        best_score,
                        best_epoch,
                        bad_epochs,
                        args,
                        amp_enabled,
                    ),
                    latest_path,
                )
            if STOP_REQUESTED or STOP_FILE.exists():
                atomic_json(
                    progress_path,
                    {
                        **progress,
                        "status": "STOPPED",
                        "phase": "checkpointed",
                        "message": "Safe checkpoint written; relaunch resumes from this step",
                    },
                )
                raise GracefulStop("Safe stop requested")
            del (
                early_image,
                recurrent_image,
                early_target,
                recurrent_target,
                valid,
                recurrent_logits,
                early_loss,
                recurrent_loss,
                loss,
                live,
            )
        training = accumulator_metrics(accumulator)
        validation_summary = validate(
            early_model,
            recurrent_model,
            validation,
            validation_store,
            device,
            epoch,
            output,
            sw_batch_size=args.sw_batch_size,
        )
        selection_score = float(validation_summary["selection_score"])
        improved = selection_score > best_score + 1e-5
        if improved:
            best_score, best_epoch, bad_epochs = selection_score, epoch, 0
            atomic_torch_save(
                {
                    "version": RUNTIME_VERSION,
                    "cv_fold": CV_FOLD,
                    "early_model": early_model.state_dict(),
                    "recurrent_model": recurrent_model.state_dict(),
                    "epoch": epoch,
                    "best_score": best_score,
                    "validation": validation_summary,
                    "pair_batch_size": batch_size,
                    "threshold": 0.5,
                    "initialization": initialization,
                    "precision": "BF16 mixed precision" if amp_enabled else "FP32",
                    "gpu_count": 1,
                },
                best_path,
            )
        else:
            bad_epochs += 1
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **training,
            **{f"val_{key}": value for key, value in validation_summary.items()},
            "selection_score": selection_score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "epoch_minutes": float((time.time() - epoch_started) / 60.0),
        }
        if history_path.is_file():
            history = pd.read_csv(history_path)
            history = history[history.epoch.astype(int) != epoch]
            history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        else:
            history = pd.DataFrame([row])
        history = history.sort_values("epoch").reset_index(drop=True)
        atomic_csv(history, history_path)
        plot_history(history, output / "history.png")
        next_epoch = epoch + 1
        atomic_torch_save(
            checkpoint_payload(
                early_model,
                recurrent_model,
                optimizer,
                scheduler,
                next_epoch,
                0,
                batch_size,
                empty_accumulator(),
                best_score,
                best_epoch,
                bad_epochs,
                args,
                amp_enabled,
            ),
            latest_path,
        )
        atomic_json(
            progress_path,
            {
                "status": "RUNNING",
                "phase": "epoch_complete",
                "epoch": epoch,
                "epochs": args.epochs,
                "loss": training["train_loss"],
                "early_soft": [training[f"train_early_soft_{region}"] for region in REGIONS],
                "recurrent_soft": [training[f"train_recurrent_soft_{region}"] for region in REGIONS],
                "validation_early": validation_summary["early_mean_dice_positive"],
                "validation_recurrent": validation_summary["recurrent_mean_dice_positive"],
                "validation_segmentation": validation_summary["segmentation_mean_dice_positive"],
                "validation_infiltration_dice": validation_summary["infiltration_dice_nonempty"],
                "validation_infiltration_auprc": validation_summary["infiltration_auprc_nonempty"],
                "validation_empty_accuracy": validation_summary["empty_accuracy"],
                "selection_score": selection_score,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "patience": args.patience,
                "epoch_minutes": row["epoch_minutes"],
            },
        )
        print(
            f"EPOCH {epoch} COMPLETE | loss={training['train_loss']:.4f} | "
            f"seg={validation_summary['segmentation_mean_dice_positive']:.4f} | "
            f"infiltration Dice/AUPRC={validation_summary['infiltration_dice_nonempty']:.4f}/"
            f"{validation_summary['infiltration_auprc_nonempty']:.4f} | "
            f"empty={validation_summary['empty_accuracy']:.4f} | best={best_score:.4f}@{best_epoch}",
            flush=True,
        )
        start_step = 0
        if epoch >= args.minimum_epochs and bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best={best_score:.4f}@{best_epoch}", flush=True)
            break
    summary = {
        "status": "PASS",
        "cv_fold": CV_FOLD,
        "best_epoch": best_epoch,
        "best_epoch_role": "pretrained_baseline" if best_epoch == 0 else "finetuned",
        "best_score": best_score,
        "best_checkpoint": str(best_path.resolve()),
        "last_completed_epoch": int(pd.read_csv(history_path).epoch.max()),
        "pair_batch_size": batch_size,
    }
    atomic_json(output / "summary.json", summary)
    (output / "TRAINING_COMPLETE").touch()
    atomic_json(progress_path, {"status": "PASS", "phase": "complete", **summary})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EXPERIMENT_ROOT / "training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-epoch", type=int, default=480)
    parser.add_argument("--pair-batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--sw-batch-size", type=int, default=8)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--cache-pairs", type=int, default=4)
    parser.add_argument("--preload-workers", type=int, default=1)
    parser.add_argument("--batch-workers", type=int, default=8)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output / "run.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another training process already holds the run lock")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        run(args)
    except GracefulStop as error:
        print(str(error), flush=True)
        raise SystemExit(75)
    except Exception as error:
        atomic_json(
            args.output / "progress.json",
            {"status": "FAILED", "phase": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
