#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import ants
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import distance_transform_edt, label
from skimage.exposure import match_histograms

from path_config import DATA_ROOT, OUTPUT_ROOT


DATASET_NAME = "MU-Glioma-Post"
EARLIER_REFERENCE_LABELS = (1, 2, 3)
RECURRENT_TARGET_LABELS = (1, 2, 3)
ALL_LESION_LABELS = (1, 2, 3, 4)
RUNTIME_VERSION = "2026-08-15-exact-v2"

EARLIER_COLOR = "#ff2b2b"
RECURRENT_COLOR = "#ffd400"
INFILTRATION_COLOR = "#0c099e"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--earlier-timepoint", required=True)
    parser.add_argument("--recurrent-timepoint", required=True)
    parser.add_argument("--modality", default="t1n", choices=("t1n",))
    parser.add_argument("--margin-mm", type=float, default=2.0)
    parser.add_argument("--local-max-mm", type=float, default=20.0)
    parser.add_argument("--registration-exclusion-mm", type=float, default=10.0)
    return parser.parse_args()


def tp_tag(value: str) -> str:
    number = int(str(value).split("_")[-1])
    return f"TP{number:02d}"


def canonical_stem(case_id: str, patient: str, earlier: str, recurrent: str) -> str:
    return (
        f"{case_id}"
        f"__{patient}"
        f"__{tp_tag(earlier)}-to-{tp_tag(recurrent)}"
        f"__T1N__Rigid__WT123__Local2to20mm"
    )


def ants_like(reference: ants.ANTsImage, array: np.ndarray) -> ants.ANTsImage:
    return ants.from_numpy(
        np.asarray(array, dtype=np.float32),
        origin=reference.origin,
        spacing=reference.spacing,
        direction=reference.direction,
    )


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_ants_atomic(image: ants.ANTsImage, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f"{path.name}.tmp.nii.gz"
    ants.image_write(image, str(temporary))
    os.replace(temporary, path)


def ncc(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(image_a) & np.isfinite(image_b)
    if int(valid.sum()) < 100:
        return float("nan")

    a = image_a[valid].astype(np.float64)
    b = image_b[valid].astype(np.float64)

    a -= a.mean()
    b -= b.mean()

    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator <= 0:
        return float("nan")

    return float(np.sum(a * b) / denominator)


def prepare_display_volumes(fixed_array: np.ndarray, warped_array: np.ndarray):
    fixed_array = np.asarray(fixed_array, dtype=np.float32)
    warped_array = np.asarray(warped_array, dtype=np.float32)

    fixed_foreground = fixed_array != 0
    warped_foreground = warped_array != 0

    warped_matched = np.zeros_like(warped_array, dtype=np.float32)

    if fixed_foreground.any() and warped_foreground.any():
        warped_matched[warped_foreground] = match_histograms(
            warped_array[warped_foreground],
            fixed_array[fixed_foreground],
            channel_axis=None,
        )
    else:
        warped_matched = warped_array.copy()

    values = np.concatenate(
        (
            fixed_array[fixed_foreground].ravel(),
            warped_matched[warped_foreground].ravel(),
        )
    )

    if values.size == 0:
        low, high = 0.0, 1.0
    else:
        low, high = np.percentile(values, [1.0, 99.0])
        if (not np.isfinite(low)) or (not np.isfinite(high)) or high <= low:
            low = float(values.min())
            high = float(values.max())
            if high <= low:
                high = low + 1.0

    def normalize(img):
        return np.clip((img - low) / (high - low), 0.0, 1.0)

    return normalize(fixed_array), normalize(warped_matched)


def add_contour(axis, mask_slice, color, linewidth=2.0):
    if not mask_slice.any():
        return
    axis.contour(
        mask_slice.T,
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
        origin="lower",
    )


def choose_qc_slice(earlier_reference, recurrent_target, infiltration):
    earlier_area = earlier_reference.sum(axis=(0, 1)).astype(np.float64)
    recurrent_area = recurrent_target.sum(axis=(0, 1)).astype(np.float64)
    infiltration_area = infiltration.sum(axis=(0, 1)).astype(np.float64)

    eligible = earlier_area > 0

    if not eligible.any():
        return int(np.argmax(recurrent_area))

    score = 10.0 * infiltration_area + recurrent_area + 0.1 * earlier_area
    score[~eligible] = -np.inf
    return int(np.argmax(score))


def main():
    args = parse_args()

    if args.local_max_mm <= args.margin_mm:
        raise ValueError("--local-max-mm must be greater than --margin-mm.")

    patient_dir = DATA_ROOT / args.patient
    earlier_dir = patient_dir / args.earlier_timepoint
    recurrent_dir = patient_dir / args.recurrent_timepoint

    modality_suffixes = {
        "t1": "t1n",
        "t1ce": "t1c",
        "t2": "t2w",
        "flair": "t2f",
    }
    earlier_modalities = {
        key: earlier_dir
        / f"{args.patient}_{args.earlier_timepoint}_brain_{suffix}.nii.gz"
        for key, suffix in modality_suffixes.items()
    }
    recurrent_modalities = {
        key: recurrent_dir
        / f"{args.patient}_{args.recurrent_timepoint}_brain_{suffix}.nii.gz"
        for key, suffix in modality_suffixes.items()
    }
    fixed_path = earlier_modalities["t1"]
    moving_path = recurrent_modalities["t1"]
    earlier_seg_path = earlier_dir / f"{args.patient}_{args.earlier_timepoint}_tumorMask.nii.gz"
    recurrent_seg_path = recurrent_dir / f"{args.patient}_{args.recurrent_timepoint}_tumorMask.nii.gz"

    for p in (
        *earlier_modalities.values(),
        *recurrent_modalities.values(),
        earlier_seg_path,
        recurrent_seg_path,
    ):
        if not p.is_file():
            raise FileNotFoundError(p)

    dirs = {
        "infiltration": OUTPUT_ROOT / "infiltration_labels_nifti",
        "excluded": OUTPUT_ROOT / "excluded_regions_nifti",
        "qc": OUTPUT_ROOT / "qc_images",
        "metrics": OUTPUT_ROOT / "metrics_json",
        "transforms": OUTPUT_ROOT / "registration_transforms" / args.case_id,
        "registered_4mod": OUTPUT_ROOT / "registered_recurrent_4mod" / args.case_id,
        "registered_seg": OUTPUT_ROOT / "registered_recurrent_segmentation" / args.case_id,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    stem = canonical_stem(
        args.case_id,
        args.patient,
        args.earlier_timepoint,
        args.recurrent_timepoint,
    )

    fixed = ants.image_read(str(fixed_path), pixeltype="float")
    moving = ants.image_read(str(moving_path), pixeltype="float")
    earlier_seg = ants.image_read(str(earlier_seg_path), pixeltype="float")
    recurrent_seg = ants.image_read(str(recurrent_seg_path), pixeltype="float")

    fixed_array = fixed.numpy().astype(np.float32, copy=False)
    moving_array = moving.numpy().astype(np.float32, copy=False)
    earlier_labels = np.rint(earlier_seg.numpy()).astype(np.int16)
    recurrent_labels = np.rint(recurrent_seg.numpy()).astype(np.int16)

    earlier_visible_tumor = np.isin(earlier_labels, EARLIER_REFERENCE_LABELS)
    earlier_cavity = earlier_labels == 4

    if earlier_visible_tumor.any():
        earlier_reference = earlier_visible_tumor
        earlier_reference_type = "visible_labels_1_2_3"
    elif earlier_cavity.any():
        earlier_reference = earlier_cavity
        earlier_reference_type = "resection_cavity_label_4"
    else:
        raise RuntimeError("Earlier segmentation contains neither labels 1-3 nor cavity label 4.")

    fixed_brain = np.isfinite(fixed_array) & (fixed_array != 0)
    moving_brain = np.isfinite(moving_array) & (moving_array != 0)

    fixed_exclusion = np.isin(earlier_labels, ALL_LESION_LABELS)
    moving_exclusion = np.isin(recurrent_labels, ALL_LESION_LABELS)

    spacing = tuple(float(v) for v in fixed.spacing)

    fixed_exclusion_dilated = distance_transform_edt(~fixed_exclusion, sampling=spacing) <= args.registration_exclusion_mm
    moving_exclusion_dilated = distance_transform_edt(~moving_exclusion, sampling=spacing) <= args.registration_exclusion_mm

    fixed_registration_mask_array = fixed_brain & ~fixed_exclusion_dilated
    moving_registration_mask_array = moving_brain & ~moving_exclusion_dilated

    fixed_registration_mask = ants_like(fixed, fixed_registration_mask_array)
    moving_registration_mask = ants_like(moving, moving_registration_mask_array)

    registration = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid",
        mask=fixed_registration_mask,
        moving_mask=moving_registration_mask,
        mask_all_stages=True,
        aff_metric="mattes",
        aff_sampling=32,
        aff_random_sampling_rate=0.2,
        random_seed=42,
        singleprecision=True,
        verbose=False,
    )

    forward_transforms = [Path(value) for value in registration.get("fwdtransforms", [])]
    if len(forward_transforms) != 1 or not forward_transforms[0].is_file():
        raise RuntimeError(
            f"Expected one saved rigid affine transform; found {forward_transforms}"
        )
    transform_path = dirs["transforms"] / f"{stem}__recurrent_to_earlier_0GenericAffine.mat"
    transform_temporary = transform_path.with_suffix(transform_path.suffix + ".tmp")
    shutil.copy2(forward_transforms[0], transform_temporary)
    os.replace(transform_temporary, transform_path)

    registered_modality_paths = {
        "t1": dirs["registered_4mod"] / "recurrent_t1_registered.nii.gz",
        "t1ce": dirs["registered_4mod"] / "recurrent_t1ce_registered.nii.gz",
        "t2": dirs["registered_4mod"] / "recurrent_t2_registered.nii.gz",
        "flair": dirs["registered_4mod"] / "recurrent_flair_registered.nii.gz",
    }
    registered_modalities: dict[str, ants.ANTsImage] = {}
    for key, native_path in recurrent_modalities.items():
        moving_modality = ants.image_read(str(native_path), pixeltype="float")
        warped_modality = ants.apply_transforms(
            fixed=fixed,
            moving=moving_modality,
            transformlist=[str(transform_path)],
            interpolator="linear",
            singleprecision=True,
        )
        write_ants_atomic(warped_modality, registered_modality_paths[key])
        registered_modalities[key] = warped_modality

    warped_recurrent = registered_modalities["t1"]

    warped_seg = ants.apply_transforms(
        fixed=fixed,
        moving=recurrent_seg,
        transformlist=[str(transform_path)],
        interpolator="genericLabel",
        singleprecision=True,
    )

    warped_array = warped_recurrent.numpy().astype(np.float32, copy=False)
    warped_labels = np.rint(warped_seg.numpy()).astype(np.int16)

    registered_seg_path = (
        dirs["registered_seg"] / "recurrent_seg_registered_genericLabel.nii.gz"
    )
    write_ants_atomic(warped_seg, registered_seg_path)

    recurrent_target = np.isin(warped_labels, RECURRENT_TARGET_LABELS)

    distance_from_earlier_mm = distance_transform_edt(~earlier_reference, sampling=spacing)

    recurrent_components, component_count = label(
        recurrent_target,
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )

    anchored_recurrent = np.zeros_like(recurrent_target, dtype=bool)
    retained_component_ids = []
    excluded_component_ids = []

    for component_id in range(1, component_count + 1):
        component = recurrent_components == component_id
        if not component.any():
            continue

        minimum_distance = float(distance_from_earlier_mm[component].min())

        if minimum_distance <= args.margin_mm + 1e-6:
            anchored_recurrent |= component
            retained_component_ids.append(component_id)
        else:
            excluded_component_ids.append(component_id)

    local_distance_band = (
        (distance_from_earlier_mm > args.margin_mm)
        & (distance_from_earlier_mm <= args.local_max_mm)
    )

    infiltration = anchored_recurrent & local_distance_band
    disconnected_recurrent = recurrent_target & ~anchored_recurrent
    far_connected_recurrent = anchored_recurrent & (distance_from_earlier_mm > args.local_max_mm)
    excluded_region = disconnected_recurrent | far_connected_recurrent

    infiltration_img = ants_like(fixed, infiltration)
    excluded_img = ants_like(fixed, excluded_region)

    infiltration_path = dirs["infiltration"] / f"{stem}__InfiltrationLabel.nii.gz"
    excluded_path = dirs["excluded"] / f"{stem}__ExcludedRegions.nii.gz"
    metrics_path = dirs["metrics"] / f"{stem}__metrics.json"
    qc_path = dirs["qc"] / f"{stem}__QC.png"

    write_ants_atomic(infiltration_img, infiltration_path)
    write_ants_atomic(excluded_img, excluded_path)

    qc_slice = choose_qc_slice(earlier_reference, recurrent_target, infiltration)

    fixed_display, recurrent_display = prepare_display_volumes(fixed_array, warped_array)
    overlay_display = 0.5 * fixed_display + 0.5 * recurrent_display

    earlier_slice = earlier_reference[:, :, qc_slice]
    recurrent_slice = recurrent_target[:, :, qc_slice]
    infiltration_slice = infiltration[:, :, qc_slice]

    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "savefig.facecolor": "black",
        "text.color": "white",
        "font.family": "sans-serif",
    })

    fig, axes = plt.subplots(1, 4, figsize=(22, 6.7), facecolor="black")

    axes[0].imshow(fixed_display[:, :, qc_slice].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    add_contour(axes[0], earlier_slice, EARLIER_COLOR)
    axes[0].set_title("Earlier Scan", fontsize=18, fontweight="bold", color="white", pad=14)

    axes[1].imshow(recurrent_display[:, :, qc_slice].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    add_contour(axes[1], recurrent_slice, RECURRENT_COLOR)
    axes[1].set_title("Recurrent Scan", fontsize=18, fontweight="bold", color="white", pad=14)

    axes[2].imshow(overlay_display[:, :, qc_slice].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    add_contour(axes[2], earlier_slice, EARLIER_COLOR)
    add_contour(axes[2], recurrent_slice, RECURRENT_COLOR)
    axes[2].set_title("Longitudinal Overlay", fontsize=18, fontweight="bold", color="white", pad=14)

    axes[3].imshow(recurrent_display[:, :, qc_slice].T, cmap="gray", origin="lower", vmin=0, vmax=1)
    if infiltration_slice.any():
        masked = np.ma.masked_where(~infiltration_slice.T, infiltration_slice.T.astype(np.uint8))
        axes[3].imshow(
            masked,
            cmap=ListedColormap([INFILTRATION_COLOR]),
            origin="lower",
            alpha=0.72,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )
        add_contour(axes[3], infiltration_slice, INFILTRATION_COLOR, linewidth=2.2)
    axes[3].set_title("Infiltration Zone", fontsize=18, fontweight="bold", color="white", pad=14)

    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("black")

    fig.suptitle(
        f"Dataset: {DATASET_NAME}   |   Patient: {args.patient}",
        fontsize=17,
        fontweight="bold",
        color="white",
        y=0.97,
    )

    fig.text(
        0.5,
        0.02,
        "Red = Earlier whole tumor   |   Yellow = Recurrent whole tumor   |   Blue = Infiltration Label",
        ha="center",
        va="center",
        color="white",
        fontsize=11,
        fontweight="bold",
    )

    fig.tight_layout(rect=(0.01, 0.05, 0.99, 0.93), w_pad=1.4)

    fig.savefig(qc_path, dpi=220, bbox_inches="tight", facecolor="black", edgecolor="none")
    plt.close(fig)

    before_metric_mask = fixed_registration_mask_array & moving_registration_mask_array
    after_metric_mask = fixed_registration_mask_array & (warped_array != 0)

    ncc_before = ncc(fixed_array, moving_array, before_metric_mask)
    ncc_after = ncc(fixed_array, warped_array, after_metric_mask)

    voxel_volume_ml = float(np.prod(spacing) / 1000.0)

    metrics = {
        "dataset": DATASET_NAME,
        "runtime_version": RUNTIME_VERSION,
        "case_id": args.case_id,
        "patient": args.patient,
        "earlier_timepoint": args.earlier_timepoint,
        "recurrent_timepoint": args.recurrent_timepoint,
        "modality": args.modality,
        "method_name": "T1N_Rigid_WT123_Local2to20mm",
        "registration": "Rigid",
        "registration_transform_path": str(transform_path),
        "registration_transform_sha256": sha256(transform_path),
        "registration_interpolator_mri": "linear",
        "registration_interpolator_segmentation": "genericLabel",
        "earlier_reference_labels": [1, 2, 3],
        "recurrent_target_labels": [1, 2, 3],
        "recurrent_target_definition": "whole_tumor_labels_1_2_3",
        "earlier_reference_type": earlier_reference_type,
        "registration_exclusion_mm": args.registration_exclusion_mm,
        "infiltration_margin_mm": args.margin_mm,
        "local_maximum_distance_mm": args.local_max_mm,
        "ncc_before": ncc_before,
        "ncc_after": ncc_after,
        "ncc_gain": ncc_after - ncc_before if np.isfinite(ncc_before) and np.isfinite(ncc_after) else float("nan"),
        "earlier_reference_volume_ml": float(earlier_reference.sum() * voxel_volume_ml),
        "earlier_visible_tumor_volume_ml": float(earlier_visible_tumor.sum() * voxel_volume_ml),
        "registered_recurrent_tumor_volume_ml": float(recurrent_target.sum() * voxel_volume_ml),
        "infiltration_volume_ml": float(infiltration.sum() * voxel_volume_ml),
        "excluded_new_or_distant_volume_ml": float(excluded_region.sum() * voxel_volume_ml),
        "disconnected_recurrent_volume_ml": float(disconnected_recurrent.sum() * voxel_volume_ml),
        "far_connected_beyond20mm_volume_ml": float(far_connected_recurrent.sum() * voxel_volume_ml),
        "recurrent_component_count": int(component_count),
        "retained_local_component_ids": retained_component_ids,
        "excluded_new_or_distant_component_ids": excluded_component_ids,
        "qc_axial_slice": int(qc_slice),
        "qc_image": str(qc_path),
        "infiltration_label_path": str(infiltration_path),
        "excluded_region_path": str(excluded_path),
        "earlier_t1": str(earlier_modalities["t1"]),
        "earlier_t1ce": str(earlier_modalities["t1ce"]),
        "earlier_t2": str(earlier_modalities["t2"]),
        "earlier_flair": str(earlier_modalities["flair"]),
        "earlier_seg": str(earlier_seg_path),
        "recurrent_t1_native": str(recurrent_modalities["t1"]),
        "recurrent_t1ce_native": str(recurrent_modalities["t1ce"]),
        "recurrent_t2_native": str(recurrent_modalities["t2"]),
        "recurrent_flair_native": str(recurrent_modalities["flair"]),
        "recurrent_seg_native": str(recurrent_seg_path),
        "recurrent_t1_registered": str(registered_modality_paths["t1"]),
        "recurrent_t1ce_registered": str(registered_modality_paths["t1ce"]),
        "recurrent_t2_registered": str(registered_modality_paths["t2"]),
        "recurrent_flair_registered": str(registered_modality_paths["flair"]),
        "recurrent_seg_registered": str(registered_seg_path),
        "exact_registration_cache_complete": True,
    }

    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"Done: {stem}")
    print(f"NCC before: {ncc_before:.4f}")
    print(f"NCC after:  {ncc_after:.4f}")
    print(f"Infiltration volume: {metrics['infiltration_volume_ml']:.3f} mL")
    print(f"QC image: {qc_path}")
    print(f"Label: {infiltration_path}")


if __name__ == "__main__":
    main()
