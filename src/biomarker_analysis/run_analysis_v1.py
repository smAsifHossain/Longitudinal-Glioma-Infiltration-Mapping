#!/usr/bin/env python3
"""Run the compact five-model OOF infiltration-volume agreement analysis."""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biomarker_common_v1 import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    GROUP_LABELS,
    GROUP_ORDER,
    MODEL_SPECS,
    OUTPUT_ROOT,
    PRIMARY_MODEL,
    add_volumes,
    agreement_metrics,
    atomic_csv,
    atomic_json,
    infer_voxel_volume_ml,
    load_model_cases,
    model_case_path,
    patient_bootstrap,
    sha256_file,
    write_progress,
)


COLORS = {
    "Swin-UNETR": "#1f77b4",
    "nnU-Net ResEnc-L": "#ff7f0e",
    "MedNeXt-S": "#2ca02c",
    "SegResNet": "#d62728",
    "3D U-Net": "#9467bd",
}


def ci_lookup(intervals: pd.DataFrame, metric: str) -> tuple[float, float]:
    row = intervals.loc[intervals.metric == metric]
    if len(row) != 1:
        raise RuntimeError(f"Missing bootstrap interval for {metric}")
    return float(row.iloc[0].ci95_low), float(row.iloc[0].ci95_high)


def model_table(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    interval_frames = []
    for index, (model, frame) in enumerate(frames.items(), start=1):
        point = agreement_metrics(frame.reference_volume_ml, frame.predicted_volume_ml)
        intervals = patient_bootstrap(
            frame,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED + index,
        )
        rho_ci = ci_lookup(intervals, "spearman_rho")
        ccc_ci = ci_lookup(intervals, "lin_ccc")
        mae_ci = ci_lookup(intervals, "mae_ml")
        bias_ci = ci_lookup(intervals, "bias_ml")
        lower_loa_ci = ci_lookup(intervals, "bland_altman_lower_ml")
        upper_loa_ci = ci_lookup(intervals, "bland_altman_upper_ml")
        summaries.append(
            {
                "model": model,
                "analysis_role": "primary" if model == PRIMARY_MODEL else "secondary comparison",
                "pairs": len(frame),
                "patients": frame.patient_id.nunique(),
                **point,
                "spearman_ci95_low": rho_ci[0],
                "spearman_ci95_high": rho_ci[1],
                "lin_ccc_ci95_low": ccc_ci[0],
                "lin_ccc_ci95_high": ccc_ci[1],
                "mae_ci95_low_ml": mae_ci[0],
                "mae_ci95_high_ml": mae_ci[1],
                "bias_ci95_low_ml": bias_ci[0],
                "bias_ci95_high_ml": bias_ci[1],
                "bland_altman_lower_ci95_low_ml": lower_loa_ci[0],
                "bland_altman_lower_ci95_high_ml": lower_loa_ci[1],
                "bland_altman_upper_ci95_low_ml": upper_loa_ci[0],
                "bland_altman_upper_ci95_high_ml": upper_loa_ci[1],
            }
        )
        intervals.insert(0, "model", model)
        interval_frames.append(intervals)
        write_progress(
            "RUNNING",
            "bootstrap",
            f"Completed patient bootstrap for {model}",
            completed_models=index,
            total_models=len(frames),
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        )
    return pd.DataFrame(summaries), pd.concat(interval_frames, ignore_index=True)


def group_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, frame in frames.items():
        for group_name in GROUP_ORDER:
            selected = frame.loc[frame.volume_group == group_name]
            if selected.empty:
                continue
            error = selected.volume_error_ml.astype(float)
            rows.append(
                {
                    "model": model,
                    "volume_group": group_name,
                    "volume_group_label": GROUP_LABELS[group_name],
                    "pairs": len(selected),
                    "patients": selected.patient_id.nunique(),
                    "reference_mean_ml": selected.reference_volume_ml.mean(),
                    "reference_median_ml": selected.reference_volume_ml.median(),
                    "predicted_mean_ml": selected.predicted_volume_ml.mean(),
                    "predicted_median_ml": selected.predicted_volume_ml.median(),
                    "mae_ml": selected.absolute_error_ml.mean(),
                    "median_absolute_error_ml": selected.absolute_error_ml.median(),
                    "bias_ml": error.mean(),
                    "predicted_positive_pairs": int((selected.predicted_volume_ml > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def all_model_group_intervals(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_index, (model, frame) in enumerate(frames.items(), start=1):
        for group_index, group_name in enumerate(GROUP_ORDER, start=1):
            selected = frame.loc[frame.volume_group == group_name]
            intervals = patient_bootstrap(
                selected,
                repetitions=BOOTSTRAP_REPETITIONS,
                seed=BOOTSTRAP_SEED + 200 + model_index * 10 + group_index,
                metrics=["mae_ml", "bias_ml"],
            ).set_index("metric")
            mae = intervals.loc["mae_ml"]
            bias = intervals.loc["bias_ml"]
            rows.append(
                {
                    "model": model,
                    "volume_group": group_name,
                    "volume_group_label": GROUP_LABELS[group_name],
                    "pairs": len(selected),
                    "patients": selected.patient_id.nunique(),
                    "mae_ml": float(mae.estimate),
                    "mae_ci95_low_ml": float(mae.ci95_low),
                    "mae_ci95_high_ml": float(mae.ci95_high),
                    "bias_ml": float(bias.estimate),
                    "bias_ci95_low_ml": float(bias.ci95_low),
                    "bias_ci95_high_ml": float(bias.ci95_high),
                    "bootstrap_repetitions": int(mae.requested_bootstrap_repetitions),
                    "resampling_unit": "patient",
                }
            )
        write_progress(
            "RUNNING",
            "group_bootstrap",
            f"Completed volume-group bootstrap for {model}",
            completed_models=model_index,
            total_models=len(frames),
            bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        )
    return pd.DataFrame(rows)


def save_figures(
    models: pd.DataFrame,
    group_intervals: pd.DataFrame,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.png", "*.pdf"):
        for existing in output.glob(pattern):
            existing.unlink()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    def save_figure(fig: plt.Figure, stem: str) -> None:
        fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    model_order = list(MODEL_SPECS)
    ordered_models = models.set_index("model").reindex(model_order)
    y_positions = np.arange(len(model_order), dtype=float)
    biases = ordered_models.bias_ml.to_numpy(float)
    lower_limits = ordered_models.bland_altman_lower_ml.to_numpy(float)
    upper_limits = ordered_models.bland_altman_upper_ml.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for position, model, bias, lower, upper in zip(
        y_positions,
        model_order,
        biases,
        lower_limits,
        upper_limits,
    ):
        ax.errorbar(
            bias,
            position,
            xerr=np.array([[bias - lower], [upper - bias]]),
            fmt="o",
            markersize=6,
            color=COLORS[model],
            ecolor=COLORS[model],
            elinewidth=2.0,
            capsize=4,
            capthick=1.4,
        )
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_order)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted − reference volume (mL)")
    ax.grid(axis="x", alpha=0.16)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "bias_limits_all_models")

    plotted_groups = GROUP_ORDER
    positions = np.arange(len(plotted_groups), dtype=float)
    width = 0.15

    def grouped_error_figure(metric: str, ylabel: str, stem: str) -> None:
        low_column = f"{metric.replace('_ml', '')}_ci95_low_ml"
        high_column = f"{metric.replace('_ml', '')}_ci95_high_ml"
        fig, ax = plt.subplots(figsize=(9.6, 5.8))
        for model_index, model in enumerate(model_order):
            selected = (
                group_intervals.loc[group_intervals.model == model]
                .set_index("volume_group")
                .reindex(plotted_groups)
            )
            values = selected[metric].to_numpy(float)
            lower = selected[low_column].to_numpy(float)
            upper = selected[high_column].to_numpy(float)
            errors = np.vstack([values - lower, upper - values])
            offset = (model_index - (len(model_order) - 1) / 2.0) * width
            ax.bar(
                positions + offset,
                values,
                width=width * 0.90,
                label=model,
                color=COLORS[model],
                edgecolor="white",
                linewidth=0.5,
                yerr=errors,
                capsize=2.0,
                error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([GROUP_LABELS[group] for group in plotted_groups])
        ax.set_xlabel("Reference infiltration volume")
        ax.set_ylabel(ylabel)
        if metric == "mae_ml":
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(0, color="black", linewidth=0.9)
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
        ax.legend(
            frameon=False,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        save_figure(fig, stem)

    grouped_error_figure(
        "mae_ml",
        "Mean absolute error (mL)",
        "volume_group_mae_all_models",
    )
    grouped_error_figure(
        "bias_ml",
        "Mean prediction error (mL)",
        "volume_group_bias_all_models",
    )


def fmt_ci(value: float, low: float, high: float) -> str:
    return f"{value:.3f} ({low:.3f}–{high:.3f})"


def write_report(
    models: pd.DataFrame,
    groups: pd.DataFrame,
    group_intervals: pd.DataFrame,
    output: Path,
) -> None:
    primary = models.loc[models.model == PRIMARY_MODEL].iloc[0]
    lines = [
        "# MU five-fold OOF infiltration-volume analysis",
        "",
        "## Cohort and analysis",
        "",
        "The analysis uses 353 pooled out-of-fold longitudinal scan-pair predictions from 146 MU-Glioma-Post patients. "
        "Every pair was predicted by a model that was not trained on that patient's outer fold. Swin-UNETR is the primary model; the remaining models are secondary comparisons.",
        "",
        "The reference and predicted infiltration masks were converted to milliliters using the verified image voxel volume. "
        "Agreement was summarized with Spearman correlation, Lin's CCC, and MAE. Confidence intervals use 2,000 patient-cluster bootstrap samples, keeping all scan pairs from a sampled patient together.",
        "",
        "## Volume agreement across models",
        "",
        "| Model | Pairs | Patients | Spearman ρ (95% CI) | Lin CCC (95% CI) | MAE, mL (95% CI) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in models.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.pairs} | {row.patients} | "
            f"{fmt_ci(row.spearman_rho, row.spearman_ci95_low, row.spearman_ci95_high)} | "
            f"{fmt_ci(row.lin_ccc, row.lin_ccc_ci95_low, row.lin_ccc_ci95_high)} | "
            f"{fmt_ci(row.mae_ml, row.mae_ci95_low_ml, row.mae_ci95_high_ml)} |"
        )
    lines += [
        "",
        "## Prediction bias and Bland–Altman agreement",
        "",
        "| Model | Pairs | Patients | Reference mean (mL) | Predicted mean (mL) | Bias, mL (95% CI) | 95% limits of agreement (mL) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.pairs} | {row.patients} | "
            f"{row.reference_mean_ml:.3f} | {row.predicted_mean_ml:.3f} | "
            f"{fmt_ci(row.bias_ml, row.bias_ci95_low_ml, row.bias_ci95_high_ml)} | "
            f"{row.bland_altman_lower_ml:.3f} to {row.bland_altman_upper_ml:.3f} |"
        )
    lines += [
        "",
        "## Volume error by reference infiltration volume",
        "",
        "| Volume group | Model | Pairs | Patients | Reference mean (mL) | Predicted mean (mL) | MAE, mL (95% CI) | Bias, mL (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    merged_groups = groups.merge(
        group_intervals[
            [
                "model",
                "volume_group",
                "mae_ci95_low_ml",
                "mae_ci95_high_ml",
                "bias_ci95_low_ml",
                "bias_ci95_high_ml",
            ]
        ],
        on=["model", "volume_group"],
        how="left",
        validate="one_to_one",
    )
    for group_name in GROUP_ORDER:
        selected_group = merged_groups.loc[merged_groups.volume_group == group_name]
        for row in selected_group.itertuples(index=False):
            lines.append(
                f"| {row.volume_group_label} | {row.model} | {row.pairs} | {row.patients} | "
                f"{row.reference_mean_ml:.3f} | {row.predicted_mean_ml:.3f} | "
                f"{fmt_ci(row.mae_ml, row.mae_ci95_low_ml, row.mae_ci95_high_ml)} | "
                f"{fmt_ci(row.bias_ml, row.bias_ci95_low_ml, row.bias_ci95_high_ml)} |"
            )
    empty = groups.loc[
        (groups.model == PRIMARY_MODEL) & (groups.volume_group == "empty")
    ].iloc[0]
    lines += [
        "",
        "## Interpretation",
        "",
        f"For the primary Swin-UNETR analysis, volume agreement was high (Spearman ρ {primary.spearman_rho:.3f}; Lin CCC {primary.lin_ccc:.3f}) with MAE {primary.mae_ml:.3f} mL. "
        f"Among {empty.pairs} empty reference pairs, {empty.pairs - empty.predicted_positive_pairs} were correctly predicted as empty.",
        "",
        "Across models, overall volume agreement was strong, but performance was weaker for absent and small infiltration. "
        "These findings support measurement agreement within this MU cohort; they do not establish prognostic or clinical biomarker validity because survival, progression, treatment response, and other clinical outcomes were not analyzed.",
        "",
        "## Files",
        "",
        "- `tables/all_models_volume_agreement.csv`: model comparison with confidence intervals.",
        "- `tables/all_models_patient_bootstrap_95ci.csv`: long-form bootstrap results.",
        "- `tables/all_models_bland_altman.csv`: all-model bias and Bland–Altman results.",
        "- `tables/all_models_volume_group_errors.csv`: all-model volume-group errors and counts.",
        "- `tables/all_models_volume_group_error_95ci.csv`: patient-bootstrap intervals for group MAE and bias.",
        "- `tables/primary_swin_oof_volumes.csv`: primary-model case-level values.",
        "- `figures/`: three publication figures in PNG and PDF formats.",
        "",
        "## Figure captions",
        "",
        "**Figure 1. Prediction bias and limits of agreement across segmentation models.** Points indicate mean predicted-minus-reference volume, and horizontal intervals indicate the Bland–Altman 95% limits of agreement.",
        "",
        "**Figure 2. Mean absolute error by reference infiltration volume.** Grouped bars compare all five models. Error bars show patient-cluster bootstrap 95% confidence intervals.",
        "",
        "**Figure 3. Prediction bias by reference infiltration volume.** Grouped bars compare mean predicted-minus-reference volume across all five models. Error bars show patient-cluster bootstrap 95% confidence intervals.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


def run() -> None:
    (OUTPUT_ROOT / "COMPLETE").unlink(missing_ok=True)
    preflight_path = OUTPUT_ROOT / "preflight" / "summary.json"
    if not preflight_path.is_file():
        raise RuntimeError("Missing preflight summary")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "PASS":
        raise RuntimeError("Preflight is not PASS")
    tables = OUTPUT_ROOT / "tables"
    figures = OUTPUT_ROOT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    write_progress("RUNNING", "loading", "Loading verified five-model OOF predictions")
    raw_primary = load_model_cases(PRIMARY_MODEL)
    voxel_volume_ml = infer_voxel_volume_ml(raw_primary)
    frames = {
        model: add_volumes(load_model_cases(model), voxel_volume_ml)
        for model in MODEL_SPECS
    }
    primary = frames[PRIMARY_MODEL]
    atomic_csv(
        primary[
            [
                "case_id",
                "patient_id",
                "cv_fold",
                "reference_volume_ml",
                "predicted_volume_ml",
                "volume_error_ml",
                "absolute_error_ml",
                "volume_group",
            ]
        ],
        tables / "primary_swin_oof_volumes.csv",
    )
    models, intervals = model_table(frames)
    atomic_csv(models, tables / "all_models_volume_agreement.csv")
    atomic_csv(intervals, tables / "all_models_patient_bootstrap_95ci.csv")
    bland_altman_columns = [
        "model",
        "pairs",
        "patients",
        "reference_mean_ml",
        "predicted_mean_ml",
        "bias_ml",
        "bias_ci95_low_ml",
        "bias_ci95_high_ml",
        "bland_altman_lower_ml",
        "bland_altman_upper_ml",
        "bland_altman_lower_ci95_low_ml",
        "bland_altman_lower_ci95_high_ml",
        "bland_altman_upper_ci95_low_ml",
        "bland_altman_upper_ci95_high_ml",
    ]
    atomic_csv(models[bland_altman_columns], tables / "all_models_bland_altman.csv")
    groups = group_table(frames)
    atomic_csv(groups, tables / "all_models_volume_group_errors.csv")
    atomic_csv(
        groups.loc[groups.model == PRIMARY_MODEL].reset_index(drop=True),
        tables / "primary_swin_volume_group_errors.csv",
    )
    group_intervals = all_model_group_intervals(frames)
    atomic_csv(group_intervals, tables / "all_models_volume_group_error_95ci.csv")
    primary_point = models.loc[models.model == PRIMARY_MODEL].iloc[0]
    write_progress("RUNNING", "figures", "Generating three publication figures")
    save_figures(models, group_intervals, figures)
    write_report(models, groups, group_intervals, OUTPUT_ROOT)
    source_files = {
        model: {
            "path": str(model_case_path(model)),
            "sha256": sha256_file(model_case_path(model)),
        }
        for model in MODEL_SPECS
    }
    summary = {
        "status": "PASS",
        "analysis_name": "MU five-fold OOF infiltration-volume agreement",
        "analysis_role": "quantitative imaging-biomarker agreement; not clinical outcome validation",
        "primary_model": PRIMARY_MODEL,
        "models": list(MODEL_SPECS),
        "oof_pairs": len(primary),
        "oof_patients": primary.patient_id.nunique(),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_unit": "patient",
        "voxel_volume_ml": voxel_volume_ml,
        "positive_volume_groups": ["≤1 mL", "1–10 mL", ">10 mL"],
        "empty_analyzed_separately": True,
        "primary_results": primary_point.to_dict(),
        "source_oof_files": source_files,
        "report": str(OUTPUT_ROOT / "REPORT.md"),
        "figures": 3,
        "clinical_outcome_analysis_performed": False,
    }
    atomic_json(OUTPUT_ROOT / "summary.json", summary)
    atomic_json(
        OUTPUT_ROOT / "provenance.json",
        {
            **summary,
            "preflight_summary": str(preflight_path),
            "preflight_summary_sha256": sha256_file(preflight_path),
            "analysis_code_directory": str(Path(__file__).resolve().parent),
        },
    )
    (OUTPUT_ROOT / "COMPLETE").touch()
    write_progress(
        "PASS",
        "complete",
        "Biomarker agreement analysis complete",
        primary_spearman_rho=primary_point.spearman_rho,
        primary_lin_ccc=primary_point.lin_ccc,
        primary_mae_ml=primary_point.mae_ml,
        figures=3,
    )


def main() -> None:
    try:
        run()
    except Exception as error:
        write_progress("FAILED", "analysis_failed", f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
