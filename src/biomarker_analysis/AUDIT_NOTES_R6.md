# r6 all-model biomarker extension

## Analysis scope

This revision performs post-hoc analysis only. It does not retrain, select, or
modify any segmentation model. It reads the locked pooled out-of-fold case
metrics produced by the five completed patient-grouped cross-validation runs.

The preflight requires the same 353 scan pairs, 146 patients, fold registry,
ground-truth volumes, and target masks for every model before analysis starts.

## Table 8 outputs

For each of the five models, the analysis records reference and predicted mean
infiltration volume, mean prediction bias, a patient-cluster bootstrap 95%
confidence interval for bias, and the Bland-Altman 95% limits of agreement.
Bootstrap intervals for both limits are retained in the CSV output.

## Table 9 outputs

For each model and each reference-volume group (empty, <=1 mL, 1-10 mL, and
>10 mL), the analysis records scan-pair and patient counts, reference and
predicted mean volume, MAE, bias, and patient-cluster bootstrap 95% confidence
intervals for MAE and bias. Patient resampling keeps all longitudinal pairs from
a sampled patient together.

## Figures

The analysis creates three figures in both PNG and PDF formats:

1. `bias_limits_all_models`: mean bias and Bland-Altman limits for all models.
2. `volume_group_mae_all_models`: groupwise MAE with bootstrap intervals.
3. `volume_group_bias_all_models`: groupwise bias with bootstrap intervals.

Figures use concise scientific labels without methodological subtitles or
interpretive phrases embedded in the artwork. Detailed explanations are kept in
the report captions.
