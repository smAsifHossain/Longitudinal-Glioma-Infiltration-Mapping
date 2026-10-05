# Five-model out-of-fold volume analysis

This package computes patient-level bootstrap confidence intervals, volume
agreement, Bland-Altman statistics, and aggregate figures from completed
five-fold out-of-fold predictions for all five models.

## Configuration

```bash
export GLIOMA_PROJECT_ROOT=/path/to/project_workspace
export BIOMARKER_OUTPUT_ROOT="$GLIOMA_PROJECT_ROOT/biomarker_analysis/experiments/MU_5fold_OOF_volume_agreement_v1"
```

The expected model experiment directories live under
`$GLIOMA_PROJECT_ROOT/finetuning/experiments`.

## Run

```bash
python self_test_v1.py
python preflight_v1.py
bash launch_v1.sh
python monitor_v1.py --once
```

The public `results/biomarker/` directory contains aggregate CSV tables and
figures only. Inputs containing case identifiers remain outside the repository.
