# Longitudinal Glioma Infiltration Mapping

Research code and aggregate results for mapping glioma infiltration beyond an
earlier whole-tumor margin in longitudinal MRI. The repository contains the
exact label-generation workflow, five segmentation pipelines, and the
out-of-fold volume-agreement analysis.

## Repository contents

- `src/label_generation/`: longitudinal registration and infiltration-target generation.
- `src/models/`: 3D U-Net, SegResNet, MedNeXt-S, Swin-UNETR, and nnU-Net ResEnc-L five-fold workflows.
- `src/biomarker_analysis/`: out-of-fold volume agreement and figure generation.
- `results/models/`: aggregate fold, pooled, bootstrap, and volume-bin metrics.
- `results/biomarker/`: aggregate volume-agreement tables and publication figures.
- `qc/`: instructions for adding de-identified QC examples.

The reported out-of-fold evaluation contains 353 longitudinal pairs from 146
patients. Only aggregate results are included. Raw MRI, derived NIfTI volumes,
case-level tables, manifests, trained weights, and identifiers are excluded.

## Setup

Python 3.10 or newer is recommended. Create an isolated environment, then run:

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

Set the paths for your HPC workspace and authorized dataset:

```bash
export GLIOMA_PROJECT_ROOT=/path/to/project_workspace
export GLIOMA_DATA_ROOT=/path/to/MU-Glioma-Post
export GLIOMA_MANIFEST=/path/to/private_mu_manifest.csv
export GLIOMA_PRETRAINING_ROOT=/path/to/pretraining_workspace
```

`GLIOMA_PROJECT_ROOT` is where generated labels, caches, experiments, and logs
are written. Keep patient data and private manifests outside the Git repository.

## Label generation

The private manifest must contain these columns:
`case_id`, `patient`, `earlier_timepoint`, `recurrent_timepoint`, and
`recurrent_target`. A synthetic schema example is provided in
`src/label_generation/manifests/manifest.example.csv`.

```bash
python src/label_generation/self_test_mu_exact_v2.py
python src/label_generation/preflight_mu_exact_v2.py
bash src/label_generation/launch_mu_exact_v2.sh
```

## Model workflows

Each model directory contains its own README and resumable launch, monitoring,
training, evaluation, and export utilities. After label generation and
model-specific preflight checks, launch from the repository root, for example:

```bash
bash src/models/swin_unetr/launch_v1.sh
```

The launch scripts require the allocated GPU/CPU resources documented in their
model README files.

## Aggregate analysis

```bash
bash src/biomarker_analysis/launch_v1.sh
```

The included `results/` directory is already populated with de-identified,
aggregate outputs from the completed five-fold evaluation.

## Data, privacy, and QC

This public export intentionally contains no source images, patient IDs,
case-level predictions, private manifests, or model checkpoints. Before adding
QC panels, verify that they are de-identified and permitted for public release;
see `qc/README.md`.

## License

No software license has been selected for this export. Add an appropriate
license before inviting external reuse or redistribution.
