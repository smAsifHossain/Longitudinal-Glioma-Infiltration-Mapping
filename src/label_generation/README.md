# Exact longitudinal infiltration target generation

This workflow registers the recurrent study to the earlier study, transfers
the recurrent whole-tumor target, and derives the connected recurrent tumor
component extending 2–20 mm beyond the earlier whole-tumor margin.

## Required configuration

```bash
export GLIOMA_PROJECT_ROOT=/path/to/project_workspace
export GLIOMA_DATA_ROOT=/path/to/MU-Glioma-Post
export GLIOMA_MANIFEST=/path/to/private_mu_manifest.csv
```

Optional overrides are `GLIOMA_LABEL_ROOT`, `GLIOMA_LABEL_CONTROL_ROOT`, and
`GLIOMA_TMP_ROOT`. Paths are resolved centrally in `path_config.py`.

## Run

```bash
python self_test_mu_exact_v2.py
python preflight_mu_exact_v2.py
bash launch_mu_exact_v2.sh
python monitor_mu_exact_v2.py --once
```

Use `stop_mu_exact_v2.sh` for a resumable stop. The real cohort manifest and
all patient data are deliberately excluded from this repository.
