#!/usr/bin/env bash
set -euo pipefail

V5=${GLIOMA_PROJECT_ROOT:?Set GLIOMA_PROJECT_ROOT to the project workspace}
EXPERIMENT="$V5/finetuning/experiments/SwinUNETR_MU_5fold_CV_v1"
AUDITS="$V5/audits"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$AUDITS/43_swinunetr_mu_5fold_cv_v1_results_${STAMP}.tar.gz"
mkdir -p "$AUDITS"
if [[ ! -d "$EXPERIMENT" ]]; then
  echo "ERROR: missing experiment $EXPERIMENT" >&2
  exit 2
fi
tar -czf "$OUTPUT" \
  --exclude='*/checkpoints/*.pt' \
  --exclude='*/predictions/*' \
  --exclude='*/qc_3x2_all_test_cases/*.png' \
  -C "$V5/finetuning/experiments" \
  "SwinUNETR_MU_5fold_CV_v1"
echo "$OUTPUT"
