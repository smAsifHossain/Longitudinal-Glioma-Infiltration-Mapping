#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
V5_ROOT=${GLIOMA_PROJECT_ROOT:-$REPOSITORY_ROOT}
OUTPUT_ROOT=${BIOMARKER_OUTPUT_ROOT:-$V5_ROOT/biomarker_analysis/experiments/MU_5fold_OOF_volume_agreement_v1}
PYTHON_BIN=${PYTHON_BIN:-python}

mkdir -p "$OUTPUT_ROOT" ${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/biomarker_mpl
export V5_ROOT BIOMARKER_OUTPUT_ROOT="$OUTPUT_ROOT"
export MPLCONFIGDIR=${MPLCONFIGDIR:-${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/biomarker_mpl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

if [[ -f "$OUTPUT_ROOT/COMPLETE" ]]; then
  echo "Biomarker analysis is already complete: $OUTPUT_ROOT"
  "$PYTHON_BIN" -u "$SCRIPT_DIR/monitor_v1.py" --once
  exit 0
fi

if [[ -f "$OUTPUT_ROOT/manager.pid" ]]; then
  existing_pid=$(tr -dc '0-9' < "$OUTPUT_ROOT/manager.pid")
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Biomarker analysis is already running as PID $existing_pid"
    "$PYTHON_BIN" -u "$SCRIPT_DIR/monitor_v1.py" --once
    exit 0
  fi
fi

nohup "$PYTHON_BIN" -u "$SCRIPT_DIR/manager_v1.py" \
  >> "$OUTPUT_ROOT/manager.log" 2>&1 < /dev/null &
manager_pid=$!
echo "$manager_pid" > "$OUTPUT_ROOT/manager.pid"
sleep 2
if ! kill -0 "$manager_pid" 2>/dev/null && [[ ! -f "$OUTPUT_ROOT/COMPLETE" ]]; then
  echo "Manager failed immediately; recent log:" >&2
  tail -n 80 "$OUTPUT_ROOT/manager.log" >&2
  exit 1
fi

echo "Started MU five-fold OOF biomarker analysis as PID $manager_pid"
echo "Primary model: Swin-UNETR | Secondary comparison: all five completed models"
echo "CPU-only analysis; no GPU is required."
echo "Experiment: $OUTPUT_ROOT"
"$PYTHON_BIN" -u "$SCRIPT_DIR/monitor_v1.py" --once
