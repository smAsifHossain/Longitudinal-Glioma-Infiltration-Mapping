#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
V5_ROOT=${GLIOMA_PROJECT_ROOT:-$REPOSITORY_ROOT}
OUTPUT_ROOT=${BIOMARKER_OUTPUT_ROOT:-$V5_ROOT/biomarker_analysis/experiments/MU_5fold_OOF_volume_agreement_v1}
PYTHON_BIN=${PYTHON_BIN:-python}

export V5_ROOT BIOMARKER_OUTPUT_ROOT="$OUTPUT_ROOT"
export MPLCONFIGDIR=${MPLCONFIGDIR:-${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/biomarker_mpl}
mkdir -p "$MPLCONFIGDIR"

if [[ -f "$OUTPUT_ROOT/manager.pid" ]]; then
  manager_pid=$(tr -dc '0-9' < "$OUTPUT_ROOT/manager.pid")
  if [[ -n "$manager_pid" ]] && kill -0 "$manager_pid" 2>/dev/null; then
    echo "ERROR: manager PID $manager_pid is still running." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -u "$SCRIPT_DIR/prepare_figure_regeneration_r4.py"
bash "$SCRIPT_DIR/launch_v1.sh"
