#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
V5_ROOT=${GLIOMA_PROJECT_ROOT:-$REPOSITORY_ROOT}
OUTPUT_ROOT=${BIOMARKER_OUTPUT_ROOT:-$V5_ROOT/biomarker_analysis/experiments/MU_5fold_OOF_volume_agreement_v1}

if [[ -f "$OUTPUT_ROOT/manager.pid" ]]; then
  manager_pid=$(tr -dc '0-9' < "$OUTPUT_ROOT/manager.pid")
  if [[ -n "$manager_pid" ]] && kill -0 "$manager_pid" 2>/dev/null; then
    echo "ERROR: manager PID $manager_pid is still running." >&2
    exit 1
  fi
fi

if [[ ! -f "$OUTPUT_ROOT/preflight/PASS" ]]; then
  echo "ERROR: verified preflight marker is missing: $OUTPUT_ROOT/preflight/PASS" >&2
  echo "Use prepare_fresh_start.sh and launch_v1.sh for a complete rerun." >&2
  exit 1
fi

archive="$OUTPUT_ROOT/archive_r5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$archive"
for path in \
  "$OUTPUT_ROOT/REPORT.md" \
  "$OUTPUT_ROOT/summary.json" \
  "$OUTPUT_ROOT/provenance.json" \
  "$OUTPUT_ROOT/progress.json" \
  "$OUTPUT_ROOT/manager_progress.json" \
  "$OUTPUT_ROOT/analysis.log"; do
  if [[ -f "$path" ]]; then
    mv "$path" "$archive/"
  fi
done
for directory in "$OUTPUT_ROOT/tables" "$OUTPUT_ROOT/figures"; do
  if [[ -d "$directory" ]]; then
    mv "$directory" "$archive/"
  fi
done
if [[ -f "$OUTPUT_ROOT/COMPLETE" ]]; then
  mv "$OUTPUT_ROOT/COMPLETE" "$archive/"
fi

echo "Archived the previous report, tables, and figures at: $archive"
echo "Verified preflight inputs were preserved. The r6 analysis is ready to run."
