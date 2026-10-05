#!/usr/bin/env bash
set -euo pipefail

CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$CODE/../.." && pwd)
V5=${GLIOMA_PROJECT_ROOT:-$REPOSITORY_ROOT}
CONTROL=${GLIOMA_LABEL_CONTROL_ROOT:-$V5/label/control_mu_exact_v2}
TMP_ROOT=${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/mu_exact_v2
mkdir -p "$CONTROL" "$TMP_ROOT/tmp" "$TMP_ROOT/cache"

if [[ -f "$CONTROL/manager.pid" ]]; then
  pid=$(tr -dc '0-9' < "$CONTROL/manager.pid")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "MU exact generation is already running as PID $pid"
    exec watch -n 20 python -u "$CODE/monitor_mu_exact_v2.py" --once
  fi
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$OMP_NUM_THREADS"
export TMPDIR="$TMP_ROOT/tmp"
export XDG_CACHE_HOME="$TMP_ROOT/cache"
export MPLCONFIGDIR="$TMP_ROOT/cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"
rm -f "$CONTROL/STOP_REQUESTED"

nohup setsid python -u "$CODE/manager_mu_exact_v2.py" \
  >> "$CONTROL/manager.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$CONTROL/manager.pid"
disown "$pid" 2>/dev/null || true
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: manager exited immediately." >&2
  tail -n 100 "$CONTROL/manager.log" >&2
  exit 1
fi
echo "Started detached exact MU label/cache generation as PID $pid"
echo "Output: $V5/label/MU-Glioma-Post"
echo "Monitor: watch -n 20 python -u $CODE/monitor_mu_exact_v2.py --once"
