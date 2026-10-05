#!/usr/bin/env bash
set -euo pipefail
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "$CODE/../.." && pwd)
V5=${GLIOMA_PROJECT_ROOT:-$REPOSITORY_ROOT}
CONTROL=${GLIOMA_LABEL_CONTROL_ROOT:-$V5/label/control_mu_exact_v2}
mkdir -p "$CONTROL"
touch "$CONTROL/STOP_REQUESTED"
if [[ -f "$CONTROL/manager.pid" ]]; then
  pid=$(tr -dc '0-9' < "$CONTROL/manager.pid")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    echo "Stop requested from MU exact manager PID $pid. Completed cases remain resumable."
    exit 0
  fi
fi
echo "No live MU exact manager was found."
