#!/usr/bin/env bash
set -euo pipefail

V5=${GLIOMA_PROJECT_ROOT:?Set GLIOMA_PROJECT_ROOT to the project workspace}
EXPERIMENT="$V5/finetuning/experiments/nnUNetResEncL_MU_5fold_CV_v1"
mkdir -p "$EXPERIMENT"
touch "$EXPERIMENT/STOP_REQUESTED"
if [[ -f "$EXPERIMENT/manager.pid" ]]; then
  pid=$(tr -dc '0-9' < "$EXPERIMENT/manager.pid")
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    echo "Safe stop requested from scheduler PID $pid. Active folds will checkpoint."
  else
    echo "Scheduler is not currently running; stop marker written."
  fi
else
  echo "Scheduler PID file is absent; stop marker written."
fi
echo "Monitor: $GLIOMA_PRETRAINING_ROOT/environments/nnunetv2_2_8_1/bin/python -u $V5/finetuning/scripts/nnunet_mu_5fold_cv_v1/monitor_v1.py --once"
