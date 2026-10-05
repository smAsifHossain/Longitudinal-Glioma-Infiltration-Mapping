#!/usr/bin/env bash
set -euo pipefail

V5=${GLIOMA_PROJECT_ROOT:?Set GLIOMA_PROJECT_ROOT to the project workspace}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXPERIMENT="$V5/finetuning/experiments/SwinUNETR_MU_5fold_CV_v1"

if [[ ! -f "$SCRIPT_DIR/manager_v1.py" ]]; then
  echo "ERROR: package is not installed at $SCRIPT_DIR" >&2
  exit 2
fi
if [[ ! -f "$V5/label/MU-Glioma-Post/EXACT_CACHE_COMPLETE" ]]; then
  echo "ERROR: finalized MU registration/label/cache generation is incomplete." >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: expose two GPUs, for example CUDA_VISIBLE_DEVICES=0,1" >&2
  exit 2
fi
IFS=',' read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#visible_gpus[@]} -ne 2 ]]; then
  echo "ERROR: expose exactly two GPUs; found $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
available_cores=$(nproc)
if (( available_cores < 32 )); then
  echo "ERROR: 32 allocated CPU cores are required; found $available_cores." >&2
  exit 2
fi
cgroup_path=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)
memory_max=$(cat "/sys/fs/cgroup${cgroup_path}/memory.max" 2>/dev/null || echo max)
minimum_memory=$((400 * 1024 * 1024 * 1024))
if [[ "$memory_max" != "max" ]] && (( memory_max < minimum_memory )); then
  echo "ERROR: request 512 GB RAM (>=400 GiB usable); current limit is $memory_max bytes." >&2
  exit 2
fi

mkdir -p "$EXPERIMENT"
mkdir -p ${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/swinunetr_mu_5fold/root_cuda
mkdir -p ${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/swinunetr_mu_5fold/root_cache
mkdir -p ${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/swinunetr_mu_5fold/root_tmp
mkdir -p ${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/swinunetr_mu_5fold/root_mpl
export CUDA_CACHE_PATH=${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/swinunetr_mu_5fold/root_cuda
export XDG_CACHE_HOME=${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/swinunetr_mu_5fold/root_cache
export TMPDIR=${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/swinunetr_mu_5fold/root_tmp
export MPLCONFIGDIR=${GLIOMA_TMP_ROOT:-/tmp/glioma_infiltration}/swinunetr_mu_5fold/root_mpl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export MALLOC_ARENA_MAX=4
unset SWINUNETR_CV_FOLD || true

if [[ -f "$EXPERIMENT/manager.pid" ]]; then
  existing_pid=$(tr -dc '0-9' < "$EXPERIMENT/manager.pid")
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Five-fold scheduler already running as PID $existing_pid."
    echo "Monitor: watch -n 20 python -u $SCRIPT_DIR/monitor_v1.py --once"
    exit 0
  fi
fi

rm -f "$EXPERIMENT/STOP_REQUESTED"
nohup setsid python -u "$SCRIPT_DIR/manager_v1.py" \
  >> "$EXPERIMENT/manager.log" 2>&1 < /dev/null &
manager_pid=$!
echo "$manager_pid" > "$EXPERIMENT/manager.pid"
disown "$manager_pid" 2>/dev/null || true
sleep 1
if ! kill -0 "$manager_pid" 2>/dev/null; then
  echo "ERROR: scheduler exited immediately. Last log lines:" >&2
  tail -n 80 "$EXPERIMENT/manager.log" >&2
  exit 1
fi

echo "Started detached Swin-UNETR five-fold CV scheduler as PID $manager_pid"
echo "Execution: two independent folds concurrently, one per GPU; next fold starts automatically"
echo "Split: patient-grouped outer five-fold CV with inner validation"
echo "Resources: 32 CPU cores, 512 GB RAM, GPUs $CUDA_VISIBLE_DEVICES"
echo "Experiment: $EXPERIMENT"
echo "Rerun this command after disconnection/allocation loss to resume incomplete folds."
echo "Dashboard: watch -n 20 python -u $SCRIPT_DIR/monitor_v1.py --once"
echo "Detailed fold: python -u $SCRIPT_DIR/live_epoch_v1.py --fold 0"
