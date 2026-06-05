#!/usr/bin/env bash
# Launch the OPSDGate teacher reward gateway (single backbone, multi-worker).
# Required env:
#   TEACHER_MODEL_PATH  Local path to the teacher checkpoint.
# Optional env (defaults shown):
#   NUM_GPUS=8 TP_SIZE=1 GATEWAY_PORT=8000 START_PORT=8001
#   GPU_UTIL=0.60 DTYPE=bfloat16 MAX_MODEL_LEN=8192 INFERENCE_BATCH_SIZE=6
#   REWARD_WORKER_LOGDIR=./reward_worker_logs
set -euo pipefail

: "${TEACHER_MODEL_PATH:?need TEACHER_MODEL_PATH}"
export NUM_GPUS="${NUM_GPUS:-8}"
export TP_SIZE="${TP_SIZE:-1}"
export GATEWAY_PORT="${GATEWAY_PORT:-8000}"
export START_PORT="${START_PORT:-8001}"
export GPU_UTIL="${GPU_UTIL:-0.60}"
export DTYPE="${DTYPE:-bfloat16}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-6}"
export REWARD_WORKER_LOGDIR="${REWARD_WORKER_LOGDIR:-./reward_worker_logs}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

exec python -m reward_service.server_gateway
