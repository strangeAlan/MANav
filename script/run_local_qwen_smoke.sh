#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
UNIGOAL_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "${UNIGOAL_ROOT}"

UNIGOAL_ENV="${UNIGOAL_ENV:-/home/hsy/miniconda3/envs/unigoal-habitat}"
NAV_GPU="${NAV_GPU:-0}"
EPISODE_ID="${EPISODE_ID:-0}"
GOAL_TYPE="${GOAL_TYPE:-ins-image}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-420}"
CONFIG_FILE="${CONFIG_FILE:-configs/config_local_qwen.yaml}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-${UNIGOAL_ENV}}"
export LD_LIBRARY_PATH="${UNIGOAL_ENV}/lib:${UNIGOAL_ENV}/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

if ! curl -fsS --max-time 2 "http://127.0.0.1:18080/health" >/dev/null 2>&1; then
  echo "[run_local_qwen_smoke] local Qwen is not healthy; starting it first"
  RUN_IN_BACKGROUND=1 script/start_local_vlm.sh
fi

echo "[run_local_qwen_smoke] config=${CONFIG_FILE} episode=${EPISODE_ID} gpu=${NAV_GPU}"
CUDA_VISIBLE_DEVICES="${NAV_GPU}" timeout "${TIMEOUT_SECONDS}s" \
  "${UNIGOAL_ENV}/bin/python" main.py \
    --config-file "${CONFIG_FILE}" \
    --goal_type "${GOAL_TYPE}" \
    --episode_id "${EPISODE_ID}"
