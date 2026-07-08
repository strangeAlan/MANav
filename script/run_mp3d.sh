#!/usr/bin/env bash
set -euo pipefail

UNIGOAL_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
cd "${UNIGOAL_ROOT}"

UNIGOAL_ENV="${UNIGOAL_ENV:-/home/hsy/miniconda3/envs/unigoal-habitat}"
NAV_GPU="${NAV_GPU:-0}"
EPISODE_ID="${EPISODE_ID:--1}"
GOAL_TYPE="${GOAL_TYPE:-object}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-999999}"
CONFIG_FILE="${CONFIG_FILE:-configs/config_objectnav_mp3d_local_qwen.yaml}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-${UNIGOAL_ENV}}"
export LD_LIBRARY_PATH="${UNIGOAL_ENV}/lib:${UNIGOAL_ENV}/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

if ! curl -fsS --max-time 2 "http://127.0.0.1:18080/health" >/dev/null 2>&1; then
  echo "[run_mp3d] local Qwen is not healthy; starting it first"
  RUN_IN_BACKGROUND=1 script/start_local_vlm.sh
fi

RUN_CONFIG="${CONFIG_FILE}"
if [[ -n "${NUM_EVAL_EPISODES}" || "${EPISODE_ID}" != "-1" ]]; then
  RUN_CONFIG="/tmp/unigoal_objectnav_mp3d_${USER:-user}_$$.yaml"
  CONFIG_FILE="${CONFIG_FILE}" RUN_CONFIG="${RUN_CONFIG}" NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES}" EPISODE_ID="${EPISODE_ID}" \
    "${UNIGOAL_ENV}/bin/python" - <<'PY'
import os
import yaml

with open(os.environ["CONFIG_FILE"], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
num_eval = os.environ.get("NUM_EVAL_EPISODES")
if num_eval:
    cfg["num_eval_episodes"] = int(num_eval)
elif os.environ.get("EPISODE_ID") != "-1":
    cfg["num_eval_episodes"] = 1
with open(os.environ["RUN_CONFIG"], "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
fi

echo "[run_mp3d] ObjectNav-MP3D config=${RUN_CONFIG} episode=${EPISODE_ID} gpu=${NAV_GPU}"
CUDA_VISIBLE_DEVICES="${NAV_GPU}" timeout "${TIMEOUT_SECONDS}s" \
  "${UNIGOAL_ENV}/bin/python" main.py \
    --config-file "${RUN_CONFIG}" \
    --goal_type "${GOAL_TYPE}" \
    --episode_id "${EPISODE_ID}"
