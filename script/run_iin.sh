#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
UNIGOAL_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "${UNIGOAL_ROOT}"

UNIGOAL_ENV="${UNIGOAL_ENV:-/home/hsy/miniconda3/envs/unigoal-habitat}"
NAV_GPU="${NAV_GPU:-0}"
EPISODE_ID="${EPISODE_ID:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
CONFIG_FILE="${CONFIG_FILE:-configs/config_local_qwen.yaml}"
EXPERIMENT_ID="${EXPERIMENT_ID:-iin_rgbd_local_qwen}"
QWEN_PORT="${QWEN_PORT:-18080}"
QWEN_STARTED_BY_RUN=0

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-${UNIGOAL_ENV}}"
export LD_LIBRARY_PATH="${UNIGOAL_ENV}/lib:${UNIGOAL_ENV}/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

if ! curl -fsS --max-time 2 "http://127.0.0.1:${QWEN_PORT}/health" >/dev/null 2>&1; then
  echo "[run_iin] local Qwen is not healthy; starting it first"
  QWEN_STARTED_BY_RUN=1
  RUN_IN_BACKGROUND=1 QWEN_PORT="${QWEN_PORT}" script/start_local_vlm.sh
fi

RUN_CONFIG="${CONFIG_FILE}"
if [[ -n "${NUM_EVAL_EPISODES:-}" || "${EPISODE_ID}" != "-1" || "${EXPERIMENT_ID}" != "local_qwen_smoke" ]]; then
  RUN_CONFIG="/tmp/unigoal_iin_${USER:-user}_$$.yaml"
  CONFIG_FILE="${CONFIG_FILE}" RUN_CONFIG="${RUN_CONFIG}" EPISODE_ID="${EPISODE_ID}" EXPERIMENT_ID="${EXPERIMENT_ID}" \
    NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-}" "${UNIGOAL_ENV}/bin/python" - <<'PY'
import os
import yaml

with open(os.environ["CONFIG_FILE"], "r") as f:
    cfg = yaml.safe_load(f)

cfg["experiment_id"] = os.environ["EXPERIMENT_ID"]
num_eval = os.environ.get("NUM_EVAL_EPISODES")
if num_eval:
    cfg["num_eval_episodes"] = int(num_eval)
elif os.environ["EPISODE_ID"] != "-1":
    cfg["num_eval_episodes"] = 1

with open(os.environ["RUN_CONFIG"], "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
fi

cleanup() {
  if [[ "${RUN_CONFIG}" == /tmp/unigoal_iin_* && -f "${RUN_CONFIG}" ]]; then
    rm -f "${RUN_CONFIG}"
  fi
  if [[ "${QWEN_STARTED_BY_RUN}" == "1" ]]; then
    echo "[run_iin] stopping local Qwen started by this run"
    QWEN_PORT="${QWEN_PORT}" script/stop_local_vlm.sh || true
  fi
}
trap cleanup EXIT

echo "[run_iin] config=${RUN_CONFIG} episode=${EPISODE_ID} gpu=${NAV_GPU}"
CUDA_VISIBLE_DEVICES="${NAV_GPU}" timeout "${TIMEOUT_SECONDS}s" \
  "${UNIGOAL_ENV}/bin/python" main.py \
    --config-file "${RUN_CONFIG}" \
    --goal_type ins-image \
    --episode_id "${EPISODE_ID}"
