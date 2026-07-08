#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
UNIGOAL_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "${UNIGOAL_ROOT}"

QWEN_ENV="${QWEN_ENV:-/home/hsy/miniconda3/envs/multi-agent-nav}"
QWEN_SERVER_SCRIPT="${QWEN_SERVER_SCRIPT:-${UNIGOAL_ROOT}/script/qwen_backend_server.py}"
DEFAULT_QWEN_MODEL_PATH="${UNIGOAL_ROOT}/data/models/Qwen3-VL-8B-Instruct"
if [[ ! -e "${DEFAULT_QWEN_MODEL_PATH}" && -e "/home/hsy/model/qwen/Qwen3-VL-8B-Instruct" ]]; then
  DEFAULT_QWEN_MODEL_PATH="/home/hsy/model/qwen/Qwen3-VL-8B-Instruct"
fi
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${DEFAULT_QWEN_MODEL_PATH}}"
QWEN_GPU="${QWEN_GPU:-1}"
QWEN_PORT="${QWEN_PORT:-18080}"
QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-256}"
QWEN_LOG_DIR="${QWEN_LOG_DIR:-${UNIGOAL_ROOT}/logs/qwen_backend_server}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

if [[ ! -x "${QWEN_ENV}/bin/python" ]]; then
  echo "[start_local_vlm] missing python: ${QWEN_ENV}/bin/python" >&2
  exit 1
fi

if [[ ! -f "${QWEN_SERVER_SCRIPT}" ]]; then
  echo "[start_local_vlm] missing qwen server script: ${QWEN_SERVER_SCRIPT}" >&2
  exit 1
fi

if curl -fsS --max-time 2 "http://127.0.0.1:${QWEN_PORT}/health" >/dev/null 2>&1; then
  echo "[start_local_vlm] server already healthy: http://127.0.0.1:${QWEN_PORT}"
  exit 0
fi

mkdir -p "${QWEN_LOG_DIR}"
echo "[start_local_vlm] model=${QWEN_MODEL_PATH}"
echo "[start_local_vlm] gpu=${QWEN_GPU} port=${QWEN_PORT}"
echo "[start_local_vlm] server_script=${QWEN_SERVER_SCRIPT}"
echo "[start_local_vlm] mode=$([[ "${RUN_IN_BACKGROUND}" == "1" ]] && echo background || echo foreground)"
echo "[start_local_vlm] press Ctrl-C to stop in foreground mode"

cmd=(
  "${QWEN_ENV}/bin/python" "${QWEN_SERVER_SCRIPT}"
  --host 127.0.0.1
  --port "${QWEN_PORT}"
  --model-path "${QWEN_MODEL_PATH}"
  --dtype bfloat16
  --device-map auto
  --max-new-tokens "${QWEN_MAX_NEW_TOKENS}"
  --temperature 0.0
  --log-dir "${QWEN_LOG_DIR}"
)

if [[ "${RUN_IN_BACKGROUND}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${QWEN_GPU}" QWEN_MODEL_PATH="${QWEN_MODEL_PATH}" exec "${cmd[@]}"
fi

log_file="${QWEN_LOG_DIR}/server-${QWEN_PORT}.log"
pid_file="${QWEN_LOG_DIR}/server-${QWEN_PORT}.pid"
echo "[start_local_vlm] log=${log_file}"
CUDA_VISIBLE_DEVICES="${QWEN_GPU}" QWEN_MODEL_PATH="${QWEN_MODEL_PATH}" "${cmd[@]}" \
  > "${log_file}" 2>&1 &

pid="$!"
echo "${pid}" > "${pid_file}"
echo "[start_local_vlm] pid=${pid}"
echo "[start_local_vlm] waiting for health..."

for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${QWEN_PORT}/health" >/dev/null 2>&1; then
    echo "[start_local_vlm] ready: http://127.0.0.1:${QWEN_PORT}"
    exit 0
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[start_local_vlm] server exited early; log:" >&2
    tail -120 "${log_file}" || true
    exit 1
  fi
  sleep 5
done

echo "[start_local_vlm] timed out waiting for server; log:" >&2
tail -120 "${log_file}" || true
exit 1
