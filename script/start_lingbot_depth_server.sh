#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
UNIGOAL_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "${UNIGOAL_ROOT}"

LINGBOT_ENV="${LINGBOT_ENV:-/home/hsy/miniconda3/envs/lingbot-map}"
LINGBOT_GPU="${LINGBOT_GPU:-0}"
LINGBOT_PORT="${LINGBOT_PORT:-18180}"
LINGBOT_MODEL_PATH="${LINGBOT_MODEL_PATH:-/home/hsy/model/lingbot-map/lingbot-map.pt}"
LINGBOT_LOG_DIR="${LINGBOT_LOG_DIR:-${UNIGOAL_ROOT}/logs/lingbot_depth_server}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-0}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy

if curl -fsS --max-time 2 "http://127.0.0.1:${LINGBOT_PORT}/health" >/dev/null 2>&1; then
  echo "[start_lingbot_depth_server] already healthy: http://127.0.0.1:${LINGBOT_PORT}"
  exit 0
fi

mkdir -p "${LINGBOT_LOG_DIR}"
cmd=(
  "${LINGBOT_ENV}/bin/python" script/lingbot_depth_server.py
  --host 127.0.0.1
  --port "${LINGBOT_PORT}"
  --model-path "${LINGBOT_MODEL_PATH}"
  --device cuda
  --use-sdpa
)

if [[ "${RUN_IN_BACKGROUND}" != "1" ]]; then
  CUDA_VISIBLE_DEVICES="${LINGBOT_GPU}" exec "${cmd[@]}"
fi

log_file="${LINGBOT_LOG_DIR}/server-${LINGBOT_PORT}.log"
pid_file="${LINGBOT_LOG_DIR}/server-${LINGBOT_PORT}.pid"
setsid env CUDA_VISIBLE_DEVICES="${LINGBOT_GPU}" "${cmd[@]}" > "${log_file}" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${pid_file}"
echo "[start_lingbot_depth_server] pid=${pid} log=${log_file}"

for _ in $(seq 1 120); do
  if curl -fsS --max-time 2 "http://127.0.0.1:${LINGBOT_PORT}/health" >/dev/null 2>&1; then
    echo "[start_lingbot_depth_server] ready"
    exit 0
  fi
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "[start_lingbot_depth_server] server exited early; log:" >&2
    tail -120 "${log_file}" || true
    exit 1
  fi
  sleep 5
done

echo "[start_lingbot_depth_server] timed out; log:" >&2
tail -120 "${log_file}" || true
exit 1
