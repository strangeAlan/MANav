#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
UNIGOAL_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "${UNIGOAL_ROOT}"

QWEN_PORT="${QWEN_PORT:-18080}"
QWEN_LOG_DIR="${QWEN_LOG_DIR:-${UNIGOAL_ROOT}/logs/qwen_backend_server}"
PID_FILE="${QWEN_LOG_DIR}/server-${QWEN_PORT}.pid"

pid=""
if [[ -f "${PID_FILE}" ]]; then
  pid="$(tr -dc '0-9' < "${PID_FILE}")"
fi

if [[ -z "${pid}" ]]; then
  pid="$(ss -ltnp 2>/dev/null | sed -n "s/.*127\\.0\\.0\\.1:${QWEN_PORT}.*pid=\\([0-9][0-9]*\\).*/\\1/p" | head -1)"
fi

if [[ -z "${pid}" ]]; then
  echo "[stop_local_vlm] no server found on port ${QWEN_PORT}"
  rm -f "${PID_FILE}"
  exit 0
fi

if ! kill -0 "${pid}" >/dev/null 2>&1; then
  echo "[stop_local_vlm] stale pid ${pid}"
  rm -f "${PID_FILE}"
  exit 0
fi

echo "[stop_local_vlm] stopping pid=${pid} port=${QWEN_PORT}"
kill "${pid}" >/dev/null 2>&1 || true

for _ in $(seq 1 30); do
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    rm -f "${PID_FILE}"
    echo "[stop_local_vlm] stopped"
    exit 0
  fi
  sleep 1
done

echo "[stop_local_vlm] pid ${pid} did not exit; sending SIGKILL"
kill -9 "${pid}" >/dev/null 2>&1 || true
rm -f "${PID_FILE}"
