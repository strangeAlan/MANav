#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
UNIGOAL_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "${UNIGOAL_ROOT}"

LINGBOT_PORT="${LINGBOT_PORT:-18180}"
LINGBOT_LOG_DIR="${LINGBOT_LOG_DIR:-${UNIGOAL_ROOT}/logs/lingbot_depth_server}"
PID_FILE="${LINGBOT_LOG_DIR}/server-${LINGBOT_PORT}.pid"

pid=""
if [[ -f "${PID_FILE}" ]]; then
  pid="$(tr -dc '0-9' < "${PID_FILE}")"
fi
if [[ -z "${pid}" ]]; then
  pid="$(ss -ltnp 2>/dev/null | sed -n "s/.*127\\.0\\.0\\.1:${LINGBOT_PORT}.*pid=\\([0-9][0-9]*\\).*/\\1/p" | head -1)"
fi
if [[ -z "${pid}" ]]; then
  echo "[stop_lingbot_depth_server] no server found on port ${LINGBOT_PORT}"
  rm -f "${PID_FILE}"
  exit 0
fi
echo "[stop_lingbot_depth_server] stopping pid=${pid} port=${LINGBOT_PORT}"
kill "${pid}" >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    rm -f "${PID_FILE}"
    echo "[stop_lingbot_depth_server] stopped"
    exit 0
  fi
  sleep 1
done
kill -9 "${pid}" >/dev/null 2>&1 || true
rm -f "${PID_FILE}"
