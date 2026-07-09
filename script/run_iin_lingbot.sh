#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
UNIGOAL_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "${UNIGOAL_ROOT}"

LINGBOT_PORT="${LINGBOT_PORT:-18180}"
LINGBOT_STARTED_BY_RUN=0

cleanup() {
  if [[ "${LINGBOT_STARTED_BY_RUN}" == "1" ]]; then
    echo "[run_iin_lingbot] stopping LingBot depth server started by this run"
    LINGBOT_PORT="${LINGBOT_PORT}" script/stop_lingbot_depth_server.sh || true
  fi
}
trap cleanup EXIT

if ! curl -fsS --max-time 2 "http://127.0.0.1:${LINGBOT_PORT}/health" >/dev/null 2>&1; then
  echo "[run_iin_lingbot] LingBot depth server is not healthy; starting it first"
  LINGBOT_STARTED_BY_RUN=1
  RUN_IN_BACKGROUND=1 LINGBOT_PORT="${LINGBOT_PORT}" script/start_lingbot_depth_server.sh
fi

CONFIG_FILE="${CONFIG_FILE:-configs/config_iin_lingbot_local_qwen.yaml}" \
EXPERIMENT_ID="${EXPERIMENT_ID:-iin_rgb_only_lingbot_local_qwen}" \
./script/run_iin.sh
