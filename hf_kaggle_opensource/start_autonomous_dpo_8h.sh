#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
LOG_FILE="${ROOT}/autonomous_dpo.log"
PID_FILE="${ROOT}/autonomous_dpo.pid"
STATE_FILE="${ROOT}/autonomous_dpo_state.json"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Autonomous pipeline already running with PID ${OLD_PID}" >&2
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"

: > "${LOG_FILE}"
rm -f "${STATE_FILE}" "${PID_FILE}"

nohup "${PYTHON_BIN}" -u "${ROOT}/autonomous_dpo_pipeline.py" --hours 8 >> "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Started autonomous DPO pipeline"
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
echo "State: ${ROOT}/autonomous_dpo_state.json"
