#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

mkdir -p logs/remote-control

export PYTHONUNBUFFERED=1
export RC_HOST="${RC_HOST:-0.0.0.0}"
export RC_PORT="${RC_PORT:-7777}"

exec /usr/bin/python3 "${SCRIPT_DIR}/server.py"
