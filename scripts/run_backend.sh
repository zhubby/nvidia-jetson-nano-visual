#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/backend"
export JETSON_HOST="${JETSON_HOST:-0.0.0.0}"
export JETSON_PORT="${JETSON_PORT:-8000}"
export JETSON_CONFIG="${JETSON_CONFIG:-$ROOT_DIR/backend/config.example.json}"
export JETSON_FRONTEND_DIST="${JETSON_FRONTEND_DIST:-$ROOT_DIR/frontend/dist}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

exec "$PYTHON" -m jetson_yolo_web.app
