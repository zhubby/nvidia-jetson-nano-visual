#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$ROOT_DIR/frontend/dist" ]; then
  (cd "$ROOT_DIR/frontend" && npm install && npm run build)
fi

export JETSON_SOURCE=synthetic
export JETSON_DETECTOR=mock
export JETSON_FRONTEND_DIST="$ROOT_DIR/frontend/dist"

exec "$ROOT_DIR/scripts/run_backend.sh"
