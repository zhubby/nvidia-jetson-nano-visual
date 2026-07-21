#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
SERVICE_NAME="jetson-yolo-web.service"

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    libopenblas-base \
    python3-pip \
    python3-venv \
    python3-opencv \
    v4l-utils
fi

python3 -m venv --system-site-packages "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/backend/requirements.txt"

NODE_MAJOR=""
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node --version | sed 's/^v//' | cut -d. -f1)"
fi

if command -v npm >/dev/null 2>&1 && [ "${NODE_MAJOR:-0}" -ge 20 ]; then
  (
    cd "$ROOT_DIR/frontend"
    if [ -f package-lock.json ]; then
      npm ci
    else
      npm install
    fi
    npm run build
  )
elif [ -f "$ROOT_DIR/frontend/dist/index.html" ]; then
  echo "Using committed frontend/dist assets. Install Node.js >=20 and npm only if you need to rebuild on-device."
else
  echo "No usable npm/Node.js build toolchain and frontend/dist is missing." >&2
  echo "Build frontend on another machine, commit frontend/dist, or install Node.js >=20." >&2
fi

if [ "${INSTALL_SYSTEMD:-0}" = "1" ]; then
  sudo cp "$ROOT_DIR/deploy/systemd/$SERVICE_NAME" "/etc/systemd/system/$SERVICE_NAME"
  sudo systemctl daemon-reload
  echo "Installed $SERVICE_NAME. Run: sudo systemctl enable --now jetson-yolo-web"
fi

echo "Jetson setup complete."
