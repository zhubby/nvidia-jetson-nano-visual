#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/models/yolov5n.onnx}"
MODEL_URL="${MODEL_URL:-https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

if command -v curl >/dev/null 2>&1; then
  curl -L --fail --retry 3 -o "$OUTPUT_PATH" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUTPUT_PATH" "$MODEL_URL"
else
  echo "curl or wget is required to download $MODEL_URL" >&2
  exit 1
fi

echo "Downloaded YOLO ONNX model to $OUTPUT_PATH"
