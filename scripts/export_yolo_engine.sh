#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-yolov8n.pt}"
OUTPUT_DIR="${2:-models}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$ROOT_DIR/$OUTPUT_DIR"

if command -v yolo >/dev/null 2>&1; then
  yolo export "model=$MODEL_PATH" format=engine half=True "imgsz=$IMAGE_SIZE" device=0
else
  if ! command -v docker >/dev/null 2>&1; then
    echo "Neither yolo CLI nor docker is available. Install Ultralytics in a compatible export environment." >&2
    exit 1
  fi
  docker run --rm --runtime nvidia --network host \
    -v "$ROOT_DIR:/workspace" \
    -w /workspace \
    ultralytics/ultralytics:latest-jetson-jetpack4 \
    yolo export "model=$MODEL_PATH" format=engine half=True "imgsz=$IMAGE_SIZE" device=0
fi

ENGINE_PATH="${MODEL_PATH%.*}.engine"
if [ -f "$ENGINE_PATH" ]; then
  cp "$ENGINE_PATH" "$ROOT_DIR/$OUTPUT_DIR/yolov8n_fp16.engine"
  echo "Engine copied to $OUTPUT_DIR/yolov8n_fp16.engine"
else
  echo "Export completed, but $ENGINE_PATH was not found. Check the exporter output path." >&2
  exit 1
fi
