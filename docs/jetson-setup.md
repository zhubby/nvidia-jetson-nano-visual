# Jetson Setup

## 1. System Packages

Run:

```bash
./scripts/install_jetson.sh
```

The installer uses apt for JetPack-provided hardware packages:

- `python3-opencv`
- `v4l-utils`
- `python3-venv`
- `python3-pip`
- `libopenblas-base`

The Python virtual environment is created with `--system-site-packages` so it can import JetPack's OpenCV, TensorRT, and CUDA packages.

This repository includes committed `frontend/dist` assets because JetPack 4.x images often ship old Node.js versions. On-device frontend rebuilds require Node.js 20 or newer; otherwise the installer uses the committed assets.

## 2. Camera Check

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

If the camera is not `/dev/video0`, set either:

```bash
export JETSON_CAMERA_INDEX=1
```

or:

```bash
export JETSON_CAMERA_DEVICE=/dev/video2
```

## 3. Model Engine

For this Jetson Nano class of device, start with OpenCV DNN YOLO if PyCUDA or a TensorRT engine is not ready:

```bash
./scripts/download_yolo_onnx.sh
export JETSON_DETECTOR=opencv
export JETSON_MODEL_PATH=models/yolov5n.onnx
```

For optimized TensorRT, generate an engine on the target Jetson or a matching JetPack 4 container:

```bash
./scripts/export_yolo_engine.sh yolov8n.pt models
```

The default TensorRT config expects:

```text
models/yolov8n_fp16.engine
```

If the engine export fails or FPS is too low, use the OpenCV ONNX fallback, a smaller YOLO variant, or lower image size. Keep `detector_backend=auto` during bring-up so the web app remains usable while the engine is missing.

## 4. Run Manually

```bash
./scripts/run_backend.sh
```

Then open:

```text
http://<jetson-ip>:8000
```

## 5. Install systemd Service

```bash
sudo INSTALL_SYSTEMD=1 ./scripts/install_jetson.sh
sudo systemctl enable --now jetson-yolo-web
sudo journalctl -u jetson-yolo-web -f
```

The service runs from the repository path by default. For production, place the repo at `/opt/jetson-yolo-web` or edit `deploy/systemd/jetson-yolo-web.service` before installing.
