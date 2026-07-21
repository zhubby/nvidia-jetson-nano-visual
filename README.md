# Jetson Nano USB YOLO Web

Jetson Nano project for USB-camera object detection with a YOLO TensorRT runtime and an NVIDIA-themed web dashboard. The backend streams an annotated MJPEG feed and REST status/config endpoints; the frontend shows live detections, system health, metrics, snapshots, and runtime controls.

## Project Layout

- `backend/` - Flask backend, camera worker, detector adapters, MJPEG stream, REST API, tests.
- `frontend/` - React/Vite/TypeScript dashboard using NVIDIA black/green visual styling.
- `models/` - TensorRT engine and model export notes. Engine binaries are ignored by git.
- `scripts/` - Jetson install, engine export, and local run helpers.
- `deploy/systemd/` - systemd unit for boot-time service recovery.
- `docs/` - architecture and Jetson deployment notes.

## Local Demo

Use the synthetic camera and mock detector when developing without a Jetson or USB camera.

```bash
cd backend
python3 -m pip install -r requirements-dev.txt
python3 -m pytest

cd ../frontend
npm install
npm test
npm run build

cd ../backend
JETSON_SOURCE=synthetic JETSON_DETECTOR=mock JETSON_FRONTEND_DIST=../frontend/dist python3 -m jetson_yolo_web.app
```

Open `http://127.0.0.1:8000`.

## Jetson Nano Runtime

Target baseline:

- Jetson Nano 4GB
- JetPack 4.6.x / L4T R32.x
- USB UVC camera at `/dev/video0`
- TensorRT FP16 YOLO engine generated on the Jetson or a matching L4T container

```bash
./scripts/install_jetson.sh
./scripts/export_yolo_engine.sh yolov8n.pt models
sudo INSTALL_SYSTEMD=1 ./scripts/install_jetson.sh
sudo systemctl enable --now jetson-yolo-web
```

The web UI listens on `0.0.0.0:8000` by default. From another device on the same LAN, open `http://<jetson-ip>:8000`.

## Backend API

- `GET /stream.mjpg` - annotated MJPEG stream.
- `GET /api/status` - camera/model/FPS/latency/temperature/memory status.
- `GET /api/detections/latest` - latest frame detections.
- `GET /api/config` - runtime config.
- `PUT /api/config` - update thresholds, source, model path, allowlist, and resolution.
- `POST /api/snapshot` - save current frame and detection JSON into `captures/`.

## Notes

Runtime does not depend on the latest `ultralytics` package. New YOLO models should be trained/exported outside the constrained JetPack Python environment, then deployed as `.engine` files under `models/`.
