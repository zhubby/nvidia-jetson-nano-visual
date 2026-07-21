import os

import numpy as np

from jetson_yolo_web.app import create_app
from jetson_yolo_web.config import DEFAULT_CONFIG
from jetson_yolo_web.runtime import RuntimeService


def make_runtime():
    config = dict(DEFAULT_CONFIG, source="synthetic", detector_backend="mock")
    runtime = RuntimeService(config=config)
    runtime.buffer.publish(
        np.zeros((16, 16, 3), dtype=np.uint8),
        b"\xff\xd8\xff\xd9",
        [{"label": "person", "confidence": 0.91, "bbox": {"x1": 1, "y1": 1, "x2": 8, "y2": 8}, "frame_ts": 1.0}],
        1.0,
    )
    return runtime


def test_status_endpoint_returns_runtime_state():
    runtime = make_runtime()
    app = create_app(runtime=runtime, auto_start=False)
    app.config["TESTING"] = True

    response = app.test_client().get("/api/status")

    assert response.status_code == 200
    assert response.get_json()["health"] == "starting"


def test_latest_detections_endpoint_returns_last_frame_detections():
    runtime = make_runtime()
    app = create_app(runtime=runtime, auto_start=False)
    app.config["TESTING"] = True

    response = app.test_client().get("/api/detections/latest")

    assert response.status_code == 200
    assert response.get_json()["detections"][0]["label"] == "person"


def test_update_config_rejects_invalid_confidence():
    runtime = make_runtime()
    app = create_app(runtime=runtime, auto_start=False)
    app.config["TESTING"] = True

    response = app.test_client().put("/api/config", json={"confidence": -0.1})

    assert response.status_code == 400
    assert "confidence" in response.get_json()["error"]


def test_update_config_accepts_valid_confidence():
    runtime = make_runtime()
    app = create_app(runtime=runtime, auto_start=False)
    app.config["TESTING"] = True

    response = app.test_client().put("/api/config", json={"confidence": 0.72})

    assert response.status_code == 200
    assert response.get_json()["confidence"] == 0.72


def test_snapshot_writes_image_and_metadata(tmp_path, monkeypatch):
    runtime = make_runtime()
    app = create_app(runtime=runtime, auto_start=False)
    app.config["TESTING"] = True
    monkeypatch.setenv("JETSON_CAPTURE_DIR", str(tmp_path))

    response = app.test_client().post("/api/snapshot")

    assert response.status_code == 201
    payload = response.get_json()
    assert os.path.exists(payload["image_path"])
    assert os.path.exists(payload["metadata_path"])


def test_stream_endpoint_emits_multipart_jpeg_chunk():
    runtime = make_runtime()
    app = create_app(runtime=runtime, auto_start=False)
    app.config["TESTING"] = True

    response = app.test_client().get("/stream.mjpg", buffered=False)
    first = next(response.response)

    assert first == b"--frame\r\n"
