import json
import os
import time

import numpy as np

from jetson_yolo_web.detector import BaseDetector
from jetson_yolo_web.config import DEFAULT_CONFIG
from jetson_yolo_web.runtime import RuntimeService


PERSON_DETECTION = {
    "label": "person",
    "confidence": 0.91,
    "bbox": {"x1": 1, "y1": 2, "x2": 20, "y2": 40},
    "frame_ts": 100.0,
}


def test_auto_snapshot_writes_person_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("JETSON_CAPTURE_DIR", str(tmp_path))
    runtime = RuntimeService(config=dict(DEFAULT_CONFIG, auto_snapshot_enabled=True))

    result = runtime._maybe_auto_snapshot(b"\xff\xd8\xff\xd9", [PERSON_DETECTION], 100.0, now=100.0)

    assert result is not None
    assert os.path.exists(result["image_path"])
    assert os.path.exists(result["metadata_path"])
    assert os.path.basename(result["image_path"]).startswith("auto-person-")
    metadata = json.loads(open(result["metadata_path"]).read())
    assert metadata["snapshot_type"] == "auto"
    assert metadata["detections"][0]["label"] == "person"
    status = runtime.status()
    assert status["auto_snapshot"]["count"] == 1
    assert status["auto_snapshot"]["last_image_path"] == result["image_path"]


def test_auto_snapshot_ignores_non_person(tmp_path, monkeypatch):
    monkeypatch.setenv("JETSON_CAPTURE_DIR", str(tmp_path))
    runtime = RuntimeService(config=dict(DEFAULT_CONFIG, auto_snapshot_enabled=True))

    result = runtime._maybe_auto_snapshot(
        b"\xff\xd8\xff\xd9",
        [dict(PERSON_DETECTION, label="laptop")],
        100.0,
        now=100.0,
    )

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_auto_snapshot_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("JETSON_CAPTURE_DIR", str(tmp_path))
    runtime = RuntimeService(config=dict(DEFAULT_CONFIG, auto_snapshot_enabled=False))

    result = runtime._maybe_auto_snapshot(b"\xff\xd8\xff\xd9", [PERSON_DETECTION], 100.0, now=100.0)

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_auto_snapshot_uses_configured_label_and_case_insensitive_matching(tmp_path, monkeypatch):
    monkeypatch.setenv("JETSON_CAPTURE_DIR", str(tmp_path))
    runtime = RuntimeService(
        config=dict(DEFAULT_CONFIG, auto_snapshot_enabled=True, auto_snapshot_label="Laptop")
    )

    result = runtime._maybe_auto_snapshot(
        b"\xff\xd8\xff\xd9",
        [dict(PERSON_DETECTION, label="laptop")],
        100.0,
        now=100.0,
    )

    assert result is not None
    assert os.path.basename(result["image_path"]).startswith("auto-laptop-")


def test_auto_snapshot_respects_cooldown(tmp_path, monkeypatch):
    monkeypatch.setenv("JETSON_CAPTURE_DIR", str(tmp_path))
    runtime = RuntimeService(
        config=dict(DEFAULT_CONFIG, auto_snapshot_enabled=True, auto_snapshot_cooldown_seconds=30.0)
    )

    first = runtime._maybe_auto_snapshot(b"\xff\xd8\xff\xd9", [PERSON_DETECTION], 100.0, now=100.0)
    second = runtime._maybe_auto_snapshot(b"\xff\xd8\xff\xd9", [PERSON_DETECTION], 105.0, now=105.0)

    assert first is not None
    assert second is None
    assert len(list(tmp_path.glob("*.jpg"))) == 1


def test_auto_snapshot_failure_respects_cooldown(monkeypatch):
    runtime = RuntimeService(
        config=dict(DEFAULT_CONFIG, auto_snapshot_enabled=True, auto_snapshot_cooldown_seconds=30.0)
    )
    calls = []

    def fail_write(*args):
        calls.append(args)
        raise OSError("disk unavailable")

    monkeypatch.setattr(runtime, "_write_snapshot", fail_write)

    first = runtime._maybe_auto_snapshot(b"\xff\xd8\xff\xd9", [PERSON_DETECTION], 100.0, now=100.0)
    second = runtime._maybe_auto_snapshot(b"\xff\xd8\xff\xd9", [PERSON_DETECTION], 105.0, now=105.0)

    assert first is None
    assert second is None
    assert len(calls) == 1
    assert runtime.status()["auto_snapshot"]["last_error"] == "disk unavailable"


def test_runtime_loop_triggers_auto_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("JETSON_CAPTURE_DIR", str(tmp_path))
    runtime = RuntimeService(
        config=dict(DEFAULT_CONFIG, auto_snapshot_enabled=True, auto_snapshot_cooldown_seconds=30.0),
        frame_source=FakeFrameSource(),
        detector=PersonDetector(),
    )

    runtime.start()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and not list(tmp_path.glob("auto-person-*.jpg")):
            time.sleep(0.02)
    finally:
        runtime.stop()

    captures = list(tmp_path.glob("auto-person-*.jpg"))
    assert len(captures) == 1
    assert runtime.status()["auto_snapshot"]["count"] == 1


def test_manual_snapshot_metadata_marks_manual(tmp_path):
    runtime = RuntimeService(config=dict(DEFAULT_CONFIG, auto_snapshot_enabled=True))

    result = runtime.save_snapshot(str(tmp_path))

    metadata = json.loads(open(result["metadata_path"]).read())
    assert metadata["snapshot_type"] == "manual"


class FakeFrameSource(object):
    def read(self):
        return True, np.zeros((32, 32, 3), dtype=np.uint8)

    def close(self):
        pass


class PersonDetector(BaseDetector):
    backend_name = "fake"

    def detect(self, frame, config):
        return [PERSON_DETECTION]
