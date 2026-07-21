import numpy as np

from jetson_yolo_web.config import DEFAULT_CONFIG
from jetson_yolo_web import detector as detector_module
from jetson_yolo_web.detector import COCO_LABELS, MockDetector
from jetson_yolo_web.yolo_postprocess import parse_yolo_output


def test_mock_detector_filters_by_confidence():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    config = dict(DEFAULT_CONFIG, confidence=0.8, class_allowlist=[])

    detections = MockDetector().detect(frame, config)

    assert [item["label"] for item in detections] == ["person"]


def test_mock_detector_filters_by_class_allowlist():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    config = dict(DEFAULT_CONFIG, confidence=0.1, class_allowlist=["laptop"])

    detections = MockDetector().detect(frame, config)

    assert len(detections) == 1
    assert detections[0]["label"] == "laptop"
    assert detections[0]["bbox"]["x2"] > detections[0]["bbox"]["x1"]


def test_parse_yolov5_output_applies_objectness_and_nms():
    output = np.zeros((1, 3, 85), dtype=np.float32)
    output[0, 0, :5] = [100, 100, 40, 60, 0.9]
    output[0, 0, 5] = 0.8
    output[0, 1, :5] = [102, 102, 40, 60, 0.7]
    output[0, 1, 5] = 0.8
    output[0, 2, :5] = [20, 20, 10, 10, 0.2]
    output[0, 2, 5] = 0.9

    detections = parse_yolo_output(output, COCO_LABELS, 1.0, (0, 0), 200, 200, 0.5, 0.45)

    assert len(detections) == 1
    assert detections[0]["label"] == "person"
    assert detections[0]["confidence"] == 0.72
    assert detections[0]["bbox"] == {"x1": 80, "y1": 70, "x2": 120, "y2": 130}


def test_parse_yolov8_transposed_output_uses_class_scores():
    output = np.zeros((1, 84, 100), dtype=np.float32)
    output[0, :4, 7] = [50, 60, 20, 30]
    output[0, 5, 7] = 0.91

    detections = parse_yolo_output(output, COCO_LABELS, 1.0, (0, 0), 200, 200, 0.5, 0.45)

    assert len(detections) == 1
    assert detections[0]["label"] == "bicycle"
    assert detections[0]["confidence"] == 0.91
    assert detections[0]["bbox"] == {"x1": 40, "y1": 45, "x2": 60, "y2": 75}


def test_auto_detector_prefers_existing_tensorrt_engine(monkeypatch, tmp_path):
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"engine")
    expected = MockDetector()

    def build_tensorrt(path):
        assert path == str(engine_path)
        return expected

    monkeypatch.setattr(detector_module, "_build_tensorrt_detector", build_tensorrt)

    config = dict(DEFAULT_CONFIG, detector_backend="auto", model_path=str(engine_path))

    assert detector_module.build_detector(config) is expected
