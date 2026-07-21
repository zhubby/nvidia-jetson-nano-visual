import numpy as np

from jetson_yolo_web.config import DEFAULT_CONFIG
from jetson_yolo_web.detector import MockDetector


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
