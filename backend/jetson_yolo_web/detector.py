import os
import time


COCO_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class DetectorUnavailable(RuntimeError):
    pass


class BaseDetector(object):
    backend_name = "base"

    def detect(self, frame, config):
        raise NotImplementedError


class MockDetector(BaseDetector):
    backend_name = "mock"

    def detect(self, frame, config):
        height, width = frame.shape[:2]
        frame_ts = time.time()
        candidates = [
            _detection("person", 0.91, [0.14 * width, 0.18 * height, 0.34 * width, 0.78 * height], frame_ts, 0),
            _detection("laptop", 0.79, [0.48 * width, 0.46 * height, 0.82 * width, 0.78 * height], frame_ts, 63),
            _detection("bottle", 0.62, [0.72 * width, 0.2 * height, 0.82 * width, 0.48 * height], frame_ts, 39),
        ]
        return filter_detections(candidates, config)


class UnavailableDetector(BaseDetector):
    backend_name = "unavailable"

    def __init__(self, reason):
        self.reason = reason

    def detect(self, frame, config):
        raise DetectorUnavailable(self.reason)


def build_detector(config):
    backend = config.get("detector_backend", "auto")
    model_path = config.get("model_path", "")
    if backend == "mock":
        return MockDetector()
    if backend == "opencv":
        return _build_opencv_detector(model_path)
    if backend == "auto":
        if model_path and os.path.exists(model_path) and model_path.endswith(".onnx"):
            return _build_opencv_detector(model_path)
        if model_path and os.path.exists(model_path) and model_path.endswith(".engine"):
            detector = _build_tensorrt_detector(model_path)
            if not isinstance(detector, UnavailableDetector):
                return detector
        opencv_fallback = "models/yolov5n.onnx"
        if os.path.exists(opencv_fallback):
            return _build_opencv_detector(opencv_fallback)
        if not model_path or not os.path.exists(model_path):
            return MockDetector()
    if backend in ("auto", "tensorrt"):
        detector = _build_tensorrt_detector(model_path)
        if backend == "auto" and isinstance(detector, UnavailableDetector):
            return MockDetector()
        return detector
    return MockDetector()


def _build_opencv_detector(model_path):
    try:
        from .opencv_detector import OpenCVDNNYOLODetector

        return OpenCVDNNYOLODetector(model_path, COCO_LABELS)
    except Exception as exc:
        return UnavailableDetector(str(exc))


def _build_tensorrt_detector(model_path):
    try:
        from .tensorrt_detector import TensorRTYOLODetector

        return TensorRTYOLODetector(model_path, COCO_LABELS)
    except Exception as exc:
        return UnavailableDetector(str(exc))


def filter_detections(detections, config):
    confidence = float(config.get("confidence", 0.0))
    allowlist = set(config.get("class_allowlist") or [])
    filtered = []
    for detection in detections:
        if detection["confidence"] < confidence:
            continue
        if allowlist and detection["label"] not in allowlist:
            continue
        filtered.append(detection)
    return filtered


def _detection(label, confidence, bbox, frame_ts, class_id=None):
    x1, y1, x2, y2 = bbox
    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "bbox": {
            "x1": int(max(0, x1)),
            "y1": int(max(0, y1)),
            "x2": int(max(0, x2)),
            "y2": int(max(0, y2)),
        },
        "frame_ts": float(frame_ts),
        "class_id": class_id,
    }
