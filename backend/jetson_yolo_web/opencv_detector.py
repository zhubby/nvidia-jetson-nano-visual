import os

import numpy as np

from .detector import filter_detections
from .yolo_postprocess import parse_yolo_output


class OpenCVDNNYOLODetector(object):
    """YOLO ONNX detector using OpenCV DNN.

    This is a practical Jetson Nano fallback when TensorRT is installed but
    PyCUDA or a platform-specific engine is not available yet. It supports the
    common YOLOv5 ONNX output ``[1, 25200, 85]`` and YOLOv8 output
    ``[1, 84, 8400]`` / ``[1, 8400, 84]``.
    """

    backend_name = "opencv"

    def __init__(self, model_path, labels, input_size=640):
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("OpenCV DNN model not found: %s" % model_path)
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV import failed: %s" % exc)
        self.cv2 = cv2
        self.model_path = model_path
        self.labels = labels
        self.input_size = int(input_size)
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def detect(self, frame, config):
        image, ratio, pad = _letterbox(self.cv2, frame, self.input_size, self.input_size)
        blob = self.cv2.dnn.blobFromImage(image, 1.0 / 255.0, (self.input_size, self.input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        detections = parse_yolo_output(
            outputs,
            self.labels,
            ratio,
            pad,
            frame.shape[1],
            frame.shape[0],
            float(config.get("confidence", 0.35)),
            float(config.get("iou", 0.45)),
        )
        return filter_detections(detections, config)


def _letterbox(cv2, frame, width, height):
    src_h, src_w = frame.shape[:2]
    ratio = min(float(width) / src_w, float(height) / src_h)
    resized_w = int(round(src_w * ratio))
    resized_h = int(round(src_h * ratio))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (width - resized_w) // 2
    pad_y = (height - resized_h) // 2
    output = np.full((height, width, 3), 114, dtype=np.uint8)
    output[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    return output, ratio, (pad_x, pad_y)

