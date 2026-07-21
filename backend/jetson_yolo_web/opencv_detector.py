import os
import time

import numpy as np

from .detector import filter_detections


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
        blob = self.cv2.dnn.blobFromImage(image, 1.0 / 255.0, (self.input_size, self.input_size), swapRB=False, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward()
        detections = _parse_output(
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


def _parse_output(output, labels, ratio, pad, original_w, original_h, confidence_threshold, iou_threshold):
    output = np.squeeze(output)
    if output.ndim != 2:
        return []
    if output.shape[0] < output.shape[1] and output.shape[0] <= 256:
        output = output.T

    boxes = []
    scores = []
    class_ids = []
    for row in output:
        if row.shape[0] < 6:
            continue
        if row.shape[0] >= 85:
            objectness = float(row[4])
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            score = objectness * float(class_scores[class_id])
        else:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
        if score < confidence_threshold:
            continue
        cx, cy, w, h = [float(value) for value in row[:4]]
        x1 = (cx - w / 2.0 - pad[0]) / ratio
        y1 = (cy - h / 2.0 - pad[1]) / ratio
        x2 = (cx + w / 2.0 - pad[0]) / ratio
        y2 = (cy + h / 2.0 - pad[1]) / ratio
        boxes.append([max(0.0, x1), max(0.0, y1), min(float(original_w), x2), min(float(original_h), y2)])
        scores.append(score)
        class_ids.append(class_id)

    keep = _nms(boxes, scores, iou_threshold)
    frame_ts = time.time()
    detections = []
    for index in keep:
        class_id = class_ids[index]
        label = labels[class_id] if class_id < len(labels) else "class_%s" % class_id
        x1, y1, x2, y2 = boxes[index]
        detections.append(
            {
                "label": label,
                "confidence": round(scores[index], 4),
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "frame_ts": frame_ts,
                "class_id": class_id,
            }
        )
    return detections


def _nms(boxes, scores, threshold):
    if not boxes:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = np.array([_iou(boxes[current], boxes[int(candidate)]) for candidate in rest])
        order = rest[ious <= threshold]
    return keep


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / max(1e-9, area_a + area_b - intersection)
