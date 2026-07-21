import math
import os
import time

import numpy as np
from PIL import Image

from .detector import filter_detections


class TensorRTYOLODetector(object):
    """YOLO TensorRT runner for JetPack-provided TensorRT/PyCUDA.

    This supports common YOLOv8 export shapes: ``[1, 84, 8400]`` and
    ``[1, 8400, 84]`` for COCO models. Engine generation still belongs on the
    Jetson or a matching L4T container because TensorRT engines are platform
    specific.
    """

    backend_name = "tensorrt"

    def __init__(self, engine_path, labels):
        if not engine_path or not os.path.exists(engine_path):
            raise RuntimeError("TensorRT engine not found: %s" % engine_path)
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError as exc:
            raise RuntimeError("TensorRT/PyCUDA import failed: %s" % exc)

        self.trt = trt
        self.cuda = cuda
        self.labels = labels
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError("Unable to deserialize TensorRT engine.")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.inputs = []
        self.outputs = []
        self.input_shape = self._resolve_input_shape()
        self._allocate_buffers()

    def detect(self, frame, config):
        original_h, original_w = frame.shape[:2]
        image, ratio, pad = _letterbox(frame, self.input_shape[3], self.input_shape[2])
        tensor = image.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
        if self.inputs[0]["dtype"] == np.float16:
            tensor = tensor.astype(np.float16)

        np.copyto(self.inputs[0]["host"], tensor.ravel())
        self.cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for output in self.outputs:
            self.cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)
        self.stream.synchronize()

        raw_outputs = [output["host"].reshape(output["shape"]) for output in self.outputs]
        detections = _parse_yolov8_output(
            raw_outputs[0],
            self.labels,
            ratio,
            pad,
            original_w,
            original_h,
            float(config.get("confidence", 0.35)),
            float(config.get("iou", 0.45)),
        )
        return filter_detections(detections, config)

    def _resolve_input_shape(self):
        input_index = None
        for index in range(self.engine.num_bindings):
            if self.engine.binding_is_input(index):
                input_index = index
                break
        if input_index is None:
            raise RuntimeError("TensorRT engine has no input binding.")
        shape = tuple(self.engine.get_binding_shape(input_index))
        if any(dim < 0 for dim in shape):
            shape = (1, 3, 640, 640)
            self.context.set_binding_shape(input_index, shape)
        if len(shape) != 4:
            raise RuntimeError("Expected NCHW TensorRT input, got shape %r." % (shape,))
        return shape

    def _allocate_buffers(self):
        trt = self.trt
        cuda = self.cuda
        self.bindings = [None] * self.engine.num_bindings
        for index in range(self.engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(index))
            if any(dim < 0 for dim in shape):
                shape = tuple(self.engine.get_binding_shape(index))
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            size = int(trt.volume(shape))
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.bindings[index] = int(device)
            binding = {"index": index, "shape": shape, "dtype": dtype, "host": host, "device": device}
            if self.engine.binding_is_input(index):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
        if not self.inputs or not self.outputs:
            raise RuntimeError("TensorRT engine must have one input and at least one output.")


def _letterbox(frame, width, height):
    src_h, src_w = frame.shape[:2]
    ratio = min(float(width) / src_w, float(height) / src_h)
    resized_w = int(round(src_w * ratio))
    resized_h = int(round(src_h * ratio))
    image = Image.fromarray(frame).resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (18, 20, 22))
    pad_x = (width - resized_w) // 2
    pad_y = (height - resized_h) // 2
    canvas.paste(image, (pad_x, pad_y))
    return np.asarray(canvas), ratio, (pad_x, pad_y)


def _parse_yolov8_output(output, labels, ratio, pad, original_w, original_h, confidence_threshold, iou_threshold):
    output = np.squeeze(output)
    if output.ndim != 2:
        return []
    if output.shape[0] < output.shape[1] and output.shape[0] <= 256:
        output = output.T
    if output.shape[1] < 6:
        return []

    boxes = []
    scores = []
    class_ids = []
    for row in output:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < confidence_threshold:
            continue
        cx, cy, w, h = row[:4]
        x1 = (cx - w / 2.0 - pad[0]) / ratio
        y1 = (cy - h / 2.0 - pad[1]) / ratio
        x2 = (cx + w / 2.0 - pad[0]) / ratio
        y2 = (cy + h / 2.0 - pad[1]) / ratio
        boxes.append([max(0, x1), max(0, y1), min(original_w, x2), min(original_h, y2)])
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
    return intersection / max(math.pow(10, -9), area_a + area_b - intersection)
