import os

import numpy as np
from PIL import Image

from .detector import filter_detections
from .yolo_postprocess import parse_yolo_output


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
        except ImportError as exc:
            raise RuntimeError("TensorRT import failed: %s" % exc)

        self.trt = trt
        self.cuda = None
        self.cuda_runtime = None
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda

            self.cuda = cuda
            self.cuda_mode = "pycuda"
        except ImportError:
            from .cuda_runtime import CudaRuntime

            self.cuda_runtime = CudaRuntime()
            self.cuda_mode = "ctypes"
        self.labels = labels
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError("Unable to deserialize TensorRT engine.")
        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.Stream() if self.cuda_mode == "pycuda" else None
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
        if self.cuda_mode == "pycuda":
            self.cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            for output in self.outputs:
                self.cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)
            self.stream.synchronize()
        else:
            self.cuda_runtime.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])
            self.context.execute_v2(bindings=self.bindings)
            for output in self.outputs:
                self.cuda_runtime.memcpy_dtoh(output["host"], output["device"])

        raw_outputs = [output["host"].reshape(output["shape"]) for output in self.outputs]
        detections = parse_yolo_output(
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
        self.bindings = [None] * self.engine.num_bindings
        for index in range(self.engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(index))
            if any(dim < 0 for dim in shape):
                shape = tuple(self.engine.get_binding_shape(index))
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            size = int(trt.volume(shape))
            if self.cuda_mode == "pycuda":
                host = self.cuda.pagelocked_empty(size, dtype)
                device = self.cuda.mem_alloc(host.nbytes)
                device_pointer = int(device)
            else:
                host = np.empty(size, dtype)
                device_pointer = self.cuda_runtime.malloc(host.nbytes)
                device = device_pointer
            self.bindings[index] = int(device_pointer)
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
    image = Image.fromarray(frame[:, :, ::-1]).resize((resized_w, resized_h), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (18, 20, 22))
    pad_x = (width - resized_w) // 2
    pad_y = (height - resized_h) // 2
    canvas.paste(image, (pad_x, pad_y))
    return np.asarray(canvas), ratio, (pad_x, pad_y)

