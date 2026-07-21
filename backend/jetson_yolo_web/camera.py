import math
import time

import numpy as np


class CameraError(RuntimeError):
    pass


class BaseFrameSource(object):
    name = "base"

    def open(self):
        return None

    def read(self):
        raise NotImplementedError

    def close(self):
        return None


class SyntheticCamera(BaseFrameSource):
    name = "synthetic"

    def __init__(self, width=640, height=480):
        self.width = int(width)
        self.height = int(height)
        self._frame_no = 0
        self._opened = False

    def open(self):
        self._opened = True

    def read(self):
        if not self._opened:
            self.open()
        self._frame_no += 1
        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)[:, None]
        pulse = (math.sin(self._frame_no / 8.0) + 1.0) / 2.0
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = np.clip((20 + 45 * x + 30 * pulse), 0, 255)
        frame[:, :, 1] = np.clip((48 + 140 * y + 50 * pulse), 0, 255)
        frame[:, :, 2] = np.clip((32 + 70 * (1 - x)), 0, 255)

        box_w = max(80, self.width // 5)
        box_h = max(70, self.height // 5)
        left = int((self.width - box_w) * pulse)
        top = int(self.height * 0.34)
        frame[top : top + box_h, left : left + box_w, :] = np.array([118, 185, 0], dtype=np.uint8)
        time.sleep(1.0 / 18.0)
        return True, frame


class OpenCVCamera(BaseFrameSource):
    name = "camera"

    def __init__(self, camera_index=0, camera_device="", width=640, height=480):
        self.camera_index = int(camera_index)
        self.camera_device = camera_device
        self.width = int(width)
        self.height = int(height)
        self._capture = None

    def open(self):
        cv2 = _load_cv2()
        source = self.camera_device or self.camera_index
        self._capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise CameraError("Unable to open USB camera source %r." % source)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        if self._capture is None:
            self.open()
        cv2 = _load_cv2()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("USB camera returned an empty frame.")
        return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class VideoFileCamera(OpenCVCamera):
    name = "video"

    def __init__(self, path, width=640, height=480):
        OpenCVCamera.__init__(self, 0, path, width, height)
        self.path = path

    def open(self):
        cv2 = _load_cv2()
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise CameraError("Unable to open sample video %r." % self.path)

    def read(self):
        try:
            return OpenCVCamera.read(self)
        except CameraError:
            self.close()
            self.open()
            return OpenCVCamera.read(self)


def build_frame_source(config):
    width = config["resolution"]["width"]
    height = config["resolution"]["height"]
    source = config.get("source", "camera")
    if source == "synthetic":
        return SyntheticCamera(width, height)
    if source == "video":
        return VideoFileCamera(config.get("sample_video", ""), width, height)
    return OpenCVCamera(config.get("camera_index", 0), config.get("camera_device", ""), width, height)


def _load_cv2():
    try:
        import cv2
    except ImportError:
        raise CameraError("OpenCV is not installed. On Jetson run: sudo apt install python3-opencv v4l-utils")
    return cv2
