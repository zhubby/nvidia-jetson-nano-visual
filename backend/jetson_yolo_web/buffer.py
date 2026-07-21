import threading
import time


class LatestFrameBuffer(object):
    """Thread-safe single-slot frame buffer.

    Camera and inference work always overwrite the slot. Stream clients wait for
    the next version instead of creating a backlog, which keeps Jetson latency
    bounded under browser or network slowdowns.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._version = 0
        self._frame = None
        self._jpeg = None
        self._detections = []
        self._frame_ts = None

    def publish(self, frame, jpeg_bytes, detections, frame_ts=None):
        with self._condition:
            self._version += 1
            self._frame = frame
            self._jpeg = jpeg_bytes
            self._detections = list(detections or [])
            self._frame_ts = frame_ts if frame_ts is not None else time.time()
            self._condition.notify_all()
            return self._version

    def snapshot(self):
        with self._condition:
            return {
                "version": self._version,
                "frame": self._frame,
                "jpeg": self._jpeg,
                "detections": list(self._detections),
                "frame_ts": self._frame_ts,
            }

    def wait_for_jpeg(self, last_version=None, timeout=1.0):
        deadline = time.time() + timeout
        with self._condition:
            while self._jpeg is None or (last_version is not None and self._version <= last_version):
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return {
                "version": self._version,
                "jpeg": self._jpeg,
                "detections": list(self._detections),
                "frame_ts": self._frame_ts,
            }
