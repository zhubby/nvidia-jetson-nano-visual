import json
import os
import threading
import time

from .buffer import LatestFrameBuffer
from .camera import CameraError, build_frame_source
from .config import ConfigValidationError, load_config, merge_config, save_config
from .detector import DetectorUnavailable, build_detector
from .overlay import draw_overlay, encode_jpeg, placeholder_frame
from .sensors import read_system_metrics


class RuntimeService(object):
    def __init__(self, config_path=None, config=None, frame_source=None, detector=None):
        self.config_path = config_path
        self.config = config or load_config(config_path)
        self.frame_source = frame_source
        self.detector = detector
        self.buffer = LatestFrameBuffer()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_status = self._initial_status()
        self._frames = 0
        self._fps_window_started = time.time()

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="jetson-yolo-runtime")
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        thread = self._thread
        if thread:
            thread.join(timeout=3.0)
        if self.frame_source:
            self.frame_source.close()

    def get_config(self):
        with self._lock:
            return json.loads(json.dumps(self.config))

    def update_config(self, update):
        with self._lock:
            previous = self.config
            self.config = merge_config(self.config, update)
            if self.config_path:
                save_config(self.config_path, self.config)
            source_changed = any(
                previous.get(key) != self.config.get(key)
                for key in ("camera_index", "camera_device", "source", "sample_video", "resolution")
            )
            detector_changed = any(previous.get(key) != self.config.get(key) for key in ("model_path", "detector_backend"))
            if source_changed and self.frame_source:
                self.frame_source.close()
                self.frame_source = None
            if detector_changed:
                self.detector = None
            return self.get_config()

    def status(self):
        with self._lock:
            status = json.loads(json.dumps(self._last_status))
        metrics = read_system_metrics()
        status.update(metrics)
        if metrics.get("temperature_c") and metrics["temperature_c"] >= 78:
            status["health"] = "thermal_warning"
        return status

    def latest_detections(self):
        snapshot = self.buffer.snapshot()
        return {
            "frame_ts": snapshot["frame_ts"],
            "version": snapshot["version"],
            "detections": snapshot["detections"],
        }

    def wait_for_jpeg(self, last_version=None, timeout=1.0):
        return self.buffer.wait_for_jpeg(last_version, timeout)

    def save_snapshot(self, directory):
        os.makedirs(directory, exist_ok=True)
        snapshot = self.buffer.snapshot()
        if snapshot["jpeg"] is None:
            config = self.get_config()
            frame = placeholder_frame(
                config["resolution"]["width"], config["resolution"]["height"], "No frame available"
            )
            jpeg = encode_jpeg(frame, config["jpeg_quality"])
            detections = []
            frame_ts = time.time()
        else:
            jpeg = snapshot["jpeg"]
            detections = snapshot["detections"]
            frame_ts = snapshot["frame_ts"] or time.time()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(frame_ts))
        image_path = os.path.join(directory, "snapshot-%s.jpg" % stamp)
        json_path = os.path.join(directory, "snapshot-%s.json" % stamp)
        with open(image_path, "wb") as handle:
            handle.write(jpeg)
        with open(json_path, "w") as handle:
            json.dump({"frame_ts": frame_ts, "detections": detections}, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return {"image_path": image_path, "metadata_path": json_path, "detections": detections}

    def _run(self):
        while not self._stop_event.is_set():
            started = time.time()
            try:
                self._ensure_runtime_objects()
                ok, frame = self.frame_source.read()
                if not ok:
                    raise CameraError("Frame source returned no frame.")
                detect_started = time.time()
                detections = self.detector.detect(frame, self.config)
                latency_ms = (time.time() - detect_started) * 1000.0
                overlay = draw_overlay(frame, detections)
                jpeg = encode_jpeg(overlay, self.config["jpeg_quality"])
                self.buffer.publish(overlay, jpeg, detections, started)
                self._record_success(len(detections), latency_ms)
            except (CameraError, DetectorUnavailable, ConfigValidationError, ValueError) as exc:
                self._record_error(str(exc))
                self._publish_placeholder(str(exc))
                time.sleep(self.config.get("retry_interval_seconds", 2.0))
            except Exception as exc:
                self._record_error("Unexpected runtime error: %s" % exc)
                self._publish_placeholder("Runtime error")
                time.sleep(self.config.get("retry_interval_seconds", 2.0))

    def _ensure_runtime_objects(self):
        if self.frame_source is None:
            self.frame_source = build_frame_source(self.config)
            self.frame_source.open()
        if self.detector is None:
            self.detector = build_detector(self.config)

    def _record_success(self, detection_count, latency_ms):
        now = time.time()
        self._frames += 1
        elapsed = max(0.001, now - self._fps_window_started)
        fps = self._frames / elapsed
        if elapsed >= 5.0:
            self._frames = 0
            self._fps_window_started = now
        with self._lock:
            self._last_status.update(
                {
                    "camera": {"connected": True, "source": self.config.get("source"), "error": None},
                    "model": {
                        "backend": getattr(self.detector, "backend_name", "unknown"),
                        "path": self.config.get("model_path"),
                        "loaded": True,
                        "error": None,
                    },
                    "health": "running",
                    "fps": round(fps, 2),
                    "latency_ms": round(latency_ms, 1),
                    "detection_count": int(detection_count),
                    "last_frame_ts": now,
                }
            )

    def _record_error(self, message):
        with self._lock:
            self._last_status.update(
                {
                    "camera": {"connected": False, "source": self.config.get("source"), "error": message},
                    "health": "error",
                    "fps": 0.0,
                    "latency_ms": None,
                    "detection_count": 0,
                    "last_error": message,
                    "last_frame_ts": time.time(),
                }
            )

    def _publish_placeholder(self, message):
        config = self.get_config()
        frame = placeholder_frame(config["resolution"]["width"], config["resolution"]["height"], message[:72])
        jpeg = encode_jpeg(frame, config["jpeg_quality"])
        self.buffer.publish(frame, jpeg, [], time.time())

    def _initial_status(self):
        return {
            "health": "starting",
            "camera": {"connected": False, "source": self.config.get("source"), "error": None},
            "model": {
                "backend": self.config.get("detector_backend"),
                "path": self.config.get("model_path"),
                "loaded": False,
                "error": None,
            },
            "fps": 0.0,
            "latency_ms": None,
            "detection_count": 0,
            "dropped_frames": 0,
            "last_error": None,
            "last_frame_ts": None,
            "temperature_c": None,
            "memory": None,
        }


def mjpeg_stream(runtime):
    last_version = None
    while True:
        item = runtime.wait_for_jpeg(last_version=last_version, timeout=1.0)
        if item is None:
            config = runtime.get_config()
            jpeg = encode_jpeg(
                placeholder_frame(config["resolution"]["width"], config["resolution"]["height"], "Waiting for camera"),
                config["jpeg_quality"],
            )
        else:
            last_version = item["version"]
            jpeg = item["jpeg"]
        yield b"--frame\r\n"
        yield b"Content-Type: image/jpeg\r\n"
        yield b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
        yield jpeg
        yield b"\r\n"
