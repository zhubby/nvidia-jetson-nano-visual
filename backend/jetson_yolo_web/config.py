import json
import os
from copy import deepcopy


DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_device": "",
    "source": "camera",
    "sample_video": "",
    "resolution": {"width": 640, "height": 480},
    "confidence": 0.35,
    "iou": 0.45,
    "class_allowlist": [],
    "model_path": "models/yolov8n_fp16.engine",
    "detector_backend": "auto",
    "jpeg_quality": 78,
    "retry_interval_seconds": 2.0,
}

ALLOWED_KEYS = set(DEFAULT_CONFIG.keys())
VALID_SOURCES = set(["camera", "synthetic", "video"])
VALID_DETECTORS = set(["auto", "mock", "tensorrt"])


class ConfigValidationError(ValueError):
    """Raised when a runtime config update is invalid."""


def _env_float(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name, default):
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def config_from_env():
    return validate_config(merge_config(DEFAULT_CONFIG, env_overrides(DEFAULT_CONFIG)))


def load_config(path=None):
    config = deepcopy(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r") as handle:
            loaded = json.load(handle)
        config = merge_config(config, loaded)
    return merge_config(config, env_overrides(config))


def env_overrides(base=None):
    base = base or DEFAULT_CONFIG
    overrides = {}
    if "JETSON_CAMERA_INDEX" in os.environ:
        overrides["camera_index"] = _env_int("JETSON_CAMERA_INDEX", DEFAULT_CONFIG["camera_index"])
    if "JETSON_CAMERA_DEVICE" in os.environ:
        overrides["camera_device"] = os.environ.get("JETSON_CAMERA_DEVICE", "")
    if "JETSON_SOURCE" in os.environ:
        overrides["source"] = os.environ.get("JETSON_SOURCE", DEFAULT_CONFIG["source"])
    if "JETSON_SAMPLE_VIDEO" in os.environ:
        overrides["sample_video"] = os.environ.get("JETSON_SAMPLE_VIDEO", "")
    if "JETSON_MODEL_PATH" in os.environ:
        overrides["model_path"] = os.environ.get("JETSON_MODEL_PATH", DEFAULT_CONFIG["model_path"])
    if "JETSON_DETECTOR" in os.environ:
        overrides["detector_backend"] = os.environ.get("JETSON_DETECTOR", DEFAULT_CONFIG["detector_backend"])
    if "JETSON_CONFIDENCE" in os.environ:
        overrides["confidence"] = _env_float("JETSON_CONFIDENCE", DEFAULT_CONFIG["confidence"])
    if "JETSON_IOU" in os.environ:
        overrides["iou"] = _env_float("JETSON_IOU", DEFAULT_CONFIG["iou"])
    if "JETSON_WIDTH" in os.environ or "JETSON_HEIGHT" in os.environ:
        overrides["resolution"] = {
            "width": _env_int("JETSON_WIDTH", base["resolution"]["width"]),
            "height": _env_int("JETSON_HEIGHT", base["resolution"]["height"]),
        }
    return overrides


def save_config(path, config):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(validate_config(config), handle, indent=2, sort_keys=True)
        handle.write("\n")


def merge_config(current, update):
    if not isinstance(update, dict):
        raise ConfigValidationError("Config update must be a JSON object.")
    unknown = set(update.keys()) - ALLOWED_KEYS
    if unknown:
        raise ConfigValidationError("Unsupported config key(s): %s" % ", ".join(sorted(unknown)))
    merged = deepcopy(current)
    for key, value in update.items():
        merged[key] = value
    return validate_config(merged)


def validate_config(config):
    normalized = deepcopy(config)
    resolution = normalized.get("resolution")
    if not isinstance(resolution, dict):
        raise ConfigValidationError("resolution must be an object with width and height.")
    width = _validate_int("resolution.width", resolution.get("width"), 160, 3840)
    height = _validate_int("resolution.height", resolution.get("height"), 120, 2160)
    normalized["resolution"] = {"width": width, "height": height}
    normalized["camera_index"] = _validate_int("camera_index", normalized.get("camera_index"), 0, 32)

    source = normalized.get("source")
    if source not in VALID_SOURCES:
        raise ConfigValidationError("source must be one of: %s" % ", ".join(sorted(VALID_SOURCES)))

    detector_backend = normalized.get("detector_backend")
    if detector_backend not in VALID_DETECTORS:
        raise ConfigValidationError("detector_backend must be one of: %s" % ", ".join(sorted(VALID_DETECTORS)))

    normalized["confidence"] = _validate_float("confidence", normalized.get("confidence"), 0.0, 1.0)
    normalized["iou"] = _validate_float("iou", normalized.get("iou"), 0.0, 1.0)
    normalized["jpeg_quality"] = _validate_int("jpeg_quality", normalized.get("jpeg_quality"), 35, 95)
    normalized["retry_interval_seconds"] = _validate_float(
        "retry_interval_seconds", normalized.get("retry_interval_seconds"), 0.1, 30.0
    )

    allowlist = normalized.get("class_allowlist")
    if allowlist is None:
        normalized["class_allowlist"] = []
    elif isinstance(allowlist, list) and all(isinstance(item, str) for item in allowlist):
        normalized["class_allowlist"] = sorted(set(item.strip() for item in allowlist if item.strip()))
    else:
        raise ConfigValidationError("class_allowlist must be a list of class labels.")

    for string_key in ("camera_device", "sample_video", "model_path"):
        value = normalized.get(string_key)
        if value is None:
            normalized[string_key] = ""
        elif not isinstance(value, str):
            raise ConfigValidationError("%s must be a string." % string_key)

    return normalized


def _validate_int(name, value, minimum, maximum):
    if isinstance(value, bool):
        raise ConfigValidationError("%s must be an integer." % name)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ConfigValidationError("%s must be an integer." % name)
    if parsed < minimum or parsed > maximum:
        raise ConfigValidationError("%s must be between %s and %s." % (name, minimum, maximum))
    return parsed


def _validate_float(name, value, minimum, maximum):
    if isinstance(value, bool):
        raise ConfigValidationError("%s must be a number." % name)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ConfigValidationError("%s must be a number." % name)
    if parsed < minimum or parsed > maximum:
        raise ConfigValidationError("%s must be between %s and %s." % (name, minimum, maximum))
    return parsed
