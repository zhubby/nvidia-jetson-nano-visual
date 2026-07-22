import json

import pytest

from jetson_yolo_web.config import ConfigValidationError, DEFAULT_CONFIG, load_config, merge_config, validate_config


def test_config_update_validates_confidence_range():
    with pytest.raises(ConfigValidationError):
        merge_config(DEFAULT_CONFIG, {"confidence": 1.2})


def test_config_update_normalizes_allowlist():
    config = merge_config(DEFAULT_CONFIG, {"class_allowlist": ["person", "person", " laptop ", ""]})

    assert config["class_allowlist"] == ["laptop", "person"]


def test_config_accepts_auto_snapshot_options():
    config = merge_config(
        DEFAULT_CONFIG,
        {
            "auto_snapshot_enabled": False,
            "auto_snapshot_label": " person ",
            "auto_snapshot_cooldown_seconds": 12.5,
        },
    )

    assert config["auto_snapshot_enabled"] is False
    assert config["auto_snapshot_label"] == "person"
    assert config["auto_snapshot_cooldown_seconds"] == 12.5


def test_config_rejects_invalid_auto_snapshot_values():
    with pytest.raises(ConfigValidationError):
        merge_config(DEFAULT_CONFIG, {"auto_snapshot_enabled": "yes"})
    with pytest.raises(ConfigValidationError):
        merge_config(DEFAULT_CONFIG, {"auto_snapshot_cooldown_seconds": 0.2})


def test_config_rejects_unknown_keys():
    with pytest.raises(ConfigValidationError):
        merge_config(DEFAULT_CONFIG, {"unexpected": True})


def test_config_accepts_supported_sources_and_detector_backends():
    config = validate_config(
        dict(
            DEFAULT_CONFIG,
            source="synthetic",
            detector_backend="opencv",
            camera_fourcc="MJPG",
            camera_fps="30",
            resolution={"width": "640", "height": "480"},
        )
    )

    assert config["source"] == "synthetic"
    assert config["detector_backend"] == "opencv"
    assert config["camera_fourcc"] == "MJPG"
    assert config["camera_fps"] == 30
    assert config["resolution"] == {"width": 640, "height": 480}


def test_environment_overrides_config_file(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(dict(DEFAULT_CONFIG, source="camera", resolution={"width": 800, "height": 600})))
    monkeypatch.setenv("JETSON_SOURCE", "synthetic")
    monkeypatch.setenv("JETSON_WIDTH", "320")
    monkeypatch.setenv("JETSON_AUTO_SNAPSHOT_ENABLED", "0")
    monkeypatch.setenv("JETSON_AUTO_SNAPSHOT_LABEL", "person")
    monkeypatch.setenv("JETSON_AUTO_SNAPSHOT_COOLDOWN", "45")

    config = load_config(str(config_path))

    assert config["source"] == "synthetic"
    assert config["resolution"] == {"width": 320, "height": 600}
    assert config["auto_snapshot_enabled"] is False
    assert config["auto_snapshot_label"] == "person"
    assert config["auto_snapshot_cooldown_seconds"] == 45.0
