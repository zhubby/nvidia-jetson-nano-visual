import numpy as np

from jetson_yolo_web.buffer import LatestFrameBuffer


def test_latest_frame_buffer_keeps_only_newest_frame():
    buffer = LatestFrameBuffer()
    frame_a = np.zeros((2, 2, 3), dtype=np.uint8)
    frame_b = np.ones((2, 2, 3), dtype=np.uint8)

    first = buffer.publish(frame_a, b"one", [{"label": "person"}], 1.0)
    second = buffer.publish(frame_b, b"two", [{"label": "laptop"}], 2.0)

    snapshot = buffer.snapshot()
    assert first == 1
    assert second == 2
    assert snapshot["version"] == 2
    assert snapshot["jpeg"] == b"two"
    assert snapshot["detections"] == [{"label": "laptop"}]
    assert snapshot["frame_ts"] == 2.0


def test_latest_frame_buffer_waits_for_next_version():
    buffer = LatestFrameBuffer()
    buffer.publish(np.zeros((2, 2, 3), dtype=np.uint8), b"first", [], 1.0)

    assert buffer.wait_for_jpeg(last_version=1, timeout=0.01) is None

    buffer.publish(np.zeros((2, 2, 3), dtype=np.uint8), b"second", [], 2.0)
    item = buffer.wait_for_jpeg(last_version=1, timeout=0.01)

    assert item["version"] == 2
    assert item["jpeg"] == b"second"
