from io import BytesIO

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


ACCENT = (118, 185, 0)
WARNING = (255, 197, 61)
ERROR = (255, 82, 82)
TEXT = (244, 247, 242)
SURFACE = (16, 18, 20)


def draw_overlay(frame, detections, status_text=None):
    if Image is None:
        return _draw_overlay_numpy(frame, detections)
    image = Image.fromarray(_as_rgb_uint8(frame)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for detection in detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        label = "%s %.0f%%" % (detection["label"], detection["confidence"] * 100)
        color = _color_for_label(detection["label"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_h - 8)
        draw.rectangle([x1, label_y, x1 + text_w + 10, label_y + text_h + 6], fill=color)
        draw.text((x1 + 5, label_y + 3), label, fill=(0, 0, 0), font=font)
    if status_text:
        draw.rectangle([14, 14, 14 + len(status_text) * 7 + 18, 42], fill=(0, 0, 0))
        draw.text((24, 23), status_text, fill=ACCENT, font=font)
    return np.asarray(image)


def encode_jpeg(frame, quality=78):
    if Image is None:
        return _FALLBACK_JPEG
    image = Image.fromarray(_as_rgb_uint8(frame)).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return buffer.getvalue()


def placeholder_frame(width=640, height=480, message="Waiting for camera"):
    if Image is None:
        frame = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        stripe = int(width * 0.18)
        frame[:, :, :] = np.array(SURFACE, dtype=np.uint8)
        frame[:, :stripe, :] = np.array(ACCENT, dtype=np.uint8)
        frame[:, stripe : stripe + 8, :] = 255
        return frame
    image = Image.new("RGB", (int(width), int(height)), SURFACE)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    stripe = int(width * 0.18)
    draw.rectangle([0, 0, stripe, height], fill=(118, 185, 0))
    draw.rectangle([stripe, 0, stripe + 8, height], fill=(255, 255, 255))
    draw.text((stripe + 32, height // 2 - 10), message, fill=TEXT, font=font)
    return np.asarray(image)


def _as_rgb_uint8(frame):
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB frame shaped HxWx3.")
    return array


def _color_for_label(label):
    seed = sum(ord(char) for char in label)
    palette = [ACCENT, WARNING, (31, 156, 255), (170, 238, 170), (232, 232, 232)]
    return palette[seed % len(palette)]


def _draw_overlay_numpy(frame, detections):
    output = _as_rgb_uint8(frame).copy()
    height, width = output.shape[:2]
    for detection in detections:
        bbox = detection["bbox"]
        x1 = max(0, min(width - 1, int(bbox["x1"])))
        y1 = max(0, min(height - 1, int(bbox["y1"])))
        x2 = max(0, min(width - 1, int(bbox["x2"])))
        y2 = max(0, min(height - 1, int(bbox["y2"])))
        color = np.array(_color_for_label(detection["label"]), dtype=np.uint8)
        output[y1 : min(y1 + 3, height), x1:x2] = color
        output[max(y2 - 3, 0) : y2, x1:x2] = color
        output[y1:y2, x1 : min(x1 + 3, width)] = color
        output[y1:y2, max(x2 - 3, 0) : x2] = color
    return output


_FALLBACK_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08"
    b"\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e"
    b"\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14"
    b"\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00"
    b"\x00?\x00\xbf\xff\xd9"
)
