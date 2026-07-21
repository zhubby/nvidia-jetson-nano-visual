import time

import numpy as np


MAX_NMS_CANDIDATES = 1000
MAX_DETECTIONS = 100


def parse_yolo_output(output, labels, ratio, pad, original_w, original_h, confidence_threshold, iou_threshold):
    output = np.squeeze(output)
    if output.ndim != 2:
        return []
    transposed_columns = set([len(labels) + 4, len(labels) + 5])
    if output.shape[0] in transposed_columns and output.shape[1] > output.shape[0]:
        output = output.T
    if output.shape[1] < 6:
        return []

    output = output.astype(np.float32, copy=False)
    if output.shape[1] >= 85:
        objectness = output[:, 4]
        class_scores = output[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
    else:
        class_scores = output[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= float(confidence_threshold)
    if not np.any(mask):
        return []

    boxes_xywh = output[mask, :4]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if scores.shape[0] > MAX_NMS_CANDIDATES:
        order = np.argsort(scores)[-MAX_NMS_CANDIDATES:]
        boxes_xywh = boxes_xywh[order]
        scores = scores[order]
        class_ids = class_ids[order]

    cx = boxes_xywh[:, 0]
    cy = boxes_xywh[:, 1]
    width = boxes_xywh[:, 2]
    height = boxes_xywh[:, 3]
    boxes = np.column_stack(
        (
            (cx - width / 2.0 - pad[0]) / ratio,
            (cy - height / 2.0 - pad[1]) / ratio,
            (cx + width / 2.0 - pad[0]) / ratio,
            (cy + height / 2.0 - pad[1]) / ratio,
        )
    )
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(original_w))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(original_h))
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not np.any(valid):
        return []

    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    keep = nms(boxes, scores, iou_threshold)[:MAX_DETECTIONS]
    frame_ts = time.time()
    detections = []
    for index in keep:
        class_id = int(class_ids[index])
        label = labels[class_id] if class_id < len(labels) else "class_%s" % class_id
        x1, y1, x2, y2 = boxes[index]
        detections.append(
            {
                "label": label,
                "confidence": round(float(scores[index]), 4),
                "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                "frame_ts": frame_ts,
                "class_id": class_id,
            }
        )
    return detections


def nms(boxes, scores, threshold):
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.size == 0 or scores.size == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        intersection = inter_w * inter_h
        denominator = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(denominator, 1e-9)
        order = rest[iou <= float(threshold)]

    return keep
