export type BBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type Detection = {
  label: string;
  confidence: number;
  bbox: BBox;
  frame_ts: number;
  class_id?: number;
};

export type RuntimeConfig = {
  camera_index: number;
  camera_device: string;
  camera_fourcc: string;
  camera_fps: number;
  source: "camera" | "synthetic" | "video";
  sample_video: string;
  resolution: {
    width: number;
    height: number;
  };
  confidence: number;
  iou: number;
  class_allowlist: string[];
  model_path: string;
  detector_backend: "auto" | "mock" | "opencv" | "tensorrt";
  jpeg_quality: number;
  retry_interval_seconds: number;
  auto_snapshot_enabled: boolean;
  auto_snapshot_label: string;
  auto_snapshot_cooldown_seconds: number;
};

export type RuntimeStatus = {
  health: "starting" | "running" | "error" | "thermal_warning" | string;
  camera: {
    connected: boolean;
    source: string;
    error: string | null;
  };
  model: {
    backend: string;
    path: string;
    loaded: boolean;
    error: string | null;
  };
  fps: number;
  latency_ms: number | null;
  detection_count: number;
  dropped_frames: number;
  last_error: string | null;
  last_frame_ts: number | null;
  temperature_c: number | null;
  memory: {
    total_mb: number;
    used_mb: number;
    percent: number;
  } | null;
  auto_snapshot: {
    enabled: boolean;
    label: string;
    cooldown_seconds: number;
    count: number;
    last_trigger_ts: number | null;
    last_image_path: string | null;
    last_metadata_path: string | null;
    last_error: string | null;
  };
};

export type LatestDetectionsResponse = {
  frame_ts: number | null;
  version: number;
  detections: Detection[];
};
