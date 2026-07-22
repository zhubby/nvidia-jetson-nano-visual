import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Activity,
  Camera,
  CameraOff,
  Cpu,
  Gauge,
  HardDrive,
  RefreshCw,
  Save,
  Settings,
  SlidersHorizontal,
  Thermometer,
  Zap
} from "lucide-react";

import { API_BASE, captureSnapshot, getConfig, getLatestDetections, getStatus, updateConfig } from "./api";
import type { Detection, RuntimeConfig, RuntimeStatus } from "./types";

const emptyStatus: RuntimeStatus = {
  health: "starting",
  camera: { connected: false, source: "camera", error: null },
  model: { backend: "auto", path: "models/yolov8n_fp16.engine", loaded: false, error: null },
  fps: 0,
  latency_ms: null,
  detection_count: 0,
  dropped_frames: 0,
  last_error: null,
  last_frame_ts: null,
  temperature_c: null,
  memory: null,
  auto_snapshot: {
    enabled: true,
    label: "person",
    cooldown_seconds: 30,
    count: 0,
    last_trigger_ts: null,
    last_image_path: null,
    last_metadata_path: null,
    last_error: null
  }
};

const defaultConfig: RuntimeConfig = {
  camera_index: 0,
  camera_device: "",
  camera_fourcc: "MJPG",
  camera_fps: 30,
  source: "camera",
  sample_video: "",
  resolution: { width: 640, height: 480 },
  confidence: 0.35,
  iou: 0.45,
  class_allowlist: [],
  model_path: "models/yolov8n_fp16.engine",
  detector_backend: "auto",
  jpeg_quality: 78,
  retry_interval_seconds: 2,
  auto_snapshot_enabled: true,
  auto_snapshot_label: "person",
  auto_snapshot_cooldown_seconds: 30
};

export default function App() {
  const [status, setStatus] = useState<RuntimeStatus>(emptyStatus);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [config, setConfig] = useState<RuntimeConfig>(defaultConfig);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [streamKey, setStreamKey] = useState(0);
  const [streamError, setStreamError] = useState(false);
  const [notice, setNotice] = useState("");
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [nextStatus, nextDetections] = await Promise.all([getStatus(), getLatestDetections()]);
        if (!cancelled) {
          setStatus(normalizeStatus(nextStatus));
          setDetections(nextDetections.detections);
          setApiError("");
        }
      } catch (error) {
        if (!cancelled) {
          setApiError(error instanceof Error ? error.message : "API unavailable");
        }
      }
    }

    load();
    const timer = window.setInterval(load, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then((nextConfig) => {
        if (!cancelled) {
          setConfig(normalizeConfig(nextConfig));
        }
      })
      .catch((error) => setApiError(error instanceof Error ? error.message : "Config unavailable"));
    return () => {
      cancelled = true;
    };
  }, []);

  const classStats = useMemo(() => {
    const counts = new Map<string, number>();
    detections.forEach((item) => counts.set(item.label, (counts.get(item.label) || 0) + 1));
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  }, [detections]);

  const healthTone = status.health === "running" ? "ok" : status.health === "thermal_warning" ? "warn" : "bad";
  const streamUrl = `${API_BASE}/stream.mjpg?view=${streamKey}`;
  const visibleError = apiError || status.last_error || status.auto_snapshot.last_error;

  async function saveSettings() {
    try {
      const next = await updateConfig(config);
      setConfig(next);
      setNotice("Settings saved");
      setApiError("");
      window.setTimeout(() => setNotice(""), 1800);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Unable to save settings");
    }
  }

  async function saveSnapshot() {
    try {
      const result = await captureSnapshot();
      setNotice(`Snapshot saved: ${shortPath(result.image_path)}`);
      setApiError("");
      window.setTimeout(() => setNotice(""), 2600);
    } catch (error) {
      setApiError(error instanceof Error ? error.message : "Snapshot failed");
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup" aria-label="Jetson Vision Console">
          <span className="brand-mark">NVIDIA</span>
          <span className="brand-divider" />
          <span className="brand-title">Jetson Vision Console</span>
        </div>
        <div className="topbar-status">
          <StatusPill tone={healthTone} label={labelForHealth(status.health)} />
          <MetricChip icon={<Activity size={16} />} label="FPS" value={formatNumber(status.fps, 1)} />
          <MetricChip icon={<Gauge size={16} />} label="Latency" value={formatLatency(status.latency_ms)} />
          <button className="icon-button" type="button" aria-label="Reload stream" title="Reload stream" onClick={() => setStreamKey((value) => value + 1)}>
            <RefreshCw size={18} />
          </button>
          <button className="icon-button" type="button" aria-label="Open settings" title="Open settings" onClick={() => setSettingsOpen(true)}>
            <Settings size={18} />
          </button>
        </div>
      </header>

      <section className="workspace">
        <section className="video-column" aria-label="Live detection stream">
          <div className="video-stage">
            {!streamError ? (
              <img
                className="video-stream"
                src={streamUrl}
                alt="Live YOLO detection stream"
                onLoad={() => setStreamError(false)}
                onError={() => setStreamError(true)}
              />
            ) : (
              <div className="stream-placeholder" role="status">
                <CameraOff size={44} />
                <span>Stream offline</span>
              </div>
            )}
            <div className="video-hud top-left">
              <span className={`live-dot ${status.camera.connected ? "on" : "off"}`} />
              <span>{status.camera.connected ? "LIVE" : "CAMERA OFFLINE"}</span>
            </div>
            <div className="video-hud bottom-right">
              <span>{config.resolution.width}x{config.resolution.height}</span>
              <span>{status.model.backend}</span>
            </div>
          </div>

          <div className="metrics-grid" aria-label="Runtime metrics">
            <MetricPanel icon={<Camera size={18} />} label="Camera" value={status.camera.connected ? status.camera.source : "Offline"} tone={status.camera.connected ? "ok" : "bad"} />
            <MetricPanel icon={<Zap size={18} />} label="Detections" value={String(status.detection_count)} tone="ok" />
            <MetricPanel icon={<Thermometer size={18} />} label="Thermal" value={status.temperature_c == null ? "N/A" : `${status.temperature_c.toFixed(1)} C`} tone={status.health === "thermal_warning" ? "warn" : "neutral"} />
            <MetricPanel icon={<HardDrive size={18} />} label="Memory" value={status.memory ? `${status.memory.percent.toFixed(1)}%` : "N/A"} tone="neutral" />
          </div>
        </section>

        <aside className="inspector" aria-label="Detection inspector">
          <section className="panel detection-panel">
            <div className="panel-heading">
              <div>
                <h2>Detections</h2>
                <p>{detections.length} active objects</p>
              </div>
              <button className="icon-button compact" type="button" aria-label="Save snapshot" title="Save snapshot" onClick={saveSnapshot}>
                <Camera size={17} />
              </button>
            </div>
            <div className="detection-list">
              {detections.length === 0 ? (
                <div className="empty-state">No objects above threshold</div>
              ) : (
                detections.map((item, index) => <DetectionRow key={`${item.label}-${index}`} detection={item} />)
              )}
            </div>
          </section>

          <section className="panel">
            <div className="panel-heading">
              <div>
                <h2>Class Mix</h2>
                <p>Current frame distribution</p>
              </div>
            </div>
            <div className="class-bars">
              {classStats.length === 0 ? (
                <div className="empty-state">No classes detected</div>
              ) : (
                classStats.map(([label, count]) => (
                  <div className="class-bar" key={label}>
                    <span>{label}</span>
                    <div className="bar-track">
                      <div style={{ width: `${Math.max(18, (count / detections.length) * 100)}%` }} />
                    </div>
                    <strong>{count}</strong>
                  </div>
                ))
              )}
            </div>
          </section>

          <section className="panel system-panel">
            <div className="panel-heading">
              <div>
                <h2>System</h2>
                <p>{status.model.loaded ? status.model.path : "Model pending"}</p>
              </div>
              <Cpu size={18} aria-hidden="true" />
            </div>
            <dl className="system-list">
              <div><dt>Detector</dt><dd>{status.model.backend}</dd></div>
              <div><dt>Auto Snap</dt><dd>{status.auto_snapshot.enabled ? `${status.auto_snapshot.count} saved` : "Off"}</dd></div>
              <div><dt>Threshold</dt><dd>{Math.round(config.confidence * 100)}%</dd></div>
              <div><dt>IOU</dt><dd>{Math.round(config.iou * 100)}%</dd></div>
              <div><dt>Dropped</dt><dd>{status.dropped_frames}</dd></div>
            </dl>
          </section>
        </aside>
      </section>

      {(notice || visibleError) && (
        <div className={`toast ${visibleError ? "error" : "success"}`} role="status">
          {visibleError || notice}
        </div>
      )}

      <SettingsDrawer
        open={settingsOpen}
        config={config}
        setConfig={setConfig}
        onClose={() => setSettingsOpen(false)}
        onSave={saveSettings}
      />
    </main>
  );
}

function DetectionRow({ detection }: { detection: Detection }) {
  const confidence = Math.round(detection.confidence * 100);
  return (
    <article className="detection-row">
      <div className="class-token">{detection.label.slice(0, 2).toUpperCase()}</div>
      <div>
        <h3>{detection.label}</h3>
        <p>
          x{detection.bbox.x1} y{detection.bbox.y1} · {detection.bbox.x2 - detection.bbox.x1}x{detection.bbox.y2 - detection.bbox.y1}
        </p>
      </div>
      <strong>{confidence}%</strong>
    </article>
  );
}

function SettingsDrawer({
  open,
  config,
  setConfig,
  onClose,
  onSave
}: {
  open: boolean;
  config: RuntimeConfig;
  setConfig: (config: RuntimeConfig) => void;
  onClose: () => void;
  onSave: () => void;
}) {
  function update<K extends keyof RuntimeConfig>(key: K, value: RuntimeConfig[K]) {
    setConfig({ ...config, [key]: value });
  }

  return (
    <aside className={`settings-drawer ${open ? "open" : ""}`} aria-hidden={!open} aria-label="Runtime settings">
      <div className="drawer-header">
        <div>
          <h2>Runtime</h2>
          <p>Camera, model, thresholds</p>
        </div>
        <button className="icon-button" type="button" aria-label="Close settings" title="Close settings" onClick={onClose}>
          <SlidersHorizontal size={18} />
        </button>
      </div>

      <label className="field">
        <span>Confidence</span>
        <input
          aria-label="Confidence"
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={config.confidence}
          onChange={(event) => update("confidence", Number(event.target.value))}
        />
        <strong>{Math.round(config.confidence * 100)}%</strong>
      </label>

      <label className="field">
        <span>IOU</span>
        <input
          aria-label="IOU"
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={config.iou}
          onChange={(event) => update("iou", Number(event.target.value))}
        />
        <strong>{Math.round(config.iou * 100)}%</strong>
      </label>

      <label className="field stacked">
        <span>Source</span>
        <select aria-label="Source" value={config.source} onChange={(event) => update("source", event.target.value as RuntimeConfig["source"])}>
          <option value="camera">USB camera</option>
          <option value="synthetic">Synthetic</option>
          <option value="video">Sample video</option>
        </select>
      </label>

      <label className="field stacked">
        <span>Detector</span>
        <select aria-label="Detector" value={config.detector_backend} onChange={(event) => update("detector_backend", event.target.value as RuntimeConfig["detector_backend"])}>
          <option value="auto">Auto</option>
          <option value="opencv">OpenCV DNN</option>
          <option value="tensorrt">TensorRT</option>
          <option value="mock">Mock</option>
        </select>
      </label>

      <label className="field stacked">
        <span>Model path</span>
        <input aria-label="Model path" value={config.model_path} onChange={(event) => update("model_path", event.target.value)} />
      </label>

      <label className="field stacked">
        <span>Class allowlist</span>
        <input
          aria-label="Class allowlist"
          value={config.class_allowlist.join(", ")}
          onChange={(event) =>
            update(
              "class_allowlist",
              event.target.value
                .split(",")
                .map((item) => item.trim())
                .filter(Boolean)
            )
          }
        />
      </label>

      <label className="toggle-field">
        <span>Auto person snapshot</span>
        <input
          aria-label="Auto person snapshot"
          type="checkbox"
          checked={config.auto_snapshot_enabled}
          onChange={(event) => update("auto_snapshot_enabled", event.target.checked)}
        />
      </label>

      <div className="resolution-row">
        <label className="field stacked">
          <span>Snapshot class</span>
          <input
            aria-label="Snapshot class"
            value={config.auto_snapshot_label}
            onChange={(event) => update("auto_snapshot_label", event.target.value)}
          />
        </label>
        <label className="field stacked">
          <span>Cooldown seconds</span>
          <input
            aria-label="Cooldown seconds"
            type="number"
            min="1"
            max="3600"
            value={config.auto_snapshot_cooldown_seconds}
            onChange={(event) => update("auto_snapshot_cooldown_seconds", Number(event.target.value))}
          />
        </label>
      </div>

      <div className="resolution-row">
        <label className="field stacked">
          <span>Width</span>
          <input
            aria-label="Width"
            type="number"
            min="160"
            max="3840"
            value={config.resolution.width}
            onChange={(event) => update("resolution", { ...config.resolution, width: Number(event.target.value) })}
          />
        </label>
        <label className="field stacked">
          <span>Height</span>
          <input
            aria-label="Height"
            type="number"
            min="120"
            max="2160"
            value={config.resolution.height}
            onChange={(event) => update("resolution", { ...config.resolution, height: Number(event.target.value) })}
          />
        </label>
      </div>

      <button className="primary-action" type="button" onClick={onSave}>
        <Save size={18} />
        <span>Save</span>
      </button>
    </aside>
  );
}

function StatusPill({ tone, label }: { tone: "ok" | "warn" | "bad"; label: string }) {
  return (
    <div className={`status-pill ${tone}`}>
      <span />
      {label}
    </div>
  );
}

function MetricChip({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="metric-chip">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricPanel({ icon, label, value, tone }: { icon: ReactNode; label: string; value: string; tone: "ok" | "warn" | "bad" | "neutral" }) {
  return (
    <article className={`metric-panel ${tone}`}>
      <div>{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function formatLatency(latency: number | null) {
  return latency == null ? "N/A" : `${Math.round(latency)} ms`;
}

function formatNumber(value: number, digits: number) {
  return Number.isFinite(value) ? value.toFixed(digits) : "0.0";
}

function labelForHealth(health: string) {
  if (health === "running") return "RUNNING";
  if (health === "thermal_warning") return "THERMAL";
  if (health === "starting") return "STARTING";
  return "ERROR";
}

function shortPath(path: string) {
  return path.split("/").slice(-1)[0] || path;
}

function normalizeStatus(status: RuntimeStatus): RuntimeStatus {
  return {
    ...emptyStatus,
    ...status,
    camera: { ...emptyStatus.camera, ...status.camera },
    model: { ...emptyStatus.model, ...status.model },
    auto_snapshot: { ...emptyStatus.auto_snapshot, ...(status.auto_snapshot || {}) }
  };
}

function normalizeConfig(config: RuntimeConfig): RuntimeConfig {
  return {
    ...defaultConfig,
    ...config,
    resolution: { ...defaultConfig.resolution, ...config.resolution },
    class_allowlist: config.class_allowlist || []
  };
}
