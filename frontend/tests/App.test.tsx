import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../src/App";

const statusPayload = {
  health: "running",
  camera: { connected: true, source: "synthetic", error: null },
  model: { backend: "mock", path: "models/yolov8n_fp16.engine", loaded: true, error: null },
  fps: 11.4,
  latency_ms: 47.8,
  detection_count: 2,
  dropped_frames: 0,
  last_error: null,
  last_frame_ts: 1,
  temperature_c: 52.2,
  memory: { total_mb: 4096, used_mb: 1600, percent: 39.1 },
  auto_snapshot: {
    enabled: true,
    label: "person",
    cooldown_seconds: 30,
    count: 2,
    last_trigger_ts: 1,
    last_image_path: "/tmp/auto-person.jpg",
    last_metadata_path: "/tmp/auto-person.json",
    last_error: null
  }
};

const detectionsPayload = {
  frame_ts: 1,
  version: 7,
  detections: [
    { label: "person", confidence: 0.91, bbox: { x1: 12, y1: 30, x2: 220, y2: 360 }, frame_ts: 1, class_id: 0 },
    { label: "laptop", confidence: 0.79, bbox: { x1: 280, y1: 210, x2: 510, y2: 390 }, frame_ts: 1, class_id: 63 }
  ]
};

const configPayload = {
  camera_index: 0,
  camera_device: "",
  camera_fourcc: "MJPG",
  camera_fps: 30,
  source: "synthetic",
  sample_video: "",
  resolution: { width: 640, height: 480 },
  confidence: 0.35,
  iou: 0.45,
  class_allowlist: [],
  model_path: "models/yolov8n_fp16.engine",
  detector_backend: "mock",
  jpeg_quality: 78,
  retry_interval_seconds: 2,
  auto_snapshot_enabled: true,
  auto_snapshot_label: "person",
  auto_snapshot_cooldown_seconds: 30
};

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/api/status")) return jsonResponse(statusPayload);
        if (url.includes("/api/detections/latest")) return jsonResponse(detectionsPayload);
        if (url.includes("/api/config") && init?.method === "PUT") {
          return jsonResponse({ ...configPayload, confidence: 0.72 });
        }
        if (url.includes("/api/config")) return jsonResponse(configPayload);
        if (url.includes("/api/snapshot")) return jsonResponse({ image_path: "/tmp/snapshot.jpg", metadata_path: "/tmp/snapshot.json" });
        return jsonResponse({}, 404);
      })
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders live runtime status and detections", async () => {
    render(<App />);

    expect(await screen.findByText("RUNNING")).toBeInTheDocument();
    expect(screen.getAllByText("person").length).toBeGreaterThan(0);
    expect(screen.getAllByText("laptop").length).toBeGreaterThan(0);
    expect(screen.getByText("11.4")).toBeInTheDocument();
    expect(screen.getByText("48 ms")).toBeInTheDocument();
    expect(screen.getByText("2 saved")).toBeInTheDocument();
  });

  it("opens settings and saves threshold updates", async () => {
    render(<App />);
    await screen.findByText("RUNNING");

    fireEvent.click(screen.getByLabelText("Open settings"));
    fireEvent.change(screen.getByLabelText("Confidence"), { target: { value: "0.72" } });
    fireEvent.click(screen.getByLabelText("Auto person snapshot"));
    fireEvent.change(screen.getByLabelText("Cooldown seconds"), { target: { value: "45" } });
    fireEvent.click(screen.getByText("Save"));

    await waitFor(() => {
      expect(screen.getByText("Settings saved")).toBeInTheDocument();
    });
    const request = putConfigRequest();
    expect(request.confidence).toBe(0.72);
    expect(request.auto_snapshot_enabled).toBe(false);
    expect(request.auto_snapshot_label).toBe("person");
    expect(request.auto_snapshot_cooldown_seconds).toBe(45);
  });

  it("handles older status payloads without auto snapshot fields", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/api/status")) {
          const legacyStatus = { ...statusPayload } as Partial<typeof statusPayload>;
          delete legacyStatus.auto_snapshot;
          return jsonResponse(legacyStatus);
        }
        if (url.includes("/api/detections/latest")) return jsonResponse(detectionsPayload);
        if (url.includes("/api/config") && init?.method === "PUT") return jsonResponse(configPayload);
        if (url.includes("/api/config")) return jsonResponse(configPayload);
        return jsonResponse({}, 404);
      })
    );

    render(<App />);

    expect(await screen.findByText("RUNNING")).toBeInTheDocument();
    expect(screen.getByText("0 saved")).toBeInTheDocument();
  });
});

function putConfigRequest() {
  const fetchMock = fetch as unknown as { mock: { calls: Array<[RequestInfo | URL, RequestInit?]> } };
  const call = fetchMock.mock.calls.find(([url, init]) => String(url).includes("/api/config") && init?.method === "PUT");
  if (!call || typeof call[1]?.body !== "string") {
    throw new Error("PUT /api/config was not called with a JSON body");
  }
  return JSON.parse(call[1].body);
}

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(payload), {
      status,
      headers: { "Content-Type": "application/json" }
    })
  );
}
