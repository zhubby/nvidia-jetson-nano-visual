import type { LatestDetectionsResponse, RuntimeConfig, RuntimeStatus } from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE || "";

export async function getStatus(): Promise<RuntimeStatus> {
  return getJson<RuntimeStatus>("/api/status");
}

export async function getLatestDetections(): Promise<LatestDetectionsResponse> {
  return getJson<LatestDetectionsResponse>("/api/detections/latest");
}

export async function getConfig(): Promise<RuntimeConfig> {
  return getJson<RuntimeConfig>("/api/config");
}

export async function updateConfig(config: Partial<RuntimeConfig>): Promise<RuntimeConfig> {
  return getJson<RuntimeConfig>("/api/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config)
  });
}

export async function captureSnapshot(): Promise<{ image_path: string; metadata_path: string }> {
  return getJson<{ image_path: string; metadata_path: string }>("/api/snapshot", { method: "POST" });
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = (await response.json()) as { error?: string };
      if (payload.error) {
        message = payload.error;
      }
    } catch {
      // Keep the HTTP status message when the server returned non-JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
