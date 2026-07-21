import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/stream.mjpg": "http://127.0.0.1:8000"
    }
  },
  test: {
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    environment: "jsdom"
  }
});
