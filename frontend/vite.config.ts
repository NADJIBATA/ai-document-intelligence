import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the API is reached through the dev server (same origin), exactly
// like behind nginx in production: no CORS, and the HttpOnly refresh cookie just works.
const apiTarget = process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: false,
    restoreMocks: true,
  },
});
