import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev the API runs on :8000; proxying keeps requests same-origin so the
// session cookie works exactly as in production.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://localhost:8000" } },
});
