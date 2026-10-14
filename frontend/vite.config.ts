import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../src/autokinetics/static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 5000,
    rollupOptions: { output: { manualChunks: { plotly: ["plotly.js-dist-min"] } } },
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
