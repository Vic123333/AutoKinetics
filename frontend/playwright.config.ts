import { defineConfig } from "@playwright/test";

// End-to-end tests against the real backend serving the built frontend.
// Build first: `npm run build` (writes to ../src/autokinetics/static).
export default defineConfig({
  testDir: "e2e",
  testMatch: /.*\.spec\.ts/,
  timeout: 120_000,
  use: { baseURL: "http://127.0.0.1:8799", viewport: { width: 1360, height: 900 } },
  webServer: {
    command: "cd .. && PYTHONPATH=src AUTOKINETICS_WORKERS=2 ${PYTHON:-.venv/bin/python} -m autokinetics.cli serve --port 8799",
    url: "http://127.0.0.1:8799/api/health",
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
