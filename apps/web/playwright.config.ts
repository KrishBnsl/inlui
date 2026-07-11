import { defineConfig, devices } from "@playwright/test";

const port = 3100;
const baseURL = `http://127.0.0.1:${port}`;

export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.mock.spec.ts",
  // A single cold Next.js dev compiler serves all browser contracts. Running
  // six pages against it in parallel can consume the entire 30s test budget
  // before hydration, which is environmental flakiness rather than product
  // behavior. Serial execution is deterministic locally and in CI.
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: "line",
  outputDir: "test-results",
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      // Use Playwright's bundled full Chromium (new headless mode). This avoids
      // a separate headless-shell dependency while exercising the same engine.
      use: { ...devices["Desktop Chrome"], channel: "chromium" },
    },
  ],
  webServer: {
    command: `npm run dev -- --hostname 127.0.0.1 --port ${port}`,
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      NEXT_PUBLIC_ML_URL: `${baseURL}/mock-ml`,
      NEXT_PUBLIC_RAG_URL: `${baseURL}/mock-rag`,
    },
  },
});
