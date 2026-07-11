import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { chromium } from "@playwright/test";

const appUrl = process.env.SEATCRAFT_SCREENSHOT_URL ?? "http://localhost:3000";
const outputDirectory = resolve(process.cwd(), "../../docs/images");
await mkdir(outputDirectory, { recursive: true });

const browser = await chromium.launch({ channel: "chromium" });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
const pageErrors = [];
const consoleErrors = [];
page.on("pageerror", (error) => pageErrors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") consoleErrors.push(message.text());
});

await page.goto(appUrl, { waitUntil: "networkidle", timeout: 90_000 });
const heroHeading = page.getByRole("heading", {
  name: "Build a JoSAA list with uncertainty you can inspect.",
});
await heroHeading.waitFor({ state: "visible", timeout: 90_000 });
await page.waitForFunction(() => {
  const heading = [...document.querySelectorAll("h1")].find((element) =>
    element.textContent?.includes("Build a JoSAA list with uncertainty you can inspect.")
  );
  if (!(heading instanceof HTMLElement)) return false;

  let element = heading;
  while (element) {
    const style = window.getComputedStyle(element);
    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      Number(style.opacity) < 0.99
    ) {
      return false;
    }
    element = element.parentElement;
  }
  return true;
});
await page.evaluate(
  () => new Promise((resolveFrame) => requestAnimationFrame(() => requestAnimationFrame(resolveFrame)))
);
await page.screenshot({ path: resolve(outputDirectory, "seatcraft-home.png") });

await page.getByLabel("JEE Main rank").fill("3000");
await page.getByLabel("JEE Advanced rank").fill("6700");
await page.getByLabel("Home state").selectOption("Delhi");
await page.getByRole("button", { name: "Calculate options" }).click();
await page.getByRole("heading", { name: "Model recommendations" }).waitFor({
  state: "visible",
  timeout: 90_000,
});
await page.getByRole("heading", { name: "Model recommendations" }).scrollIntoViewIfNeeded();
await page.screenshot({ path: resolve(outputDirectory, "seatcraft-results.png") });

console.log(
  JSON.stringify(
    {
      appUrl,
      screenshots: ["docs/images/seatcraft-home.png", "docs/images/seatcraft-results.png"],
      pageErrors,
      consoleErrors,
    },
    null,
    2
  )
);
await browser.close();
