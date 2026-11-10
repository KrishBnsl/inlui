import { expect, test } from "@playwright/test";

test("live frontend reaches the configured ML service", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("JEE Main rank").fill("3000");
  await page.getByLabel("JEE Advanced rank").fill("6700");
  await page.getByLabel("Home state").selectOption("Delhi");
  await page.getByRole("button", { name: "Calculate options" }).click();

  await expect(page.getByRole("heading", { name: "Model recommendations" })).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.locator("[data-testid^='result-row-']").first()).toBeVisible();
  await expect(
    page.getByText(
      "Decision support only; the uncalibrated admission-likelihood estimate is not an individual admission probability and is not an admission guarantee."
    )
  ).toBeVisible();
});
