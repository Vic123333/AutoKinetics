import { expect, test, type Page } from "@playwright/test";

async function collectErrors(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  return errors;
}

async function loadExample(page: Page, title: RegExp) {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Beispielbibliothek" })).toBeVisible();
  await page.getByRole("button", { name: title }).click();
  await expect(page.getByText(/geladen/)).toBeVisible();
}

test("example → simulate → results with physically correct half-life", async ({ page }) => {
  const errors = await collectErrors(page);
  await loadExample(page, /N₂O₅-Zerfall/);
  await expect(page.getByText(/Mechanismus vollständig und konsistent/)).toBeVisible();
  await page.getByRole("button", { name: "▶ Simulieren" }).first().click();
  await expect(page.getByRole("heading", { name: "Konzentrationsverlauf" })).toBeVisible();
  // ln2 / k with k = 4.94e13 exp(-103.4 kJ/mol / RT) at 318.15 K = 1328.1 s = 22.13 min
  await expect(page.locator(".tile", { hasText: "Halbwertszeit N2O5" })).toContainText("22,13 min");
  expect(errors).toEqual([]);
});

test("build a mechanism from scratch, validation catches element imbalance", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Schließen" }).click();
  await page.getByRole("button", { name: "Neu" }).click().catch(() => undefined);
  page.on("dialog", (d) => d.accept());
  const eq = page.getByRole("textbox", { name: "Reaktion" });
  if (await eq.isVisible()) {
    await eq.fill("H2 + O2 => H2O");
    await page.getByRole("button", { name: "Anlegen" }).click();
  }
  await expect(page.getByText(/Elementbilanz verletzt/).first()).toBeVisible({ timeout: 10_000 });
  await page.getByRole("textbox", { name: "Gleichung 1" }).fill("2 H2 + O2 => 2 H2O");
  await expect(page.getByText(/Mechanismus vollständig und konsistent/)).toBeVisible({ timeout: 10_000 });
});

test("numerics auto-optimisation recommends a stiff solver for Robertson", async ({ page }) => {
  const errors = await collectErrors(page);
  await loadExample(page, /Robertson/);
  await page.getByRole("tab", { name: "Numerik" }).click();
  await page.getByRole("button", { name: "Analyse starten" }).click();
  await expect(page.getByRole("heading", { name: "Empfehlung" })).toBeVisible({ timeout: 90_000 });
  await expect(page.locator(".tile", { hasText: "Problem" })).toContainText("steif");
  await expect(page.locator(".tile", { hasText: "Verfahren" }).first()).toContainText(/Radau|BDF|SDIRK|Gauß|Rosenbrock|Euler/);
  await page.getByRole("button", { name: "Übernehmen & simulieren" }).click();
  await expect(page.getByRole("heading", { name: "Konzentrationsverlauf" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("equilibrium analysis for N2O4 gives Kp ≈ 0.14", async ({ page }) => {
  await loadExample(page, /N₂O₄/);
  await page.getByRole("tab", { name: "Analyse" }).click();
  await page.getByRole("button", { name: "Berechnen" }).click();
  await expect(page.getByRole("table")).toContainText("NO2");
});
