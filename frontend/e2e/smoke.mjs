// Manual visual smoke run: node e2e/smoke.mjs <baseURL> <outdir>
import { chromium } from "@playwright/test";
const [base, out] = process.argv.slice(2);
const browser = await chromium.launch();
for (const scheme of ["light", "dark"]) {
  const page = await browser.newPage({ viewport: { width: 1360, height: 900 }, colorScheme: scheme });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(base);
  await page.getByRole("heading", { name: "Beispielbibliothek" }).waitFor();
  await page.screenshot({ path: `${out}/${scheme}-0-examples.png` });
  await page.getByRole("button", { name: /Methan\/Luft/ }).click();
  await page.getByText("Prüfung").waitFor();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${out}/${scheme}-1-mechanism.png` });
  await page.getByRole("tab", { name: "Reaktor & Löser" }).click();
  await page.screenshot({ path: `${out}/${scheme}-2-reactor.png`, fullPage: true });
  await page.getByRole("button", { name: "▶ Simulieren" }).first().click();
  await page.getByText("Konzentrationsverlauf").waitFor({ timeout: 60000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${out}/${scheme}-3-results.png`, fullPage: true });
  await page.getByRole("tab", { name: "Netzwerk" }).click();
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${out}/${scheme}-4-network.png` });
  console.log(scheme, "errors:", errors);
  await page.close();
}
await browser.close();
