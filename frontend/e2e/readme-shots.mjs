// Generates the README screenshots: node e2e/readme-shots.mjs http://127.0.0.1:8000 ../docs/screenshots
import { chromium } from "@playwright/test";
const [base, out] = process.argv.slice(2);
const browser = await chromium.launch();

async function open(scheme, example) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, colorScheme: scheme, deviceScaleFactor: 2 });
  await page.goto(base);
  await page.addStyleTag({ content: ".topbar, .tabs { position: static !important; } .toast { display: none !important; }" });
  await page.getByRole("heading", { name: "Beispielbibliothek" }).waitFor();
  if (example) {
    await page.getByRole("button", { name: example }).click();
    await page.getByRole("heading", { name: "Prüfung" }).waitFor();
    await page.waitForTimeout(800);
  }
  return page;
}
const card = (page, title) => page.locator("section.card", { has: page.getByRole("heading", { name: title, exact: true }) });

// 1) Example library
{
  const page = await open("light", null);
  await page.locator(".example").first().waitFor();
  await page.waitForTimeout(500);
  await page.locator("dialog").screenshot({ path: `${out}/examples.png` });
  await page.close();
}
// 2) Methane ignition, dark mode: concentrations + temperature
{
  const page = await open("dark", /Methan\/Luft/);
  await page.getByRole("button", { name: "▶ Simulieren" }).first().click();
  await page.getByRole("heading", { name: "Konzentrationsverlauf" }).waitFor({ timeout: 60000 });
  await page.waitForTimeout(1500);
  await page.locator(".tiles").first().screenshot({ path: `${out}/ignition-metrics.png` });
  const conc = card(page, "Konzentrationsverlauf");
  for (const sp of ["O2", "CO"]) await conc.locator("label.chip", { hasText: new RegExp(`^${sp}$`) }).click();
  for (const sp of ["H", "O", "HO2", "CH3"]) await conc.locator("label.chip", { hasText: new RegExp(`^${sp}$`) }).click();
  await conc.locator("label.check", { hasText: "log t" }).click();
  await page.waitForTimeout(1200);
  await conc.locator(".plot").screenshot({ path: `${out}/ignition-species.png` });
  await card(page, "Temperatur").screenshot({ path: `${out}/ignition-temperature.png` });
  await page.close();
}
// 3) Oregonator oscillations
{
  const page = await open("light", /Oregonator/);
  await page.getByRole("tab", { name: "Reaktor & Löser" }).click();
  const tEnd = page.locator(".field", { hasText: "Simulationsdauer" }).locator("input");
  await tEnd.fill("1500");
  await page.getByRole("button", { name: "▶ Simulieren" }).first().click();
  await page.getByRole("heading", { name: "Konzentrationsverlauf" }).waitFor({ timeout: 60000 });
  await page.waitForTimeout(1500);
  await card(page, "Konzentrationsverlauf").screenshot({ path: `${out}/oregonator.png` });
  await page.close();
}
// 4) H2/O2: mechanism editor, network, sensitivity, numerics
{
  const page = await open("light", /H₂\/O₂/);
  await page.waitForTimeout(800);
  await page.locator("main").screenshot({ path: `${out}/mechanism.png`, clip: undefined });
  await page.getByRole("tab", { name: "Netzwerk" }).click();
  await page.waitForTimeout(800);
  await card(page, "Reaktionsnetzwerk").screenshot({ path: `${out}/network.png` });
  await page.getByRole("tab", { name: "Analyse" }).click();
  await page.getByRole("tab", { name: "Sensitivität" }).click();
  await page.getByRole("button", { name: "Berechnen" }).click();
  await page.getByText(/Zielgröße:/).waitFor({ timeout: 90000 });
  await page.waitForTimeout(1000);
  await card(page, "Lokale Sensitivitätsanalyse").screenshot({ path: `${out}/sensitivity.png` });
  await page.getByRole("tab", { name: "Numerik" }).click();
  await page.getByRole("button", { name: "Analyse starten" }).click();
  await page.getByRole("heading", { name: "Empfehlung" }).waitFor({ timeout: 120000 });
  await page.waitForTimeout(1500);
  await card(page, "Empfehlung").screenshot({ path: `${out}/numerics-recommendation.png` });
  await card(page, "Stabilität").screenshot({ path: `${out}/numerics-stability.png` });
  await card(page, "Konsistenz & Konvergenz").screenshot({ path: `${out}/numerics-order.png` });
  await page.close();
}
await browser.close();
console.log("done");
