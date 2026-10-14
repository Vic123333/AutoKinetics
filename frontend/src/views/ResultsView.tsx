import type { Data } from "plotly.js";
import { useEffect, useMemo, useRef, useState } from "react";
import type { AppContext } from "../App";
import { chartTheme, ColorAssigner, MAX_SERIES } from "../lib/palette";
import type { SimulationResult } from "../lib/types";
import { Plot } from "../components/Plot";
import { useDarkMode } from "../components/theme";
import { Alert, Card, fmtNum, fmtTime } from "../components/ui";

function defaultSelection(r: SimulationResult, hint?: string[]): string[] {
  const valid = (hint ?? []).filter((s) => r.species.includes(s));
  if (valid.length) return valid.slice(0, MAX_SERIES);
  const range = r.species.map((s, i) => {
    const c = r.concentrations[i];
    return [s, Math.max(...c) - Math.min(...c)] as const;
  });
  return range.sort((a, b) => b[1] - a[1]).filter(([, v]) => v > 0).slice(0, 6).map(([s]) => s);
}

export function ResultsView({ ctx }: { ctx: AppContext }) {
  const { result, project } = ctx;
  const dark = useDarkMode();
  const theme = chartTheme(dark);
  const colors = useRef(new ColorAssigner());
  const [selected, setSelected] = useState<string[]>([]);
  const [logY, setLogY] = useState(!!project.meta.plot?.log_y);
  const [logX, setLogX] = useState(!!project.meta.plot?.log_time || project.simulation.log_time);
  const [quantity, setQuantity] = useState<"c" | "x">("c");
  const [showTable, setShowTable] = useState(false);
  const [filter, setFilter] = useState("");

  useEffect(() => {
    if (result) setSelected(defaultSelection(result, project.meta.plot?.species));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  const tScale = useMemo(() => {
    if (!result) return { f: 1, unit: "s" };
    const tEnd = result.t[result.t.length - 1];
    if (tEnd < 1e-3) return { f: 1e6, unit: "µs" };
    if (tEnd < 1) return { f: 1e3, unit: "ms" };
    if (tEnd > 7200) return { f: 1 / 3600, unit: "h" };
    if (tEnd > 300) return { f: 1 / 60, unit: "min" };
    return { f: 1, unit: "s" };
  }, [result]);

  const traces = useMemo<Data[]>(() => {
    if (!result) return [];
    colors.current.release(selected);
    const t = result.t.map((v) => v * tScale.f);
    const src = quantity === "x" && result.mole_fractions ? result.mole_fractions : result.concentrations;
    return selected.map((name) => {
      const i = result.species.indexOf(name);
      const y = logY ? src[i].map((v) => (v > 0 ? v : null)) : src[i];
      return {
        x: logX ? t.slice(1) : t, y: logX ? y.slice(1) : y, name, type: "scattergl", mode: "lines",
        line: { width: 2, color: colors.current.color(name, theme) ?? theme.muted },
        hovertemplate: `${name}: %{y:.4~g}<extra></extra>`,
      } as Data;
    });
  }, [result, selected, logY, logX, quantity, theme, tScale]);

  if (!result) {
    return (
      <Card title="Ergebnisse">
        <p>Noch keine Simulation. Reaktor und Löser einstellen, dann <b>Simulieren</b> (Strg+Enter).</p>
        <button className="primary" onClick={() => void ctx.runSimulation()} disabled={!project.species.length}>▶ Simulieren</button>
        {ctx.simError && <Alert kind="error">{ctx.simError}</Alert>}
      </Card>
    );
  }

  const tAxis = { title: { text: `Zeit t [${tScale.unit}]` }, type: logX ? "log" : "linear" } as const;
  const varyingT = Math.max(...result.T) - Math.min(...result.T) > 1e-6;
  const varyingP = Math.max(...result.P) - Math.min(...result.P) > 1e-6 * Math.max(...result.P);
  const unit = quantity === "x" ? "Stoffmengenanteil" : `c [${result.concentration_unit}]`;
  const shownSpecies = result.species.filter((s) => !filter || s.toLowerCase().includes(filter.toLowerCase()));

  const downloadCsv = () => {
    const header = ["t [s]", "T [K]", "P [Pa]", ...result.species.map((s) => `${s} [${result.concentration_unit}]`)];
    const rows = result.t.map((t, k) => [t, result.T[k], result.P[k], ...result.species.map((_, i) => result.concentrations[i][k])].map((v) => String(v)).join(","));
    const url = URL.createObjectURL(new Blob([[header.join(","), ...rows].join("\n")], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${project.name.replace(/\W+/g, "_")}_ergebnis.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="stack">
      {!result.success && <Alert kind="error">Integration nicht erfolgreich: {result.message}</Alert>}
      <Metrics r={result} />
      <Card
        title="Konzentrationsverlauf"
        actions={
          <div className="inline">
            {result.mole_fractions && (
              <select value={quantity} aria-label="Größe" onChange={(e) => setQuantity(e.target.value as "c" | "x")}>
                <option value="c">Konzentration</option>
                <option value="x">Stoffmengenanteil</option>
              </select>
            )}
            <label className="check"><input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> log y</label>
            <label className="check"><input type="checkbox" checked={logX} onChange={(e) => setLogX(e.target.checked)} /> log t</label>
            <button className="ghost small" onClick={() => setShowTable(!showTable)}>{showTable ? "Diagramm" : "Tabelle"}</button>
            <button className="ghost small" onClick={downloadCsv}>CSV</button>
          </div>
        }
      >
        <div className="species-picker" role="group" aria-label="Spezies auswählen">
          {result.species.length > 12 && <input type="search" placeholder="Spezies suchen …" value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Spezies suchen" />}
          {shownSpecies.map((s) => {
            const on = selected.includes(s);
            const disabled = !on && selected.length >= MAX_SERIES;
            const color = on ? colors.current.color(s, theme) : undefined;
            return (
              <label key={s} className={`chip ${on ? "on" : ""} ${disabled ? "disabled" : ""}`} title={disabled ? `Höchstens ${MAX_SERIES} Kurven gleichzeitig` : undefined}>
                <input type="checkbox" checked={on} disabled={disabled} onChange={() => setSelected(on ? selected.filter((x) => x !== s) : [...selected, s])} />
                {on && <span className="swatch" style={{ background: color ?? undefined }} aria-hidden="true" />}
                {s}
              </label>
            );
          })}
        </div>
        {showTable ? (
          <DataTable r={result} species={selected} tScale={tScale} />
        ) : (
          <Plot ariaLabel="Konzentrationen über der Zeit" data={traces} layout={{ xaxis: tAxis, yaxis: { title: { text: unit }, type: logY ? "log" : "linear", exponentformat: "power" } }} />
        )}
      </Card>
      {varyingT && (
        <Card title="Temperatur">
          <Plot ariaLabel="Temperatur über der Zeit" height={260}
            data={[{ x: result.t.map((v) => v * tScale.f), y: result.T, type: "scattergl", mode: "lines", name: "T", line: { width: 2, color: theme.series[1] }, hovertemplate: "T = %{y:.2f} K<extra></extra>" }]}
            layout={{ xaxis: { ...tAxis }, yaxis: { title: { text: "T [K]" } }, showlegend: false }} />
        </Card>
      )}
      {varyingP && (
        <Card title="Druck">
          <Plot ariaLabel="Druck über der Zeit" height={240}
            data={[{ x: result.t.map((v) => v * tScale.f), y: result.P.map((p) => p / 1e5), type: "scattergl", mode: "lines", name: "P", line: { width: 2, color: theme.series[0] }, hovertemplate: "p = %{y:.4f} bar<extra></extra>" }]}
            layout={{ xaxis: { ...tAxis }, yaxis: { title: { text: "p [bar]" } }, showlegend: false }} />
        </Card>
      )}
      {result.heat_release_rate && (
        <Card title="Wärmefreisetzungsrate">
          <Plot ariaLabel="Wärmefreisetzungsrate" height={240}
            data={[{ x: result.t.map((v) => v * tScale.f), y: result.heat_release_rate.map((q) => q / 1e6), type: "scattergl", mode: "lines", name: "q", line: { width: 2, color: theme.series[7] }, hovertemplate: "%{y:.4~g} MW/m³<extra></extra>" }]}
            layout={{ xaxis: { ...tAxis }, yaxis: { title: { text: "q̇ [MW/m³]" } }, showlegend: false }} />
        </Card>
      )}
    </div>
  );
}

function Metrics({ r }: { r: SimulationResult }) {
  const d = r.diagnostics as Record<string, any>;
  const tiles: { label: string; value: string; note?: string }[] = [];
  if (r.metrics.ignition_delay !== undefined) tiles.push({ label: "Zündverzug (max. dT/dt)", value: fmtTime(r.metrics.ignition_delay) });
  if (r.metrics.T_final !== undefined) tiles.push({ label: "Endtemperatur", value: `${fmtNum(r.metrics.T_final, 5)} K`, note: `ΔT = ${fmtNum(r.metrics.T_rise ?? 0, 4)} K` });
  for (const [s, v] of Object.entries(r.metrics.half_life ?? {}).slice(0, 4)) tiles.push({ label: `Halbwertszeit ${s}`, value: fmtTime(v) });
  for (const [s, v] of Object.entries(r.metrics.conversion ?? {}).slice(0, 4)) tiles.push({ label: `Umsatz ${s}`, value: `${fmtNum(v * 100, 4)} %` });
  for (const [s, v] of Object.entries(r.metrics.maxima ?? {}).slice(0, 3)) tiles.push({ label: `Maximum ${s}`, value: `${fmtNum(v.value, 4)} ${r.concentration_unit}`, note: `bei t = ${fmtTime(v.t)}` });
  return (
    <div className="stack">
      <div className="tiles">
        {tiles.map((t) => (
          <div className="tile" key={t.label}>
            <div className="tile-label">{t.label}</div>
            <div className="tile-value">{t.value}</div>
            {t.note && <div className="tile-note">{t.note}</div>}
          </div>
        ))}
      </div>
      <details className="sub">
        <summary>Numerik & Erhaltungsgrößen – {r.method}, {fmtTime(r.wall_time)} Rechenzeit</summary>
        <table className="grid compact">
          <tbody>
            {Object.entries(r.stats).map(([k, v]) => <tr key={k}><td>{STAT_LABEL[k] ?? k}</td><td className="num">{v}</td></tr>)}
            {d.element_conservation_max !== undefined && <tr><td>Elementerhaltung (max. rel. Abweichung)</td><td className="num">{fmtNum(d.element_conservation_max, 3)}</td></tr>}
            {d.mass_conservation !== undefined && <tr><td>Massenerhaltung (rel. Abweichung)</td><td className="num">{fmtNum(d.mass_conservation, 3)}</td></tr>}
            <tr><td>Kleinster Wert / Maximum (negativ = Unterschwinger)</td><td className="num">{fmtNum(d.min_concentration_relative, 3)}</td></tr>
          </tbody>
        </table>
        {d.negative_concentrations && <Alert kind="warning">Deutlich negative Konzentrationen – Toleranzen verschärfen oder steifen Löser wählen.</Alert>}
      </details>
    </div>
  );
}

const STAT_LABEL: Record<string, string> = {
  rhs: "Auswertungen rechte Seite", jacobian: "Jacobi-Matrizen", lu: "LU-Zerlegungen", steps: "Zeitschritte", segments: "Integrationsabschnitte",
  accepted: "akzeptierte Schritte", rejected: "verworfene Schritte", newton_iterations: "Newton-Iterationen", newton_failures: "Newton-Fehlschläge",
};

function DataTable({ r, species, tScale }: { r: SimulationResult; species: string[]; tScale: { f: number; unit: string } }) {
  const step = Math.max(1, Math.floor(r.t.length / 200));
  const idx = r.t.map((_, k) => k).filter((k) => k % step === 0 || k === r.t.length - 1);
  return (
    <div className="table-wrap tall">
      <table className="grid compact">
        <thead>
          <tr><th className="num">t [{tScale.unit}]</th><th className="num">T [K]</th>{species.map((s) => <th key={s} className="num">{s}</th>)}</tr>
        </thead>
        <tbody>
          {idx.map((k) => (
            <tr key={k}>
              <td className="num">{fmtNum(r.t[k] * tScale.f, 5)}</td>
              <td className="num">{fmtNum(r.T[k], 6)}</td>
              {species.map((s) => <td key={s} className="num">{fmtNum(r.concentrations[r.species.indexOf(s)][k], 5)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
