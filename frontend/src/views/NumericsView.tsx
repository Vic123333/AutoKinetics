import type { Data } from "plotly.js";
import { useMemo, useState } from "react";
import { mechanismPayload, type AppContext } from "../App";
import { api } from "../lib/api";
import { chartTheme } from "../lib/palette";
import { reactorPayload, simulationPayload } from "../lib/project";
import { Plot } from "../components/Plot";
import { useDarkMode } from "../components/theme";
import { Alert, Card, Field, Icon, NumInput, Spinner, fmtNum, fmtTime, useElapsed } from "../components/ui";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Report = any;

export function NumericsView({ ctx }: { ctx: AppContext }) {
  const [budget, setBudget] = useState(20);
  const [target, setTarget] = useState(1e-4);
  const [report, setReport] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const elapsed = useElapsed(busy);
  const run = async () => {
    setBusy(true);
    setErr(null);
    try {
      setReport(await api.post("/api/numerics/analyze", {
        mechanism: mechanismPayload(ctx.project), reactor: reactorPayload(ctx.project), simulation: simulationPayload(ctx.project),
        budget_s: budget, target_rtol: target,
      }));
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const label = (id: string) => (ctx.methods.methods.find((m) => m.id === id)?.label_de as string | undefined) ?? id;
  return (
    <div className="stack">
      <Card title="Automatische Löseroptimierung">
        <p className="muted">
          Analysiert das aktuelle Problem: <b>Stabilität</b> (Spektrum der Jacobi-Matrix entlang der Lösung, Steifheit, Stabilitätsgebiete),
          <b> Konsistenz</b> (Ordnungsbedingungen, empirische lokale Fehlerordnung), <b>Konvergenz</b> (globale Fehlerordnung,
          Toleranzproportionalität, Newton-Kontraktion) und wählt Verfahren, Gleichungslöser und Toleranz mit dem geringsten Aufwand
          für die gewünschte <i>globale</i> Genauigkeit.
        </p>
        <div className="row">
          <Field label="Globale Zielgenauigkeit (relativ)" hint="z. B. 1e-4 = 0,01 %"><NumInput value={target} min={1e-12} onChange={setTarget} /></Field>
          <Field label="Zeitbudget [s]"><NumInput value={budget} min={1} onChange={setBudget} /></Field>
          <button className="primary" disabled={busy || !ctx.project.species.length} onClick={() => void run()}>Analyse starten</button>
          {busy && <Spinner label={`analysiert … ${elapsed.toFixed(0)} s`} />}
        </div>
        {err && <Alert kind="error">{err}</Alert>}
      </Card>
      {report && <ReportView ctx={ctx} r={report} label={label} />}
    </div>
  );
}

function ReportView({ ctx, r, label }: { ctx: AppContext; r: Report; label: (id: string) => string }) {
  const theme = chartTheme(useDarkMode());
  const rec = r.recommendation;
  const apply = () => {
    ctx.update((p) => {
      p.simulation.method = rec.options.method;
      p.simulation.rtol = Number(rec.options.rtol);
      if (rec.options.linear_solver) p.simulation.linear_solver = rec.options.linear_solver;
      if (rec.options.nonlinear_solver) p.simulation.nonlinear_solver = rec.options.nonlinear_solver;
      p.simulation.fixed_step = null;
    });
    ctx.notify(`Empfehlung übernommen: ${label(rec.options.method)}, rtol = ${fmtNum(rec.options.rtol, 2)}`, "success");
  };
  const stab = r.stability ?? {};
  return (
    <>
      <Card title="Empfehlung" className={rec.fallback ? "card-error" : ""}>
        <div className="tiles">
          <div className="tile"><div className="tile-label">Verfahren</div><div className="tile-value small">{label(rec.options.method)}</div></div>
          <div className="tile"><div className="tile-label">rtol (lokal)</div><div className="tile-value">{fmtNum(rec.options.rtol, 2)}</div></div>
          {rec.wall_time !== undefined && <div className="tile"><div className="tile-label">Rechenzeit (Pilotintervall)</div><div className="tile-value">{fmtTime(rec.wall_time)}</div></div>}
          {rec.error !== undefined && <div className="tile"><div className="tile-label">Fehler / Ziel</div><div className="tile-value">{fmtNum(rec.error / r.problem.error_target_factor, 3)}</div><div className="tile-note">≤ 1 erfüllt die Zielgenauigkeit</div></div>}
          <div className="tile"><div className="tile-label">Problem</div><div className="tile-value small">{stab.stiff ? "steif" : "nicht steif"}</div><div className="tile-note">Steifheitsverhältnis {fmtNum(stab.stiffness_ratio_max, 2)}</div></div>
        </div>
        <p>{rec.reasoning_de}</p>
        <div className="row"><button className="primary" onClick={apply}>Empfehlung übernehmen</button>
          <button className="ghost" onClick={() => { apply(); void ctx.runSimulation(); }}>Übernehmen & simulieren</button></div>
        {rec.fallback && <Alert kind="warning">Keine Konfiguration erreichte alle Kriterien – Rückfall auf ein robustes Standardverfahren.</Alert>}
        <details className="sub"><summary>Begründung Schritt für Schritt</summary><ol>{(r.reasoning_de ?? []).map((t: string, k: number) => <li key={k}>{t}</li>)}</ol></details>
        <p className="muted">Analysezeit {fmtTime(r.budget.elapsed_s)} von {fmtTime(r.budget.budget_s)} Budget · Referenz: {label(r.reference.method)} bei rtol {fmtNum(r.reference.rtol, 2)}{r.reference.pilot_t_end && r.reference.pilot_t_end < r.problem.t_span[1] ? ` (Pilotintervall bis t = ${fmtTime(r.reference.pilot_t_end)})` : ""}</p>
      </Card>
      <StabilityCard r={r} theme={theme} label={label} />
      <OrderCard r={r} theme={theme} label={label} />
      <BenchmarkCard r={r} theme={theme} label={label} />
      {r.skipped?.length > 0 && (
        <Card title="Übersprungen">
          <table className="grid compact"><tbody>{r.skipped.map((s: any, k: number) => <tr key={k}><td>{s.stage}</td><td>{label(s.item)}</td><td>{s.reason_de}</td></tr>)}</tbody></table>
        </Card>
      )}
    </>
  );
}

function StabilityCard({ r, theme, label }: { r: Report; theme: ReturnType<typeof chartTheme>; label: (id: string) => string }) {
  const stab = r.stability ?? {};
  const regions = stab.stability_regions ?? {};
  const methods = Object.keys(regions);
  const [method, setMethod] = useState<string>(methods.includes(r.recommendation.options.method) ? r.recommendation.options.method : methods[0]);
  const benchRow = (r.benchmark ?? []).find((b: any) => b.method === method && b.success);
  const limits = (stab.explicit_limits ?? []).find((l: any) => l.method === method);
  const h = limits?.h_max_stable ?? benchRow?.stats?.h_mean ?? null;
  const region = regions[method];
  const spectra = stab.spectra ?? [];
  const data = useMemo<Data[]>(() => {
    if (!region) return [];
    const out: Data[] = [{
      type: "contour", x: region.x, y: region.y, z: region.abs_R, showscale: false, hoverinfo: "skip",
      contours: { start: 0, end: 1, size: 1, coloring: "none" }, line: { color: theme.ink2, width: 2 }, name: "|R(z)| = 1",
    } as Data, {
      type: "contour", x: region.x, y: region.y, z: region.abs_R.map((row: number[]) => row.map((v) => (v <= 1 ? 1 : 0))), showscale: false,
      hoverinfo: "skip", contours: { start: 0.5, end: 1.5, size: 1, coloring: "fill" }, colorscale: [[0, "rgba(0,0,0,0)"], [1, theme.dark ? "rgba(57,135,229,0.22)" : "rgba(42,120,214,0.16)"]], line: { width: 0 }, name: "stabil",
    } as Data];
    if (h) {
      const xs: number[] = [], ys: number[] = [];
      for (const s of spectra) s.re.forEach((re: number, k: number) => { xs.push(h * re); ys.push(h * s.im[k]); });
      out.push({ type: "scatter", mode: "markers", x: xs, y: ys, name: "h·λ", marker: { size: 8, color: theme.series[1], line: { width: 1, color: theme.surface } }, hovertemplate: "h·λ = %{x:.3g} + %{y:.3g}i<extra></extra>" } as Data);
    }
    return out;
  }, [region, spectra, h, theme]);
  if (!methods.length) return null;
  const xr = region ? [Math.min(...region.x), Math.max(...region.x)] : undefined;
  const yr = region ? [Math.min(...region.y), Math.max(...region.y)] : undefined;
  const props = stab.method_properties?.[method];
  return (
    <Card title="Stabilität" actions={<select value={method} onChange={(e) => setMethod(e.target.value)} aria-label="Verfahren">{methods.map((m) => <option key={m} value={m}>{label(m)}</option>)}</select>}>
      <p>{stab.reasoning_de}</p>
      <div className="grid2">
        <div>
          <Plot ariaLabel={`Stabilitätsgebiet ${label(method)}`} height={360} data={data}
            layout={{ xaxis: { title: { text: "Re(z)" }, range: xr, zeroline: true }, yaxis: { title: { text: "Im(z)" }, range: yr, scaleanchor: "x", zeroline: true }, hovermode: "closest", showlegend: true }} />
          <p className="muted">Blau: Stabilitätsgebiet |R(z)| ≤ 1. Punkte: Eigenwerte der Jacobi-Matrix entlang der Lösung, skaliert mit {limits ? "der maximal stabilen" : "der mittleren"} Schrittweite h = {h ? fmtTime(h) : "–"} (außerhalb des Bildausschnitts liegende Punkte liegen weit in der linken Halbebene).</p>
        </div>
        <div>
          <Plot ariaLabel="Spektralradius und Zerfallsraten entlang der Lösung" height={360}
            data={[
              { type: "scatter", mode: "lines+markers", x: spectra.map((s: any) => s.t), y: spectra.map((s: any) => s.spectral_radius), name: "Spektralradius ρ(J)", line: { width: 2, color: theme.series[0] }, marker: { size: 8 } },
              { type: "scatter", mode: "lines+markers", x: spectra.map((s: any) => s.t), y: spectra.map((s: any) => Math.max(s.max_decay_rate, 1e-300)), name: "max. Zerfallsrate", line: { width: 2, color: theme.series[1] }, marker: { size: 8 } },
            ] as Data[]}
            layout={{ xaxis: { title: { text: "t [s]" } }, yaxis: { title: { text: "Rate [1/s]" }, type: "log", exponentformat: "power" } }} />
          <table className="grid compact">
            <tbody>
              <tr><td>A-stabil</td><td>{props?.a_stable ? "ja" : "nein"}</td></tr>
              <tr><td>L-stabil</td><td>{props?.l_stable ? "ja" : "nein"}</td></tr>
              <tr><td>|R(∞)|</td><td className="num">{fmtNum(props?.r_infinity, 3)}</td></tr>
              {limits && <tr><td>max. stabile Schrittweite</td><td className="num">{fmtTime(limits.h_max_stable)} (≈ {fmtNum(limits.estimated_steps, 2)} Schritte)</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
    </Card>
  );
}

function OrderCard({ r, theme, label }: { r: Report; theme: ReturnType<typeof chartTheme>; label: (id: string) => string }) {
  const methods = Object.keys(r.consistency ?? {});
  const [method, setMethod] = useState(methods.includes(r.recommendation.options.method) ? r.recommendation.options.method : methods[0]);
  if (!methods.length) return null;
  const c = r.consistency[method];
  const g = r.convergence?.[method];
  const study = (s: any, color: string, name: string): Data[] => {
    if (!s) return [];
    const pts = s.h.map((h: number, k: number) => [h, s.errors[k], s.used[k]]).filter((p: any) => p[1] !== null && p[1] > 0);
    const out: Data[] = [{ type: "scatter", mode: "markers", x: pts.map((p: any) => p[0]), y: pts.map((p: any) => p[1]), name, marker: { size: 9, color, symbol: pts.map((p: any) => (p[2] ? "circle" : "circle-open")) }, hovertemplate: "h = %{x:.3g}<br>Fehler = %{y:.3g}<extra></extra>" } as Data];
    if (s.fitted && s.slope !== null) out.push({ type: "scatter", mode: "lines", x: s.h, y: s.fitted, name: `${name}: Steigung ${s.slope.toFixed(2)}`, line: { width: 2, color, dash: "dot" }, hoverinfo: "skip" } as Data);
    return out;
  };
  const tol = g?.tolerance;
  return (
    <Card title="Konsistenz & Konvergenz" actions={<select value={method} onChange={(e) => setMethod(e.target.value)} aria-label="Verfahren">{methods.map((m) => <option key={m} value={m}>{label(m)}</option>)}</select>}>
      <div className="grid2">
        <div>
          <Plot ariaLabel="Fehler über Schrittweite" height={320} data={[...study(c.local, theme.series[0], "lokaler Fehler"), ...study(g?.global, theme.series[1], "globaler Fehler")]}
            layout={{ xaxis: { title: { text: "Schrittweite h" }, type: "log", exponentformat: "power" }, yaxis: { title: { text: "Fehler (skaliert)" }, type: "log", exponentformat: "power" }, hovermode: "closest" }} />
          <p className="muted">Lokaler Fehler ∝ h^(p+1) (Konsistenzordnung p), globaler Fehler ∝ h^p (Konvergenzordnung). Hohle Punkte: nicht in den Fit einbezogen (Rundungs-/Referenzgenauigkeit).</p>
        </div>
        <div>
          <table className="grid compact">
            <tbody>
              <tr><td>Theoretische Ordnung</td><td className="num">{c.theoretical_order}</td></tr>
              <tr><td>Algebraisch (Ordnungsbedingungen)</td><td className="num">{c.algebraic_order}</td></tr>
              <tr><td>Empirische Konsistenzordnung</td><td className="num">{fmtNum(c.empirical_order, 3)}</td></tr>
              <tr><td>Empirische Konvergenzordnung</td><td className="num">{fmtNum(g?.empirical_order, 3)}</td></tr>
              <tr><td>Toleranzproportionalität (Steigung)</td><td className="num">{fmtNum(tol?.slope, 3)}</td></tr>
              {g?.newton && <tr><td>Newton: Iterationen/Schritt, ⌀ Kontraktion θ</td><td className="num">{fmtNum(g.newton.iterations_per_step, 3)} / {fmtNum(g.newton.theta_mean, 2)}</td></tr>}
            </tbody>
          </table>
          <p><Icon kind={c.consistent === false ? "error" : c.consistent ? "success" : "info"} /> {c.reasoning_de}</p>
          {g && <p><Icon kind={g.convergent === false ? "error" : g.convergent ? "success" : "info"} /> {g.reasoning_de}</p>}
          {tol && <p className="muted">{tol.reasoning_de}</p>}
          {g?.newton?.reasoning_de && <p className="muted">{g.newton.reasoning_de}</p>}
        </div>
      </div>
    </Card>
  );
}

function BenchmarkCard({ r, theme, label }: { r: Report; theme: ReturnType<typeof chartTheme>; label: (id: string) => string }) {
  const rows = [...(r.benchmark ?? [])].sort((a: any, b: any) => a.wall_time - b.wall_time);
  const best = r.recommendation.options;
  return (
    <Card title={`Vergleich (${rows.length} Konfigurationen)`}>
      <Plot ariaLabel="Rechenzeit gegen Fehler" height={340}
        data={[true, false].map((valid) => {
          const sel = rows.filter((b: any) => b.success && b.valid === valid && b.error !== null);
          return { type: "scatter", mode: "markers", name: valid ? "gültig" : "ungültig", x: sel.map((b: any) => b.wall_time * 1e3), y: sel.map((b: any) => b.error),
            text: sel.map((b: any) => `${label(b.method)} · rtol ${fmtNum(b.options.rtol, 2)} · ${b.nonlinear_solver}/${b.linear_solver}`),
            marker: { size: 10, color: valid ? theme.series[0] : theme.muted, symbol: valid ? "circle" : "x" }, hovertemplate: "%{text}<br>%{x:.3g} ms, Fehler %{y:.3g}<extra></extra>" } as Data;
        })}
        layout={{ xaxis: { title: { text: "Rechenzeit [ms]" }, type: "log" }, yaxis: { title: { text: "Fehler (Vielfache der Zielgenauigkeit)" }, type: "log", exponentformat: "power" }, hovermode: "closest",
          shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: r.problem.error_target_factor, y1: r.problem.error_target_factor, line: { color: theme.ink2, dash: "dash", width: 1 } }] }} />
      <p className="muted">Gestrichelt: Genauigkeitsgrenze. Gesucht ist der Punkt unterhalb der Linie mit der kleinsten Rechenzeit (Arbeit-Genauigkeits-Diagramm).</p>
      <div className="table-wrap tall">
        <table className="grid compact">
          <thead><tr><th>Verfahren</th><th>Nichtlinear / Linear</th><th className="num">rtol</th><th className="num">Zeit</th><th className="num">Fehler</th><th>gültig</th><th>Begründung</th></tr></thead>
          <tbody>{rows.map((b: any, k: number) => {
            const isBest = b.method === best.method && b.options.rtol === best.rtol && b.linear_solver === best.linear_solver && b.nonlinear_solver === best.nonlinear_solver;
            return (
              <tr key={k} className={isBest ? "best" : ""}>
                <td>{label(b.method)}</td><td>{b.nonlinear_solver} / {b.linear_solver}</td><td className="num">{fmtNum(b.options.rtol, 2)}</td>
                <td className="num">{fmtTime(b.wall_time)}</td><td className="num">{fmtNum(b.error, 3)}</td>
                <td><Icon kind={b.valid ? "success" : "error"} /><span className="sr-only">{b.valid ? "gültig" : "ungültig"}</span></td>
                <td>{b.reasoning_de}</td>
              </tr>
            );
          })}</tbody>
        </table>
      </div>
    </Card>
  );
}
