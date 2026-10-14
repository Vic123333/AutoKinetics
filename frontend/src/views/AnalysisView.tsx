import type { Data } from "plotly.js";
import { useState } from "react";
import { mechanismPayload, type AppContext } from "../App";
import { api } from "../lib/api";
import { chartTheme } from "../lib/palette";
import { reactorPayload, simulationPayload } from "../lib/project";
import { Plot } from "../components/Plot";
import { useDarkMode } from "../components/theme";
import { Alert, Card, Field, NumInput, Spinner, fmtNum, useElapsed } from "../components/ui";

type Sub = "eq" | "sens" | "rop" | "fit" | "order";
const SUBS: [Sub, string][] = [["eq", "Gleichgewicht"], ["sens", "Sensitivität"], ["rop", "Reaktionsflüsse"], ["fit", "Parameter anpassen"], ["order", "Ordnung & Arrhenius"]];

export function AnalysisView({ ctx }: { ctx: AppContext }) {
  const [sub, setSub] = useState<Sub>("eq");
  return (
    <div className="stack">
      <div className="segmented" role="tablist" aria-label="Analyse">
        {SUBS.map(([id, label]) => <button key={id} role="tab" aria-selected={sub === id} className={sub === id ? "active" : ""} onClick={() => setSub(id)}>{label}</button>)}
      </div>
      {sub === "eq" && <Equilibrium ctx={ctx} />}
      {sub === "sens" && <Sensitivity ctx={ctx} />}
      {sub === "rop" && <Rop ctx={ctx} />}
      {sub === "fit" && <Fit ctx={ctx} />}
      {sub === "order" && <OrderArrhenius />}
    </div>
  );
}

function useJob<T>() {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const elapsed = useElapsed(busy);
  const run = async (path: string, body: unknown) => {
    setBusy(true);
    setError(null);
    try {
      setData(await api.post<T>(path, body));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return { data, error, busy, elapsed, run };
}

function base(ctx: AppContext) {
  return { mechanism: mechanismPayload(ctx.project), reactor: reactorPayload(ctx.project), simulation: simulationPayload(ctx.project) };
}

function Equilibrium({ ctx }: { ctx: AppContext }) {
  const gas = ctx.project.phase === "ideal-gas";
  const [mode, setMode] = useState(gas ? "TP" : "TV");
  const job = useJob<any>();
  const sim = ctx.result;
  return (
    <Card title="Thermodynamisches Gleichgewicht">
      <p className="muted">Gibbs-/Helmholtz-Minimierung (Elementpotential-Methode). Bei reversibler, thermodynamisch konsistenter Kinetik ist das der Endzustand jeder Simulation.</p>
      <div className="row">
        <Field label="Randbedingungen">
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            {gas && <option value="TP">T, p konstant</option>}
            <option value="TV">T, V konstant</option>
            {gas && <option value="HP">H, p konstant (adiabate Flammentemperatur)</option>}
            <option value="UV">U, V konstant</option>
          </select>
        </Field>
        <button className="primary" disabled={job.busy} onClick={() => void job.run("/api/equilibrium", { mechanism: mechanismPayload(ctx.project), reactor: reactorPayload(ctx.project), mode, concentration_unit: "mol/L" })}>Berechnen</button>
        {job.busy && <Spinner label="rechnet …" />}
      </div>
      {job.error && <Alert kind="error">{job.error}</Alert>}
      {job.data && (
        <>
          <div className="tiles">
            <div className="tile"><div className="tile-label">Temperatur</div><div className="tile-value">{fmtNum(job.data.T, 6)} K</div></div>
            {gas && <div className="tile"><div className="tile-label">Druck</div><div className="tile-value">{fmtNum(job.data.P / 1e5, 5)} bar</div></div>}
            <div className="tile"><div className="tile-label">Bilanzbasis</div><div className="tile-value small">{job.data.basis === "elements" ? "Elemente" : "Reaktionsnetzwerk"}</div></div>
          </div>
          {job.data.notes?.map((n: string) => <Alert key={n} kind="info">{n}</Alert>)}
          <div className="table-wrap tall">
            <table className="grid compact">
              <thead><tr><th>Spezies</th><th className="num">Start [mol/L]</th><th className="num">Gleichgewicht [mol/L]</th>{gas && <th className="num">x_eq</th>}{sim && <th className="num">Simulation Ende [{sim.concentration_unit}]</th>}</tr></thead>
              <tbody>
                {job.data.species.map((s: string, i: number) => (job.data.concentrations[i] > 1e-14 * Math.max(...job.data.concentrations) || job.data.initial_concentrations[i] > 0) && (
                  <tr key={s}>
                    <td>{s}</td>
                    <td className="num">{fmtNum(job.data.initial_concentrations[i], 5)}</td>
                    <td className="num">{fmtNum(job.data.concentrations[i], 6)}</td>
                    {gas && <td className="num">{fmtNum(job.data.mole_fractions[i], 5)}</td>}
                    {sim && <td className="num">{fmtNum(sim.concentrations[sim.species.indexOf(s)]?.at(-1), 6)}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}

function targetOptions(ctx: AppContext): [string, string][] {
  const out: [string, string][] = [];
  if (ctx.project.reactor.energy !== "isothermal") out.push(["ignition_delay", "Zündverzugszeit"], ["T_final", "Endtemperatur"], ["T_max", "Maximaltemperatur"]);
  for (const s of ctx.project.species.map((x) => x.name)) {
    out.push([`final:${s}`, `[${s}] am Ende`], [`max:${s}`, `max. [${s}]`]);
    if ((ctx.project.reactor.composition[s] ?? 0) > 0) out.push([`half_life:${s}`, `Halbwertszeit ${s}`], [`conversion:${s}`, `Umsatz ${s}`]);
  }
  return out;
}

function Sensitivity({ ctx }: { ctx: AppContext }) {
  const opts = targetOptions(ctx);
  const [target, setTarget] = useState(opts[0]?.[0] ?? "");
  const [maxR, setMaxR] = useState(Math.min(25, ctx.project.reactions.length || 1));
  const job = useJob<any>();
  const theme = chartTheme(useDarkMode());
  const res = (job.data?.results ?? []).filter((r: any) => Number.isFinite(r.sensitivity)).slice(0, 20).reverse();
  return (
    <Card title="Lokale Sensitivitätsanalyse">
      <p className="muted">Normierte Sensitivität S = ∂ ln(Zielgröße) / ∂ ln(Aⱼ) per zentraler Differenz. Ohne Auswahl werden die Reaktionen mit dem größten Umsatz untersucht.</p>
      <div className="row">
        <Field label="Zielgröße">
          <select value={target} onChange={(e) => setTarget(e.target.value)}>{opts.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
        </Field>
        <Field label="Reaktionen (max.)"><NumInput value={maxR} min={1} onChange={(v) => setMaxR(Math.round(v))} /></Field>
        <button className="primary" disabled={job.busy || !target} onClick={() => void job.run("/api/analysis/sensitivity", { ...base(ctx), target, max_reactions: maxR })}>Berechnen</button>
        {job.busy && <Spinner label={`${2 * maxR + 1} Simulationen … ${job.elapsed.toFixed(0)} s`} />}
      </div>
      {job.error && <Alert kind="error">{job.error}</Alert>}
      {job.data && (
        <>
          <p>Zielgröße: <b>{fmtNum(job.data.value, 5)}</b>{job.data.skipped_due_to_budget ? ` – ${job.data.skipped_due_to_budget} Reaktionen wegen Zeitlimit übersprungen` : ""}</p>
          <Plot ariaLabel="Sensitivitätskoeffizienten" height={Math.max(220, 26 * res.length + 60)}
            data={[{ type: "bar", orientation: "h", x: res.map((r: any) => r.sensitivity), y: res.map((r: any) => `R${r.index + 1}: ${r.equation}`),
              marker: { color: res.map((r: any) => (r.sensitivity >= 0 ? theme.diverging[1] : theme.diverging[0])) }, hovertemplate: "%{y}<br>S = %{x:.4f}<extra></extra>" } as Data]}
            layout={{ margin: { l: 280 }, xaxis: { title: { text: "Sensitivität S" }, zeroline: true }, yaxis: { automargin: true }, showlegend: false, hovermode: "closest" }} />
          <p className="muted">Blau: Erhöhung von Aⱼ verkleinert die Zielgröße · Rot: vergrößert sie.</p>
        </>
      )}
    </Card>
  );
}

function Rop({ ctx }: { ctx: AppContext }) {
  const [species, setSpecies] = useState(ctx.project.species[0]?.name ?? "");
  const job = useJob<any>();
  const theme = chartTheme(useDarkMode());
  const rx = (job.data?.reactions ?? []).slice(0, 8);
  return (
    <Card title="Reaktionsflussanalyse (Rate of Production)">
      <div className="row">
        <Field label="Spezies">
          <select value={species} onChange={(e) => setSpecies(e.target.value)}>{ctx.project.species.map((s) => <option key={s.key}>{s.name}</option>)}</select>
        </Field>
        <button className="primary" disabled={job.busy || !species} onClick={() => void job.run("/api/analysis/rop", { ...base(ctx), species, top: 12 })}>Berechnen</button>
        {job.busy && <Spinner label="rechnet …" />}
      </div>
      {job.error && <Alert kind="error">{job.error}</Alert>}
      {job.data && (
        <>
          <Plot ariaLabel={`Beiträge zur Bildung von ${species}`}
            data={rx.map((r: any, k: number) => ({ x: job.data.t, y: r.rate, name: `R${r.index + 1}: ${r.equation}`, type: "scattergl", mode: "lines", line: { width: 2, color: theme.series[k] } }) as Data)}
            layout={{ xaxis: { title: { text: "t [s]" } }, yaxis: { title: { text: `ν·q [mol/(m³·s)] für ${species}` } } }} />
          <table className="grid compact">
            <thead><tr><th>Reaktion</th><th className="num">integrierter Beitrag [mol/m³]</th><th>Rolle</th></tr></thead>
            <tbody>{job.data.reactions.map((r: any) => (
              <tr key={r.index}><td>R{r.index + 1}: {r.equation}</td><td className="num">{fmtNum(r.integral, 4)}</td><td>{r.integral >= 0 ? "Bildung" : "Verbrauch"}</td></tr>
            ))}</tbody>
          </table>
        </>
      )}
    </Card>
  );
}

function parseTable(text: string): number[][] {
  return text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !/^[#a-zA-Z]/.test(l))
    .map((l) => l.split(/[;\t ,]+/).map((x) => Number(x.replace(",", ".")))).filter((r) => r.every(Number.isFinite));
}

function Fit({ ctx }: { ctx: AppContext }) {
  const { project } = ctx;
  const [species, setSpecies] = useState(project.species[0]?.name ?? "");
  const [text, setText] = useState("# t [s]   c [mol/L]\n");
  const [params, setParams] = useState<{ reaction: number; param: "A" | "Ea" }[]>([]);
  const job = useJob<any>();
  const theme = chartTheme(useDarkMode());
  const rows = parseTable(text);
  const toggle = (reaction: number, param: "A" | "Ea") =>
    setParams((ps) => (ps.some((p) => p.reaction === reaction && p.param === param) ? ps.filter((p) => !(p.reaction === reaction && p.param === param)) : [...ps, { reaction, param }]));
  return (
    <Card title="Parameteranpassung an Messdaten">
      <p className="muted">Gauß-Newton/Trust-Region-Ausgleichsrechnung; A wird logarithmisch angepasst. Standardfehler aus der Kovarianzmatrix (JᵀJ)⁻¹·s².</p>
      <div className="row top">
        <div className="stack grow">
          <Field label="Gemessene Spezies">
            <select value={species} onChange={(e) => setSpecies(e.target.value)}>{project.species.map((s) => <option key={s.key}>{s.name}</option>)}</select>
          </Field>
          <Field label={`Messdaten: t [s] und c [mol/L] je Zeile (${rows.length} Punkte erkannt)`} wide>
            <textarea className="code" rows={10} value={text} onChange={(e) => setText(e.target.value)} />
          </Field>
          {ctx.result && (
            <button className="ghost small" onClick={() => {
              const r = ctx.result!;
              const i = r.species.indexOf(species);
              const idx = [...Array(12).keys()].map((k) => Math.round(((k + 1) * (r.t.length - 1)) / 12));
              setText("# synthetisch aus letzter Simulation (+1 % Rauschen)\n" + idx.map((k) => `${r.t[k].toPrecision(6)}\t${(r.concentrations[i][k] * (1 + 0.01 * (Math.random() * 2 - 1))).toPrecision(6)}`).join("\n"));
            }}>Testdaten aus letzter Simulation erzeugen</button>
          )}
        </div>
        <div className="stack grow">
          <span className="field-label">Anzupassende Parameter</span>
          <div className="table-wrap tall">
            <table className="grid compact">
              <thead><tr><th>Reaktion</th><th>A</th><th>Eₐ</th></tr></thead>
              <tbody>{project.reactions.map((r, j) => (
                <tr key={r.key}><td>R{j + 1}: {r.equation}</td>
                  <td><input type="checkbox" aria-label={`A von R${j + 1}`} checked={params.some((p) => p.reaction === j && p.param === "A")} onChange={() => toggle(j, "A")} /></td>
                  <td><input type="checkbox" aria-label={`Ea von R${j + 1}`} checked={params.some((p) => p.reaction === j && p.param === "Ea")} onChange={() => toggle(j, "Ea")} /></td></tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      </div>
      <div className="row">
        <button className="primary" disabled={job.busy || rows.length < 2 || !params.length} onClick={() => void job.run("/api/analysis/fit", {
          ...base(ctx), concentration_unit: "mol/L", parameters: params,
          data: [{ species, t: rows.map((r) => r[0]), c: rows.map((r) => r[1]) }],
        })}>Anpassen</button>
        {job.busy && <Spinner label={`optimiert … ${job.elapsed.toFixed(0)} s`} />}
      </div>
      {job.error && <Alert kind="error">{job.error}</Alert>}
      {job.data && (
        <>
          <Alert kind={job.data.success ? "success" : "warning"}>{job.data.success ? "Konvergiert" : "Nicht konvergiert"} nach {job.data.n_simulations} Simulationen · χ²_red = {fmtNum(job.data.chi2_reduced, 3)}</Alert>
          <table className="grid compact">
            <thead><tr><th>Parameter</th><th className="num">Schätzwert</th><th className="num">Unsicherheit (1σ)</th></tr></thead>
            <tbody>{job.data.estimates.map((e: any) => (
              <tr key={`${e.reaction}${e.param}`}>
                <td>{e.param} von R{e.reaction + 1}</td>
                <td className="num">{fmtNum(e.value, 5)} {e.param === "Ea" ? "kJ/mol" : "(SI)"}</td>
                <td className="num">{e.param === "A" ? `± ${fmtNum(e.rel_std_error * 100, 3)} %` : `± ${fmtNum(e.std_error, 3)} kJ/mol`}</td>
              </tr>
            ))}</tbody>
          </table>
          {job.data.fit.map((s: any) => (
            <Plot key={s.species} ariaLabel={`Anpassung ${s.species}`} height={300} data={[
              { x: s.t, y: s.measured, name: "Messung", type: "scatter", mode: "markers", marker: { size: 8, color: theme.series[1] } },
              { x: s.t, y: s.model, name: "Modell", type: "scatter", mode: "lines", line: { width: 2, color: theme.series[0] } },
            ] as Data[]} layout={{ xaxis: { title: { text: "t [s]" } }, yaxis: { title: { text: `[${s.species}] [mol/L]` } } }} />
          ))}
          <p className="muted">Hinweis: Ein angepasster Wert gilt im Projekt erst nach manueller Übernahme in die Reaktionstabelle (A in SI-Einheiten mol, m³, s).</p>
        </>
      )}
    </Card>
  );
}

function OrderArrhenius() {
  const [ct, setCt] = useState("# t [s]  c [mol/L]\n0 1.0\n10 0.74\n20 0.55\n30 0.41\n40 0.30\n60 0.17");
  const [tk, setTk] = useState("# T [K]  k\n290 1.2e-4\n300 3.1e-4\n310 7.6e-4\n320 1.75e-3\n330 3.9e-3");
  const order = useJob<any>();
  const arr = useJob<any>();
  const theme = chartTheme(useDarkMode());
  const tkRows = parseTable(tk);
  return (
    <div className="grid2">
      <Card title="Reaktionsordnung aus c(t)">
        <Field label="Messreihe" wide><textarea className="code" rows={8} value={ct} onChange={(e) => setCt(e.target.value)} /></Field>
        <button className="primary" onClick={() => { const r = parseTable(ct); void order.run("/api/analysis/order", { t: r.map((x) => x[0]), c: r.map((x) => x[1]) }); }}>Ordnung bestimmen</button>
        {order.error && <Alert kind="error">{order.error}</Alert>}
        {order.data && (
          <table className="grid compact">
            <thead><tr><th>Modell</th><th className="num">n</th><th className="num">k</th><th className="num">R²</th><th className="num">AIC</th></tr></thead>
            <tbody>{order.data.fits.map((f: any, k: number) => (
              <tr key={k} className={f === order.data.best || (f.aic === order.data.best.aic) ? "best" : ""}>
                <td>{f.method}</td><td className="num">{fmtNum(f.order, 4)}</td><td className="num">{fmtNum(f.k, 4)} {f.k_unit}</td><td className="num">{fmtNum(f.r_squared, 6)}</td><td className="num">{fmtNum(f.aic, 4)}</td>
              </tr>
            ))}</tbody>
          </table>
        )}
      </Card>
      <Card title="Arrhenius-Parameter aus k(T)">
        <Field label="Messreihe" wide><textarea className="code" rows={8} value={tk} onChange={(e) => setTk(e.target.value)} /></Field>
        <button className="primary" onClick={() => void arr.run("/api/analysis/arrhenius", { T: tkRows.map((x) => x[0]), k: tkRows.map((x) => x[1]) })}>Anpassen</button>
        {arr.error && <Alert kind="error">{arr.error}</Alert>}
        {arr.data && (
          <>
            <p>Eₐ = <b>{fmtNum(arr.data.Ea / 1e3, 5)} kJ/mol</b> (95 %-KI {fmtNum(arr.data.Ea_ci95[0] / 1e3, 4)} … {fmtNum(arr.data.Ea_ci95[1] / 1e3, 4)}) · A = <b>{fmtNum(arr.data.A, 4)}</b> · R² = {fmtNum(arr.data.r_squared, 6)}</p>
            <Plot ariaLabel="Arrhenius-Auftragung" height={280} data={[
              { x: tkRows.map((r) => 1000 / r[0]), y: tkRows.map((r) => Math.log(r[1])), type: "scatter", mode: "markers", name: "Messung", marker: { size: 8, color: theme.series[1] } },
              { x: tkRows.map((r) => 1000 / r[0]), y: tkRows.map((r) => Math.log(arr.data.A) - arr.data.Ea / (8.31446261815324 * r[0])), type: "scatter", mode: "lines", name: "Ausgleichsgerade", line: { width: 2, color: theme.series[0] } },
            ] as Data[]} layout={{ xaxis: { title: { text: "1000/T [1/K]" } }, yaxis: { title: { text: "ln k" } } }} />
          </>
        )}
      </Card>
    </div>
  );
}
