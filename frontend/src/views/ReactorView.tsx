import type { AppContext } from "../App";
import type { ReactorForm, SimForm } from "../lib/types";
import { Alert, Card, Field, NumInput, Spinner, useElapsed } from "../components/ui";

export function ReactorView({ ctx }: { ctx: AppContext }) {
  const { project, update, simulating, simError } = ctx;
  const f = project.reactor;
  const gas = project.phase === "ideal-gas";
  const setR = (patch: Partial<ReactorForm>) => update((p) => void Object.assign(p.reactor, patch));
  const elapsed = useElapsed(simulating);
  const compSum = Object.values(f.composition).reduce((a, b) => a + (b > 0 ? b : 0), 0);

  return (
    <div className="stack">
      <Card title="Reaktor">
        <div className="row">
          <Field label="Reaktortyp">
            <select value={f.kind} onChange={(e) => setR({ kind: e.target.value as ReactorForm["kind"], ...(e.target.value === "cstr" ? { constraint: "volume" } : {}) })}>
              <option value="batch">Batch (geschlossen)</option>
              <option value="cstr">Rührkessel kontinuierlich (CSTR)</option>
            </select>
          </Field>
          {gas && (
            <Field label="Randbedingung">
              <select value={f.constraint} disabled={f.kind === "cstr"} onChange={(e) => setR({ constraint: e.target.value as ReactorForm["constraint"] })}>
                <option value="pressure">konstanter Druck</option>
                <option value="volume">konstantes Volumen</option>
              </select>
            </Field>
          )}
          <Field label="Energie">
            <select value={f.energy} onChange={(e) => setR({ energy: e.target.value as ReactorForm["energy"] })}>
              <option value="isothermal">isotherm / temperaturgeführt</option>
              <option value="adiabatic">adiabat (Energiebilanz)</option>
              <option value="heat-exchange">gekühlt / beheizt (UA)</option>
            </select>
          </Field>
          <Field label="Temperatur T₀ [K]" hint={`${(f.T - 273.15).toFixed(2)} °C`}>
            <NumInput value={f.T} min={1} onChange={(v) => setR({ T: v, ...(f.T_env === f.T ? { T_env: v } : {}) })} />
          </Field>
          {gas && (
            <Field label="Druck">
              <div className="inline">
                <NumInput value={f.P} min={0} onChange={(v) => setR({ P: v })} />
                <select value={f.Punit} aria-label="Druckeinheit" onChange={(e) => setR({ Punit: e.target.value as ReactorForm["Punit"] })}>
                  {["Pa", "kPa", "bar", "atm", "MPa"].map((u) => <option key={u}>{u}</option>)}
                </select>
              </div>
            </Field>
          )}
        </div>
        {f.energy === "heat-exchange" && (
          <div className="row">
            <Field label="UA/V [W/(m³·K)]" hint="Wärmedurchgang × Fläche je Volumen"><NumInput value={f.UA_V} min={0} onChange={(v) => setR({ UA_V: v })} /></Field>
            <Field label="Kühl-/Heiztemperatur [K]"><NumInput value={f.T_env} min={1} onChange={(v) => setR({ T_env: v })} /></Field>
          </div>
        )}
        {!gas && f.energy !== "isothermal" && (
          <div className="row">
            <Field label="ρ·c_p [J/(m³·K)]" hint="Wasser ≈ 4.18·10⁶"><NumInput value={f.rho_cp} min={0} onChange={(v) => setR({ rho_cp: v })} /></Field>
          </div>
        )}
        {f.kind === "cstr" && (
          <div className="row">
            <Field label="Verweilzeit τ [s]"><NumInput value={f.tau} min={0} onChange={(v) => setR({ tau: v })} /></Field>
          </div>
        )}
        {f.energy === "isothermal" && <TemperatureProgram ctx={ctx} />}
      </Card>

      <Card title={f.kind === "cstr" ? "Anfangsfüllung und Zulauf" : "Anfangszusammensetzung"}>
        <div className="row">
          {gas && (
            <Field label="Angabe als">
              <select value={f.basis} onChange={(e) => setR({ basis: e.target.value as ReactorForm["basis"] })}>
                <option value="mole-fraction">Stoffmengenanteile (werden normiert)</option>
                <option value="concentration">Konzentrationen</option>
              </select>
            </Field>
          )}
          {(f.basis === "concentration" || f.kind === "cstr") && (
            <Field label="Konzentrationseinheit">
              <select value={f.concentration_unit} onChange={(e) => setR({ concentration_unit: e.target.value as ReactorForm["concentration_unit"] })}>
                <option value="mol/L">mol/L</option>
                <option value="mmol/L">mmol/L</option>
                <option value="mol/m3">mol/m³</option>
              </select>
            </Field>
          )}
        </div>
        {f.basis === "mole-fraction" && compSum > 0 && Math.abs(compSum - 1) > 1e-9 && <p className="muted">Summe {compSum.toPrecision(5)} – wird auf 1 normiert.</p>}
        <CompositionEditor ctx={ctx} />
      </Card>

      <SolverCard ctx={ctx} />

      <div className="run-bar">
        <button className="primary big" onClick={() => void ctx.runSimulation()} disabled={simulating || !project.species.length}>▶ Simulieren</button>
        {simulating && (
          <>
            <Spinner label={`rechnet … ${elapsed.toFixed(1)} s`} />
            <button className="ghost" onClick={ctx.cancelSimulation}>Abbrechen</button>
          </>
        )}
      </div>
      {simError && <Alert kind="error">{simError}</Alert>}
    </div>
  );
}

function CompositionEditor({ ctx }: { ctx: AppContext }) {
  const { project, update } = ctx;
  const f = project.reactor;
  const used = project.species.filter((s) => (f.composition[s.name] ?? 0) > 0 || (f.feed[s.name] ?? 0) > 0 || f.fixed_species.includes(s.name) || s.name in f.composition);
  const unused = project.species.filter((s) => !used.includes(s));
  return (
    <>
      {!used.length && <Alert kind="info">Noch keine Startstoffe – unten Spezies hinzufügen.</Alert>}
      <div className="comp-grid">
        {used.map((s) => (
          <div className="comp-cell" key={s.key}>
            <span className="name" title={s.name}>{s.name}</span>
            <span className="controls">
              <NumInput value={f.composition[s.name] ?? 0} min={0} ariaLabel={`Startwert ${s.name}`} onChange={(v) => update((p) => void (p.reactor.composition[s.name] = v))} />
              {f.kind === "cstr" && <NumInput value={f.feed[s.name] ?? 0} min={0} ariaLabel={`Zulauf ${s.name}`} placeholder="Zulauf" onChange={(v) => update((p) => void (p.reactor.feed[s.name] = v))} />}
              {f.constraint === "volume" && (
                <label className="check" title="Konzentration bleibt konstant (gepuffert)">
                  <input type="checkbox" aria-label={`${s.name} gepuffert`} checked={f.fixed_species.includes(s.name)} onChange={(e) => update((p) => {
                    p.reactor.fixed_species = e.target.checked ? [...p.reactor.fixed_species, s.name] : p.reactor.fixed_species.filter((n) => n !== s.name);
                  })} />puffer
                </label>
              )}
              <button className="ghost small danger" aria-label={`${s.name} entfernen`} onClick={() => update((p) => {
                delete p.reactor.composition[s.name];
                delete p.reactor.feed[s.name];
                p.reactor.fixed_species = p.reactor.fixed_species.filter((n) => n !== s.name);
              })}>✕</button>
            </span>
          </div>
        ))}
      </div>
      {f.kind === "cstr" && <p className="muted">Je Spezies: Anfangsfüllung (links) und Zulaufkonzentration (rechts).</p>}
      {unused.length > 0 && (
        <div className="inline">
          <select aria-label="Spezies hinzufügen" value="" onChange={(e) => e.target.value && update((p) => void (p.reactor.composition[e.target.value] = 0))}>
            <option value="">+ Spezies hinzufügen …</option>
            {unused.map((s) => <option key={s.key} value={s.name}>{s.name}</option>)}
          </select>
        </div>
      )}
    </>
  );
}

function TemperatureProgram({ ctx }: { ctx: AppContext }) {
  const { project, update } = ctx;
  const prog = project.reactor.T_program;
  return (
    <details className="sub" open={prog.length > 0}>
      <summary>Temperaturprogramm (optional, stückweise linear)</summary>
      <table className="grid compact">
        <thead><tr><th className="num">t [s]</th><th className="num">T [K]</th><th /></tr></thead>
        <tbody>
          {prog.map(([t, T], k) => (
            <tr key={k}>
              <td><NumInput value={t} min={0} onChange={(v) => update((p) => void (p.reactor.T_program[k][0] = v))} /></td>
              <td><NumInput value={T} min={1} onChange={(v) => update((p) => void (p.reactor.T_program[k][1] = v))} /></td>
              <td><button className="ghost small danger" onClick={() => update((p) => void p.reactor.T_program.splice(k, 1))}>✕</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <button className="ghost small" onClick={() => update((p) => {
        const last = p.reactor.T_program[p.reactor.T_program.length - 1];
        p.reactor.T_program.push(last ? [last[0] + 10, last[1]] : [0, p.reactor.T]);
      })}>+ Stützstelle</button>
    </details>
  );
}

function SolverCard({ ctx }: { ctx: AppContext }) {
  const { project, update, methods } = ctx;
  const s = project.simulation;
  const setS = (patch: Partial<SimForm>) => update((p) => void Object.assign(p.simulation, patch));
  const list = methods.methods.length ? methods.methods : [{ id: "scipy:BDF", label_de: "SciPy BDF" }];
  const current = list.find((m) => m.id === s.method);
  const families = [...new Set(list.map((m) => String(m.family ?? "Sonstige")))];
  const implicit = current ? current.implicit !== false && String(current.family ?? "").toLowerCase() !== "explicit rk" && !String(current.id).startsWith("scipy") : false;
  const fixedOnly = current ? current.adaptive === false : false;
  return (
    <Card title="Zeit & numerischer Löser" actions={<button className="ghost small" onClick={() => ctx.goto("numerics")}>Auto-Optimierung …</button>}>
      <div className="row">
        <Field label="Simulationsdauer">
          <div className="inline">
            <NumInput value={s.t_end} min={0} onChange={(v) => setS({ t_end: v })} />
            <select value={s.t_unit} aria-label="Zeiteinheit" onChange={(e) => setS({ t_unit: e.target.value as SimForm["t_unit"] })}>
              <option value="us">µs</option><option value="ms">ms</option><option value="s">s</option><option value="min">min</option><option value="h">h</option>
            </select>
          </div>
        </Field>
        <Field label="Verfahren" hint={current?.description_de as string | undefined}>
          <select value={s.method} onChange={(e) => setS({ method: e.target.value })}>
            {families.map((fam) => (
              <optgroup key={fam} label={fam}>
                {list.filter((m) => String(m.family ?? "Sonstige") === fam).map((m) => (
                  <option key={m.id} value={m.id}>{m.label_de ?? m.id}{m.order ? ` (Ordnung ${m.order})` : ""}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </Field>
        <Field label="rtol" hint="relative Toleranz"><NumInput value={s.rtol} min={0} onChange={(v) => setS({ rtol: v })} /></Field>
        <Field label="atol (relativ zur Gesamtkonz.)"><NumInput value={s.atol_rel} min={0} onChange={(v) => setS({ atol_rel: v })} /></Field>
      </div>
      <div className="row">
        {fixedOnly && (
          <Field label="Feste Schrittweite h [s]" hint="Verfahren ohne Schrittweitensteuerung">
            <NumInput value={s.fixed_step ?? s.t_end / 1000} min={0} onChange={(v) => setS({ fixed_step: v })} />
          </Field>
        )}
        {implicit && methods.linear_solvers.length > 0 && (
          <Field label="Lineares Gleichungssystem">
            <select value={s.linear_solver} onChange={(e) => setS({ linear_solver: e.target.value })}>
              {methods.linear_solvers.map((m) => <option key={m.id} value={m.id}>{m.label_de ?? m.id}</option>)}
            </select>
          </Field>
        )}
        {implicit && methods.nonlinear_solvers.length > 0 && (
          <Field label="Nichtlineare Iteration">
            <select value={s.nonlinear_solver} onChange={(e) => setS({ nonlinear_solver: e.target.value })}>
              {methods.nonlinear_solvers.map((m) => <option key={m.id} value={m.id}>{m.label_de ?? m.id}</option>)}
            </select>
          </Field>
        )}
        <Field label="Ausgabepunkte"><NumInput value={s.max_points} min={10} onChange={(v) => setS({ max_points: Math.round(v) })} /></Field>
        <Field label="Logarithmische Zeitachse">
          <input type="checkbox" checked={s.log_time} onChange={(e) => setS({ log_time: e.target.checked })} />
        </Field>
      </div>
    </Card>
  );
}
