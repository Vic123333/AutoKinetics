import { useMemo, useState } from "react";
import type { AppContext } from "../App";
import { api } from "../lib/api";
import { concentrationUnitLabel, convertA, convertEa, equationSpecies, formatComposition, parseEquation, rateUnitLabel, reactionOrder, tryParseFormula } from "../lib/chem";
import { newReaction, newSpecies } from "../lib/project";
import type { Issue, Num, Project, ReactionRow, SpeciesRow, Units } from "../lib/types";
import { Alert, Card, Field, NumInput, QtyInput, Spinner, fmtNum } from "../components/ui";

const ATOMIC: Record<string, number> = { H: 1.008, C: 12.011, N: 14.007, O: 15.999, S: 32.06, Cl: 35.45, Ar: 39.948, He: 4.0026, F: 18.998, P: 30.974, Na: 22.99, K: 39.098, Br: 79.904, I: 126.9, Ca: 40.078, Mg: 24.305, Fe: 55.845, Cu: 63.546, Ce: 140.116, Si: 28.085, Al: 26.98, Zn: 65.38, Mn: 54.938 };

function molarMass(c: Record<string, number>): number | null {
  let m = 0;
  for (const [el, n] of Object.entries(c)) {
    if (!(el in ATOMIC)) return null;
    m += ATOMIC[el] * n;
  }
  return Object.keys(c).length ? m : null;
}

const PAGE = 150;

export function MechanismView({ ctx }: { ctx: AppContext }) {
  const { project, update, issues, validating } = ctx;
  const [filter, setFilter] = useState("");
  const [showAll, setShowAll] = useState(false);
  const errors = issues.filter((i) => i.level === "error");
  const warnings = issues.filter((i) => i.level === "warning");
  const infos = issues.filter((i) => i.level === "info");

  if (!project.species.length && !project.reactions.length) return <EmptyState ctx={ctx} />;

  const issueByReaction = new Map<number, Issue[]>();
  for (const i of issues) if (i.reaction !== null) issueByReaction.set(i.reaction, [...(issueByReaction.get(i.reaction) ?? []), i]);

  const q = filter.trim().toLowerCase();
  const rxIdx = project.reactions.map((r, i) => [r, i] as const).filter(([r]) => !q || r.equation.toLowerCase().includes(q));
  const spIdx = project.species.map((s, i) => [s, i] as const).filter(([s]) => !q || s.name.toLowerCase().includes(q) || s.formula.toLowerCase().includes(q));

  return (
    <div className="stack">
      <SettingsCard ctx={ctx} />
      <Card
        title={<>Prüfung {validating && <Spinner label="prüfe …" />}</>}
        className={errors.length ? "card-error" : ""}
      >
        {!issues.length && !validating && <Alert kind="success">Mechanismus vollständig und konsistent (Element-, Ladungsbilanz und Thermodynamik geprüft).</Alert>}
        {[...errors, ...warnings].slice(0, 30).map((i, k) => (
          <Alert key={k} kind={i.level === "error" ? "error" : "warning"}>{i.message}</Alert>
        ))}
        {infos.map((i, k) => <Alert key={`i${k}`} kind="info">{i.message}</Alert>)}
        {errors.length + warnings.length > 30 && <p className="muted">… und {errors.length + warnings.length - 30} weitere</p>}
      </Card>
      <div className="toolbar">
        <input type="search" placeholder="Spezies/Reaktionen filtern …" value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="Filter" />
      </div>
      <SpeciesTable ctx={ctx} rows={showAll ? spIdx : spIdx.slice(0, PAGE)} total={spIdx.length} />
      <ReactionsTable ctx={ctx} rows={showAll ? rxIdx : rxIdx.slice(0, PAGE)} issueByReaction={issueByReaction} />
      {(spIdx.length > PAGE || rxIdx.length > PAGE) && !showAll && (
        <button className="ghost" onClick={() => setShowAll(true)}>Alle {spIdx.length} Spezies und {rxIdx.length} Reaktionen anzeigen</button>
      )}
      <button className="ghost add" onClick={() => update((p) => void p.reactions.push(newReaction()))}>+ Reaktion</button>
    </div>
  );
}

function EmptyState({ ctx }: { ctx: AppContext }) {
  const [eq, setEq] = useState("A + B => C");
  const create = () => {
    const parsed = parseEquation(eq);
    if (!parsed) return ctx.notify("Gleichung nicht erkannt – Beispiel: 2 H2 + O2 => 2 H2O", "error");
    ctx.update((p) => {
      for (const n of equationSpecies(eq)) if (!p.species.some((s) => s.name === n)) p.species.push(newSpecies(n));
      p.reactions.push(newReaction(eq));
    });
  };
  return (
    <Card title="Neuen Mechanismus beginnen">
      <p>Geben Sie eine erste Reaktionsgleichung ein – unbekannte Spezies werden automatisch angelegt. Oder laden Sie ein Beispiel bzw. importieren eine Cantera-YAML- oder alte .kin-Datei.</p>
      <div className="row">
        <Field label="Reaktion" wide>
          <input value={eq} onChange={(e) => setEq(e.target.value)} onKeyDown={(e) => e.key === "Enter" && create()} />
        </Field>
        <Field label="Phase">
          <select value={ctx.project.phase} onChange={(e) => ctx.update((p) => setPhase(p, e.target.value as Project["phase"]))}>
            <option value="liquid">Flüssigphase (konstantes Volumen)</option>
            <option value="ideal-gas">Ideales Gas</option>
          </select>
        </Field>
        <button className="primary" onClick={create}>Anlegen</button>
      </div>
      <p className="muted">Syntax: <code>A + B =&gt; C</code> (irreversibel), <code>A &lt;=&gt; B</code> (reversibel), <code>2 O + M &lt;=&gt; O2 + M</code> (Dreierstoß), <code>H + O2 (+M) &lt;=&gt; HO2 (+M)</code> (Fall-off).</p>
    </Card>
  );
}

function setPhase(p: Project, phase: Project["phase"]) {
  p.phase = phase;
  if (phase === "liquid") {
    p.reactor.basis = "concentration";
    p.reactor.constraint = "volume";
  } else {
    p.reactor.basis = "mole-fraction";
  }
}

function SettingsCard({ ctx }: { ctx: AppContext }) {
  const { project, update } = ctx;
  const [open, setOpen] = useState(false);
  const changeUnits = (patch: Partial<Units>) =>
    update((p) => {
      const to = { ...p.units, ...patch };
      for (const r of p.reactions) {
        const conv = (rate: { A: Num; b: Num; Ea: Num } | null, order: number | null) => {
          if (!rate) return;
          if (typeof rate.A === "number" && order !== null) rate.A = convertA(rate.A, order, p.units, to);
          if (typeof rate.Ea === "number") rate.Ea = convertEa(rate.Ea, p.units.ea, to.ea);
        };
        conv(r.rate, reactionOrder(r));
        const pe = parseEquation(r.equation);
        if (r.reverse && pe) conv(r.reverse, Object.values(pe.products).reduce((a, b) => a + b, 0) + (r.type === "three-body" ? 1 : 0));
        if (r.dH !== null && typeof r.dH === "number") r.dH = convertEa(r.dH, p.units.ea, to.ea);
      }
      p.units = to;
    });
  return (
    <Card
      title="Einstellungen"
      actions={<button className="ghost small" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? "Beschreibung ausblenden" : "Beschreibung & Quellen"}</button>}
    >
      <div className="row">
        <Field label="Phase" hint={project.phase === "ideal-gas" ? "Standardzustand p° = 1 bar" : "Standardzustand c° = 1 mol/L"}>
          <select value={project.phase} onChange={(e) => update((p) => setPhase(p, e.target.value as Project["phase"]))}>
            <option value="liquid">Flüssigphase</option>
            <option value="ideal-gas">Ideales Gas</option>
          </select>
        </Field>
        <Field label="Konzentrationsbasis für A" hint={`Konzentration in ${concentrationUnitLabel(project.units)} – Werte werden umgerechnet`}>
          <select value={`${project.units.quantity}|${project.units.length}`} onChange={(e) => { const [quantity, length] = e.target.value.split("|"); changeUnits({ quantity: quantity as Units["quantity"], length: length as Units["length"] }); }}>
            <option value="mol|dm">mol/L</option>
            <option value="mol|cm">mol/cm³</option>
            <option value="mol|m">mol/m³</option>
            <option value="kmol|m">kmol/m³</option>
            <option value="molec|cm">Moleküle/cm³</option>
          </select>
        </Field>
        <Field label="Aktivierungsenergie">
          <select value={project.units.ea} onChange={(e) => changeUnits({ ea: e.target.value as Units["ea"] })}>
            {["kJ/mol", "J/mol", "kcal/mol", "cal/mol", "J/kmol", "K"].map((u) => <option key={u} value={u}>{u === "K" ? "K (Ea/R)" : u}</option>)}
          </select>
        </Field>
      </div>
      {project.meta.summary && <p className="summary">{project.meta.summary}</p>}
      {project.meta.parameterQuality && project.meta.parameterQuality !== "literature" && (
        <Alert kind="info">Parameterqualität: {project.meta.parameterQuality === "illustrative" ? "illustrativ (Lehrbeispiel, keine Messwerte)" : "gemischt – siehe Quellen"}.</Alert>
      )}
      {open && (
        <div className="stack">
          <Field label="Beschreibung" wide>
            <textarea rows={3} value={project.description} onChange={(e) => update((p) => void (p.description = e.target.value))} />
          </Field>
          <Field label="Quellen (eine pro Zeile)" wide>
            <textarea rows={4} value={project.references.join("\n")} onChange={(e) => update((p) => void (p.references = e.target.value.split("\n")))} />
          </Field>
        </div>
      )}
    </Card>
  );
}

function SpeciesTable({ ctx, rows, total }: { ctx: AppContext; rows: (readonly [SpeciesRow, number])[]; total: number }) {
  const { project, update, notify } = ctx;
  const [busy, setBusy] = useState<string | null>(null);
  const lookup = async (i: number) => {
    const s = project.species[i];
    setBusy(s.key);
    try {
      const info = await api.get<any>(`/api/species/lookup?q=${encodeURIComponent(s.formula || s.name)}`);
      update((p) => {
        const t = p.species[i];
        if (info.composition && Object.keys(info.composition).length) {
          t.composition = info.composition;
          t.charge = info.charge ?? 0;
          t.formula = info.formula ?? t.formula;
        }
        if (info.thermo) t.thermo = { ...info.thermo, note: info.thermo_source ?? "" };
        t.extra = { ...(t.extra ?? {}), ...(info.pubchem ? { pubchem: info.pubchem.cid } : {}), ...(info.cas ? { cas: info.cas } : {}) };
      });
      notify(info.thermo ? `${s.name}: Thermodaten aus ${info.thermo_source}` : `${s.name}: ${info.warnings?.[0] ?? "keine Thermodaten"}`, info.thermo ? "success" : "info");
    } catch (e) {
      notify(`${s.name}: ${(e as Error).message}`, "error");
    } finally {
      setBusy(null);
    }
  };
  return (
    <Card
      title={`Spezies (${total})`}
      actions={<button className="ghost small" onClick={() => update((p) => void p.species.push(newSpecies()))}>+ Spezies</button>}
    >
      <div className="table-wrap">
        <table className="grid">
          <thead>
            <tr>
              <th>Name</th>
              <th>Summenformel</th>
              <th>Ladung</th>
              <th className="num">M [g/mol]</th>
              <th>Thermodaten</th>
              <th aria-label="Aktionen" />
            </tr>
          </thead>
          <tbody>
            {rows.map(([s, i]) => {
              const parsed = s.formula ? tryParseFormula(s.formula) : null;
              const formulaInvalid = !!s.formula && !parsed;
              const mm = molarMass(s.composition);
              return (
                <tr key={s.key}>
                  <td><input value={s.name} aria-label="Spezies-Name" onChange={(e) => {
                    const old = s.name, nu = e.target.value;
                    update((p) => {
                      p.species[i].name = nu;
                      if (old) {
                        const re = new RegExp(`(^|[\\s+(])${old.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?=$|[\\s+)])`, "g");
                        for (const r of p.reactions) r.equation = r.equation.replace(re, `$1${nu}`);
                        if (old in p.reactor.composition) { p.reactor.composition[nu] = p.reactor.composition[old]; delete p.reactor.composition[old]; }
                      }
                    });
                  }} /></td>
                  <td>
                    <input value={s.formula} aria-label="Summenformel" aria-invalid={formulaInvalid} className={formulaInvalid ? "invalid" : ""} placeholder="z. B. C2H5OH"
                      onChange={(e) => update((p) => {
                        const t = p.species[i];
                        t.formula = e.target.value;
                        const pf = tryParseFormula(e.target.value);
                        if (pf) { t.composition = pf.composition; t.charge = pf.charge; }
                        else if (!e.target.value) { t.composition = {}; t.charge = 0; }
                      })} />
                    {formulaInvalid && <span className="field-hint error">ungültige Formel</span>}
                    {!s.formula && Object.keys(s.composition).length > 0 && <span className="field-hint">{formatComposition(s.composition)}</span>}
                  </td>
                  <td className="num">{s.charge ? (s.charge > 0 ? `+${s.charge}` : s.charge) : "0"}</td>
                  <td className="num">{mm ? fmtNum(mm, 6) : "–"}</td>
                  <td>
                    {s.thermo ? <span className="tag" title={String((s.thermo as any).note ?? "")}>{String((s.thermo as any).model)}{(s.thermo as any).model === "database" ? ` (${(s.thermo as any).name})` : ""}</span> : <span className="muted">keine</span>}
                    {" "}
                    <button className="ghost small" disabled={busy === s.key || !ctx.health?.online_data && !s.name} onClick={() => void lookup(i)} title="PubChem / NIST WebBook / NASA-Datenbank">
                      {busy === s.key ? "sucht …" : "Stoffdaten suchen"}
                    </button>
                    {s.thermo && <button className="ghost small" onClick={() => update((p) => void (p.species[i].thermo = null))} aria-label={`Thermodaten von ${s.name} entfernen`}>✕</button>}
                  </td>
                  <td><button className="ghost small danger" aria-label={`Spezies ${s.name} löschen`} onClick={() => update((p) => void p.species.splice(i, 1))}>Löschen</button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

const TYPE_LABEL: Record<ReactionRow["type"], string> = {
  elementary: "Elementar", "three-body": "Dreierstoß", falloff: "Fall-off", "pressure-dependent-Arrhenius": "PLOG",
};

function ReactionsTable({ ctx, rows, issueByReaction }: { ctx: AppContext; rows: (readonly [ReactionRow, number])[]; issueByReaction: Map<number, Issue[]> }) {
  const { project, update } = ctx;
  const [open, setOpen] = useState<string | null>(null);
  const names = useMemo(() => new Set(project.species.map((s) => s.name)), [project.species]);
  return (
    <Card title={`Reaktionen (${project.reactions.length})`} actions={<button className="ghost small" onClick={() => update((p) => void p.reactions.push(newReaction()))}>+ Reaktion</button>}>
      <div className="table-wrap">
        <table className="grid reactions">
          <thead>
            <tr>
              <th className="num">#</th>
              <th>Gleichung</th>
              <th>Typ</th>
              <th>A</th>
              <th>b</th>
              <th>Eₐ [{project.units.ea}]</th>
              <th aria-label="Details" />
            </tr>
          </thead>
          <tbody>
            {rows.map(([r, i]) => {
              const parsed = parseEquation(r.equation);
              const unknown = equationSpecies(r.equation).filter((n) => !names.has(n));
              const rowIssues = issueByReaction.get(i) ?? [];
              const order = reactionOrder(r, false);
              const isOpen = open === r.key;
              return [
                <tr key={r.key} className={rowIssues.some((x) => x.level === "error") ? "row-error" : ""}>
                  <td className="num">{i + 1}</td>
                  <td>
                    <input className={`equation ${!parsed && r.equation ? "invalid" : ""}`} value={r.equation} aria-label={`Gleichung ${i + 1}`} placeholder="A + B => C"
                      onChange={(e) => update((p) => void (p.reactions[i].equation = e.target.value))} />
                    {unknown.length > 0 && (
                      <span className="field-hint error">
                        Unbekannt: {unknown.join(", ")}{" "}
                        <button className="link" onClick={() => update((p) => { for (const n of unknown) p.species.push(newSpecies(n)); })}>anlegen</button>
                      </span>
                    )}
                    {rowIssues.map((x, k) => <span key={k} className={`field-hint ${x.level === "error" ? "error" : ""}`}>{x.message.replace(/^Reaktion \d+ \([^)]*\): /, "")}</span>)}
                  </td>
                  <td>
                    <select value={r.type} aria-label="Reaktionstyp" onChange={(e) => update((p) => void (p.reactions[i].type = e.target.value as ReactionRow["type"]))}>
                      {Object.entries(TYPE_LABEL).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                    </select>
                  </td>
                  {r.type === "pressure-dependent-Arrhenius" ? (
                    <td colSpan={3} className="muted">Druck-Tabelle (Details)</td>
                  ) : (
                    <>
                      <td><QtyInput value={r.rate.A} ariaLabel="Vorfaktor A" onChange={(v) => update((p) => void (p.reactions[i].rate.A = v))} /><span className="unit">{rateUnitLabel(order, project.units)}{r.type === "falloff" ? " (k∞)" : ""}</span></td>
                      <td><QtyInput value={r.rate.b} ariaLabel="Temperaturexponent b" onChange={(v) => update((p) => void (p.reactions[i].rate.b = v))} /></td>
                      <td><QtyInput value={r.rate.Ea} ariaLabel="Aktivierungsenergie" onChange={(v) => update((p) => void (p.reactions[i].rate.Ea = v))} /></td>
                    </>
                  )}
                  <td className="nowrap">
                    <button className="ghost small" aria-expanded={isOpen} onClick={() => setOpen(isOpen ? null : r.key)}>{isOpen ? "▴" : "▾"} Details</button>
                    <button className="ghost small danger" aria-label={`Reaktion ${i + 1} löschen`} onClick={() => update((p) => void p.reactions.splice(i, 1))}>✕</button>
                  </td>
                </tr>,
                isOpen && (
                  <tr key={`${r.key}-d`} className="details">
                    <td />
                    <td colSpan={6}><ReactionDetails ctx={ctx} i={i} /></td>
                  </tr>
                ),
              ];
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function ReactionDetails({ ctx, i }: { ctx: AppContext; i: number }) {
  const { project, update } = ctx;
  const r = project.reactions[i];
  const parsed = parseEquation(r.equation);
  const revOrder = parsed ? Object.values(parsed.products).reduce((a, b) => a + b, 0) + (r.type === "three-body" ? 1 : 0) : null;
  const extraKeys = Object.keys(r.extra);
  return (
    <div className="stack">
      {parsed?.reversible && (
        <div className="row">
          <Field label="Rückreaktion">
            <select value={r.reverse ? "arrhenius" : r.K ? "K" : "thermo"} onChange={(e) => update((p) => {
              const t = p.reactions[i];
              t.reverse = e.target.value === "arrhenius" ? { A: 1, b: 0, Ea: 0 } : null;
              t.K = e.target.value === "K" ? { K: 1, Tref: 298.15, dH: 0 } : null;
            })}>
              <option value="thermo">aus Thermodaten (ΔG°, empfohlen)</option>
              <option value="K">aus Gleichgewichtskonstante K(T)</option>
              <option value="arrhenius">eigener Arrhenius-Ausdruck</option>
            </select>
          </Field>
          {r.reverse && (
            <>
              <Field label={`A_rück [${rateUnitLabel(revOrder, project.units)}]`}><QtyInput value={r.reverse.A} onChange={(v) => update((p) => void (p.reactions[i].reverse!.A = v))} /></Field>
              <Field label="b_rück"><QtyInput value={r.reverse.b} onChange={(v) => update((p) => void (p.reactions[i].reverse!.b = v))} /></Field>
              <Field label={`Eₐ,rück [${project.units.ea}]`}><QtyInput value={r.reverse.Ea} onChange={(v) => update((p) => void (p.reactions[i].reverse!.Ea = v))} /></Field>
            </>
          )}
          {r.K && (
            <>
              <Field label={`K(T_ref) [(${concentrationUnitLabel(project.units)})^Δν]`}><QtyInput value={r.K.K} onChange={(v) => update((p) => void (p.reactions[i].K!.K = v))} /></Field>
              <Field label="T_ref [K]"><QtyInput value={r.K.Tref} onChange={(v) => update((p) => void (p.reactions[i].K!.Tref = v))} /></Field>
              <Field label={`ΔH°_R [${project.units.ea}] (van 't Hoff)`}><QtyInput value={r.K.dH} onChange={(v) => update((p) => void (p.reactions[i].K!.dH = v))} /></Field>
            </>
          )}
        </div>
      )}
      <div className="row">
        <Field label="Reaktionsordnungen (leer = Stöchiometrie)" hint="z. B. A: 0.5, B: 1 – nicht-elementare Kinetik">
          <input defaultValue={Object.entries(r.orders).map(([k, v]) => `${k}: ${v}`).join(", ")} onBlur={(e) => update((p) => {
            const o: Record<string, number> = {};
            for (const part of e.target.value.split(",")) {
              const [k, v] = part.split(":").map((x) => x.trim());
              if (k && v !== undefined && Number.isFinite(Number(v))) o[k] = Number(v);
            }
            p.reactions[i].orders = o;
          })} />
        </Field>
        {project.phase === "liquid" && (
          <Field label={`Reaktionsenthalpie ΔH_R [${project.units.ea}]`} hint="für Energiebilanz ohne Thermodaten">
            <NumInput value={typeof r.dH === "number" ? r.dH : null} allowEmpty onChange={(v) => update((p) => void (p.reactions[i].dH = Number.isNaN(v) ? null : v))} />
          </Field>
        )}
        <Field label="Duplikat">
          <input type="checkbox" checked={r.duplicate} onChange={(e) => update((p) => void (p.reactions[i].duplicate = e.target.checked))} />
        </Field>
      </div>
      {extraKeys.length > 0 && (
        <Field label="Weitere Parameter (Cantera-YAML, JSON)" hint="Fall-off (low-P, Troe/SRI), Stoßeffizienzen, PLOG-Tabellen" wide>
          <ExtraEditor value={r.extra} onChange={(v) => update((p) => void (p.reactions[i].extra = v))} />
        </Field>
      )}
      {r.type === "falloff" && !("low-P-rate-constant" in r.extra) && (
        <button className="ghost small" onClick={() => update((p) => void (p.reactions[i].extra = { ...p.reactions[i].extra, "low-P-rate-constant": { A: 1, b: 0, Ea: 0 }, Troe: { A: 0.5, T3: 100, T1: 1000 } }))}>Niederdruck-Parameter hinzufügen</button>
      )}
      {r.type === "pressure-dependent-Arrhenius" && !("rate-constants" in r.extra) && (
        <button className="ghost small" onClick={() => update((p) => void (p.reactions[i].extra = { ...p.reactions[i].extra, "rate-constants": [{ P: "1 atm", A: 1, b: 0, Ea: 0 }] }))}>PLOG-Tabelle anlegen</button>
      )}
      <Field label="Notiz" wide><input value={r.note ?? ""} onChange={(e) => update((p) => void (p.reactions[i].note = e.target.value))} /></Field>
    </div>
  );
}

function ExtraEditor({ value, onChange }: { value: Record<string, unknown>; onChange: (v: Record<string, unknown>) => void }) {
  const [text, setText] = useState(JSON.stringify(value, null, 1));
  const [err, setErr] = useState<string | null>(null);
  return (
    <>
      <textarea className="code" rows={Math.min(12, text.split("\n").length + 1)} value={text} onChange={(e) => {
        setText(e.target.value);
        try {
          onChange(JSON.parse(e.target.value));
          setErr(null);
        } catch (x) {
          setErr((x as Error).message);
        }
      }} />
      {err && <span className="field-hint error">{err}</span>}
    </>
  );
}
