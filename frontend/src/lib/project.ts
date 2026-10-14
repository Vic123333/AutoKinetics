// Project model <-> Cantera-style mechanism document, defaults and persistence.
import { defaultUnits, formatComposition, tryParseFormula } from "./chem";
import type { ExampleMeta, Num, Phase, Project, Rate, ReactionRow, ReactionType, ReactorForm, SimForm, SpeciesRow, Units } from "./types";

let counter = 0;
export const newKey = (p = "k") => `${p}${Date.now().toString(36)}${(counter++).toString(36)}`;

const RATE_KEYS = ["rate-constant", "high-P-rate-constant"];
const HANDLED_RXN_KEYS = new Set([
  "equation", "type", "rate-constant", "high-P-rate-constant", "reverse-rate-constant", "equilibrium-constant",
  "orders", "reaction-enthalpy", "duplicate", "note",
]);

export function defaultReactor(phase: Phase): ReactorForm {
  return {
    kind: "batch", constraint: phase === "ideal-gas" ? "pressure" : "volume", energy: "isothermal",
    T: 298.15, P: 1, Punit: "atm", basis: phase === "ideal-gas" ? "mole-fraction" : "concentration",
    concentration_unit: "mol/L", composition: {}, UA_V: 1e4, T_env: 298.15, rho_cp: 4.18e6, tau: 100,
    feed: {}, fixed_species: [], T_program: [],
  };
}

export function defaultSimulation(): SimForm {
  return {
    t_end: 100, t_unit: "s", method: "radau5", rtol: 1e-6, atol_rel: 1e-12, log_time: false,
    linear_solver: "lu_pivot", nonlinear_solver: "simplified_newton", fixed_step: null, max_points: 1500,
  };
}

export function emptyProject(phase: Phase = "liquid"): Project {
  return {
    version: 1, name: "Neues Projekt", description: "", references: [], phase, units: defaultUnits(phase),
    species: [], reactions: [], reactor: defaultReactor(phase), simulation: defaultSimulation(), meta: {},
  };
}

export function newSpecies(name = ""): SpeciesRow {
  const parsed = name ? tryParseFormula(name) : null;
  return {
    key: newKey("s"), name, formula: parsed ? name : "", composition: parsed?.composition ?? {},
    charge: parsed?.charge ?? 0, thermo: null,
  };
}

export function newReaction(equation = ""): ReactionRow {
  return {
    key: newKey("r"), equation, type: "elementary", rate: { A: 1, b: 0, Ea: 0 }, reverse: null, K: null,
    orders: {}, dH: null, duplicate: false, extra: {},
  };
}

function rateFrom(v: unknown): Rate {
  if (Array.isArray(v)) return { A: v[0] as Num, b: v[1] as Num, Ea: v[2] as Num };
  const o = (v ?? {}) as Record<string, Num>;
  return { A: o.A ?? 0, b: o.b ?? 0, Ea: o.Ea ?? 0 };
}

const ENERGY_UNITS = ["J/mol", "kJ/mol", "cal/mol", "kcal/mol", "J/kmol", "K"];

export function unitsFromDoc(u: Record<string, string> | undefined): Units {
  const quantity = (u?.quantity ?? "kmol") as Units["quantity"];
  let ea = u?.["activation-energy"] ?? (u?.energy ? `${u.energy}/${quantity}` : quantity === "kmol" ? "J/kmol" : "J/mol");
  if (!ENERGY_UNITS.includes(ea)) ea = "J/mol";
  return { length: (u?.length ?? "m") as Units["length"], quantity, ea: ea as Units["ea"] };
}

/** Build a project from a Cantera-style document (+ optional example metadata / preset). */
export function projectFromDocument(doc: Record<string, any>, meta?: Partial<ExampleMeta>): Project {
  const phases = (doc.phases ?? []) as Record<string, any>[];
  const phaseDef = phases.find((p) => ["ideal-gas", "liquid", "ideal-dilute-solution"].includes(p.thermo)) ?? phases[0] ?? {};
  const phase: Phase = phaseDef.thermo === "ideal-gas" || !phaseDef.thermo ? "ideal-gas" : "liquid";
  const declared: string[] | null = Array.isArray(phaseDef.species) ? phaseDef.species.map(String) : null;
  const allSpecies = (doc.species ?? []) as Record<string, any>[];
  const speciesDocs = declared ? declared.map((n) => allSpecies.find((s) => String(s.name) === n)).filter(Boolean) : allSpecies;
  const species: SpeciesRow[] = (speciesDocs as Record<string, any>[]).map((s) => {
    const { name, composition, thermo, note, charge, ...extra } = s;
    const comp = { ...(composition ?? {}) } as Record<string, number>;
    let q = Number(charge ?? 0);
    if (comp.E) {
      q = -comp.E;
      delete comp.E;
    }
    delete (extra as Record<string, unknown>).transport;
    return {
      key: newKey("s"), name: String(name), formula: (extra.formula as string) ?? formatComposition(comp),
      composition: comp, charge: q, thermo: thermo ?? null, note: note ? String(note) : undefined, extra,
    };
  });
  const reactions: ReactionRow[] = ((doc.reactions ?? []) as Record<string, any>[]).map((r) => {
    const extra: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(r)) if (!HANDLED_RXN_KEYS.has(k)) extra[k] = v;
    const type = (r.type ?? (String(r.equation).includes("(+") ? "falloff" : / \+ M\b/.test(String(r.equation)) ? "three-body"
      : r["rate-constants"] ? "pressure-dependent-Arrhenius" : "elementary")) as ReactionType;
    const ek = r["equilibrium-constant"];
    return {
      key: newKey("r"), equation: String(r.equation), type,
      rate: rateFrom(r[RATE_KEYS.find((k) => r[k] !== undefined) ?? "rate-constant"]),
      reverse: r["reverse-rate-constant"] ? rateFrom(r["reverse-rate-constant"]) : null,
      K: ek ? { K: ek.K, Tref: ek["T-ref"] ?? 298.15, dH: ek["delta-H"] ?? 0 } : null,
      orders: (r.orders ?? {}) as Record<string, number>, dH: r["reaction-enthalpy"] ?? null,
      duplicate: !!r.duplicate, note: r.note, extra,
    };
  });
  const p = emptyProject(phase);
  p.name = meta?.title ?? String(doc.name ?? phaseDef.name ?? "Mechanismus");
  p.description = String(doc.description ?? "");
  p.references = (doc.references ?? []).map(String);
  p.units = unitsFromDoc(doc.units);
  p.species = species;
  p.reactions = reactions;
  if (meta?.id) p.meta.exampleId = meta.id;
  if (meta) {
    p.meta.parameterQuality = meta.parameter_quality;
    p.meta.sources = meta.sources;
    p.meta.summary = meta.summary;
  }
  const preset = meta?.preset ?? (doc.autokinetics as ExampleMeta["preset"] | undefined);
  if (preset?.reactor) applyReactorPreset(p, preset.reactor as Record<string, any>);
  if (preset?.simulation) applySimulationPreset(p, preset.simulation as Record<string, any>);
  if (preset?.plot) p.meta.plot = preset.plot;
  return p;
}

function parseQty(v: unknown, fallbackUnit: string): [number, string] {
  if (typeof v === "number") return [v, fallbackUnit];
  const m = /^\s*([-+0-9.eE]+)\s*(\S.*)?$/.exec(String(v ?? ""));
  return m ? [parseFloat(m[1]), (m[2] ?? fallbackUnit).trim()] : [NaN, fallbackUnit];
}

export function applyReactorPreset(p: Project, r: Record<string, any>) {
  const f = p.reactor;
  if (r.kind) f.kind = r.kind;
  if (r.constraint) f.constraint = r.constraint;
  if (r.energy) f.energy = r.energy;
  if (r.T !== undefined) f.T = parseQty(r.T, "K")[0];
  if (r.P !== undefined) {
    const [v, u] = parseQty(r.P, "Pa");
    f.P = v;
    f.Punit = (["Pa", "kPa", "bar", "atm", "MPa"].includes(u) ? u : "Pa") as ReactorForm["Punit"];
  }
  if (r.basis) f.basis = r.basis;
  if (r.concentration_unit) f.concentration_unit = r.concentration_unit;
  if (r.composition) f.composition = { ...r.composition };
  if (r.UA_V !== undefined) f.UA_V = Number(r.UA_V);
  f.T_env = r.T_env !== undefined ? parseQty(r.T_env, "K")[0] : f.T;
  if (r.rho_cp !== undefined) f.rho_cp = Number(r.rho_cp);
  if (r.tau !== undefined) f.tau = parseQty(r.tau, "s")[0];
  if (r.feed) f.feed = { ...r.feed };
  if (r.fixed_species) f.fixed_species = [...r.fixed_species];
  if (r.T_program) f.T_program = r.T_program;
}

export function applySimulationPreset(p: Project, s: Record<string, any>) {
  const f = p.simulation;
  if (s.t_end !== undefined) {
    f.t_end = parseQty(s.t_end, "s")[0];
    f.t_unit = "s";
  }
  for (const k of ["rtol", "atol_rel", "max_points"] as const) if (s[k] !== undefined) f[k] = Number(s[k]);
  if (s.log_time !== undefined) f.log_time = !!s.log_time;
  if (s.method) f.method = s.method;
}

function rateOut(r: Rate) {
  const num = (v: Num) => (typeof v === "string" && v.trim() !== "" && !isNaN(Number(v)) ? Number(v) : v);
  return { A: num(r.A), b: num(r.b), Ea: num(r.Ea) };
}

export function toDocument(p: Project): Record<string, unknown> {
  const reactions = p.reactions.map((r) => {
    const out: Record<string, unknown> = { equation: r.equation.trim() };
    if (r.type !== "elementary") out.type = r.type;
    if (r.type === "falloff") out["high-P-rate-constant"] = rateOut(r.rate);
    else if (r.type !== "pressure-dependent-Arrhenius") out["rate-constant"] = rateOut(r.rate);
    if (r.reverse) out["reverse-rate-constant"] = rateOut(r.reverse);
    if (r.K) out["equilibrium-constant"] = { K: Number(r.K.K), "T-ref": Number(r.K.Tref), "delta-H": Number(r.K.dH) };
    if (Object.keys(r.orders).length) out.orders = r.orders;
    if (r.dH !== null && r.dH !== "" && r.dH !== undefined) out["reaction-enthalpy"] = Number(r.dH);
    if (r.duplicate) out.duplicate = true;
    if (r.note) out.note = r.note;
    return { ...out, ...r.extra };
  });
  return {
    description: p.description,
    references: p.references,
    units: { length: p.units.length, quantity: p.units.quantity, "activation-energy": p.units.ea },
    phases: [{ name: p.name || "phase", thermo: p.phase === "ideal-gas" ? "ideal-gas" : "liquid", species: p.species.map((s) => s.name) }],
    species: p.species.map((s) => {
      const out: Record<string, unknown> = { name: s.name, composition: s.composition };
      if (s.charge) out.charge = s.charge;
      if (s.thermo) out.thermo = s.thermo;
      if (s.note) out.note = s.note;
      return { ...out, ...(s.extra ?? {}) };
    }),
    reactions,
  };
}

const T_FACTOR: Record<SimForm["t_unit"], number> = { s: 1, ms: 1e-3, us: 1e-6, min: 60, h: 3600 };

export function reactorPayload(p: Project): Record<string, unknown> {
  const f = p.reactor;
  const comp = Object.fromEntries(Object.entries(f.composition).filter(([k, v]) => v > 0 && p.species.some((s) => s.name === k)));
  const out: Record<string, unknown> = {
    kind: f.kind, constraint: f.constraint, energy: f.energy, T: f.T, P: `${f.P} ${f.Punit}`, basis: f.basis,
    concentration_unit: f.concentration_unit, composition: comp,
  };
  if (f.energy === "heat-exchange") {
    out.UA_V = f.UA_V;
    out.T_env = f.T_env;
  }
  if (p.phase === "liquid" && f.energy !== "isothermal") out.rho_cp = f.rho_cp;
  if (f.kind === "cstr") {
    out.tau = f.tau;
    out.feed = Object.fromEntries(Object.entries(f.feed).filter(([k, v]) => v > 0 && p.species.some((s) => s.name === k)));
  }
  const fixed = f.fixed_species.filter((n) => p.species.some((s) => s.name === n));
  if (fixed.length) out.fixed_species = fixed;
  if (f.energy === "isothermal" && f.T_program.length) out.T_program = f.T_program;
  return out;
}

export function simulationPayload(p: Project): Record<string, unknown> {
  const s = p.simulation;
  const out: Record<string, unknown> = {
    t_end: s.t_end * T_FACTOR[s.t_unit], method: s.method, rtol: s.rtol, atol_rel: s.atol_rel, log_time: s.log_time,
    max_points: s.max_points, linear_solver: s.linear_solver, nonlinear_solver: s.nonlinear_solver,
  };
  if (s.fixed_step) out.fixed_step = s.fixed_step;
  return out;
}

const STORAGE_KEY = "autokinetics.project.v1";

export function saveLocal(p: Project) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
  } catch {
    /* storage unavailable or full – the project stays in memory */
  }
}

export function loadLocal(): Project | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw) as Project;
    return p.version === 1 && Array.isArray(p.species) ? p : null;
  } catch {
    return null;
  }
}
