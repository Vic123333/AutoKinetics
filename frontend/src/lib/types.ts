export type Phase = "ideal-gas" | "liquid";
export type Num = number | string;

export interface Units {
  length: "m" | "dm" | "cm" | "mm";
  quantity: "mol" | "kmol" | "molec";
  ea: "J/mol" | "kJ/mol" | "cal/mol" | "kcal/mol" | "J/kmol" | "K";
}

export interface Rate {
  A: Num;
  b: Num;
  Ea: Num;
}

export interface SpeciesRow {
  key: string;
  name: string;
  formula: string;
  composition: Record<string, number>;
  charge: number;
  thermo: Record<string, unknown> | null;
  note?: string;
  extra?: Record<string, unknown>;
}

export type ReactionType = "elementary" | "three-body" | "falloff" | "pressure-dependent-Arrhenius";

export interface ReactionRow {
  key: string;
  equation: string;
  type: ReactionType;
  rate: Rate;
  reverse: Rate | null;
  K: { K: Num; Tref: Num; dH: Num } | null;
  orders: Record<string, number>;
  dH: Num | null;
  duplicate: boolean;
  note?: string;
  extra: Record<string, unknown>;
}

export interface ReactorForm {
  kind: "batch" | "cstr";
  constraint: "volume" | "pressure";
  energy: "isothermal" | "adiabatic" | "heat-exchange";
  T: number;
  P: number;
  Punit: "Pa" | "kPa" | "bar" | "atm" | "MPa";
  basis: "concentration" | "mole-fraction";
  concentration_unit: "mol/L" | "mmol/L" | "mol/m3";
  composition: Record<string, number>;
  UA_V: number;
  T_env: number;
  rho_cp: number;
  tau: number;
  feed: Record<string, number>;
  fixed_species: string[];
  T_program: [number, number][];
}

export interface SimForm {
  t_end: number;
  t_unit: "s" | "ms" | "us" | "min" | "h";
  method: string;
  rtol: number;
  atol_rel: number;
  log_time: boolean;
  linear_solver: string;
  nonlinear_solver: string;
  fixed_step: number | null;
  max_points: number;
}

export interface Project {
  version: 1;
  name: string;
  description: string;
  references: string[];
  phase: Phase;
  units: Units;
  species: SpeciesRow[];
  reactions: ReactionRow[];
  reactor: ReactorForm;
  simulation: SimForm;
  meta: { exampleId?: string; plot?: PlotHints; parameterQuality?: string; sources?: string[]; summary?: string };
}

export interface PlotHints {
  species?: string[];
  log_time?: boolean;
  log_y?: boolean;
  show_T?: boolean;
}

export interface Issue {
  level: "error" | "warning" | "info";
  message: string;
  reaction: number | null;
  species: string | null;
}

export interface SimulationResult {
  success: boolean;
  message: string;
  t: number[];
  species: string[];
  concentration_unit: string;
  concentrations: number[][];
  mole_fractions?: number[][];
  T: number[];
  P: number[];
  V: number[];
  stats: Record<string, number>;
  wall_time: number;
  diagnostics: Record<string, unknown>;
  metrics: {
    conversion?: Record<string, number>;
    half_life?: Record<string, number>;
    ignition_delay?: number;
    T_final?: number;
    T_rise?: number;
    max_dTdt?: number;
  };
  method: string;
  heat_release_rate?: number[];
}

export interface ExampleMeta {
  id: string;
  title: string;
  category: string;
  summary: string;
  parameter_quality: string;
  sources: string[];
  n_species: number;
  n_reactions: number;
  phase: Phase;
  preset: { reactor?: Record<string, unknown>; simulation?: Record<string, unknown>; plot?: PlotHints; expected?: unknown[] };
}

export interface MethodInfo {
  id: string;
  label_de?: string;
  description_de?: string;
  family?: string;
  order?: number | null;
  implicit?: boolean;
  adaptive?: boolean;
  [k: string]: unknown;
}
