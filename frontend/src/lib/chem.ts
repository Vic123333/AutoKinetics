// Chemistry helpers mirroring the Python implementation (immediate UI feedback;
// the backend validates authoritatively).
import type { Phase, ReactionRow, Units } from "./types";

const ELEMENTS = new Set(
  ("H D T He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr " +
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu " +
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U").split(" "),
);

export interface ParsedFormula {
  composition: Record<string, number>;
  charge: number;
}

function parseGroup(f: string): Record<string, number> {
  const stack: Record<string, number>[] = [{}];
  let i = 0;
  const num = () => {
    const m = /^\d+(?:\.\d+)?/.exec(f.slice(i));
    if (!m) return 1;
    i += m[0].length;
    return parseFloat(m[0]);
  };
  while (i < f.length) {
    const ch = f[i];
    if (ch === "(") {
      stack.push({});
      i++;
    } else if (ch === ")") {
      if (stack.length === 1) throw new Error("Klammerfehler");
      i++;
      const mult = num();
      const inner = stack.pop()!;
      for (const [el, n] of Object.entries(inner)) stack[stack.length - 1][el] = (stack[stack.length - 1][el] ?? 0) + n * mult;
    } else {
      const m = /^[A-Z][a-z]?/.exec(f.slice(i));
      if (!m || !ELEMENTS.has(m[0])) throw new Error(`Unbekanntes Element bei '${f.slice(i)}'`);
      i += m[0].length;
      const n = num();
      const top = stack[stack.length - 1];
      top[m[0]] = (top[m[0]] ?? 0) + n;
    }
  }
  if (stack.length !== 1) throw new Error("Klammerfehler");
  return stack[0];
}

/** Parse 'Ca(OH)2', 'CuSO4·5H2O', 'SO4^2-', 'NH4+', 'Fe+++', 'SO4-2'. Throws on invalid input. */
export function parseFormula(formula: string): ParsedFormula {
  let text = formula.replace(/\s+/g, "");
  if (!text) throw new Error("Leere Formel");
  let charge = 0;
  let m: RegExpExecArray | null;
  if ((m = /\^(\d*)([+-])$/.exec(text))) {
    charge = (m[2] === "+" ? 1 : -1) * (m[1] ? parseInt(m[1]) : 1);
    text = text.slice(0, m.index);
  } else if ((m = /([+-]+)$/.exec(text)) && m.index > 0) {
    charge = (m[1][0] === "+" ? 1 : -1) * m[1].length;
    text = text.slice(0, m.index);
  } else if ((m = /([+-])(\d+)$/.exec(text)) && m.index > 0) {
    charge = (m[1] === "+" ? 1 : -1) * parseInt(m[2]);
    text = text.slice(0, m.index);
  }
  const composition: Record<string, number> = {};
  for (let part of text.split(/[·*.](?=\d*[A-Z(])/)) {
    let mult = 1;
    const mm = /^(\d+)(?=[A-Z(])/.exec(part);
    if (mm) {
      mult = parseInt(mm[1]);
      part = part.slice(mm[0].length);
    }
    for (const [el, n] of Object.entries(parseGroup(part))) composition[el] = (composition[el] ?? 0) + n * mult;
  }
  return { composition, charge };
}

export function tryParseFormula(formula: string): ParsedFormula | null {
  try {
    return parseFormula(formula);
  } catch {
    return null;
  }
}

export function formatComposition(c: Record<string, number>): string {
  return Object.entries(c)
    .filter(([, n]) => n)
    .map(([el, n]) => (n === 1 ? el : `${el}${n}`))
    .join("");
}

export interface ParsedEquation {
  reactants: Record<string, number>;
  products: Record<string, number>;
  reversible: boolean;
  thirdBody: boolean;
  falloff: string | null;
}

const ARROWS: [string, boolean][] = [
  ["<=>", true], ["<->", true], ["⇌", true], ["=>", false], ["->", false], ["→", false], ["=", true],
];

/** Lightweight equation parser (species names, coefficients, arrow, third bodies). */
export function parseEquation(eq: string): ParsedEquation | null {
  let text = eq.trim().replace(/\s+/g, " ");
  if (!text) return null;
  let falloff: string | null = null;
  const fo = /\(\s*\+\s*([^()\s]+)\s*\)/g;
  const matches = [...text.matchAll(fo)];
  if (matches.length === 2) {
    falloff = matches[0][1];
    text = text.replace(fo, " ").replace(/\s+/g, " ").trim();
  }
  let lhs = "", rhs = "", reversible = false, found = false;
  for (const [arrow, rev] of ARROWS) {
    const i = text.indexOf(arrow);
    if (i >= 0) {
      lhs = text.slice(0, i);
      rhs = text.slice(i + arrow.length);
      reversible = rev;
      found = true;
      break;
    }
  }
  if (!found) return null;
  let thirdBody = false;
  const side = (s: string) => {
    const out: Record<string, number> = {};
    for (const tok of s.split(/\s\+\s|\s\+$|^\+\s/).map((x) => x.trim()).filter(Boolean)) {
      const m = /^(\d+(?:\.\d*)?|\.\d+)(?:\s+|\*)?(.+)$/.exec(tok);
      let nu = 1, name = tok;
      if (m) {
        nu = parseFloat(m[1]);
        name = m[2].trim();
      }
      if (name === "M") {
        thirdBody = true;
        continue;
      }
      out[name] = (out[name] ?? 0) + nu;
    }
    return out;
  };
  const reactants = side(lhs);
  const products = side(rhs);
  if (!Object.keys(reactants).length) return null;
  return { reactants, products, reversible, thirdBody, falloff };
}

export function equationSpecies(eq: string): string[] {
  const p = parseEquation(eq);
  if (!p) return [];
  const names = new Set([...Object.keys(p.reactants), ...Object.keys(p.products)]);
  if (p.falloff && p.falloff !== "M") names.add(p.falloff);
  return [...names];
}

/** Overall concentration order of the rate expression (incl. third body / low-P limit). */
export function reactionOrder(r: ReactionRow, lowPressure = false): number | null {
  const p = parseEquation(r.equation);
  if (!p) return null;
  const orders = { ...p.reactants, ...r.orders };
  let n = Object.values(orders).reduce((a, b) => a + b, 0);
  if (r.type === "three-body" || (r.type === "falloff" && lowPressure)) n += 1;
  return n;
}

const SUP: Record<string, string> = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", ".": "·", "-": "⁻" };
const sup = (x: number) => String(+x.toFixed(2)).split("").map((c) => SUP[c] ?? c).join("");

/** Unit label of the pre-exponential factor (k = C^(1-n)/time) for concentration order n. */
export function rateUnitLabel(order: number | null, units: Units, timeUnit = "s"): string {
  if (order === null) return "";
  const e = Math.round((1 - order) * 1e9) / 1e9; // exponent of the concentration unit
  const q = units.quantity;
  const vol = (k: number) => (units.length === "dm" ? `L${k === 1 ? "" : sup(k)}` : `${units.length}${sup(3 * k)}`);
  if (e === 0) return `1/${timeUnit}`;
  if (!Number.isInteger(e)) return `(${q}/${vol(1)})${sup(e)}/${timeUnit}`;
  if (e < 0) return `${vol(-e)}/(${q}${e === -1 ? "" : sup(-e)}·${timeUnit})`;
  return `${q}${e === 1 ? "" : sup(e)}/(${vol(e)}·${timeUnit})`;
}

export function concentrationUnitLabel(units: Units): string {
  return units.length === "dm" ? `${units.quantity}/L` : `${units.quantity}/${units.length}³`;
}

export function defaultUnits(phase: Phase): Units {
  return phase === "ideal-gas"
    ? { length: "cm", quantity: "mol", ea: "kJ/mol" }
    : { length: "dm", quantity: "mol", ea: "kJ/mol" };
}

const LENGTH_M: Record<string, number> = { m: 1, dm: 0.1, cm: 0.01, mm: 0.001 };
const QTY_MOL: Record<string, number> = { mol: 1, kmol: 1000, molec: 1 / 6.02214076e23 };
const EA_JMOL: Record<string, number> = {
  "J/mol": 1, "kJ/mol": 1000, "cal/mol": 4.184, "kcal/mol": 4184, "J/kmol": 1e-3, K: 8.31446261815324,
};

/** Convert a pre-exponential factor between unit systems for a given order. */
export function convertA(A: number, order: number, from: Units, to: Units): number {
  const cf = (u: Units) => QTY_MOL[u.quantity] / LENGTH_M[u.length] ** 3; // to mol/m³
  return (A * cf(from) ** (1 - order)) / cf(to) ** (1 - order);
}

export function convertEa(Ea: number, from: Units["ea"], to: Units["ea"]): number {
  return (Ea * EA_JMOL[from]) / EA_JMOL[to];
}
