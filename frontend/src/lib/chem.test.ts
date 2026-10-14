import { describe, expect, it } from "vitest";
import { convertA, convertEa, parseEquation, parseFormula, rateUnitLabel, reactionOrder } from "./chem";
import type { ReactionRow } from "./types";

const rxn = (equation: string, type: ReactionRow["type"] = "elementary", orders = {}): ReactionRow => ({
  key: "r", equation, type, rate: { A: 1, b: 0, Ea: 0 }, reverse: null, K: null, orders, dH: null, duplicate: false, extra: {},
});

describe("parseFormula", () => {
  it("parses formulas, hydrates and charges like the backend", () => {
    expect(parseFormula("Ca(OH)2").composition).toEqual({ Ca: 1, O: 2, H: 2 });
    expect(parseFormula("CuSO4·5H2O").composition).toEqual({ Cu: 1, S: 1, O: 9, H: 10 });
    expect(parseFormula("NH4+")).toEqual({ composition: { N: 1, H: 4 }, charge: 1 });
    expect(parseFormula("SO4^2-")).toEqual({ composition: { S: 1, O: 4 }, charge: -2 });
    expect(parseFormula("Fe(CN)6-4").charge).toBe(-4);
    expect(() => parseFormula("Xy2")).toThrow();
  });
});

describe("parseEquation", () => {
  it("handles arrows, coefficients and third bodies", () => {
    const e = parseEquation("2 H2 + O2 => 2 H2O")!;
    expect(e.reactants).toEqual({ H2: 2, O2: 1 });
    expect(e.reversible).toBe(false);
    expect(parseEquation("2 O + M <=> O2 + M")!.thirdBody).toBe(true);
    expect(parseEquation("H + O2 (+M) <=> HO2 (+M)")!.falloff).toBe("M");
    expect(parseEquation("CH2(S) + N2 = CH2 + N2")!.reactants).toEqual({ "CH2(S)": 1, N2: 1 });
    expect(parseEquation("nonsense")).toBeNull();
  });
  it("computes reaction orders", () => {
    expect(reactionOrder(rxn("2 A + B => C"))).toBe(3);
    expect(reactionOrder(rxn("2 O + M <=> O2 + M", "three-body"))).toBe(3);
    expect(reactionOrder(rxn("A => B", "elementary", { A: 0.5 }))).toBe(0.5);
  });
});

describe("units", () => {
  const liq = { length: "dm", quantity: "mol", ea: "kJ/mol" } as const;
  const gas = { length: "cm", quantity: "mol", ea: "cal/mol" } as const;
  it("labels rate constants", () => {
    expect(rateUnitLabel(1, liq)).toBe("1/s");
    expect(rateUnitLabel(2, liq)).toBe("L/(mol·s)");
    expect(rateUnitLabel(3, gas)).toBe("cm⁶/(mol²·s)");
    expect(rateUnitLabel(0, liq)).toBe("mol/(L·s)");
  });
  it("converts A and Ea", () => {
    expect(convertA(1e-3, 2, { length: "m", quantity: "mol", ea: "J/mol" }, liq)).toBeCloseTo(1, 12);
    expect(convertA(1e6, 2, gas, { length: "m", quantity: "mol", ea: "J/mol" })).toBeCloseTo(1, 12);
    expect(convertEa(1, "kcal/mol", "kJ/mol")).toBeCloseTo(4.184, 12);
  });
});
