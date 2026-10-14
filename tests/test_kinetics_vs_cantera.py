"""Differential tests of the C++ kernel and the numpy reference against Cantera."""

import numpy as np
import pytest
import yaml

from autokinetics import kinetics_ref
from autokinetics.compiled import compile_mechanism
from autokinetics.io import load_mechanism
from helpers import cantera_data

ct = pytest.importorskip("cantera")

FALLOFF_PLOG_YAML = """
units: {length: cm, quantity: mol, activation-energy: cal/mol}
phases:
- {name: gas, thermo: ideal-gas, elements: [H, O, Ar], species: [H, O2, HO2, AR, H2O, OH, H2, O], kinetics: gas, reactions: all}
species:
- name: H
  composition: {H: 1}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 3500.0], data: [[2.5, 7.05332819e-13, -1.99591964e-15, 2.30081632e-18, -9.27732332e-22, 2.54736599e+04, -0.446682853], [2.50000001, -2.30842973e-11, 1.61561948e-14, -4.73515235e-18, 4.98197357e-22, 2.54736599e+04, -0.446682914]]}
- name: O2
  composition: {O: 2}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 3500.0], data: [[3.78245636, -2.99673416e-03, 9.84730201e-06, -9.68129509e-09, 3.24372837e-12, -1063.94356, 3.65767573], [3.28253784, 1.48308754e-03, -7.57966669e-07, 2.09470555e-10, -2.16717794e-14, -1088.45772, 5.45323129]]}
- name: HO2
  composition: {H: 1, O: 2}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 3500.0], data: [[4.30179801, -4.74912051e-03, 2.11582891e-05, -2.42763894e-08, 9.29225124e-12, 294.80804, 3.71666245], [4.0172109, 2.23982013e-03, -6.3365815e-07, 1.1424637e-10, -1.07908535e-14, 111.856713, 3.78510215]]}
- name: AR
  composition: {Ar: 1}
  thermo: {model: NASA7, temperature-ranges: [300.0, 5000.0], data: [[2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.366]]}
- name: H2O
  composition: {H: 2, O: 1}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 3500.0], data: [[4.19864056, -2.0364341e-03, 6.52040211e-06, -5.48797062e-09, 1.77197817e-12, -3.02937267e+04, -0.849032208], [3.03399249, 2.17691804e-03, -1.64072518e-07, -9.7041987e-11, 1.68200992e-14, -3.00042971e+04, 4.9667701]]}
- name: OH
  composition: {H: 1, O: 1}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 3500.0], data: [[3.99201543, -2.40131752e-03, 4.61793841e-06, -3.88113333e-09, 1.3641147e-12, 3615.08056, -0.103925458], [3.09288767, 5.48429716e-04, 1.26505228e-07, -8.79461556e-11, 1.17412376e-14, 3858.657, 4.4766961]]}
- name: H2
  composition: {H: 2}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 3500.0], data: [[2.34433112, 7.98052075e-03, -1.9478151e-05, 2.01572094e-08, -7.37611761e-12, -917.935173, 0.683010238], [3.3372792, -4.94024731e-05, 4.99456778e-07, -1.79566394e-10, 2.00255376e-14, -950.158922, -3.20502331]]}
- name: O
  composition: {O: 1}
  thermo: {model: NASA7, temperature-ranges: [200.0, 1000.0, 3500.0], data: [[3.1682671, -3.27931884e-03, 6.64306396e-06, -6.12806624e-09, 2.11265971e-12, 2.91222592e+04, 2.05193346], [2.56942078, -8.59741137e-05, 4.19484589e-08, -1.00177799e-11, 1.22833691e-15, 2.92175791e+04, 4.78433864]]}
reactions:
- equation: H + O2 (+M) <=> HO2 (+M)
  type: falloff
  low-P-rate-constant: {A: 6.366e+20, b: -1.72, Ea: 524.8}
  high-P-rate-constant: {A: 4.65084e+12, b: 0.44, Ea: 0.0}
  Troe: {A: 0.5, T3: 1.0e-30, T1: 1.0e+30}
  efficiencies: {H2O: 14.0, O2: 0.78, AR: 0.67}
- equation: H + O2 (+AR) <=> HO2 (+AR)
  type: falloff
  low-P-rate-constant: {A: 6.81e+18, b: -1.2, Ea: 0.0}
  high-P-rate-constant: {A: 4.65e+12, b: 0.44, Ea: 0.0}
- equation: H2O + OH (+M) <=> HO2 + H2 (+M)
  type: falloff
  low-P-rate-constant: {A: 1.0e+15, b: 0.0, Ea: 3.0e+04}
  high-P-rate-constant: {A: 1.0e+12, b: 0.0, Ea: 2.5e+04}
  SRI: {A: 0.5, B: 300.0, C: 1500.0, D: 1.1, E: 0.2}
- equation: H2 + O2 <=> HO2 + H
  type: pressure-dependent-Arrhenius
  rate-constants:
  - {P: 0.01 atm, A: 1.0e+11, b: 0.5, Ea: 4.0e+04}
  - {P: 1.0 atm, A: 4.0e+12, b: 0.3, Ea: 4.2e+04}
  - {P: 1.0 atm, A: 1.0e+11, b: 0.0, Ea: 3.0e+04}
  - {P: 100.0 atm, A: 3.0e+13, b: 0.1, Ea: 4.5e+04}
- equation: 2 OH + M <=> H2O + O + M
  type: three-body
  rate-constant: {A: 1.0e+15, b: -0.5, Ea: 0.0}
  efficiencies: {H2O: 5.0}
"""


def _ideal_gas_phase(path):
    with open(path) as f:
        return [p["name"] for p in yaml.safe_load(f)["phases"] if p.get("thermo") == "ideal-gas"][0]


def _compare(mech, gas, n_states=25, seed=0):
    from autokinetics._ext._core import KineticsModel

    cm = compile_mechanism(mech)
    km = KineticsModel(cm)
    rng = np.random.default_rng(seed)
    worst = {"qf": 0.0, "qr": 0.0, "wdot": 0.0, "ref": 0.0}
    for _ in range(n_states):
        T, P = rng.uniform(400, 2800), 10 ** rng.uniform(3, 7)
        X = rng.random(gas.n_species) ** 3
        gas.TPX = T, P, X / X.sum()
        c = gas.concentrations * 1e3
        qf, qr, _ = km.rates_of_progress(T, P, c)
        w = km.production_rates(T, P, c)
        scale = np.max(np.abs(gas.net_production_rates * 1e3))
        for name, ours, ref in (("qf", qf, gas.forward_rates_of_progress * 1e3),
                                ("qr", qr, gas.reverse_rates_of_progress * 1e3)):
            worst[name] = max(worst[name], np.max(np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-12 * max(np.abs(ref).max(), 1e-300))))
        worst["wdot"] = max(worst["wdot"], np.max(np.abs(w - gas.net_production_rates * 1e3)) / scale)
        w_ref = kinetics_ref.production_rates(cm, T, P, c)
        worst["ref"] = max(worst["ref"], np.max(np.abs(w - w_ref)) / scale)
    return worst


@pytest.mark.parametrize("fname", ["h2o2.yaml", "gri30.yaml", "nDodecane_Reitz.yaml"])
def test_kernel_matches_cantera(fname):
    path = cantera_data(fname)
    mech = load_mechanism(path)
    gas = ct.Solution(path, _ideal_gas_phase(path))
    worst = _compare(mech, gas, n_states=10 if "Dodecane" in fname else 25)
    assert worst["qf"] < 1e-10
    assert worst["qr"] < 1e-9
    assert worst["wdot"] < 1e-10
    assert worst["ref"] < 1e-10


def test_falloff_sri_plog_threebody_match_cantera():
    mech = load_mechanism(FALLOFF_PLOG_YAML)
    gas = ct.Solution(yaml=FALLOFF_PLOG_YAML)
    worst = _compare(mech, gas, n_states=40, seed=7)
    assert worst["qf"] < 1e-10 and worst["qr"] < 1e-9 and worst["wdot"] < 1e-10


def test_thermo_matches_cantera_at_one_bar():
    path = cantera_data("gri30.yaml")
    from autokinetics._ext._core import KineticsModel

    mech = load_mechanism(path)
    km = KineticsModel(compile_mechanism(mech))
    gas = ct.Solution(path)
    for T in (300.0, 1000.0, 2500.0):
        gas.TP = T, 1e5
        cp, h, s = km.species_thermo(T)
        R = ct.gas_constant / 1e3
        np.testing.assert_allclose(cp, gas.standard_cp_R * R, rtol=1e-12)
        np.testing.assert_allclose(h, gas.standard_enthalpies_RT * R * T, rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(s, gas.standard_entropies_R * R, rtol=1e-12)


@pytest.mark.parametrize("fname", ["h2o2.yaml", "gri30.yaml"])
def test_analytic_jacobian_matches_finite_differences(fname):
    from autokinetics._ext._core import KineticsModel

    mech = load_mechanism(cantera_data(fname))
    km = KineticsModel(compile_mechanism(mech))
    rng = np.random.default_rng(2)
    for _ in range(5):
        T, P = rng.uniform(800, 2500), rng.uniform(1e4, 1e6)
        c = rng.random(mech.n_species) * 10 + 1e-3
        J = km.jacobian_conc(T, P, c)
        Jfd = np.empty_like(J)
        for i in range(mech.n_species):
            h = 1e-6 * c[i]
            cp, cm = c.copy(), c.copy()
            cp[i] += h
            cm[i] -= h
            Jfd[:, i] = (km.production_rates(T, P, cp) - km.production_rates(T, P, cm)) / (2 * h)
        assert np.max(np.abs(J - Jfd)) / np.max(np.abs(Jfd)) < 1e-6


def test_reactor_jacobian_energy_and_constant_pressure(h2o2):
    """Full reactor Jacobian (species + temperature, const P / const V) vs finite differences."""
    from autokinetics.reactor import ReactorSpec, build_reactor

    for constraint in ("pressure", "volume"):
        spec = ReactorSpec(constraint=constraint, energy="adiabatic", T0=1200.0, P0=2e5, basis="mole-fraction",
                           initial={"H2": 2, "O2": 1, "AR": 5, "OH": 0.01, "H": 0.01, "O": 0.01, "HO2": 0.001})
        b = build_reactor(h2o2, spec)
        y = b.y0.copy()
        J = b.system.eval_jacobian(0.0, y)
        Jfd = np.empty_like(J)
        for i in range(len(y)):
            h = 1e-6 * max(abs(y[i]), 1e-3)
            yp, ym = y.copy(), y.copy()
            yp[i] += h
            ym[i] -= h
            Jfd[:, i] = (b.system.eval_rhs(0.0, yp) - b.system.eval_rhs(0.0, ym)) / (2 * h)
        err = np.abs(J - Jfd) / (np.abs(Jfd) + 1e-6 * np.max(np.abs(Jfd)))
        assert np.max(err) < 1e-4, (constraint, np.unravel_index(np.argmax(err), err.shape))
