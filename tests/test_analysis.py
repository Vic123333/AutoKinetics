import numpy as np
import pytest

from autokinetics import analysis
from autokinetics.library import load_example
from autokinetics.presets import options_from_dict, reactor_spec_from_dict
from autokinetics.simulation import simulate


@pytest.fixture(scope="module")
def saponification():
    mech, meta, _ = load_example("saponification")
    spec = reactor_spec_from_dict(meta.preset["reactor"], mech)
    opts = options_from_dict(meta.preset["simulation"])
    return mech, spec, opts, simulate(mech, spec, opts)


def test_reaction_order_detected(saponification):
    _, _, _, r = saponification
    fit = analysis.fit_reaction_order(r.t, r.species("CH3COOC2H5") / 1e3)
    assert fit["best"]["order"] == pytest.approx(2.0, abs=1e-3)
    assert fit["best"]["k"] == pytest.approx(0.11, rel=1e-3)
    t = np.linspace(0, 100, 30)
    fit1 = analysis.fit_reaction_order(t, 2.0 * np.exp(-0.03 * t))
    assert fit1["best"]["order"] == pytest.approx(1.0, abs=1e-3)


def test_arrhenius_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    T = np.linspace(280, 360, 9)
    k = 3e9 * np.exp(-60000 / (8.31446261815324 * T)) * np.exp(0.005 * rng.standard_normal(T.size))
    fit = analysis.fit_arrhenius(T, k)
    assert fit["Ea_ci95"][0] < 60000 < fit["Ea_ci95"][1]
    assert fit["Ea"] == pytest.approx(60000, rel=0.02)


def test_parameter_estimation_recovers_rate_constant(saponification):
    mech, spec, opts, r = saponification
    true_A = mech.reactions[0].rate.A
    rng = np.random.default_rng(0)
    t = np.linspace(60, 3000, 15)
    c = np.interp(t, r.t, r.species("CH3COOC2H5")) * (1 + 0.01 * rng.standard_normal(t.size))
    wrong, _, _ = load_example("saponification")
    wrong.reactions[0].rate.A *= 3.0
    est = analysis.estimate_parameters(wrong, spec, opts, [{"species": "CH3COOC2H5", "t": t, "c": c}],
                                       [{"reaction": 0, "param": "A"}])
    assert est["success"]
    v = est["estimates"][0]
    assert v["value"] == pytest.approx(true_A, rel=4 * v["rel_std_error"] + 1e-3)


def test_sensitivity_identifies_chain_branching(h2o2):
    mech, meta, _ = load_example("hydrogen_oxygen_ignition")
    spec = reactor_spec_from_dict(meta.preset["reactor"], mech)
    opts = options_from_dict(meta.preset["simulation"])
    s = analysis.sensitivity(mech, spec, opts, "ignition_delay", max_reactions=8)
    top = s["results"][0]
    assert top["equation"] == "H + O2 <=> O + OH"
    assert top["sensitivity"] < -0.5  # faster branching -> shorter ignition delay


def test_rates_of_production(saponification):
    _, _, _, r = saponification
    rop = analysis.rates_of_production(r, "CH3COO-")
    assert rop["reactions"][0]["integral"] == pytest.approx(r.species("CH3COO-")[-1], rel=1e-3)
