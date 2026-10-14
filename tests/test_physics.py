"""Physical validation: analytical solutions, conservation laws, thermodynamic
consistency, literature values and agreement with Cantera reactor simulations."""

import math

import numpy as np
import pytest

from autokinetics.constants import GAS_CONSTANT as R
from autokinetics.equilibrium import equilibrate
from autokinetics.io import load_mechanism
from autokinetics.library import load_example
from autokinetics.presets import options_from_dict, reactor_spec_from_dict
from autokinetics.reactor import ReactorSpec
from autokinetics.simulation import SimulationOptions, simulate

LIQ = """
units: {length: dm, quantity: mol, activation-energy: kJ/mol}
phases: [{name: l, thermo: liquid, species: %s}]
species: %s
reactions: %s
"""


def liquid(species, reactions):
    sp = "[" + ", ".join("{name: " + s + "}" for s in species) + "]"
    return load_mechanism(LIQ % ("[" + ", ".join(species) + "]", sp, reactions))


def run(mech, initial_molL, t_end, T=298.15, rtol=1e-10, **kw):
    spec = ReactorSpec(T0=T, initial={k: v * 1e3 for k, v in initial_molL.items()}, **kw)
    opts = SimulationOptions(t_end=t_end, rtol=rtol, atol_rel=1e-14,
                             t_eval=np.linspace(0, t_end, 201), max_points=10000)
    return simulate(mech, spec, opts)


# ----------------------------------------------------------------------------- analytical solutions
def test_first_order_decay():
    m = liquid(["A", "B"], "[{equation: A => B, rate-constant: {A: 0.3, b: 0, Ea: 0}}]")
    r = run(m, {"A": 1.0}, 20.0)
    np.testing.assert_allclose(r.species("A") / 1e3, np.exp(-0.3 * r.t), rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose((r.species("A") + r.species("B")) / 1e3, 1.0, rtol=1e-12)


def test_second_order_equal_and_unequal_concentrations():
    k = 0.5  # L/(mol s)
    m = liquid(["A", "B", "P"], f"[{{equation: A + B => P, rate-constant: {{A: {k}, b: 0, Ea: 0}}}}]")
    r = run(m, {"A": 1.0, "B": 1.0}, 20.0)
    np.testing.assert_allclose(r.species("A") / 1e3, 1.0 / (1.0 + k * r.t), rtol=1e-7)
    a0, b0 = 1.0, 0.4
    r = run(m, {"A": a0, "B": b0}, 20.0)
    expected = a0 * (a0 - b0) / (a0 - b0 * np.exp(-(a0 - b0) * k * r.t))
    np.testing.assert_allclose(r.species("A") / 1e3, expected, rtol=1e-7)


def test_zero_order_and_fractional_order():
    m = liquid(["A", "B"], "[{equation: A => B, rate-constant: {A: 0.01, b: 0, Ea: 0}, orders: {A: 0.5}}]")
    r = run(m, {"A": 1.0}, 50.0)
    k_si = 0.01 * (1e3) ** 0.5  # (mol/L)^0.5/s -> (mol/m³)^0.5/s
    expected = (np.sqrt(1e3) - 0.5 * k_si * r.t) ** 2  # sqrt(c) = sqrt(c0) - k t / 2
    np.testing.assert_allclose(r.species("A"), expected, rtol=1e-6)


def test_consecutive_reactions_bateman():
    k1, k2 = 1e-3, 5e-4
    m = liquid(["A", "B", "C"], f"[{{equation: A => B, rate-constant: {{A: {k1}, b: 0, Ea: 0}}}}, "
                                f"{{equation: B => C, rate-constant: {{A: {k2}, b: 0, Ea: 0}}}}]")
    r = run(m, {"A": 1.0}, 10000.0)
    b = k1 / (k2 - k1) * (np.exp(-k1 * r.t) - np.exp(-k2 * r.t))
    np.testing.assert_allclose(r.species("B") / 1e3, b, rtol=1e-6, atol=1e-12)


def test_reversible_first_order_relaxation():
    kf, kr = 2.0, 0.5
    m = liquid(["A", "B"], f"[{{equation: A <=> B, rate-constant: {{A: {kf}, b: 0, Ea: 0}}, "
                           f"reverse-rate-constant: {{A: {kr}, b: 0, Ea: 0}}}}]")
    r = run(m, {"A": 1.0}, 5.0)
    a_eq = kr / (kf + kr)
    expected = a_eq + (1 - a_eq) * np.exp(-(kf + kr) * r.t)
    np.testing.assert_allclose(r.species("A") / 1e3, expected, rtol=1e-7)


def test_cstr_steady_state_and_startup():
    m = liquid(["A", "B"], "[{equation: A => B, rate-constant: {A: 0.01, b: 0, Ea: 0}}]")
    spec = ReactorSpec(kind="cstr", tau=100.0, T0=298.15, initial={}, feed={"A": 1000.0})
    r = simulate(m, spec, SimulationOptions(t_end=1000.0, rtol=1e-10, atol_rel=1e-14,
                                            t_eval=np.linspace(0, 1000, 101)))
    k, tau = 0.01, 100.0
    css = 1000.0 / (1 + k * tau)
    expected = css * (1 - np.exp(-(1 / tau + k) * r.t))
    np.testing.assert_allclose(r.species("A"), expected, rtol=1e-7, atol=1e-9)


def test_temperature_program_follows_arrhenius():
    """Isothermal steps: dln[A]/dt must equal -k(T(t)) with the programmed T."""
    m = liquid(["A", "B"], "[{equation: A => B, rate-constant: {A: 1.0e10, b: 0, Ea: 70.0}}]")
    spec = ReactorSpec(T0=300.0, initial={"A": 1000.0}, T_program=[(0.0, 300.0), (100.0, 300.0), (100.0001, 330.0), (400.0, 330.0)])
    # A decays by ~25 e-folds: atol must lie far below the final value (7.5e-9 mol/m³)
    r = simulate(m, spec, SimulationOptions(t_end=400.0, rtol=1e-10, atol_rel=1e-20, t_eval=np.array([0, 100, 400.0])))
    assert r.stats["segments"] == 3  # integration restarted at the programme breakpoints
    k = lambda T: 1e10 * math.exp(-70000 / (R * T))  # noqa: E731
    a100 = 1000 * math.exp(-k(300) * 100)
    a400 = a100 * math.exp(-k(330) * (300.0 - 0.00005))  # linear ramp over 1e-4 s ≈ midpoint T
    assert r.species("A")[1] == pytest.approx(a100, rel=1e-6)
    assert r.species("A")[2] == pytest.approx(a400, rel=1e-5)


def test_michaelis_menten_quasi_steady_state():
    """Elementary-step model vs integrated Michaelis–Menten equation (sQSSA valid: E0 ≪ S0 + K_M).

    The sQSSA neglects substrate bound in ES; its relative error grows like (E0/K_M)·(V_max t/K_M)
    in the exponential tail, so the comparison is restricted to S ≥ 1e-3 S0."""
    from scipy.special import lambertw

    mech, meta, _ = load_example("michaelis_menten")
    r = simulate(mech, reactor_spec_from_dict(meta.preset["reactor"], mech),
                 SimulationOptions(t_end=400.0, rtol=1e-10, atol_rel=1e-16, t_eval=np.linspace(0, 400, 41)))
    k1, km1, kcat = 1e6, 1e3, 100.0
    KM = (km1 + kcat) / k1  # mol/L
    E0, S0 = 1e-6, 1e-2
    Vmax = kcat * E0
    S_mm = KM * np.real(lambertw(S0 / KM * np.exp((S0 - Vmax * r.t) / KM)))
    valid = (r.t > 1.0) & (S_mm > 1e-3 * S0)
    assert valid.sum() > 10
    np.testing.assert_allclose(r.species("S")[valid] / 1e3, S_mm[valid], rtol=3e-3)


# ----------------------------------------------------------------------------- conservation laws
def test_robertson_reference_solution_and_mass_conservation():
    """Hairer & Wanner reference values of the Robertson problem at t = 40."""
    mech, _, _ = load_example("robertson")
    spec = ReactorSpec(T0=298.15, initial={"A": 1000.0})
    r = simulate(mech, spec, SimulationOptions(t_end=40.0, rtol=1e-11, atol_rel=1e-16, t_eval=np.array([0.0, 40.0])))
    y = r.concentrations[-1] / 1e3
    assert y[0] == pytest.approx(0.7158270687193, rel=1e-6)
    assert y[1] == pytest.approx(0.9185534764529e-5, rel=1e-5)
    assert y[2] == pytest.approx(0.2841637457262, rel=1e-6)
    assert y.sum() == pytest.approx(1.0, abs=1e-10)


def test_ignition_conserves_elements_mass_and_enthalpy(h2o2):
    from autokinetics._ext._core import KineticsModel

    from autokinetics.compiled import compile_mechanism

    spec = ReactorSpec(constraint="pressure", energy="adiabatic", T0=1000.0, P0=101325.0, basis="mole-fraction",
                       initial={"H2": 2.0, "O2": 1.0, "AR": 7.0})
    r = simulate(h2o2, spec, SimulationOptions(t_end=2e-3, rtol=1e-9, atol_rel=1e-15))
    assert r.success
    assert r.diagnostics["element_conservation_max"] < 1e-12
    assert r.diagnostics["mass_conservation"] < 1e-12
    km = KineticsModel(compile_mechanism(h2o2))
    n = r.amounts()
    H = np.array([n[k] @ km.species_thermo(r.T[k])[1] for k in range(len(r.t))])
    assert np.max(np.abs(H - H[0])) / np.max(np.abs(n[0] @ km.species_thermo(r.T[0])[0] * r.T[0])) < 1e-7
    np.testing.assert_allclose(r.P, 101325.0, rtol=1e-12)


def test_constant_volume_conserves_internal_energy(h2o2):
    from autokinetics._ext._core import KineticsModel

    from autokinetics.compiled import compile_mechanism

    spec = ReactorSpec(constraint="volume", energy="adiabatic", T0=1100.0, P0=101325.0, basis="mole-fraction",
                       initial={"H2": 2.0, "O2": 1.0, "N2": 4.0})
    r = simulate(h2o2, spec, SimulationOptions(t_end=1e-3, rtol=1e-9, atol_rel=1e-15))
    km = KineticsModel(compile_mechanism(h2o2))
    U = np.array([r.concentrations[k] @ (km.species_thermo(r.T[k])[1] - R * r.T[k]) for k in range(len(r.t))])
    assert np.max(np.abs(U - U[0])) / np.max(np.abs(r.concentrations[0] @ km.species_thermo(r.T[0])[0]) * r.T[0]) < 1e-7
    np.testing.assert_allclose(r.V, 1.0)


def test_liquid_adiabatic_temperature_rise():
    """Full conversion of an exothermic liquid reaction: ΔT_ad = c0 (−ΔH) / (ρ cp)."""
    m = load_mechanism("""
units: {length: dm, quantity: mol, activation-energy: kJ/mol}
phases: [{name: l, thermo: liquid, species: [A, B]}]
species: [{name: A}, {name: B}]
reactions: [{equation: A => B, rate-constant: {A: 1.0e8, b: 0, Ea: 60}, reaction-enthalpy: -80}]
""")
    spec = ReactorSpec(energy="adiabatic", T0=320.0, initial={"A": 2000.0}, rho_cp=4.0e6)
    r = simulate(m, spec, SimulationOptions(t_end=5e4, rtol=1e-10, atol_rel=1e-14))
    assert r.metrics["conversion"]["A"] == pytest.approx(1.0, abs=1e-8)
    assert r.T[-1] - 320.0 == pytest.approx(2000.0 * 80e3 / 4.0e6, rel=1e-8)


def test_thermal_runaway_semenov_transition():
    mech, meta, _ = load_example("thermal_runaway")
    peaks = {}
    for UA in (3e4, 1e4):
        d = dict(meta.preset["reactor"], UA_V=UA)
        r = simulate(mech, reactor_spec_from_dict(d, mech), options_from_dict({"t_end": 20000, "rtol": 1e-8}))
        peaks[UA] = r.T.max() - 340.0
    assert peaks[3e4] < 15.0  # controlled
    assert peaks[1e4] > 50.0  # runaway: large part of ΔT_ad = 100 K released quickly


def test_oregonator_oscillates():
    from scipy.signal import find_peaks

    mech, meta, _ = load_example("oregonator")
    r = simulate(mech, reactor_spec_from_dict(meta.preset["reactor"], mech),
                 SimulationOptions(t_end=2000.0, rtol=1e-8, max_points=20000))
    peaks, _ = find_peaks(np.log(r.species("Ce4+")), prominence=2.0)
    assert len(peaks) >= 3
    periods = np.diff(r.t[peaks])
    assert np.std(periods[-2:]) / np.mean(periods[-2:]) < 0.05  # limit cycle: regular period


# ----------------------------------------------------------------------------- thermodynamics
def test_kinetic_long_time_limit_equals_gibbs_equilibrium(h2o2):
    """Reversible, thermodynamically consistent kinetics must relax to the Gibbs minimum."""
    spec = ReactorSpec(constraint="volume", energy="isothermal", T0=2500.0, P0=101325.0, basis="mole-fraction",
                       initial={"H2": 2.0, "O2": 1.0, "AR": 3.0})
    r = simulate(h2o2, spec, SimulationOptions(t_end=1.0, rtol=1e-10, atol_rel=1e-16))
    eq = equilibrate(h2o2, r.concentrations[0], 2500.0, V=1.0, mode="TV")
    np.testing.assert_allclose(r.mole_fractions()[-1], eq.mole_fractions, atol=1e-8)


def test_adiabatic_flame_temperatures_literature(gri30):
    """Adiabatic flame temperatures at 298 K, 1 atm (textbook: CH4/air ≈ 2226 K, H2/air ≈ 2390 K
    incl. dissociation; values depend slightly on thermo data)."""
    air = {"O2": 1.0, "N2": 3.76}
    for fuel, nu_o2, expected, tol in (("CH4", 2.0, 2226.0, 10.0), ("H2", 0.5, 2390.0, 15.0)):
        x = {fuel: 1.0, "O2": nu_o2 * air["O2"], "N2": nu_o2 * air["N2"]}
        n0 = np.zeros(gri30.n_species)
        for k, v in x.items():
            n0[gri30.species_index(k)] = v
        eq = equilibrate(gri30, n0, 298.15, P=101325.0, mode="HP")
        assert eq.T == pytest.approx(expected, abs=tol), fuel


def test_n2o4_equilibrium_constant_and_le_chatelier():
    mech, meta, _ = load_example("n2o4_dissociation")
    alphas = {}
    for P in (101325.0, 10 * 101325.0):
        spec = ReactorSpec(constraint="pressure", T0=298.15, P0=P, basis="mole-fraction", initial={"N2O4": 1.0})
        r = simulate(mech, spec, SimulationOptions(t_end=1e-2, rtol=1e-10, atol_rel=1e-15))
        x = r.mole_fractions()[-1]
        Kp = x[1] ** 2 / x[0] * P / 1e5
        assert Kp == pytest.approx(0.15, rel=0.10)  # textbook value at 298 K
        alpha = r.amounts()[-1][1] / 2 / r.amounts()[0][0]
        assert alpha == pytest.approx(math.sqrt(Kp / (Kp + 4 * P / 1e5)), rel=1e-6)
        alphas[P] = alpha
    assert alphas[10 * 101325.0] < alphas[101325.0]  # Le Chatelier: pressure suppresses dissociation


def test_esterification_equilibrium_conversion():
    mech, meta, _ = load_example("esterification")
    r = simulate(mech, reactor_spec_from_dict(meta.preset["reactor"], mech),
                 SimulationOptions(t_end=1e6, rtol=1e-10, atol_rel=1e-15))
    assert r.metrics["conversion"]["CH3COOH"] == pytest.approx(2 / 3, rel=1e-6)
    n0 = r.concentrations[0]
    eq = equilibrate(mech, n0, 298.15, V=1.0, mode="TV")
    np.testing.assert_allclose(eq.concentrations, r.concentrations[-1], rtol=1e-6)


def test_detailed_balance_reverse_rate_constants(gri30):
    """k_f / k_r = K_c(T) from ΔG° for every thermo-reversible reaction."""
    from autokinetics._ext._core import KineticsModel

    from autokinetics.compiled import compile_mechanism

    km = KineticsModel(compile_mechanism(gri30))
    T = 1500.0
    c = np.ones(gri30.n_species)
    kf, kr, _ = km.rate_constants(T, 1e5, c)
    Kc = km.equilibrium_constants(T)
    _, h, s = km.species_thermo(T)
    nr, np_ = gri30.stoichiometric_matrices()
    nu = np_ - nr
    dG = nu @ (h - T * s)
    Kc_expected = np.exp(-dG / (R * T)) * (1e5 / (R * T)) ** nu.sum(axis=1)
    rev = np.array([r.reversible for r in gri30.reactions])
    np.testing.assert_allclose(Kc[rev], Kc_expected[rev], rtol=1e-10)
    ok = rev & (kr > 0)
    np.testing.assert_allclose(kf[ok] / kr[ok], Kc[ok], rtol=1e-12)


# ----------------------------------------------------------------------------- vs. Cantera reactors
@pytest.mark.parametrize("constraint", ["pressure", "volume"])
def test_ignition_delay_matches_cantera(h2o2, constraint):
    from helpers import cantera_data

    ct = pytest.importorskip("cantera")
    T0, P0, X = 1000.0, 101325.0, "H2:2, O2:1, AR:7"
    spec = ReactorSpec(constraint=constraint, energy="adiabatic", T0=T0, P0=P0, basis="mole-fraction",
                       initial={"H2": 2.0, "O2": 1.0, "AR": 7.0})
    r = simulate(h2o2, spec, SimulationOptions(t_end=2e-3, rtol=1e-10, atol_rel=1e-16, max_points=100000))
    gas = ct.Solution(cantera_data("h2o2.yaml"))
    gas.TPX = T0, P0, X
    reac = (ct.IdealGasConstPressureReactor if constraint == "pressure" else ct.IdealGasReactor)(gas, clone=False)
    net = ct.ReactorNet([reac])
    net.rtol, net.atol = 1e-12, 1e-22
    ts, Ts = [0.0], [T0]
    while net.time < 2e-3:
        net.step()
        ts.append(net.time)
        Ts.append(reac.T)
    ts, Ts = np.array(ts), np.array(Ts)
    tau_ct = ts[np.argmax(np.gradient(Ts, ts))]
    assert r.metrics["ignition_delay"] == pytest.approx(tau_ct, rel=2e-3)
    assert r.T[-1] == pytest.approx(Ts[-1], rel=1e-6)


@pytest.mark.slow
def test_library_expected_values():
    """Every example's documented reference value is reproduced by the simulation."""
    from autokinetics.library import list_examples

    for ex in list_examples():
        if not ex.preset.get("expected"):
            continue
        mech, meta, _ = load_example(ex.id)
        spec = reactor_spec_from_dict(meta.preset["reactor"], mech)
        sim = dict(meta.preset["simulation"], rtol=1e-10, atol_rel=1e-15, max_points=20000)
        r = simulate(mech, spec, options_from_dict(sim))
        for e in meta.preset["expected"]:
            sp = e.get("species")
            if e["quantity"] == "half_life":
                v = r.metrics["half_life"][sp]
            elif e["quantity"] == "conversion":
                v = r.metrics["conversion"][sp]
            elif e["quantity"] == "max_time":
                v = r.metrics["maxima"][sp]["t"]
            elif e["quantity"] == "max_value":
                v = r.metrics["maxima"][sp]["value"] / 1e3
            elif e["quantity"] == "final_value":
                v = float(r.species(sp)[-1]) / 1e3
            else:
                raise AssertionError(e)
            assert v == pytest.approx(e["value"], rel=e["tolerance"]), (ex.id, e)
