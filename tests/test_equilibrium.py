import numpy as np
import pytest

from autokinetics.equilibrium import conservation_matrix, equilibrate
from autokinetics.errors import SimulationError
from autokinetics.io import load_mechanism
from helpers import cantera_data

ct = pytest.importorskip("cantera")


@pytest.mark.parametrize("mode", ["TP", "HP", "UV", "TV"])
@pytest.mark.parametrize("mixture", ["CH4:1, O2:2, N2:7.52", "H2:2, O2:1, AR:3", "C2H6:1, O2:2, N2:5"])
def test_equilibrium_matches_cantera(gri30, mode, mixture):
    gas = ct.Solution(cantera_data("gri30.yaml"))
    T0 = 300.0 if mode in ("HP", "UV") else 2200.0
    gas.TPX = T0, 2e5, mixture
    n0 = gas.X.copy()
    kw = {"P": 2e5} if mode in ("TP", "HP") else {"V": n0.sum() * 8.314462618 * T0 / 2e5}
    eq = equilibrate(gri30, n0, T0, mode=mode, **kw)
    gas.equilibrate(mode)
    assert eq.T == pytest.approx(gas.T, rel=1e-9)
    np.testing.assert_allclose(eq.mole_fractions, gas.X, atol=1e-10)
    assert eq.P == pytest.approx(gas.P, rel=1e-8)


def test_reaction_based_conservation_laws():
    nu = np.array([[-1, 1, 0], [0, -1, 1]], float)  # A -> B -> C : one conserved total
    W = conservation_matrix(nu)
    assert W.shape == (1, 3)
    np.testing.assert_allclose(W @ nu.T, 0, atol=1e-12)


def test_equilibrium_from_equilibrium_constants_and_wegscheider():
    ok = load_mechanism("""
units: {length: dm, quantity: mol, activation-energy: kJ/mol}
phases: [{name: l, thermo: liquid, species: [A, B, C]}]
species: [{name: A}, {name: B}, {name: C}]
reactions:
- {equation: A <=> B, rate-constant: {A: 1, b: 0, Ea: 0}, equilibrium-constant: {K: 2.0}}
- {equation: B <=> C, rate-constant: {A: 1, b: 0, Ea: 0}, equilibrium-constant: {K: 3.0}}
- {equation: A <=> C, rate-constant: {A: 1, b: 0, Ea: 0}, equilibrium-constant: {K: 6.0}}
""")
    eq = equilibrate(ok, np.array([1.0, 0, 0]), 298.15, V=1.0, mode="TV")
    np.testing.assert_allclose(eq.amounts, np.array([1, 2, 6]) / 9.0, rtol=1e-10)
    bad = load_mechanism("""
units: {length: dm, quantity: mol, activation-energy: kJ/mol}
phases: [{name: l, thermo: liquid, species: [A, B, C]}]
species: [{name: A}, {name: B}, {name: C}]
reactions:
- {equation: A <=> B, rate-constant: {A: 1, b: 0, Ea: 0}, equilibrium-constant: {K: 2.0}}
- {equation: B <=> C, rate-constant: {A: 1, b: 0, Ea: 0}, equilibrium-constant: {K: 3.0}}
- {equation: A <=> C, rate-constant: {A: 1, b: 0, Ea: 0}, equilibrium-constant: {K: 5.0}}
""")
    with pytest.raises(SimulationError, match="Wegscheider"):
        equilibrate(bad, np.array([1.0, 0, 0]), 298.15, V=1.0, mode="TV")
