import math

import pytest

from autokinetics.composition import hill_formula, molar_mass, parse_formula
from autokinetics.equation import parse_equation
from autokinetics.errors import MechanismError, UnitError
from autokinetics.io.yaml_format import composite_unit_factor
from autokinetics.units import UnitSystem, activation_energy_to_J_per_mol, concentration_to_SI


@pytest.mark.parametrize("formula,comp,charge", [
    ("H2O", {"H": 2, "O": 1}, 0),
    ("C2H5OH", {"C": 2, "H": 6, "O": 1}, 0),
    ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}, 0),
    ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}, 0),
    ("Fe(CN)6-4", {"Fe": 1, "C": 6, "N": 6}, -4),
    ("SO4^2-", {"S": 1, "O": 4}, -2),
    ("Na+", {"Na": 1}, 1),
    ("NH4+", {"N": 1, "H": 4}, 1),
])
def test_formula_parsing(formula, comp, charge):
    c, q = parse_formula(formula)
    assert c == pytest.approx(comp)
    assert q == charge


def test_formula_errors():
    for bad in ["Xx2", "C2H(", "H2)O", ""]:
        with pytest.raises(MechanismError):
            parse_formula(bad)


def test_molar_mass_and_hill():
    assert molar_mass({"C": 1, "O": 2}) == pytest.approx(44.009e-3, rel=1e-4)
    assert molar_mass({"H": 2, "O": 1}) * 1e3 == pytest.approx(18.015, rel=1e-4)
    assert hill_formula({"O": 1, "H": 6, "C": 2}) == "C2H6O"


def test_equation_variants():
    sp = ["H2", "O2", "H2O", "H", "O", "OH", "HO2", "AR", "CH2(S)", "N2"]
    e = parse_equation("2 H2 + O2 => 2 H2O", sp)
    assert e.reactants == {"H2": 2, "O2": 1} and e.products == {"H2O": 2} and not e.reversible
    e = parse_equation("2 O + M <=> O2 + M", sp)
    assert e.third_body == "M" and e.reversible
    e = parse_equation("H + O2 (+M) <=> HO2 (+M)", sp)
    assert e.falloff_collider == "M"
    e = parse_equation("H + O2 (+ AR) = HO2 (+AR)", sp)
    assert e.falloff_collider == "AR" and e.reversible
    e = parse_equation("CH2(S) + N2 -> H2 + N2", sp)
    assert e.reactants == {"CH2(S)": 1, "N2": 1}
    e = parse_equation("H2+O2⇌2OH", sp)
    assert e.reactants == {"H2": 1, "O2": 1} and e.products == {"OH": 2} and e.reversible
    with pytest.raises(MechanismError):
        parse_equation("H2 + X => H2O", sp)
    with pytest.raises(MechanismError):
        parse_equation("H2 + O2", sp)
    with pytest.raises(MechanismError):
        parse_equation("2 O + M <=> O2", sp)


def test_units():
    assert concentration_to_SI(1.0, "mol/L") == 1000.0
    assert activation_energy_to_J_per_mol(1.0, "kcal/mol") == pytest.approx(4184.0)
    assert activation_energy_to_J_per_mol(1000.0, "K") == pytest.approx(8314.462618, rel=1e-9)
    us = UnitSystem.from_mapping({"length": "cm", "quantity": "mol", "activation-energy": "cal/mol"})
    assert us.concentration_factor == pytest.approx(1e6)  # mol/cm³ -> mol/m³
    assert us.rate_constant_factor(2.0) == pytest.approx(1e-6)  # cm³/mol/s -> m³/mol/s
    assert composite_unit_factor("cm^3/mol/s") == pytest.approx(1e-6)
    assert composite_unit_factor("L/mol/s") == pytest.approx(1e-3)
    assert composite_unit_factor("m^6/kmol^2/s") == pytest.approx(1e-6)
    with pytest.raises(UnitError):
        UnitSystem.from_mapping({"length": "furlong"})
    assert math.isclose(UnitSystem().concentration_factor, 1000.0)  # Cantera default kmol/m³
