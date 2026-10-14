import pytest

from autokinetics.data import nasa_db, nist
from autokinetics.thermo import from_shomate
from helpers import FIXTURES


def test_nist_parser_methane_fixture():
    info = nist.parse_thermo_page((FIXTURES / "nist_methane.html").read_text(encoding="utf-8"))
    assert info["formula"] == "CH4" and info["cas"] == "74-82-8"
    assert info["dHf_gas"] == pytest.approx(-74.87e3)
    assert info["S_gas"] == pytest.approx(186.25)
    th = nist.thermo_from_page(info)
    assert th.h(298.15) == pytest.approx(-74.87e3, abs=20)  # Shomate fit reproduces ΔfH°
    assert th.s(298.15) == pytest.approx(186.25, abs=0.05)
    assert th.cp(298.15) == pytest.approx(35.69, abs=0.1)


def test_nist_cp_points_polynomial_fixture():
    info = nist.parse_thermo_page((FIXTURES / "nist_ethyl_acetate.html").read_text(encoding="utf-8"))
    assert info["shomate"] is None and len(info["cp_points"]) >= 5
    th = nist.thermo_from_cp_points(info["dHf_gas"], info["S_gas"], info["cp_points"], "test")
    assert th.h(298.15) == pytest.approx(info["dHf_gas"], abs=1e-6)
    assert th.s(298.15) == pytest.approx(info["S_gas"], abs=1e-9)
    for T, cp in info["cp_points"]:
        assert th.cp(T) == pytest.approx(cp, rel=0.03)


def test_shomate_conversion_is_exact():
    # water vapour, NIST/Chase 1998, 500–1700 K
    data = [[30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967]]
    th = from_shomate([500.0, 1700.0], data)
    t = 1.0
    A, B, C, D, E, F, G = data[0]
    assert th.cp(1000.0) == pytest.approx(A + B * t + C * t**2 + D * t**3 + E / t**2, rel=1e-12)
    assert th.h(1000.0) == pytest.approx(1e3 * (A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F), rel=1e-12)
    import math
    assert th.s(1000.0) == pytest.approx(A * math.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G, rel=1e-12)


def test_nasa_database_offline():
    e, th = nasa_db.by_name("H2O")
    assert th.at_standard_pressure().h(298.15) == pytest.approx(-241.83e3, abs=100)
    names = {x[0]["name"] for x in nasa_db.by_composition({"C": 2, "H": 6, "O": 1})}
    assert {"C2H5OH", "CH3OCH3"} <= names


@pytest.mark.network
def test_live_lookup_pubchem_and_nist():
    from autokinetics.data import lookup_species

    info = lookup_species("methane")
    assert info.pubchem and info.pubchem["cid"] == 297 and info.cas == "74-82-8"
    assert info.dHf_298 == pytest.approx(-74.87e3, abs=200)
    eth = lookup_species("ethanol")
    assert eth.dHf_298 == pytest.approx(-234.95e3, abs=2e3)  # isomer resolved via NIST ΔfH°
    ac = lookup_species("acetone")
    assert any("anderes Isomer" in w for w in ac.warnings)  # wrong NASA isomer rejected
