import pytest
from fastapi.testclient import TestClient

from autokinetics.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["extension"] is True


def test_examples_and_simulation_roundtrip(client):
    exs = client.get("/api/examples").json()
    assert any(e["id"] == "n2o5_decomposition" for e in exs)
    ex = client.get("/api/examples/n2o5_decomposition").json()
    preset = ex["meta"]["preset"]
    r = client.post("/api/simulate", json={"mechanism": {"document": ex["document"]},
                                          "reactor": preset["reactor"], "simulation": preset["simulation"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] and body["species"][0] == "N2O5"
    assert body["metrics"]["half_life"]["N2O5"] == pytest.approx(1328.1, rel=1e-3)


def test_parse_reports_validation_issues(client):
    yaml_text = """
units: {length: m, quantity: mol, activation-energy: J/mol}
phases: [{name: g, thermo: ideal-gas, species: [H2, O2, H2O]}]
species: [{name: H2, composition: {H: 2}}, {name: O2, composition: {O: 2}}, {name: H2O, composition: {H: 2, O: 1}}]
reactions: [{equation: H2 + O2 => H2O, rate-constant: {A: 1, b: 0, Ea: 0}}]
"""
    r = client.post("/api/mechanism/parse", json={"yaml": yaml_text})
    assert r.status_code == 200
    assert any(i["level"] == "error" and "Elementbilanz" in i["message"] for i in r.json()["issues"])
    sim = client.post("/api/simulate", json={"mechanism": {"yaml": yaml_text}, "reactor": {"composition": {"H2": 1}},
                                            "simulation": {"t_end": 1}})
    assert sim.status_code == 422 and "Elementbilanz" in sim.json()["detail"]


def test_invalid_inputs_give_422(client):
    ex = client.get("/api/examples/n2o5_decomposition").json()
    bad = client.post("/api/simulate", json={"mechanism": {"document": ex["document"]},
                                            "reactor": {"composition": {"XYZ": 1}}, "simulation": {"t_end": 10}})
    assert bad.status_code == 422 and "unbekannte Spezies" in bad.json()["detail"]
    bad = client.post("/api/simulate", json={"mechanism": {"document": ex["document"]},
                                            "reactor": {"composition": {"N2O5": 1}}, "simulation": {"t_end": -1}})
    assert bad.status_code == 422
    assert client.post("/api/mechanism/parse", json={"yaml": "a: [unclosed"}).status_code == 422


def test_equilibrium_endpoint(client):
    ex = client.get("/api/examples/n2o4_dissociation").json()
    r = client.post("/api/equilibrium", json={"mechanism": {"document": ex["document"]},
                                             "reactor": ex["meta"]["preset"]["reactor"], "mode": "TP"})
    assert r.status_code == 200, r.text
    x = r.json()["mole_fractions"]
    assert x[1] ** 2 / x[0] * 1.01325 == pytest.approx(0.144, rel=0.02)


def test_order_and_arrhenius_endpoints(client):
    import math

    t = [0, 10, 20, 30, 40, 50]
    r = client.post("/api/analysis/order", json={"t": t, "c": [math.exp(-0.05 * x) for x in t]})
    assert r.json()["best"]["order"] == pytest.approx(1.0, abs=1e-3)
    T = [300, 310, 320, 330]
    r = client.post("/api/analysis/arrhenius", json={"T": T, "k": [1e8 * math.exp(-50000 / (8.314462618 * x)) for x in T]})
    assert r.json()["Ea"] == pytest.approx(50000, rel=1e-6)


def test_nasa_thermo_endpoint(client):
    assert client.get("/api/thermo/nasa", params={"name": "H2O"}).status_code == 200
    assert client.get("/api/thermo/nasa", params={"name": "unobtainium"}).status_code == 404
