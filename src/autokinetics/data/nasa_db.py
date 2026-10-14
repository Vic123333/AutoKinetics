"""Bundled NASA thermodynamic database (McBride, Gordon & Reno, NASA TM-4513, 1993).

748 gas-phase species as NASA-7 polynomials (via Cantera's ``nasa_gas.yaml``).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

from ..composition import hill_formula
from ..io.yaml_format import safe_load_yaml
from ..thermo import Nasa9Thermo, from_nasa7


@lru_cache(maxsize=1)
def _database() -> list[dict]:
    text = resources.files("autokinetics.library").joinpath("thermo/nasa_gas.yaml").read_text(encoding="utf-8")
    doc = safe_load_yaml(text)
    out = []
    for sp in doc["species"]:
        comp = {str(k): float(v) for k, v in sp.get("composition", {}).items()}
        out.append({"name": str(sp["name"]), "composition": comp, "thermo": sp["thermo"],
                    "note": sp["thermo"].get("note", "")})
    return out


def _thermo(entry: dict) -> Nasa9Thermo:
    th = entry["thermo"]
    return from_nasa7(th["temperature-ranges"], th["data"],
                      source=f"NASA TM-4513 (McBride et al. 1993), {entry['note']}".strip(", "))


def by_name(name: str) -> tuple[dict, Nasa9Thermo] | None:
    key = name.strip().upper()
    for e in _database():
        if e["name"].upper() == key:
            return e, _thermo(e)
    return None


def by_composition(composition: dict[str, float], charge: float = 0.0) -> list[tuple[dict, Nasa9Thermo]]:
    target = {k: float(v) for k, v in composition.items() if v}
    if charge:
        target["E"] = -float(charge)
    hits = []
    for e in _database():
        comp = {k: v for k, v in e["composition"].items() if v}
        if comp == target:
            hits.append((e, _thermo(e)))
    return hits


def formula_of(entry: dict) -> str:
    return hill_formula({k: v for k, v in entry["composition"].items() if k != "E"})
