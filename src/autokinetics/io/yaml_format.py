"""Cantera-compatible YAML mechanism format.

Supported subset of Cantera's YAML (https://cantera.org/documentation/docs-3.0/sphinx/html/yaml):

* top-level ``units``, ``phases`` (thermo ``ideal-gas``; AutoKinetics extension ``liquid``),
  ``species``, ``reactions`` (or named reaction sections)
* species thermo models ``NASA7``, ``NASA9``, ``Shomate``, ``constant-cp``
* reaction types: elementary, three-body, falloff (Lindemann, Troe, SRI),
  pressure-dependent-Arrhenius (PLOG); ``efficiencies``, ``default-efficiency``,
  ``orders``, ``duplicate``, ``negative-A``
* quantities with inline units (``Ea: 15 kcal/mol``, ``P: 1 atm``, ``A: 1e13 cm^3/mol/s``)

AutoKinetics extensions (ignored by Cantera when prefixed ``x-``/unknown):
``reverse-rate-constant``, ``equilibrium-constant``, ``reaction-enthalpy``, top-level
``description``, ``references``, ``autokinetics`` (reactor presets & metadata).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..composition import try_parse_formula
from ..equation import parse_equation
from ..errors import MechanismError, UnitError
from ..mechanism import Arrhenius, EquilibriumConstant, Falloff, Mechanism, Reaction, Species
from ..thermo import Nasa9Thermo, from_constant_cp, from_nasa7, from_nasa9, from_shomate
from ..units import (
    ENERGY_PER_QUANTITY,
    LENGTH,
    PRESSURE,
    QUANTITY,
    TIME,
    UnitSystem,
    parse_quantity,
)


class _Yaml12Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false, so species such as NO,
    ON or Y stay strings (PyYAML's YAML 1.1 default would turn NO into False)."""


_Yaml12Loader.yaml_implicit_resolvers = {
    key: [(tag, regex) for tag, regex in resolvers if tag != "tag:yaml.org,2002:bool"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))


def safe_load_yaml(text: str):
    return yaml.load(text, Loader=_Yaml12Loader)  # noqa: S506 - SafeLoader subclass


SUPPORTED_THERMO = {"ideal-gas": "ideal-gas", "liquid": "liquid", "ideal-dilute-solution": "liquid",
                    "constant-density": "liquid"}
SUPPORTED_REACTION_TYPES = {"elementary", "three-body", "falloff", "pressure-dependent-Arrhenius"}


# ---------------------------------------------------------------------------- units
_UNIT_FACTORS: dict[str, tuple[float, str]] = {}
for _table, _dim in ((QUANTITY, "quantity"), (LENGTH, "length"), (TIME, "time")):
    for _k, _v in _table.items():
        _UNIT_FACTORS[_k] = (_v, _dim)
_UNIT_FACTORS["L"] = (1.0e-3, "volume")  # litre = dm³


def composite_unit_factor(unit: str) -> float:
    """Factor converting e.g. 'cm^3/mol/s' or 'L/mol/s' or 'm^6/kmol^2/s' to SI."""
    unit = unit.replace(" ", "").replace("**", "^")
    factor = 1.0
    sign = 1
    for token in re.split(r"([/*])", unit):
        if token in ("", "*"):
            continue
        if token == "/":
            sign = -1
            continue
        m = re.fullmatch(r"([A-Za-z]+)(?:\^?(-?\d+(?:\.\d+)?))?", token)
        if not m or m.group(1) not in _UNIT_FACTORS:
            raise UnitError(f"Unbekannte Einheit '{token}' in '{unit}'")
        f = _UNIT_FACTORS[m.group(1)][0]
        exp = float(m.group(2)) if m.group(2) else 1.0
        factor *= f ** (sign * exp)
        sign = 1
    return factor


def _number(value: Any, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise MechanismError(f"{what}: Zahl erwartet, erhalten '{value}'") from None


def _ea(value: Any, units: UnitSystem) -> float:
    if isinstance(value, str):
        v, u = parse_quantity(value, units.activation_energy)
        if u not in ENERGY_PER_QUANTITY:
            raise UnitError(f"Unbekannte Energieeinheit '{u}'")
        return v * ENERGY_PER_QUANTITY[u]
    return units.activation_energy_SI(_number(value, "Ea"))


def _pressure(value: Any, units: UnitSystem) -> float:
    if isinstance(value, str):
        v, u = parse_quantity(value, units.pressure)
        if u not in PRESSURE:
            raise UnitError(f"Unbekannte Druckeinheit '{u}'")
        return v * PRESSURE[u]
    return units.pressure_SI(_number(value, "P"))


def _arrhenius(data: Any, units: UnitSystem, conc_order: float, what: str) -> Arrhenius:
    if isinstance(data, (list, tuple)) and len(data) == 3:
        data = {"A": data[0], "b": data[1], "Ea": data[2]}
    if not isinstance(data, dict) or "A" not in data:
        raise MechanismError(f"{what}: Arrhenius-Parameter {{A, b, Ea}} erwartet")
    A = data["A"]
    if isinstance(A, str):
        v, u = parse_quantity(A, "")
        A_si = v * (composite_unit_factor(u) if u else units.rate_constant_factor(conc_order))
    else:
        A_si = _number(A, f"{what} A") * units.rate_constant_factor(conc_order)
    return Arrhenius(A_si, _number(data.get("b", 0.0), f"{what} b"), _ea(data.get("Ea", 0.0), units))


# ---------------------------------------------------------------------------- species
def _thermo(data: dict | None, name: str) -> Nasa9Thermo | None:
    if not data:
        return None
    model = data.get("model")
    src = data.get("note", "") or data.get("source", "")
    # Cantera's default reference pressure is 1 atm for all species thermo models.
    p_ref = _pressure(data["reference-pressure"], UnitSystem()) if "reference-pressure" in data else 101325.0
    try:
        if model == "NASA7":
            return from_nasa7(data["temperature-ranges"], data["data"], src, p_ref)
        if model == "NASA9":
            return from_nasa9(data["temperature-ranges"], data["data"], src, p_ref)
        if model == "Shomate":
            return from_shomate(data["temperature-ranges"], data["data"], src, p_ref)
        if model == "constant-cp":
            def q(key: str, unit: str, default: float | None = None) -> float:
                if key not in data:
                    if default is None:
                        raise MechanismError(f"Spezies {name}: '{key}' fehlt im constant-cp-Modell")
                    return default
                v = data[key]
                if isinstance(v, str):
                    val, u = parse_quantity(v, unit)
                    if u in ENERGY_PER_QUANTITY:
                        return val * ENERGY_PER_QUANTITY[u]
                    if u == "J/mol/K":
                        return val
                    if u == "kJ/mol/K":
                        return val * 1e3
                    if u in ("J/kmol/K",):
                        return val * 1e-3
                    if u == "K":
                        return val
                    raise UnitError(f"Spezies {name}: Einheit '{u}' für {key} unbekannt")
                return float(v)
            return from_constant_cp(q("h0", "J/mol", 0.0), q("s0", "J/mol/K", 0.0), q("cp0", "J/mol/K", 0.0),
                                    q("T0", "K", 298.15), q("T-min", "K", 200.0), q("T-max", "K", 5000.0), src,
                                    p_ref)
    except KeyError as e:
        raise MechanismError(f"Spezies {name}: Thermo-Feld {e} fehlt") from None
    raise MechanismError(f"Spezies {name}: Thermo-Modell '{model}' nicht unterstützt "
                         "(NASA7, NASA9, Shomate, constant-cp)")


def _species(data: dict) -> Species:
    if "name" not in data:
        raise MechanismError("Spezies ohne Namen")
    name = str(data["name"])
    comp = {str(k): float(v) for k, v in (data.get("composition") or {}).items()}
    charge = 0.0
    if "E" in comp:  # Cantera stores electrons as element E
        charge = -comp.pop("E")
    if not comp and data.get("formula"):
        parsed = try_parse_formula(str(data["formula"]))
        if parsed is None:
            raise MechanismError(f"Spezies {name}: Summenformel '{data['formula']}' ungültig")
        comp, charge = parsed[0], float(parsed[1])
    if "charge" in data:
        charge = float(data["charge"])
    mw = data.get("molecular-weight")
    meta = {k: v for k, v in data.items() if k in ("pubchem", "smiles", "cas", "inchikey", "sources", "formula")}
    return Species(name=name, composition=comp, charge=charge, thermo=_thermo(data.get("thermo"), name),
                   molecular_weight=float(mw) * 1e-3 if mw is not None else None,
                   note=str(data.get("note", "")), metadata=meta)


# ---------------------------------------------------------------------------- reactions
def _reaction(data: dict, units: UnitSystem, known: set[str], phase: str, idx: int) -> Reaction:
    if "equation" not in data:
        raise MechanismError(f"Reaktion {idx + 1}: 'equation' fehlt")
    eq_text = str(data["equation"])
    what = f"Reaktion {idx + 1} ({eq_text})"
    parsed = parse_equation(eq_text, known)
    rtype = data.get("type")
    if rtype is None:
        if parsed.falloff_collider:
            rtype = "falloff"
        elif parsed.third_body:
            rtype = "three-body"
        elif "rate-constants" in data:
            rtype = "pressure-dependent-Arrhenius"
        else:
            rtype = "elementary"
    if rtype not in SUPPORTED_REACTION_TYPES:
        raise MechanismError(f"{what}: Reaktionstyp '{rtype}' wird nicht unterstützt "
                             f"(unterstützt: {', '.join(sorted(SUPPORTED_REACTION_TYPES))})")
    if rtype == "three-body" and not parsed.third_body:
        if parsed.falloff_collider:
            raise MechanismError(f"{what}: three-body mit '(+M)' – meinten Sie 'falloff'?")
        # Cantera allows explicit third body species, e.g. "2 O + AR <=> O2 + AR"
        common = set(parsed.reactants) & set(parsed.products)
        if len(common) != 1:
            raise MechanismError(f"{what}: three-body benötigt '+ M' in der Gleichung")
        collider = common.pop()
        parsed.reactants[collider] -= 1
        parsed.products[collider] -= 1
        parsed.reactants = {k: v for k, v in parsed.reactants.items() if v > 0}
        parsed.products = {k: v for k, v in parsed.products.items() if v > 0}
        data = {**data, "default-efficiency": 0.0, "efficiencies": {collider: 1.0}}
    if rtype == "falloff" and not parsed.falloff_collider:
        raise MechanismError(f"{what}: falloff benötigt '(+M)' in der Gleichung")

    orders = {str(k): _number(v, f"{what} Ordnung") for k, v in (data.get("orders") or {}).items()}
    for name in orders:
        if name not in known:
            raise MechanismError(f"{what}: Ordnung für unbekannte Spezies '{name}'")
    fwd_orders = dict(parsed.reactants)
    fwd_orders.update(orders)
    n_conc = sum(fwd_orders.values())

    rxn = Reaction(reactants=parsed.reactants, products=parsed.products, rate=Arrhenius(0.0),
                   reversible=parsed.reversible, type=rtype, orders=orders,
                   duplicate=bool(data.get("duplicate", False)),
                   negative_A=bool(data.get("negative-A", False)),
                   id=str(data.get("id", "")), note=str(data.get("note", "")))

    if rtype == "elementary":
        rxn.rate = _arrhenius(data.get("rate-constant"), units, n_conc, what)
    elif rtype == "three-body":
        rxn.rate = _arrhenius(data.get("rate-constant"), units, n_conc + 1, what)
    elif rtype == "falloff":
        rxn.falloff_collider = parsed.falloff_collider
        rxn.rate = _arrhenius(data.get("high-P-rate-constant"), units, n_conc, what + " high-P")
        rxn.low_rate = _arrhenius(data.get("low-P-rate-constant"), units, n_conc + 1, what + " low-P")
        if "Troe" in data:
            t = data["Troe"]
            rxn.falloff = Falloff("Troe", {"A": float(t["A"]), "T3": float(t["T3"]), "T1": float(t["T1"]),
                                           **({"T2": float(t["T2"])} if "T2" in t else {})})
        elif "SRI" in data:
            s = data["SRI"]
            rxn.falloff = Falloff("SRI", {"a": float(s["A"]), "b": float(s["B"]), "c": float(s["C"]),
                                          "d": float(s.get("D", 1.0)), "e": float(s.get("E", 0.0))})
        else:
            rxn.falloff = Falloff("Lindemann", {})
        if parsed.falloff_collider != "M":
            rxn.default_efficiency = 0.0
            rxn.efficiencies = {parsed.falloff_collider: 1.0}
    else:  # PLOG
        rates = data.get("rate-constants")
        if not rates:
            raise MechanismError(f"{what}: 'rate-constants' fehlt")
        for entry in rates:
            if "P" not in entry:
                raise MechanismError(f"{what}: PLOG-Eintrag ohne Druck P")
            rxn.plog_rates.append((_pressure(entry["P"], units), _arrhenius(entry, units, n_conc, what)))
        rxn.plog_rates.sort(key=lambda pk: pk[0])
        rxn.rate = rxn.plog_rates[-1][1]

    if rtype in ("three-body", "falloff"):
        if "efficiencies" in data:
            for k, v in data["efficiencies"].items():
                rxn.efficiencies[str(k)] = float(v)
        if "default-efficiency" in data:
            rxn.default_efficiency = float(data["default-efficiency"])

    if "reverse-rate-constant" in data:
        if not rxn.reversible:
            raise MechanismError(f"{what}: Rück-Arrhenius für irreversible Reaktion angegeben")
        extra = 1 if rtype == "three-body" else 0
        rxn.reverse_rate = _arrhenius(data["reverse-rate-constant"], units,
                                      sum(parsed.products.values()) + extra, what + " reverse")
    if "equilibrium-constant" in data:
        if not rxn.reversible:
            raise MechanismError(f"{what}: Gleichgewichtskonstante für irreversible Reaktion angegeben")
        ec = data["equilibrium-constant"]
        dnu = sum(parsed.products.values()) - sum(parsed.reactants.values())
        K_ref = _number(ec["K"], f"{what} K") * units.concentration_factor ** dnu
        rxn.equilibrium_constant = EquilibriumConstant(
            K_ref, float(ec.get("T-ref", 298.15)),
            _ea(ec.get("delta-H", 0.0), units))
    if "reaction-enthalpy" in data:
        rxn.reaction_enthalpy = _ea(data["reaction-enthalpy"], units)
    return rxn


# ---------------------------------------------------------------------------- loading
def mechanism_from_dict(doc: dict, phase_name: str | None = None, name: str | None = None) -> Mechanism:
    if not isinstance(doc, dict):
        raise MechanismError("Mechanismus-Datei: Mapping auf oberster Ebene erwartet")
    units = UnitSystem.from_mapping(doc.get("units"))
    phases = doc.get("phases") or []
    if phases:
        if phase_name:
            matches = [p for p in phases if p.get("name") == phase_name]
            if not matches:
                raise MechanismError(f"Phase '{phase_name}' nicht gefunden")
            phase = matches[0]
        else:
            supported = [p for p in phases if p.get("thermo", "ideal-gas") in SUPPORTED_THERMO]
            phase = supported[0] if supported else phases[0]
    else:
        phase = {"name": "gas", "thermo": "ideal-gas"}
    thermo_model = phase.get("thermo", "ideal-gas")
    if thermo_model in SUPPORTED_THERMO:
        phase_kind = SUPPORTED_THERMO[thermo_model]
    else:
        raise MechanismError(f"Phasen-Thermo '{thermo_model}' nicht unterstützt (ideal-gas, liquid)")
    if isinstance(phase.get("kinetics"), str) and phase["kinetics"] not in ("gas", "bulk", "liquid"):
        raise MechanismError(f"Kinetik-Modell '{phase['kinetics']}' nicht unterstützt")

    all_species = {}
    for sd in doc.get("species") or []:
        sp = _species(sd)
        all_species[sp.name] = sp
    declared = phase.get("species", "all")
    if declared == "all" or declared is None:
        species = list(all_species.values())
    else:
        species = []
        for entry in declared:
            if isinstance(entry, dict):
                raise MechanismError("Externe Spezies-Referenzen (andere Dateien) werden nicht unterstützt")
            if entry not in all_species:
                raise MechanismError(f"Phase deklariert unbekannte Spezies '{entry}'")
            species.append(all_species[entry])
    known = {s.name for s in species}

    reaction_spec = phase.get("reactions", "all")
    sections: list[str]
    skip_undeclared = False
    if reaction_spec in ("all", None):
        sections = ["reactions"]
    elif reaction_spec == "none":
        sections = []
    elif reaction_spec == "declared-species":
        sections, skip_undeclared = ["reactions"], True
    else:
        sections = []
        for entry in reaction_spec:
            if isinstance(entry, dict):
                ((key, val),) = entry.items()
                if "/" in key:
                    raise MechanismError("Externe Reaktions-Referenzen werden nicht unterstützt")
                sections.append(key)
                skip_undeclared = skip_undeclared or val == "declared-species"
            else:
                sections.append(entry)

    reactions = []
    idx = 0
    for sec in sections:
        for rd in doc.get(sec) or []:
            if skip_undeclared:
                pe = parse_equation(str(rd.get("equation", "")), None)
                involved = set(pe.reactants) | set(pe.products)
                if pe.falloff_collider and pe.falloff_collider != "M":
                    involved.add(pe.falloff_collider)
                if not involved <= known:
                    continue
            reactions.append(_reaction(rd, units, known, phase_kind, idx))
            idx += 1

    # Drop efficiencies of species not in the phase (Cantera skips them with a warning).
    for r in reactions:
        r.efficiencies = {k: v for k, v in r.efficiencies.items() if k in known}

    meta = dict(doc.get("autokinetics") or {})
    if "state" in phase:
        meta.setdefault("phase_state", phase["state"])
    return Mechanism(
        name=name or str(doc.get("name") or meta.get("name") or phase.get("name") or "mechanism"),
        phase=phase_kind, species=species, reactions=reactions,
        description=str(doc.get("description", "")),
        references=[str(r) for r in doc.get("references", [])],
        metadata=meta,
    )


def load_mechanism(source: str | Path, phase_name: str | None = None) -> Mechanism:
    """Load a mechanism from a YAML/JSON file path or a YAML string."""
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source
                                    and Path(source).suffix.lower() in (".yaml", ".yml", ".json", ".kin")):
        path = Path(source)
        if not path.exists():
            raise MechanismError(f"Datei nicht gefunden: {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".kin":
            from .kin_legacy import load_kin
            return load_kin(text, name=path.stem)
        name = path.stem
    else:
        text, name = str(source), None
    try:
        doc = safe_load_yaml(text)
    except yaml.YAMLError as e:
        raise MechanismError(f"YAML-Syntaxfehler: {e}") from None
    return mechanism_from_dict(doc, phase_name, name)


# ---------------------------------------------------------------------------- writing
def _arr_out(a: Arrhenius, conc_order: float, units: UnitSystem) -> dict:
    return {"A": a.A / units.rate_constant_factor(conc_order), "b": a.b,
            "Ea": a.Ea / ENERGY_PER_QUANTITY[units.activation_energy]}


def mechanism_to_dict(mech: Mechanism, units: UnitSystem | None = None) -> dict:
    """Serialise to a Cantera-compatible YAML document (dict)."""
    units = units or (UnitSystem("mol", "cm", "s", "J/mol", "Pa") if mech.phase == "ideal-gas"
                      else UnitSystem("mol", "dm", "s", "J/mol", "Pa"))
    doc: dict[str, Any] = {
        "description": mech.description,
        "generator": "AutoKinetics 2",
        "units": {"length": units.length, "time": units.time, "quantity": units.quantity,
                  "activation-energy": units.activation_energy},
        "phases": [{"name": mech.name, "thermo": "ideal-gas" if mech.phase == "ideal-gas" else "liquid",
                    "elements": mech.elements, "species": mech.species_names,
                    "kinetics": "gas" if mech.phase == "ideal-gas" else "bulk", "reactions": "all"}],
        "species": [], "reactions": [],
    }
    if mech.references:
        doc["references"] = list(mech.references)
    if mech.metadata:
        doc["autokinetics"] = dict(mech.metadata)
    for s in mech.species:
        sd: dict[str, Any] = {"name": s.name, "composition": dict(s.composition)}
        if s.charge:
            sd["charge"] = s.charge
        if s.thermo is not None:
            sd["thermo"] = s.thermo.to_dict()
        if s.molecular_weight is not None and not s.composition:
            sd["molecular-weight"] = s.molecular_weight * 1e3
        if s.note:
            sd["note"] = s.note
        sd.update({k: v for k, v in s.metadata.items() if k not in sd})
        doc["species"].append(sd)
    cf = units.concentration_factor
    for r in mech.reactions:
        rd: dict[str, Any] = {"equation": r.equation}
        n_conc = sum(r.forward_orders().values())
        if r.type == "elementary":
            rd["rate-constant"] = _arr_out(r.rate, n_conc, units)
        elif r.type == "three-body":
            rd["type"] = "three-body"
            rd["rate-constant"] = _arr_out(r.rate, n_conc + 1, units)
        elif r.type == "falloff":
            rd["type"] = "falloff"
            rd["low-P-rate-constant"] = _arr_out(r.low_rate, n_conc + 1, units)  # type: ignore[arg-type]
            rd["high-P-rate-constant"] = _arr_out(r.rate, n_conc, units)
            if r.falloff and r.falloff.kind == "Troe":
                rd["Troe"] = dict(r.falloff.params)
            elif r.falloff and r.falloff.kind == "SRI":
                p = r.falloff.params
                rd["SRI"] = {"A": p["a"], "B": p["b"], "C": p["c"], "D": p["d"], "E": p["e"]}
        else:
            rd["type"] = "pressure-dependent-Arrhenius"
            rd["rate-constants"] = [{"P": f"{P} Pa", **_arr_out(k, n_conc, units)} for P, k in r.plog_rates]
        if r.type in ("three-body", "falloff") and (r.falloff_collider in (None, "M")):
            if r.efficiencies:
                rd["efficiencies"] = dict(r.efficiencies)
            if r.default_efficiency != 1.0:
                rd["default-efficiency"] = r.default_efficiency
        if r.orders:
            rd["orders"] = dict(r.orders)
        if r.duplicate:
            rd["duplicate"] = True
        if r.negative_A:
            rd["negative-A"] = True
        if r.reverse_rate is not None:
            extra = 1 if r.type == "three-body" else 0
            rd["reverse-rate-constant"] = _arr_out(r.reverse_rate, sum(r.products.values()) + extra, units)
        if r.equilibrium_constant is not None:
            dnu = sum(r.products.values()) - sum(r.reactants.values())
            ec = r.equilibrium_constant
            rd["equilibrium-constant"] = {"K": ec.K_ref / cf**dnu, "T-ref": ec.T_ref,
                                          "delta-H": ec.dH / ENERGY_PER_QUANTITY[units.activation_energy]}
        if r.reaction_enthalpy is not None:
            rd["reaction-enthalpy"] = r.reaction_enthalpy / ENERGY_PER_QUANTITY[units.activation_energy]
        if r.id:
            rd["id"] = r.id
        if r.note:
            rd["note"] = r.note
        doc["reactions"].append(rd)
    return doc


def dump_mechanism(mech: Mechanism, units: UnitSystem | None = None) -> str:
    return yaml.safe_dump(mechanism_to_dict(mech, units), sort_keys=False, allow_unicode=True, width=100)


__all__ = ["composite_unit_factor", "dump_mechanism", "load_mechanism", "mechanism_from_dict",
           "mechanism_to_dict"]
