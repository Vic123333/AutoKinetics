"""Import of AutoKinetics 1.x ``.kin`` files (PyQt6 GUI, JSON).

Format: ``species`` (name, start_concentration [mol/L], ...), ``groups`` (id, title,
items = species indices), ``arrows`` (start_id/end_id = species index or group id,
arrow_type Forward/Backward/Equilibrium, arrhenius_A, temperature_exponent_n,
activation_energy_Ea [J/mol], reaction_order, stoichiometry {reactants/products:
{index: factor}}). Concentrations are interpreted as mol/L and rate constants in
L, mol, s units – as in the 1.x simulator.
"""

from __future__ import annotations

import json
from collections import Counter

from ..errors import MechanismError
from ..mechanism import Arrhenius, Mechanism, Reaction, Species


def _f(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_kin(text: str, name: str = "kin-import") -> Mechanism:
    try:
        data = json.loads(text)
    except ValueError as e:
        raise MechanismError(f".kin-Datei ist kein gültiges JSON: {e}") from None
    if "species" not in data or "arrows" not in data:
        raise MechanismError(".kin-Datei: 'species' oder 'arrows' fehlt")
    species = []
    initial = {}
    for k, sd in enumerate(data["species"]):
        nm = str(sd.get("name") or f"S{k + 1}").strip()
        species.append(Species(name=nm, note="importiert aus AutoKinetics 1.x"))
        initial[nm] = _f(sd.get("start_concentration"), 0.0)
    names = [s.name for s in species]
    groups = {g["id"]: g for g in data.get("groups", [])}

    def resolve(node) -> list[int]:
        if isinstance(node, int):
            return [node]
        if isinstance(node, str) and node in groups:
            return list(groups[node].get("items", []))
        return []

    reactions = []
    warnings = []
    for a in data["arrows"]:
        start, end = resolve(a.get("start_id")), resolve(a.get("end_id"))
        if a.get("arrow_type") == "Backward":
            start, end = end, start
        lhs, rhs = Counter(start), Counter(end)
        st = a.get("stoichiometry") or {}
        role_map = (("reactants", lhs), ("products", rhs)) if a.get("arrow_type") != "Backward" else \
            (("reactants", rhs), ("products", lhs))
        for role, counter in role_map:
            for idx, factor in (st.get(role) or {}).items():
                counter[int(idx)] = _f(factor, 1.0)
        if not lhs or not rhs:
            warnings.append(f"Pfeil '{a.get('rate_constant', '?')}' ohne Edukte/Produkte übersprungen")
            continue
        n_order = sum(lhs.values())
        orders = {}
        explicit = a.get("reaction_order")
        if explicit not in (None, ""):
            o = _f(explicit, n_order)
            orders = {names[i]: o for i in lhs}
            n_order = o * len(lhs)
        conc = 1e-3  # mol/L -> mol/m³ factor inverse: k_SI = k_L * (1e-3)^(n-1)
        A = _f(a.get("arrhenius_A"), 1.0) * conc ** (n_order - 1)
        rxn = Reaction(reactants={names[i]: float(v) for i, v in lhs.items()},
                       products={names[i]: float(v) for i, v in rhs.items()},
                       rate=Arrhenius(A, _f(a.get("temperature_exponent_n"), 0.0),
                                      _f(a.get("activation_energy_Ea"), 0.0)),
                       reversible=False, orders=orders, id=str(a.get("rate_constant", "")))
        if a.get("arrow_type") == "Equilibrium":
            warnings.append(f"Gleichgewichtspfeil '{rxn.id}' als Hinreaktion importiert "
                            "(1.x enthielt keine Rückreaktionsparameter)")
        reactions.append(rxn)
    mech = Mechanism(name=name, phase="liquid", species=species, reactions=reactions,
                     description="Importiert aus AutoKinetics 1.x (.kin)")
    mech.metadata = {"import_warnings": warnings,
                     "reactor": {"basis": "concentration", "concentration_unit": "mol/L",
                                 "composition": {k: v for k, v in initial.items() if v > 0}}}
    return mech
