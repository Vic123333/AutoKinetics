"""Combined species lookup: PubChem identity + thermo from NIST (online) or NASA (offline)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..composition import hill_formula, molar_mass, try_parse_formula
from ..errors import ExternalDataError
from . import nasa_db, nist, pubchem


@dataclass
class SpeciesInfo:
    query: str
    name: str | None = None
    formula: str | None = None
    composition: dict[str, float] = field(default_factory=dict)
    charge: int = 0
    molecular_weight: float | None = None  # g/mol
    pubchem: dict | None = None
    cas: str | None = None
    thermo: dict | None = None  # Cantera-YAML thermo block
    thermo_source: str | None = None
    dHf_298: float | None = None  # J/mol
    S_298: float | None = None  # J/mol/K
    cp_298: float | None = None  # J/mol/K
    thermo_candidates: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _apply_thermo(info: SpeciesInfo, th, source: str) -> None:
    info.thermo = th.to_dict()
    info.thermo_source = source
    th1 = th.at_standard_pressure()
    info.dHf_298 = th1.h(298.15)
    info.S_298 = th1.s(298.15)
    info.cp_298 = th1.cp(298.15)


def thermo_candidates(composition: dict[str, float], charge: int = 0) -> list[dict]:
    out = []
    for e, th in nasa_db.by_composition(composition, charge):
        th1 = th.at_standard_pressure()
        out.append({"name": e["name"], "source": th.source, "dHf_298": th1.h(298.15), "S_298": th1.s(298.15),
                    "thermo": th.to_dict()})
    return out


def lookup_species(query: str, *, online: bool = True, use_nist: bool = True) -> SpeciesInfo:
    """Identify a species by name/formula and attach thermo data.

    Order: PubChem (identity, CAS) → NIST WebBook by CAS (gas-phase Shomate) →
    bundled NASA database by name, then by composition.
    """
    info = SpeciesInfo(query=query)
    parsed = try_parse_formula(query)
    pc = None
    if online:
        try:
            pc = pubchem.compound(query)
            if pc is None and parsed is not None:
                pc = pubchem.compound(query, "fastformula")
        except ExternalDataError as e:
            info.warnings.append(str(e))
    if pc:
        info.pubchem = pc
        info.name = pc["name"]
        info.formula = pc["formula"]
        info.cas = pc["cas"]
        info.charge = pc["charge"]
        info.molecular_weight = pc["molecular_weight"]
        comp = try_parse_formula(pc["formula"] or "")
        if comp:
            info.composition = comp[0]
    elif parsed is not None:
        info.composition, info.charge = parsed[0], parsed[1]
        info.formula = hill_formula(info.composition)
        info.name = query
        info.molecular_weight = molar_mass(info.composition, info.charge) * 1e3
    else:
        info.name = query

    nist_data = None
    if online and use_nist and (info.cas or info.formula):
        try:
            nist_data = nist.lookup(cas=info.cas) if info.cas else None
            if nist_data is None and not pc and info.formula:
                nist_data = nist.lookup(formula=info.formula)
            if nist_data and nist_data.get("thermo") is not None:
                _apply_thermo(info, nist_data["thermo"], nist_data["thermo"].source)
        except ExternalDataError as e:
            info.warnings.append(str(e))

    if info.composition:
        info.thermo_candidates = thermo_candidates(info.composition, info.charge)
    if info.thermo is None:
        hit = nasa_db.by_name(query)
        cands = info.thermo_candidates
        if hit is None and len(cands) == 1:
            hit = nasa_db.by_name(cands[0]["name"])
        if hit is None and len(cands) > 1 and nist_data and nist_data.get("dHf_gas") is not None:
            # identify the isomer by its formation enthalpy
            best = min(cands, key=lambda c: abs(c["dHf_298"] - nist_data["dHf_gas"]))
            others = [abs(c["dHf_298"] - nist_data["dHf_gas"]) for c in cands if c is not best]
            if abs(best["dHf_298"] - nist_data["dHf_gas"]) < 5e3 and min(others) > 15e3:
                hit = nasa_db.by_name(best["name"])
                info.warnings.append(f"Isomer anhand ΔfH° (NIST) zugeordnet: NASA-Eintrag {best['name']}")
        if hit is not None and nist_data and nist_data.get("dHf_gas") is not None:
            h_nasa = hit[1].at_standard_pressure().h(298.15)
            if abs(h_nasa - nist_data["dHf_gas"]) > 10e3:
                info.warnings.append(
                    f"NASA-Eintrag {hit[0]['name']} verworfen: ΔfH° {h_nasa / 1e3:.1f} kJ/mol weicht von "
                    f"NIST ({nist_data['dHf_gas'] / 1e3:.1f} kJ/mol) ab – vermutlich anderes Isomer")
                hit = None
        if hit is not None:
            _apply_thermo(info, hit[1], hit[1].source)
        elif nist_data and nist_data.get("dHf_gas") is not None and nist_data.get("S_gas") is not None:
            th = nist.thermo_from_cp_points(nist_data["dHf_gas"], nist_data["S_gas"],
                                            nist_data.get("cp_points") or [], "NIST WebBook (ΔfH°, S°, cp)")
            if th is not None:
                _apply_thermo(info, th, th.source)
                info.warnings.append("Thermodaten aus NIST-Einzelwerten konstruiert (keine Shomate-Daten); "
                                     f"gültig ca. {th.t_min:.0f}–{th.t_max:.0f} K")
            else:
                info.dHf_298, info.S_298 = nist_data["dHf_gas"], nist_data["S_gas"]
                info.warnings.append("NIST: nur ΔfH°/S° verfügbar, keine Wärmekapazität")
        elif len(cands) > 1:
            info.warnings.append(
                "Mehrere Isomere in der NASA-Datenbank – bitte Thermodaten auswählen: "
                + ", ".join(c["name"] for c in cands))
    if info.thermo is None and nist_data and info.dHf_298 is None:
        info.dHf_298, info.S_298 = nist_data.get("dHf_gas"), nist_data.get("S_gas")
    if info.thermo is None:
        info.warnings.append("Keine vollständigen Thermodaten gefunden (für Gleichgewicht/Energiebilanz "
                             "manuell ergänzen)")
    return info
