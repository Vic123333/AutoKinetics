"""Mechanism data model: species, reactions, rate expressions, validation.

All numeric values stored here are SI (mol, m³, s, J/mol, K, Pa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .composition import molar_mass
from .constants import GAS_CONSTANT as R
from .equation import format_equation
from .errors import MechanismError
from .thermo import Nasa9Thermo

Phase = Literal["ideal-gas", "liquid"]
ReactionType = Literal["elementary", "three-body", "falloff", "pressure-dependent-Arrhenius"]


@dataclass
class Arrhenius:
    """k = A T^b exp(-Ea / (R T)); A in SI (mol, m³, s), Ea in J/mol."""

    A: float
    b: float = 0.0
    Ea: float = 0.0

    def __call__(self, T: float) -> float:
        return self.A * T**self.b * math.exp(-self.Ea / (R * T))

    def to_dict(self) -> dict:
        return {"A": self.A, "b": self.b, "Ea": self.Ea}


@dataclass
class Falloff:
    """Fall-off function. kind: Lindemann | Troe | SRI."""

    kind: Literal["Lindemann", "Troe", "SRI"] = "Lindemann"
    # Troe: A, T3, T1, T2 (T2 optional -> None); SRI: a, b, c, d, e
    params: dict[str, float] = field(default_factory=dict)


@dataclass
class EquilibriumConstant:
    """Explicit equilibrium constant in concentration units (SI, mol/m³ based):
    Kc(T) = K_ref * exp(-dH/R * (1/T - 1/T_ref)) (van 't Hoff, constant dH)."""

    K_ref: float
    T_ref: float = 298.15
    dH: float = 0.0  # J/mol

    def __call__(self, T: float) -> float:
        return self.K_ref * math.exp(-self.dH / R * (1.0 / T - 1.0 / self.T_ref))


@dataclass
class Species:
    name: str
    composition: dict[str, float] = field(default_factory=dict)
    charge: float = 0.0
    thermo: Nasa9Thermo | None = None
    molecular_weight: float | None = None  # kg/mol; derived from composition if None
    note: str = ""
    metadata: dict = field(default_factory=dict)  # e.g. pubchem cid, smiles, data sources

    @property
    def mw(self) -> float | None:
        if self.molecular_weight is not None:
            return self.molecular_weight
        if self.composition:
            return molar_mass(self.composition, self.charge)
        return None


@dataclass
class Reaction:
    reactants: dict[str, float]
    products: dict[str, float]
    rate: Arrhenius  # high-pressure limit for fall-off reactions
    reversible: bool = False
    type: ReactionType = "elementary"
    efficiencies: dict[str, float] = field(default_factory=dict)
    default_efficiency: float = 1.0
    falloff_collider: str | None = None  # "M" or explicit species for '(+X)'
    low_rate: Arrhenius | None = None
    falloff: Falloff | None = None
    plog_rates: list[tuple[float, Arrhenius]] = field(default_factory=list)  # (P [Pa], k)
    orders: dict[str, float] = field(default_factory=dict)  # explicit forward orders
    reverse_rate: Arrhenius | None = None  # explicit reverse Arrhenius
    equilibrium_constant: EquilibriumConstant | None = None
    reaction_enthalpy: float | None = None  # J/mol, for liquid energy balance w/o thermo
    duplicate: bool = False
    negative_A: bool = False
    id: str = ""
    note: str = ""

    @property
    def equation(self) -> str:
        third = "M" if self.type == "three-body" else None
        collider = self.falloff_collider if self.type == "falloff" else None
        return format_equation(self.reactants, self.products, self.reversible, third, collider)

    def forward_orders(self) -> dict[str, float]:
        orders = dict(self.reactants)
        orders.update(self.orders)
        return orders

    @property
    def is_mass_action(self) -> bool:
        return not self.orders or all(self.orders.get(k, v) == v for k, v in self.reactants.items()) and set(self.orders) <= set(self.reactants)


@dataclass
class ValidationIssue:
    level: Literal["error", "warning", "info"]
    message: str
    reaction: int | None = None
    species: str | None = None

    def to_dict(self) -> dict:
        return {"level": self.level, "message": self.message, "reaction": self.reaction,
                "species": self.species}


@dataclass
class Mechanism:
    name: str
    phase: Phase
    species: list[Species]
    reactions: list[Reaction]
    description: str = ""
    references: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._index = {s.name: i for i, s in enumerate(self.species)}
        if len(self._index) != len(self.species):
            seen: set[str] = set()
            dup = [s.name for s in self.species if s.name in seen or seen.add(s.name)]
            raise MechanismError(f"Doppelte Spezies: {sorted(set(dup))}")

    # -- lookup ---------------------------------------------------------------
    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    @property
    def species_names(self) -> list[str]:
        return [s.name for s in self.species]

    def species_index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError:
            raise MechanismError(f"Unbekannte Spezies '{name}'") from None

    @property
    def elements(self) -> list[str]:
        els: list[str] = []
        for s in self.species:
            for el in s.composition:
                if el not in els:
                    els.append(el)
        return els

    @property
    def has_thermo(self) -> bool:
        return all(s.thermo is not None for s in self.species)

    # -- matrices ---------------------------------------------------------------
    def stoichiometric_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """(nu_reactants, nu_products), shape (n_reactions, n_species)."""
        nr = np.zeros((self.n_reactions, self.n_species))
        np_ = np.zeros((self.n_reactions, self.n_species))
        for j, r in enumerate(self.reactions):
            for name, nu in r.reactants.items():
                nr[j, self.species_index(name)] += nu
            for name, nu in r.products.items():
                np_[j, self.species_index(name)] += nu
        return nr, np_

    def element_matrix(self) -> np.ndarray:
        """Shape (n_elements, n_species): atoms of element k in species i."""
        els = self.elements
        E = np.zeros((len(els), self.n_species))
        for i, s in enumerate(self.species):
            for el, n in s.composition.items():
                E[els.index(el), i] = n
        return E

    def molecular_weights(self) -> np.ndarray:
        return np.array([s.mw if s.mw is not None else np.nan for s in self.species])

    # -- validation -------------------------------------------------------------
    def validate(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if not self.species:
            issues.append(ValidationIssue("error", "Mechanismus enthält keine Spezies"))
        if not self.reactions:
            issues.append(ValidationIssue("warning", "Mechanismus enthält keine Reaktionen"))
        names = set(self.species_names)
        without_comp = [s.name for s in self.species if not s.composition]
        if without_comp:
            issues.append(ValidationIssue(
                "info", "Ohne Summenformel (Elementbilanz nicht prüfbar): " + ", ".join(without_comp[:20])
                + (" …" if len(without_comp) > 20 else "")))
        for s in self.species:
            if s.thermo is not None and s.thermo.t_min >= s.thermo.t_max:
                issues.append(ValidationIssue("error", "Ungültiger Temperaturbereich", species=s.name))

        signatures: dict[tuple, int] = {}
        for j, r in enumerate(self.reactions):
            label = f"Reaktion {j + 1} ({r.equation})"
            for name in list(r.reactants) + list(r.products) + list(r.efficiencies) + list(r.orders):
                if name not in names:
                    issues.append(ValidationIssue("error", f"{label}: unbekannte Spezies '{name}'", j, name))
            if r.falloff_collider not in (None, "M") and r.falloff_collider not in names:
                issues.append(ValidationIssue("error", f"{label}: unbekannter Stoßpartner", j))
            if r.rate.A < 0 and not r.negative_A:
                issues.append(ValidationIssue("error", f"{label}: negativer Vorfaktor A ohne 'negative-A'", j))
            if r.type == "falloff" and (r.low_rate is None or r.falloff is None):
                issues.append(ValidationIssue("error", f"{label}: Fall-off ohne Niederdruck-Parameter", j))
            if r.type == "pressure-dependent-Arrhenius" and len(r.plog_rates) < 1:
                issues.append(ValidationIssue("error", f"{label}: PLOG ohne Stützstellen", j))
            if r.type in ("three-body", "falloff", "pressure-dependent-Arrhenius") and self.phase != "ideal-gas":
                issues.append(ValidationIssue("error", f"{label}: druckabhängige Reaktionen nur in der Gasphase", j))
            for name, order in r.orders.items():
                if order < 0:
                    issues.append(ValidationIssue("warning", f"{label}: negative Ordnung für {name}", j, name))
            if r.reversible and r.reverse_rate is None and r.equilibrium_constant is None:
                missing = [n for n in list(r.reactants) + list(r.products)
                           if n in names and self.species[self._index[n]].thermo is None]
                if missing:
                    issues.append(ValidationIssue(
                        "error", f"{label}: reversibel, aber Rückreaktion nicht bestimmbar – "
                        f"Thermodaten fehlen für {', '.join(missing)} (alternativ Rück-Arrhenius "
                        "oder Gleichgewichtskonstante angeben)", j))
            if r.reversible and r.orders and r.reverse_rate is None:
                issues.append(ValidationIssue(
                    "warning", f"{label}: reversible Reaktion mit expliziten Ordnungen – Rückreaktion "
                    "nutzt Massenwirkung, thermodynamische Konsistenz nur im Gleichgewicht gewährleistet", j))
            # element & charge balance
            species_in = [self.species[self._index[n]] for n in list(r.reactants) + list(r.products) if n in names]
            if species_in and all(s.composition for s in species_in):
                balance: dict[str, float] = {}
                charge = 0.0
                for name, nu in r.reactants.items():
                    s = self.species[self._index[name]]
                    charge -= nu * s.charge
                    for el, n in s.composition.items():
                        balance[el] = balance.get(el, 0.0) - nu * n
                for name, nu in r.products.items():
                    s = self.species[self._index[name]]
                    charge += nu * s.charge
                    for el, n in s.composition.items():
                        balance[el] = balance.get(el, 0.0) + nu * n
                bad = {el: v for el, v in balance.items() if abs(v) > 1e-9}
                if bad:
                    desc = ", ".join(f"{el}: {v:+g}" for el, v in bad.items())
                    issues.append(ValidationIssue("error", f"{label}: Elementbilanz verletzt ({desc})", j))
                if abs(charge) > 1e-9:
                    issues.append(ValidationIssue("error", f"{label}: Ladungsbilanz verletzt ({charge:+g})", j))
            sig = (tuple(sorted(r.reactants.items())), tuple(sorted(r.products.items())), r.type,
                   r.falloff_collider)
            if sig in signatures and not (r.duplicate and self.reactions[signatures[sig]].duplicate):
                issues.append(ValidationIssue(
                    "warning", f"{label}: identisch mit Reaktion {signatures[sig] + 1} (als 'duplicate' markieren?)", j))
            signatures.setdefault(sig, j)
        return issues

    def check(self) -> None:
        errors = [i for i in self.validate() if i.level == "error"]
        if errors:
            raise MechanismError("; ".join(e.message for e in errors[:10])
                                 + (f" (+{len(errors) - 10} weitere)" if len(errors) > 10 else ""))
