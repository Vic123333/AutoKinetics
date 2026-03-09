"""
Core data model for chemical reaction networks.

Scientific basis:
  Rate law:    v_j = k_j(T) · ∏_r [X_r]^m_r
  Modified Arrhenius: k(T) = A · T^n · exp(-Ea / (R·T))
  ODE system:  d[Xi]/dt = Σ_j ν_ij · v_j

References:
  - IUPAC Gold Book: https://goldbook.iupac.org/
  - Cantera documentation on Arrhenius rate laws
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

R_GAS = 8.314462618   # J mol⁻¹ K⁻¹  (CODATA 2018)


@dataclass
class Species:
    """A chemical species with its initial concentration."""
    name: str
    start_concentration: float = 1.0    # mol L⁻¹  (Molar)
    is_intermediate: bool = False        # True ⇒ apply QSSA

    def __post_init__(self):
        if self.start_concentration < 0:
            raise ValueError(f"Initial concentration of '{self.name}' must be ≥ 0")
        if not self.name.strip():
            raise ValueError("Species name must not be empty")


@dataclass
class Reaction:
    """
    One elementary (or pseudo-elementary) reaction step.

    reactants / products : list of (species_index, stoichiometric_coefficient)
    reaction_order       : dict {species_index: partial_order}
                           If not provided, defaults to stoichiometric order.
    """
    reactants:             List[Tuple[int, float]]
    products:              List[Tuple[int, float]]
    rate_label:            str
    arrhenius_A:           float = 1.0
    activation_energy_Ea:  float = 0.0   # J mol⁻¹
    temperature_exponent_n: float = 0.0
    reaction_order:        Dict[int, float] = field(default_factory=dict)

    # ── extra kwargs from JSON are silently accepted ──
    def __init__(self, reactants, products, rate_label, **kwargs):
        self.reactants            = reactants
        self.products             = products
        self.rate_label           = rate_label
        self.arrhenius_A          = float(kwargs.get('arrhenius_A', 1.0))
        self.activation_energy_Ea = float(kwargs.get('activation_energy_Ea', 0.0))
        self.temperature_exponent_n = float(kwargs.get('temperature_exponent_n', 0.0))
        raw_order = kwargs.get('reaction_order', {})
        if isinstance(raw_order, str):
            raw_order = {}
        self.reaction_order = {int(k): float(v) for k, v in raw_order.items()}

    def calculate_k(self, temperature: float) -> float:
        """
        Modified Arrhenius rate constant:
            k(T) = A · T^n · exp(-Ea / (R·T))

        Returns 0 instead of raising for very small values (numerical safety).
        """
        A  = self.arrhenius_A
        Ea = self.activation_energy_Ea
        n  = self.temperature_exponent_n
        try:
            exponent = -Ea / (R_GAS * temperature)
            if exponent < -700:        # underflow prevention
                return 0.0
            k = A * (temperature ** n) * np.exp(exponent)
            return max(k, 0.0)
        except (OverflowError, ZeroDivisionError):
            return 0.0

    @property
    def overall_order(self) -> float:
        """Sum of all partial reaction orders (used for unit labelling)."""
        if self.reaction_order:
            return float(sum(self.reaction_order.values()))
        return float(sum(s for _, s in self.reactants))

    @property
    def k_unit(self) -> str:
        """IUPAC-correct unit string for this rate constant."""
        n = self.overall_order
        if abs(n)    < 1e-9: return 'mol L⁻¹ s⁻¹'
        if abs(n - 1) < 1e-9: return 's⁻¹'
        if abs(n - 2) < 1e-9: return 'L mol⁻¹ s⁻¹'
        if abs(n - 3) < 1e-9: return 'L² mol⁻² s⁻¹'
        return f'M^(1-{n:.4g}) s⁻¹'


class ReactionSystem:
    """Container for all species and reactions forming a kinetic network."""

    def __init__(self, species: List[Species], reactions: List[Reaction]):
        self.species   = species
        self.reactions = reactions

    def get_initial_concentrations(self) -> np.ndarray:
        return np.array([s.start_concentration for s in self.species], dtype=float)

    def get_rate_law_equations(self) -> str:
        """Return a human-readable ODE system string."""
        lines = []
        for i, sp in enumerate(self.species):
            terms = []
            for rxn in self.reactions:
                net_coeff = 0.0
                for idx, s in rxn.reactants:
                    if idx == i:
                        net_coeff -= s
                for idx, s in rxn.products:
                    if idx == i:
                        net_coeff += s
                if abs(net_coeff) < 1e-12:
                    continue
                sign = '+' if net_coeff > 0 else '-'
                mag  = abs(net_coeff)
                coeff_str = '' if abs(mag - 1.0) < 1e-9 else f'{mag:.4g}·'
                rate_factors = []
                for idx, _ in rxn.reactants:
                    order = rxn.reaction_order.get(idx, 1.0)
                    exp   = f'^{order:.4g}' if abs(order - 1.0) > 1e-9 else ''
                    rate_factors.append(f'[{self.species[idx].name}]{exp}')
                term = f'{coeff_str}{rxn.rate_label}·{"·".join(rate_factors)}'
                terms.append(f'{sign} {term}')
            rhs = ' '.join(terms).lstrip('+ ') if terms else '0'
            lines.append(f'd[{sp.name}]/dt = {rhs}')
        return '\n'.join(lines)
