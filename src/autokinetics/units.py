"""Unit handling. Internally everything is SI (mol, m, s, J, K, Pa)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .constants import CAL, GAS_CONSTANT, ONE_ATM, ONE_BAR
from .errors import UnitError

QUANTITY = {"mol": 1.0, "kmol": 1.0e3, "mmol": 1.0e-3, "molec": 1.0 / 6.02214076e23,
            "molecule": 1.0 / 6.02214076e23}
LENGTH = {"m": 1.0, "dm": 0.1, "cm": 0.01, "mm": 1.0e-3}
TIME = {"s": 1.0, "ms": 1.0e-3, "us": 1.0e-6, "ns": 1.0e-9, "min": 60.0, "h": 3600.0}
# Activation energy -> J/mol. "K" means Ea/R is given in Kelvin.
ENERGY_PER_QUANTITY = {
    "J/mol": 1.0, "kJ/mol": 1.0e3, "cal/mol": CAL, "kcal/mol": CAL * 1.0e3,
    "J/kmol": 1.0e-3, "kJ/kmol": 1.0, "cal/kmol": CAL * 1.0e-3, "kcal/kmol": CAL,
    "K": GAS_CONSTANT, "eV": 96485.33212,
}
PRESSURE = {"Pa": 1.0, "kPa": 1.0e3, "MPa": 1.0e6, "bar": ONE_BAR, "atm": ONE_ATM,
            "torr": ONE_ATM / 760.0, "mbar": 100.0}
# Concentration units -> mol/m³
CONCENTRATION = {"mol/m3": 1.0, "mol/L": 1.0e3, "mmol/L": 1.0, "umol/L": 1.0e-3,
                 "kmol/m3": 1.0e3, "mol/cm3": 1.0e6, "M": 1.0e3, "mM": 1.0}


def _lookup(table: dict[str, float], unit: str, kind: str) -> float:
    try:
        return table[unit]
    except KeyError:
        raise UnitError(f"Unbekannte {kind}-Einheit '{unit}'. Erlaubt: {', '.join(table)}") from None


def activation_energy_to_J_per_mol(value: float, unit: str) -> float:
    return float(value) * _lookup(ENERGY_PER_QUANTITY, unit, "Aktivierungsenergie")


def pressure_to_Pa(value: float, unit: str = "Pa") -> float:
    return float(value) * _lookup(PRESSURE, unit, "Druck")


def concentration_to_SI(value: float, unit: str = "mol/L") -> float:
    return float(value) * _lookup(CONCENTRATION, unit, "Konzentrations")


def time_to_s(value: float, unit: str = "s") -> float:
    return float(value) * _lookup(TIME, unit, "Zeit")


_QTY_RE = re.compile(r"^\s*([-+0-9.eE]+)\s*(\S.*)?$")


def parse_quantity(text: str | float | int, default_unit: str) -> tuple[float, str]:
    """Parse '83.5 kJ/mol' -> (83.5, 'kJ/mol'). Plain numbers get ``default_unit``."""
    if isinstance(text, (int, float)):
        return float(text), default_unit
    m = _QTY_RE.match(str(text))
    if not m:
        raise UnitError(f"Kann Größe '{text}' nicht lesen")
    return float(m.group(1)), (m.group(2) or default_unit).strip()


@dataclass(frozen=True)
class UnitSystem:
    """Unit system of a mechanism file (Cantera-compatible ``units`` block).

    Cantera's defaults are kmol, m, s, J/kmol; files usually override them.
    """

    quantity: str = "kmol"
    length: str = "m"
    time: str = "s"
    activation_energy: str = "J/kmol"
    pressure: str = "Pa"

    @classmethod
    def from_mapping(cls, data: dict | None, base: UnitSystem | None = None) -> UnitSystem:
        base = base or cls()
        if not data:
            return base
        known = {"quantity", "length", "time", "activation-energy", "pressure", "energy",
                 "temperature", "mass"}
        unknown = set(data) - known
        if unknown:
            raise UnitError(f"Unbekannte Einträge im units-Block: {sorted(unknown)}")
        ea = data.get("activation-energy")
        if ea is None and "energy" in data:
            ea = f"{data['energy']}/{data.get('quantity', base.quantity)}"
        us = cls(
            quantity=data.get("quantity", base.quantity),
            length=data.get("length", base.length),
            time=data.get("time", base.time),
            activation_energy=ea or base.activation_energy,
            pressure=data.get("pressure", base.pressure),
        )
        # Validate eagerly so errors surface at load time.
        _lookup(QUANTITY, us.quantity, "Stoffmengen")
        _lookup(LENGTH, us.length, "Längen")
        _lookup(TIME, us.time, "Zeit")
        _lookup(ENERGY_PER_QUANTITY, us.activation_energy, "Aktivierungsenergie")
        _lookup(PRESSURE, us.pressure, "Druck")
        return us

    @property
    def concentration_factor(self) -> float:
        """Factor converting a concentration in file units to mol/m³."""
        return QUANTITY[self.quantity] / LENGTH[self.length] ** 3

    def rate_constant_factor(self, concentration_order: float) -> float:
        """Factor converting a pre-exponential factor to SI.

        k has units conc^(1-n)/time with n the overall concentration order of the
        rate expression (reaction orders + 1 for third bodies / low-pressure limit).
        """
        return self.concentration_factor ** (1.0 - concentration_order) / TIME[self.time]

    def activation_energy_SI(self, value: float) -> float:
        return float(value) * ENERGY_PER_QUANTITY[self.activation_energy]

    def pressure_SI(self, value: float) -> float:
        return float(value) * PRESSURE[self.pressure]
