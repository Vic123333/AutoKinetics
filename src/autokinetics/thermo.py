"""Species thermodynamics.

All models are converted to the NASA 9-coefficient form (piecewise in T), which
contains NASA-7, Shomate (NIST WebBook) and constant-cp exactly:

    cp/R  = a1 T^-2 + a2 T^-1 + a3 + a4 T + a5 T^2 + a6 T^3 + a7 T^4
    h/RT  = -a1 T^-2 + a2 ln(T)/T + a3 + a4 T/2 + a5 T^2/3 + a6 T^3/4 + a7 T^4/5 + b1/T
    s/R   = -a1 T^-2/2 - a2/T + a3 ln T + a4 T + a5 T^2/2 + a6 T^3/3 + a7 T^4/4 + b2

Reference state: elements in their standard state have h(298.15 K) = 0, so h is
the absolute (formation-referenced) molar enthalpy, consistent across sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import GAS_CONSTANT as R
from .constants import STANDARD_PRESSURE, T_REF
from .errors import MechanismError


@dataclass
class Nasa9Thermo:
    """Piecewise NASA-9 polynomial. ``coeffs[k]`` = (a1..a7, b1, b2) on
    ``[t_ranges[k], t_ranges[k+1]]``."""

    t_ranges: list[float]
    coeffs: list[list[float]]
    source: str = ""
    model: str = "NASA9"
    original: dict = field(default_factory=dict)  # original definition for round-tripping
    reference_pressure: float = STANDARD_PRESSURE  # Pa; pressure the entropy refers to

    def __post_init__(self) -> None:
        if len(self.t_ranges) != len(self.coeffs) + 1:
            raise MechanismError("Thermo: Anzahl Temperaturbereiche passt nicht zu Koeffizienten")
        if any(len(c) != 9 for c in self.coeffs):
            raise MechanismError("Thermo: NASA9 benötigt 9 Koeffizienten pro Bereich")
        if any(b <= a for a, b in zip(self.t_ranges, self.t_ranges[1:], strict=False)):
            raise MechanismError("Thermo: Temperaturbereiche müssen aufsteigend sein")

    @property
    def t_min(self) -> float:
        return self.t_ranges[0]

    @property
    def t_max(self) -> float:
        return self.t_ranges[-1]

    def _coeffs_at(self, T: float) -> list[float]:
        # Outside the fitted range the nearest polynomial is extrapolated (as Cantera does).
        for k in range(len(self.coeffs)):
            if T <= self.t_ranges[k + 1]:
                return self.coeffs[k]
        return self.coeffs[-1]

    def cp(self, T: float) -> float:
        """Molar heat capacity in J/(mol K)."""
        a = self._coeffs_at(T)
        return R * (a[0] / T**2 + a[1] / T + a[2] + T * (a[3] + T * (a[4] + T * (a[5] + T * a[6]))))

    def h(self, T: float) -> float:
        """Molar enthalpy in J/mol."""
        a = self._coeffs_at(T)
        h_rt = (-a[0] / T**2 + a[1] * math.log(T) / T + a[2] + T * (a[3] / 2 + T * (a[4] / 3 + T * (a[5] / 4 + T * a[6] / 5))) + a[7] / T)
        return R * T * h_rt

    def s(self, T: float) -> float:
        """Standard molar entropy in J/(mol K)."""
        a = self._coeffs_at(T)
        return R * (-a[0] / (2 * T**2) - a[1] / T + a[2] * math.log(T) + T * (a[3] + T * (a[4] / 2 + T * (a[5] / 3 + T * a[6] / 4))) + a[8])

    def g(self, T: float) -> float:
        """Standard molar Gibbs energy in J/mol."""
        return self.h(T) - T * self.s(T)

    def at_standard_pressure(self) -> Nasa9Thermo:
        """Copy whose entropy refers to p° = 1 bar (ideal gas: s(p°) = s(p_ref) + R ln(p_ref/p°))."""
        if self.reference_pressure == STANDARD_PRESSURE:
            return self
        shift = math.log(self.reference_pressure / STANDARD_PRESSURE)
        coeffs = [c[:8] + [c[8] + shift] for c in self.coeffs]
        return Nasa9Thermo(list(self.t_ranges), coeffs, self.source, self.model, dict(self.original),
                           STANDARD_PRESSURE)

    def to_dict(self) -> dict:
        if self.original:
            d = dict(self.original)
            if self.reference_pressure != 101325.0:
                d["reference-pressure"] = f"{self.reference_pressure} Pa"
            return d
        return {"model": "NASA9", "temperature-ranges": list(self.t_ranges),
                "data": [list(c) for c in self.coeffs]}

    def packed(self) -> tuple[np.ndarray, np.ndarray]:
        """(t_ranges, coeffs[n_ranges, 9]) for the C++ kernel."""
        return np.asarray(self.t_ranges, float), np.asarray(self.coeffs, float)


def from_nasa7(t_ranges: list[float], data: list[list[float]], source: str = "",
             reference_pressure: float = 101325.0) -> Nasa9Thermo:
    t_ranges = [float(t) for t in t_ranges]
    if len(data) != len(t_ranges) - 1:
        raise MechanismError("NASA7: Anzahl Datensätze muss Anzahl Temperaturbereiche entsprechen")
    coeffs = []
    for row in data:
        if len(row) != 7:
            raise MechanismError("NASA7: 7 Koeffizienten pro Bereich erwartet")
        n = [float(x) for x in row]
        coeffs.append([0.0, 0.0, n[0], n[1], n[2], n[3], n[4], n[5], n[6]])
    return Nasa9Thermo(t_ranges, coeffs, source=source, model="NASA7",
                       original={"model": "NASA7", "temperature-ranges": t_ranges,
                                 "data": [list(map(float, r)) for r in data]},
                       reference_pressure=reference_pressure)


def from_nasa9(t_ranges: list[float], data: list[list[float]], source: str = "",
             reference_pressure: float = 101325.0) -> Nasa9Thermo:
    coeffs = [[float(x) for x in row] for row in data]
    return Nasa9Thermo([float(t) for t in t_ranges], coeffs, source=source, model="NASA9",
                       original={"model": "NASA9", "temperature-ranges": [float(t) for t in t_ranges],
                                 "data": coeffs}, reference_pressure=reference_pressure)


def from_shomate(t_ranges: list[float], data: list[list[float]], source: str = "",
             reference_pressure: float = STANDARD_PRESSURE) -> Nasa9Thermo:
    """NIST WebBook Shomate parameters (A..H, t = T/1000; Cp in J/mol/K, H in kJ/mol)."""
    coeffs = []
    for row in data:
        if len(row) < 7:
            raise MechanismError("Shomate: Parameter A–G benötigt")
        A, B, C, D, E, F, G = (float(x) for x in row[:7])
        a1 = E * 1.0e6 / R
        a3 = A / R
        a4 = B / (1.0e3 * R)
        a5 = C / (1.0e6 * R)
        a6 = D / (1.0e9 * R)
        b1 = 1.0e3 * F / R
        b2 = (G - A * math.log(1000.0)) / R
        coeffs.append([a1, 0.0, a3, a4, a5, a6, 0.0, b1, b2])
    t_ranges = [float(t) for t in t_ranges]
    return Nasa9Thermo(t_ranges, coeffs, source=source, model="Shomate",
                       original={"model": "Shomate", "temperature-ranges": t_ranges,
                                 "data": [list(map(float, r)) for r in data]},
                       reference_pressure=reference_pressure)


def from_constant_cp(h0: float, s0: float, cp0: float, T0: float = T_REF,
                     t_min: float = 200.0, t_max: float = 5000.0, source: str = "",
                     reference_pressure: float = STANDARD_PRESSURE) -> Nasa9Thermo:
    """Constant heat capacity: h0 [J/mol], s0 [J/mol/K], cp0 [J/mol/K] at T0."""
    a3 = cp0 / R
    b1 = (h0 - cp0 * T0) / R
    b2 = (s0 - cp0 * math.log(T0)) / R
    return Nasa9Thermo([t_min, t_max], [[0.0, 0.0, a3, 0.0, 0.0, 0.0, 0.0, b1, b2]], source=source,
                       model="constant-cp",
                       original={"model": "constant-cp", "T0": T0, "h0": h0, "s0": s0, "cp0": cp0,
                                 "T-min": t_min, "T-max": t_max},
                       reference_pressure=reference_pressure)
