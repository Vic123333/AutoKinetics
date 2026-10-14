"""Reactor specification (user-facing, SI) and construction of the C++ system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .compiled import CompiledMechanism, compile_mechanism
from .constants import GAS_CONSTANT as R
from .errors import SimulationError
from .mechanism import Mechanism

ReactorKind = Literal["batch", "cstr"]
Constraint = Literal["volume", "pressure"]
EnergyMode = Literal["isothermal", "adiabatic", "heat-exchange"]
Basis = Literal["concentration", "mole-fraction"]


@dataclass
class ReactorSpec:
    """Reactor definition. All values SI (K, Pa, mol/m³, s, W/(m³K), J/(m³K))."""

    kind: ReactorKind = "batch"
    constraint: Constraint = "volume"
    energy: EnergyMode = "isothermal"
    T0: float = 298.15
    P0: float = 101325.0
    initial: dict[str, float] = field(default_factory=dict)
    basis: Basis = "concentration"
    T_program: list[tuple[float, float]] | None = None  # (t, T) for isothermal operation
    UA_V: float = 0.0
    T_env: float | None = None  # defaults to T0
    T_env_program: list[tuple[float, float]] | None = None
    rho_cp: float = 4.18e6  # water
    tau: float | None = None
    feed: dict[str, float] = field(default_factory=dict)  # mol/m³
    T_feed: float | None = None

    def validate(self, mech: Mechanism) -> None:
        gas = mech.phase == "ideal-gas"
        if self.kind not in ("batch", "cstr"):
            raise SimulationError(f"Unbekannter Reaktortyp '{self.kind}'")
        if self.constraint not in ("volume", "pressure"):
            raise SimulationError(f"Unbekannte Randbedingung '{self.constraint}'")
        if self.energy not in ("isothermal", "adiabatic", "heat-exchange"):
            raise SimulationError(f"Unbekannter Energiemodus '{self.energy}'")
        if not self.T0 > 0:
            raise SimulationError("Starttemperatur muss > 0 K sein")
        if gas and not self.P0 > 0:
            raise SimulationError("Druck muss > 0 Pa sein")
        if self.constraint == "pressure" and not gas:
            raise SimulationError("Konstanter Druck ist nur für Gasphasen-Mechanismen sinnvoll")
        if self.basis == "mole-fraction" and not gas:
            raise SimulationError("Molenbrüche als Startbedingung nur für die Gasphase")
        if self.energy != "isothermal" and gas and not mech.has_thermo:
            missing = [s.name for s in mech.species if s.thermo is None]
            raise SimulationError("Energiebilanz benötigt Thermodaten für alle Spezies; fehlend: "
                                  + ", ".join(missing[:10]))
        if self.energy != "isothermal" and not gas:
            missing = [r.equation for r in mech.reactions if r.reaction_enthalpy is None and any(
                mech.species[mech.species_index(n)].thermo is None for n in list(r.reactants) + list(r.products))]
            if missing:
                raise SimulationError("Energiebilanz (Flüssigphase) benötigt Reaktionsenthalpien oder "
                                      "Thermodaten; fehlend für: " + "; ".join(missing[:5]))
            if not self.rho_cp > 0:
                raise SimulationError("ρ·cp muss > 0 sein")
        if self.kind == "cstr":
            if not (self.tau and self.tau > 0):
                raise SimulationError("CSTR benötigt eine Verweilzeit τ > 0")
            if self.constraint != "volume":
                raise SimulationError("CSTR nur bei konstantem Volumen")
            if self.energy != "isothermal" and gas:
                raise SimulationError("CSTR mit Energiebilanz wird nur für Flüssigphasen unterstützt")
        if self.energy == "heat-exchange" and not self.UA_V > 0:
            raise SimulationError("Wärmeaustausch benötigt UA/V > 0")
        if self.energy == "adiabatic" and self.UA_V:
            raise SimulationError("Adiabat bedeutet UA/V = 0")
        if self.T_program and self.energy != "isothermal":
            raise SimulationError("Temperaturprogramm nur im isothermen (temperaturgeführten) Betrieb")
        names = set(mech.species_names)
        for d, what in ((self.initial, "Startzusammensetzung"), (self.feed, "Zulauf")):
            for k, v in d.items():
                if k not in names:
                    raise SimulationError(f"{what}: unbekannte Spezies '{k}'")
                if v < 0 or not np.isfinite(v):
                    raise SimulationError(f"{what}: Wert für '{k}' muss endlich und ≥ 0 sein")
        if not any(v > 0 for v in self.initial.values()) and self.kind == "batch":
            raise SimulationError("Startzusammensetzung ist leer")

    def initial_concentrations(self, mech: Mechanism) -> np.ndarray:
        c = np.zeros(mech.n_species)
        for k, v in self.initial.items():
            c[mech.species_index(k)] = v
        if self.basis == "mole-fraction":
            total = c.sum()
            if total <= 0:
                raise SimulationError("Molenbrüche summieren sich zu 0")
            c = c / total * self.P0 / (R * self.T0)
        return c


@dataclass
class BuiltReactor:
    system: object  # autokinetics._ext._core.Reactor
    y0: np.ndarray
    compiled: CompiledMechanism
    mechanism: Mechanism
    spec: ReactorSpec
    energy: bool
    atol: np.ndarray


def build_reactor(mech: Mechanism, spec: ReactorSpec, rtol: float = 1e-6,
                  atol_rel: float = 1e-12, compiled: CompiledMechanism | None = None) -> BuiltReactor:
    from ._ext._core import KineticsModel, Reactor

    spec.validate(mech)
    cm = compiled or compile_mechanism(mech)
    km = KineticsModel(cm)
    energy = spec.energy != "isothermal"
    c0 = spec.initial_concentrations(mech)
    c_in = np.zeros(mech.n_species)
    for k, v in spec.feed.items():
        c_in[mech.species_index(k)] = v
    T_env = spec.T_env if spec.T_env is not None else spec.T0
    T_prog = spec.T_program or [(0.0, spec.T0)]
    T_env_prog = spec.T_env_program or [(0.0, T_env)]
    try:
        system = Reactor(
            km, energy=1 if energy else 0, constraint=1 if spec.constraint == "pressure" else 0,
            flow=1 if spec.kind == "cstr" else 0, P0=spec.P0, T_program=T_prog,
            UA_V=spec.UA_V if spec.energy == "heat-exchange" else 0.0, T_env_program=T_env_prog,
            rho_cp=spec.rho_cp, tau=spec.tau or 0.0, c_in=c_in,
            T_in=spec.T_feed if spec.T_feed is not None else spec.T0)
    except ValueError as e:
        raise SimulationError(f"Reaktor ungültig: {e}") from None
    # constant pressure: amounts per m³ of initial volume (V0 = 1 m³) -> numerically n = c0
    y0 = np.concatenate([c0, [spec.T0]]) if energy else c0.copy()
    scale = max(float(np.sum(c0)), float(np.sum(c_in)), 1e-30)
    atol = np.full(len(y0), atol_rel * scale)
    if energy:
        atol[-1] = 1e-6
    return BuiltReactor(system, y0, cm, mech, spec, energy, atol)
