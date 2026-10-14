"""High-level simulation driver: mechanism + reactor spec -> time series + diagnostics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .constants import GAS_CONSTANT as R
from .errors import SimulationError
from .mechanism import Mechanism
from .reactor import BuiltReactor, ReactorSpec, build_reactor

DEFAULT_METHOD = "scipy:BDF"


@dataclass
class SimulationOptions:
    t_end: float
    method: str = DEFAULT_METHOD
    rtol: float = 1e-6
    atol_rel: float = 1e-12  # absolute tolerance relative to total initial concentration
    max_points: int = 1500
    t_eval: np.ndarray | None = None
    log_time: bool = False  # sample output logarithmically in time (t_eval generated)
    linear_solver: str = "lu_pivot"
    nonlinear_solver: str = "simplified_newton"
    fixed_step: float | None = None
    max_wall_time: float = 120.0

    def validate(self) -> None:
        if not (self.t_end > 0 and math.isfinite(self.t_end)):
            raise SimulationError("Simulationsdauer muss > 0 sein")
        if not (0 < self.rtol < 1):
            raise SimulationError("rtol muss zwischen 0 und 1 liegen")
        if not (0 < self.atol_rel < 1):
            raise SimulationError("atol muss zwischen 0 und 1 liegen")
        if self.max_points < 10:
            raise SimulationError("max_points muss ≥ 10 sein")


@dataclass
class SimulationResult:
    mechanism: Mechanism
    spec: ReactorSpec
    options: SimulationOptions
    t: np.ndarray
    concentrations: np.ndarray  # (m, ns) mol/m³
    T: np.ndarray
    P: np.ndarray
    V: np.ndarray  # m³ per m³ initial volume
    success: bool
    message: str
    stats: dict
    wall_time: float
    diagnostics: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def species_names(self) -> list[str]:
        return self.mechanism.species_names

    def mole_fractions(self) -> np.ndarray:
        tot = self.concentrations.sum(axis=1, keepdims=True)
        return np.divide(self.concentrations, tot, out=np.zeros_like(self.concentrations), where=tot > 0)

    def amounts(self) -> np.ndarray:
        return self.concentrations * self.V[:, None]

    def species(self, name: str) -> np.ndarray:
        return self.concentrations[:, self.mechanism.species_index(name)]

    def to_dict(self, concentration_unit: str = "mol/L") -> dict:
        factor = {"mol/L": 1e-3, "mol/m3": 1.0, "mmol/L": 1.0}[concentration_unit]
        out = {
            "success": self.success,
            "message": self.message,
            "t": self.t.tolist(),
            "species": self.species_names,
            "concentration_unit": concentration_unit,
            "concentrations": (self.concentrations * factor).T.tolist(),
            "T": self.T.tolist(),
            "P": self.P.tolist(),
            "V": self.V.tolist(),
            "stats": self.stats,
            "wall_time": self.wall_time,
            "diagnostics": self.diagnostics,
            "metrics": self.metrics,
            "method": self.options.method,
        }
        if self.mechanism.phase == "ideal-gas":
            out["mole_fractions"] = self.mole_fractions().T.tolist()
        return out


# ---------------------------------------------------------------------------- integration
def _integrate(built: BuiltReactor, opts: SimulationOptions, t_eval: np.ndarray | None):
    system = built.system
    if not opts.method.startswith("scipy:"):
        try:
            from .numerics import integrate as num_integrate
        except ImportError as e:  # pragma: no cover - numerics package missing
            raise SimulationError(f"Löser '{opts.method}' nicht verfügbar: {e}") from None
        kwargs = dict(method=opts.method, rtol=opts.rtol, atol=built.atol, t_eval=t_eval,
                      linear_solver=opts.linear_solver, nonlinear_solver=opts.nonlinear_solver)
        if opts.fixed_step:
            kwargs["fixed_step"] = opts.fixed_step
        res = num_integrate(system, (0.0, opts.t_end), built.y0, **kwargs)
        return res.t, res.y, bool(res.success), str(res.message), dict(res.stats)

    from scipy.integrate import solve_ivp

    method = opts.method.split(":", 1)[1]
    if method not in ("BDF", "Radau", "LSODA", "RK45", "RK23", "DOP853"):
        raise SimulationError(f"Unbekannter scipy-Löser '{method}'")
    deadline = time.perf_counter() + opts.max_wall_time

    def f(t, y):
        if time.perf_counter() > deadline:
            raise TimeoutError
        return system.eval_rhs(t, y)

    jac = (lambda t, y: system.eval_jacobian(t, y)) if method in ("BDF", "Radau", "LSODA") else None
    try:
        sol = solve_ivp(f, (0.0, opts.t_end), built.y0, method=method, rtol=opts.rtol, atol=built.atol,
                        jac=jac, t_eval=t_eval)
    except TimeoutError:
        raise SimulationError(f"Zeitlimit von {opts.max_wall_time:.0f} s überschritten") from None
    except ValueError as e:
        raise SimulationError(f"Integration abgebrochen: {e}") from None
    stats = {"rhs": int(sol.nfev), "jacobian": int(sol.njev), "lu": int(sol.nlu), "steps": len(sol.t) - 1}
    return sol.t, sol.y.T, bool(sol.success), str(sol.message), stats


def _thin(t: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(t) <= max_points:
        return t, y
    idx = np.unique(np.linspace(0, len(t) - 1, max_points).round().astype(int))
    return t[idx], y[idx]


# ---------------------------------------------------------------------------- diagnostics
def _diagnostics(mech: Mechanism, spec: ReactorSpec, C: np.ndarray, V: np.ndarray, T: np.ndarray) -> dict:
    d: dict = {}
    scale = max(float(np.max(np.abs(C))), 1e-300)
    d["min_concentration_relative"] = float(np.min(C) / scale)
    d["negative_concentrations"] = bool(np.min(C) < -1e-8 * scale)
    if spec.kind == "batch":
        species_ok = [bool(s.composition) for s in mech.species]
        if mech.elements and all(species_ok):
            E = mech.element_matrix()
            moles = (C * V[:, None]) @ E.T  # (m, n_el)
            ref = np.maximum(np.abs(moles[0]), 1e-300)
            drift = np.max(np.abs(moles - moles[0]) / ref, axis=0)
            d["element_conservation"] = {el: float(v) for el, v in zip(mech.elements, drift, strict=True)
                                         if moles[0][mech.elements.index(el)] > 0}
            d["element_conservation_max"] = float(max(d["element_conservation"].values(), default=0.0))
        mw = mech.molecular_weights()
        if np.all(np.isfinite(mw)):
            mass = (C * V[:, None]) @ mw
            d["mass_conservation"] = float(np.max(np.abs(mass - mass[0])) / max(abs(mass[0]), 1e-300))
    if spec.energy != "isothermal" or spec.T_program:
        d["T_min"] = float(T.min())
        d["T_max"] = float(T.max())
    return d


def _metrics(mech: Mechanism, spec: ReactorSpec, t: np.ndarray, C: np.ndarray, T: np.ndarray,
             V: np.ndarray) -> dict:
    m: dict = {}
    n = C * V[:, None]
    conv = {}
    half = {}
    for i, s in enumerate(mech.species):
        n0 = n[0, i]
        if n0 > 0 and spec.kind == "batch":
            conv[s.name] = float(1.0 - n[-1, i] / n0)
            below = np.nonzero(n[:, i] <= 0.5 * n0)[0]
            if below.size:
                k = int(below[0])
                if k > 0:  # linear interpolation for the crossing time
                    f = (n[k - 1, i] - 0.5 * n0) / (n[k - 1, i] - n[k, i])
                    half[s.name] = float(t[k - 1] + f * (t[k] - t[k - 1]))
                else:
                    half[s.name] = float(t[0])
    m["conversion"] = conv
    m["half_life"] = half
    if spec.energy != "isothermal" and len(t) > 2:
        dTdt = np.gradient(T, t)
        k = int(np.argmax(dTdt))
        m["max_dTdt"] = float(dTdt[k])
        m["T_final"] = float(T[-1])
        m["T_rise"] = float(T.max() - T[0])
        if T.max() - T[0] > 50.0 and mech.phase == "ideal-gas":
            m["ignition_delay"] = float(t[k])
    return m


# ---------------------------------------------------------------------------- driver
def simulate(mech: Mechanism, spec: ReactorSpec, options: SimulationOptions,
             built: BuiltReactor | None = None) -> SimulationResult:
    options.validate()
    t0 = time.perf_counter()
    built = built or build_reactor(mech, spec, options.rtol, options.atol_rel)
    t_eval = options.t_eval
    if t_eval is None and options.log_time:
        t_first = options.t_end * 1e-9
        t_eval = np.concatenate([[0.0], np.geomspace(t_first, options.t_end, options.max_points - 1)])
    try:
        t, y, ok, msg, stats = _integrate(built, options, t_eval)
    except (ValueError, RuntimeError, ArithmeticError) as e:
        raise SimulationError(f"Integration fehlgeschlagen: {e}") from None
    if len(t) == 0:
        raise SimulationError(f"Integration lieferte keine Werte: {msg}")
    t, y = _thin(np.asarray(t), np.asarray(y), options.max_points)
    C, T, P, V = built.system.observe(t, y)
    if not np.all(np.isfinite(C)) or not np.all(np.isfinite(T)):
        ok, msg = False, "Lösung enthält nicht-endliche Werte (Instabilität?)"
    stats.update(built.system.counters)
    result = SimulationResult(mech, spec, options, t, C, T, P, V, ok, msg, stats,
                              time.perf_counter() - t0)
    result.diagnostics = _diagnostics(mech, spec, C, V, T)
    result.metrics = _metrics(mech, spec, t, C, T, V)
    return result


def heat_release_rate(result: SimulationResult) -> np.ndarray:
    """Volumetric heat release rate -sum(h_i wdot_i) [W/m³] along the solution (gas phase)."""
    from ._ext._core import KineticsModel
    from .compiled import compile_mechanism

    km = KineticsModel(compile_mechanism(result.mechanism))
    out = np.zeros(len(result.t))
    for k in range(len(result.t)):
        _, h, _ = km.species_thermo(result.T[k])
        w = km.production_rates(result.T[k], result.P[k], result.concentrations[k])
        out[k] = -float(np.dot(np.nan_to_num(h), w))
    return out


__all__ = ["SimulationOptions", "SimulationResult", "simulate", "heat_release_rate", "R"]
