"""High-level simulation driver: mechanism + reactor spec -> time series + diagnostics."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .constants import GAS_CONSTANT as R
from .errors import SimulationError
from .mechanism import Mechanism
from .reactor import BuiltReactor, ReactorSpec, build_reactor

DEFAULT_METHOD = "auto"
AUTO_DENSE_LIMIT = 40  # radau5 factorises the dense 3n x 3n stage system; beyond this BDF is cheaper


def resolve_method(method: str, n_state: int) -> str:
    """'auto': C++ Radau IIA (order 5) for small systems, SciPy BDF for large ones."""
    if method != "auto":
        return method
    return "radau5" if n_state <= AUTO_DENSE_LIMIT else "scipy:BDF"


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
            "metrics": {**self.metrics, "maxima": {k: {"t": v["t"], "value": v["value"] * factor}
                                                   for k, v in self.metrics.get("maxima", {}).items()}},
            "method": self.options.method,
        }
        if self.mechanism.phase == "ideal-gas":
            out["mole_fractions"] = self.mole_fractions().T.tolist()
        return out


# ---------------------------------------------------------------------------- integration
def _integrate(built: BuiltReactor, opts: SimulationOptions, t_eval: np.ndarray | None, t0: float = 0.0):
    system = built.system
    if not opts.method.startswith("scipy:"):
        try:
            from .numerics import integrate as num_integrate
        except ImportError as e:  # pragma: no cover - numerics package missing
            raise SimulationError(f"Löser '{opts.method}' nicht verfügbar: {e}") from None
        # With prescribed output times the steps land exactly on them: Hermite dense output
        # would limit the accuracy of high-order methods taking large steps to O(h^4).
        kwargs = dict(method=opts.method, rtol=opts.rtol, atol=built.atol, t_eval=t_eval,
                      t_eval_exact=t_eval is not None,
                      max_wall_time=opts.max_wall_time,
                      linear_solver=opts.linear_solver, nonlinear_solver=opts.nonlinear_solver)
        if opts.fixed_step:
            kwargs["fixed_step"] = opts.fixed_step
        res = num_integrate(system, (t0, opts.t_end), built.y0, **kwargs)
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
        sol = solve_ivp(f, (t0, opts.t_end), built.y0, method=method, rtol=opts.rtol, atol=built.atol,
                        jac=jac, t_eval=t_eval)
    except TimeoutError:
        raise SimulationError(f"Zeitlimit von {opts.max_wall_time:.0f} s überschritten") from None
    except ValueError as e:
        raise SimulationError(f"Integration abgebrochen: {e}") from None
    stats = {"rhs": int(sol.nfev), "jacobian": int(sol.njev), "lu": int(sol.nlu), "steps": len(sol.t) - 1}
    return sol.t, sol.y.T, bool(sol.success), str(sol.message), stats


def _breakpoints(spec: ReactorSpec, t_end: float) -> list[float]:
    """Interior times where the temperature programs have kinks/jumps; the integrator
    is restarted there instead of stepping across a non-smooth right-hand side."""
    pts: set[float] = set()
    for prog in (spec.T_program, spec.T_env_program):
        for t, _ in prog or []:
            if 0.0 < t < t_end:
                pts.add(float(t))
    return sorted(pts)


def _integrate_piecewise(built: BuiltReactor, opts: SimulationOptions, t_eval: np.ndarray | None):
    bps = _breakpoints(built.spec, opts.t_end)
    if not bps:
        return _integrate(built, opts, t_eval)
    edges = [0.0, *bps, opts.t_end]
    ts, ys, stats, ok, msg = [], [], {}, True, ""
    y_start = built.y0
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        seg_eval = None
        if t_eval is not None:
            sel = t_eval[(t_eval >= a) & (t_eval <= b)]
            seg_eval = np.unique(np.concatenate([[a], sel, [b]]))
        seg_opts = copy.copy(opts)
        seg_opts.t_end = b
        seg_built = copy.copy(built)
        seg_built.y0 = np.asarray(y_start, float)
        t, y, ok, msg, st = _integrate(seg_built, seg_opts, seg_eval, t0=a)
        for key, v in st.items():
            stats[key] = stats.get(key, 0) + v
        if k > 0:
            t, y = t[1:], y[1:]
        ts.append(np.asarray(t))
        ys.append(np.asarray(y))
        y_start = ys[-1][-1]
        if not ok:
            break
    t_all, y_all = np.concatenate(ts), np.concatenate(ys)
    if t_eval is not None:
        keep = np.isin(t_all, t_eval)
        t_all, y_all = t_all[keep], y_all[keep]
    stats["segments"] = len(ts)
    return t_all, y_all, ok, msg, stats


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
    if spec.kind == "batch" and not spec.fixed_species:
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


def _crossing_time(t: np.ndarray, y: np.ndarray, level: float, k: int) -> float:
    """Time where y first falls to ``level`` (index k is the first point at/below it).

    Uses a monotone cubic (PCHIP) interpolant through the neighbouring points, which is
    far more accurate than linear interpolation on coarse adaptive output."""
    if k == 0:
        return float(t[0])
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq

    lo, hi = max(0, k - 3), min(len(t), k + 3)
    tt, yy = t[lo:hi], y[lo:hi]
    if len(tt) < 3 or np.any(np.diff(tt) <= 0):
        f = (y[k - 1] - level) / (y[k - 1] - y[k])
        return float(t[k - 1] + f * (t[k] - t[k - 1]))
    p = PchipInterpolator(tt, yy)
    return float(brentq(lambda x: float(p(x)) - level, t[k - 1], t[k], xtol=1e-14 * max(t[k], 1e-300)))


def _metrics(mech: Mechanism, spec: ReactorSpec, t: np.ndarray, C: np.ndarray, T: np.ndarray,
             V: np.ndarray) -> dict:
    m: dict = {}
    n = C * V[:, None]
    conv = {}
    half = {}
    for i, s in enumerate(mech.species):
        n0 = n[0, i]
        if n0 > 0 and spec.kind == "batch" and s.name not in spec.fixed_species:
            conv[s.name] = float(1.0 - n[-1, i] / n0)
            below = np.nonzero(n[:, i] <= 0.5 * n0)[0]
            if below.size:
                half[s.name] = _crossing_time(t, n[:, i], 0.5 * n0, int(below[0]))
    m["conversion"] = conv
    m["half_life"] = half
    # interior maxima of intermediates (start at ~0, peak, decay), located by parabolic
    # interpolation through the three output points around the discrete maximum
    maxima = {}
    for i, s in enumerate(mech.species):
        y = C[:, i]
        k = int(np.argmax(y))
        if 0 < k < len(t) - 1 and y[k] > 1.01 * max(y[0], y[-1]) and y[k] > 0:
            t0_, t1_, t2_ = t[k - 1], t[k], t[k + 1]
            y0_, y1_, y2_ = y[k - 1], y[k], y[k + 1]
            denom = (t0_ - t1_) * (t0_ - t2_) * (t1_ - t2_)
            a = (t2_ * (y1_ - y0_) + t1_ * (y0_ - y2_) + t0_ * (y2_ - y1_)) / denom
            b = (t2_**2 * (y0_ - y1_) + t1_**2 * (y2_ - y0_) + t0_**2 * (y1_ - y2_)) / denom
            if a < 0:
                tv = -b / (2 * a)
                if t0_ <= tv <= t2_:
                    c = y1_ - a * t1_**2 - b * t1_
                    maxima[s.name] = {"t": float(tv), "value": float(a * tv**2 + b * tv + c)}
                    continue
            maxima[s.name] = {"t": float(t1_), "value": float(y1_)}
    m["maxima"] = maxima
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
    if options.method == "auto":
        options = copy.copy(options)
        options.method = resolve_method("auto", len(built.y0))
    t_eval = options.t_eval
    if t_eval is None and options.log_time:
        t_first = options.t_end * 1e-9
        t_eval = np.concatenate([[0.0], np.geomspace(t_first, options.t_end, options.max_points - 1)])
    try:
        t, y, ok, msg, stats = _integrate_piecewise(built, options, t_eval)
        if (t_eval is None and ok and not options.method.startswith("scipy:")
                and len(t) < options.max_points // 2):
            # High-order methods take few, large steps. For smooth curves add evenly spaced
            # output points; steps land exactly on them, so every point has full accuracy
            # (Hermite dense output would only be O(h^4)).
            n_extra = min(options.max_points - len(t), 400)
            grid = (np.geomspace(max(t[1], options.t_end * 1e-9), options.t_end, n_extra)
                    if options.log_time and len(t) > 1 else np.linspace(0.0, options.t_end, n_extra))
            dense_eval = np.unique(np.concatenate([np.asarray(t), grid]))
            dense_opts = copy.copy(options)
            dense_opts.t_eval = dense_eval
            t2, y2, ok2, msg2, stats2 = _integrate_piecewise(built, dense_opts, dense_eval)
            if ok2:
                t, y, msg, stats = t2, y2, msg2, stats2
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
