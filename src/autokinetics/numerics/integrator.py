"""Public integration entry point: :func:`integrate` and :class:`IntegrationResult`."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autokinetics._ext import _core
from autokinetics.numerics.methods import LINEAR_SOLVERS, METHODS, NONLINEAR_SOLVERS
from autokinetics.numerics.tableaux import ButcherTableau

__all__ = ["IntegrationResult", "integrate", "as_system", "STATUS_MESSAGES_DE"]

OdeSystemHandle = _core.OdeSystemHandle

STATUS_MESSAGES_DE = {
    "success": "Integration erfolgreich abgeschlossen.",
    "max_steps": "Maximale Schrittzahl erreicht – Integration abgebrochen.",
    "wall_time": "Zeitbudget ausgeschöpft – Integration abgebrochen.",
    "step_size_too_small": "Schrittweite unter die Mindestschrittweite gefallen – das Verfahren kommt nicht "
    "voran (Problem zu steif für das Verfahren oder nichtlinearer Löser konvergiert nicht).",
    "nonlinear_solver_failure": "Nichtlinearer Löser (Newton/Fixpunkt) bzw. linearer Löser im "
    "Festschrittmodus gescheitert – kleinere Schrittweite oder anderen Löser wählen.",
    "non_finite": "Lösung wurde nicht-endlich (NaN/Inf) – Verfahren instabil für diese Schrittweite?",
    "scipy_failure": "SciPy-Integrator meldet einen Fehler.",
}


@dataclass
class IntegrationResult:
    """Solution y[k] ≈ y(t[k]) of an initial value problem plus diagnostics."""

    t: np.ndarray  # (m,)
    y: np.ndarray  # (m, n)
    success: bool
    status: str  # machine-readable status code
    message: str  # German, user-facing
    method: str
    stats: dict[str, Any]
    wall_time: float
    options: dict[str, Any] = field(default_factory=dict)
    detail: str = ""  # technical (English) detail from the solver

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        from autokinetics.numerics._json import jsonable

        return jsonable(
            {
                "t": self.t,
                "y": self.y,
                "success": self.success,
                "status": self.status,
                "message": self.message,
                "detail": self.detail,
                "method": self.method,
                "stats": self.stats,
                "wall_time": self.wall_time,
                "options": self.options,
            }
        )


class _FunctionSystem:
    """Adapter for a plain callable f(t, y) (optionally with jac(t, y))."""

    def __init__(self, f, jac=None):
        self._f = f
        if jac is not None:
            self.jacobian = jac

    def rhs(self, t, y):
        return self._f(t, y)


def as_system(system, n: int) -> OdeSystemHandle:
    """Returns a compiled handle for ``system``: an :class:`OdeSystemHandle` is used
    as is; objects with ``rhs(t, y)`` [and ``jacobian(t, y)``] or plain callables
    ``f(t, y)`` are wrapped in a C++ callback system."""
    if isinstance(system, OdeSystemHandle):
        if system.size != n:
            raise ValueError(f"y0 has length {n}, system has dimension {system.size}")
        return system
    if not callable(getattr(system, "rhs", None)):
        if callable(system):
            system = _FunctionSystem(system)
        else:
            raise TypeError(
                "system must be an OdeSystemHandle, an object with rhs(t, y), or a callable f(t, y)"
            )
    mask = getattr(system, "nonnegative_mask", None)
    if callable(mask):
        mask = mask()
    return _core.PythonOdeSystem(system, n, mask)


def _check_choice(value: str, registry: dict, what: str) -> None:
    if value not in registry:
        raise ValueError(f"unknown {what} {value!r}; available: {sorted(registry)}")


def integrate(
    system,
    t_span,
    y0,
    method: str | ButcherTableau = "dopri5",
    *,
    rtol: float = 1e-6,
    atol=1e-12,
    linear_solver: str = "lu_pivot",
    nonlinear_solver: str = "simplified_newton",
    fixed_step: float | None = None,
    t_eval=None,
    t_eval_exact: bool = False,
    max_steps: int = 100_000,
    h0: float | None = None,
    h_min: float | None = None,
    h_max: float | None = None,
    max_wall_time: float | None = None,
    nonnegative: bool = False,
    newton_tol: float = 0.03,
    max_newton_iter: int | None = None,
    linear_tol: float = 1e-12,
    linear_max_iter: int | None = None,
) -> IntegrationResult:
    """Integrates y' = f(t, y), y(t0) = y0 over t_span = (t0, tf).

    Parameters
    ----------
    system : OdeSystemHandle, object with ``rhs(t, y)`` (and optional ``jacobian(t, y)``)
        or callable ``f(t, y)``.
    method : id from :data:`METHODS` (C++ methods or ``"scipy:..."``) or a
        :class:`ButcherTableau` (custom explicit or implicit RK method).
    rtol, atol : local error tolerances (atol scalar or per component). In
        fixed-step mode they only scale the termination test of the nonlinear solver.
    linear_solver, nonlinear_solver : ids from :data:`LINEAR_SOLVERS` /
        :data:`NONLINEAR_SOLVERS` (used by implicit methods).
    fixed_step : constant step size; adjusted to (tf - t0)/N with N = ceil((tf - t0)/fixed_step).
    t_eval : output times (Hermite dense output); default: every accepted step.
    t_eval_exact : adaptive mode only: shorten steps so that they end exactly on the
        t_eval points (no interpolation error at the output times).
    max_steps, max_wall_time : step and wall-clock budgets (the result then has
        ``success=False`` and holds the solution computed so far).
    nonnegative : enforce the system's non-negativity mask (reject steps with
        significantly negative components, clip round-off negatives).
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    if y0.ndim != 1:
        raise ValueError("y0 must be one-dimensional")
    t0, tf = (float(v) for v in t_span)
    if not tf > t0:
        raise ValueError("t_span must satisfy t0 < tf")
    _check_choice(linear_solver, LINEAR_SOLVERS, "linear solver")
    _check_choice(nonlinear_solver, NONLINEAR_SOLVERS, "nonlinear solver")
    options = {
        "method": method.name if isinstance(method, ButcherTableau) else method,
        "rtol": rtol,
        "atol": atol,
        "linear_solver": linear_solver,
        "nonlinear_solver": nonlinear_solver,
        "fixed_step": fixed_step,
        "t_eval_exact": t_eval_exact,
        "max_steps": max_steps,
        "h0": h0,
        "h_min": h_min,
        "h_max": h_max,
        "max_wall_time": max_wall_time,
        "nonnegative": nonnegative,
        "newton_tol": newton_tol,
        "max_newton_iter": max_newton_iter,
        "linear_tol": linear_tol,
        "linear_max_iter": linear_max_iter,
    }
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
    if isinstance(method, str) and method.startswith("scipy:"):
        _check_choice(method, METHODS, "method")
        return _integrate_scipy(
            system,
            t0,
            tf,
            y0,
            method,
            rtol,
            atol,
            t_eval,
            h0,
            h_max,
            fixed_step,
            max_wall_time,
            options,
        )
    tableau = None
    if isinstance(method, ButcherTableau):
        tableau = method.to_cpp_dict()
        method_name = method.name
    else:
        _check_choice(method, METHODS, "method")
        method_name = method
    handle = as_system(system, y0.size)
    start = time.perf_counter()
    raw = _core.num_integrate(
        handle,
        method_name,
        t0,
        tf,
        y0,
        tableau=tableau,
        rtol=rtol,
        atol=atol,
        h0=h0 or 0.0,
        h_min=h_min or 0.0,
        h_max=h_max if h_max is not None else math.inf,
        fixed_step=fixed_step or 0.0,
        max_steps=int(max_steps),
        max_wall_time=max_wall_time or 0.0,
        t_eval=t_eval,
        exact_t_eval=t_eval_exact,
        nonnegative=nonnegative,
        nonlinear_solver=nonlinear_solver,
        linear_solver=linear_solver,
        newton_tol=newton_tol,
        max_newton_iter=max_newton_iter or 0,
        linear_tol=linear_tol,
        linear_max_iter=linear_max_iter or 0,
    )
    wall = time.perf_counter() - start
    status = raw["status"]
    return IntegrationResult(
        t=raw["t"],
        y=raw["y"],
        success=status == "success",
        status=status,
        message=STATUS_MESSAGES_DE[status],
        detail=raw["message"],
        method=options["method"],
        stats=raw["stats"],
        wall_time=wall,
        options=options,
    )


class _WallTimeExceeded(Exception):
    pass


def _integrate_scipy(
    system, t0, tf, y0, method, rtol, atol, t_eval, h0, h_max, fixed_step, max_wall_time, options
) -> IntegrationResult:
    from scipy.integrate import solve_ivp

    if fixed_step:
        raise ValueError("scipy methods do not support fixed_step")
    handle = as_system(system, y0.size)
    counters = {"rhs": 0, "jac": 0}
    deadline = time.perf_counter() + max_wall_time if max_wall_time else None

    def f(t, y):
        counters["rhs"] += 1
        if deadline is not None and time.perf_counter() > deadline:
            raise _WallTimeExceeded
        return handle.rhs(t, y)

    def jac(t, y):
        counters["jac"] += 1
        return handle.jacobian(t, y)

    kwargs: dict[str, Any] = {"rtol": rtol, "atol": atol}
    if h0:
        kwargs["first_step"] = h0
    if h_max is not None:
        kwargs["max_step"] = h_max
    if t_eval is not None:
        kwargs["t_eval"] = t_eval
    name = method.split(":", 1)[1]
    if name != "RK45" and handle.has_jacobian:
        kwargs["jac"] = jac
    start = time.perf_counter()
    try:
        sol = solve_ivp(f, (t0, tf), y0, method=name, **kwargs)
    except _WallTimeExceeded:
        wall = time.perf_counter() - start
        return IntegrationResult(
            t=np.array([t0]),
            y=y0[None, :].copy(),
            success=False,
            status="wall_time",
            message=STATUS_MESSAGES_DE["wall_time"],
            detail="wall-clock budget exhausted",
            method=method,
            stats={"n_rhs": counters["rhs"], "n_jac": counters["jac"]},
            wall_time=wall,
            options=options,
        )
    wall = time.perf_counter() - start
    n_steps = max(len(sol.t) - 1, 0) if t_eval is None else None
    stats = {
        "n_rhs": int(sol.nfev),
        "n_jac": int(sol.njev),
        "n_factorizations": int(sol.nlu),
        "n_accepted": n_steps,
    }
    status = "success" if sol.success else "scipy_failure"
    return IntegrationResult(
        t=sol.t,
        y=sol.y.T.copy(),
        success=bool(sol.success),
        status=status,
        message=STATUS_MESSAGES_DE[status],
        detail=str(sol.message),
        method=method,
        stats=stats,
        wall_time=wall,
        options=options,
    )
