"""Convergence analysis (ODE script Kap. 2.3 / Satz 2.13, Kap. 4; Numerik I Kap. 5).

* Global convergence under step refinement (error -> 0 with rate p).
* Tolerance proportionality of adaptive methods: error vs. rtol sweep.
* Newton / fixed-point statistics of an integration run and an explicit
  record of one stage solve (||ΔZ_k||, contraction θ_k, observed order).
* Diagnostics hooks for invariants (e.g. mass conservation) and
  non-negativity of concentrations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from autokinetics.numerics.consistency import (
    OrderStudy,
    fit_order,
    global_error_study,
    theoretical_order,
)
from autokinetics.numerics.integrator import IntegrationResult, as_system, integrate
from autokinetics.numerics.linalg import solve
from autokinetics.numerics.methods import METHODS
from autokinetics.numerics.tableaux import ButcherTableau

__all__ = [
    "global_convergence",
    "tolerance_proportionality",
    "newton_statistics",
    "stage_iteration_history",
    "invariant_drift",
    "nonnegativity_violation",
]


def global_convergence(
    system,
    t_span,
    y0,
    method,
    n_steps=(8, 16, 32, 64, 128),
    *,
    y_ref=None,
    weights=None,
    nonlinear_solver="simplified_newton",
    linear_solver="lu_pivot",
    max_wall_time: float | None = None,
) -> dict:
    """Global error at t_end for h = (t_end - t0)/N and the fitted convergence order.

    Verdict: "konvergent" if the fitted order is >= 0.7 (errors decrease at least
    roughly linearly in h), "ordnung_bestaetigt" if |p_emp - p| <= 0.3."""
    study = global_error_study(
        system,
        t_span,
        y0,
        method,
        n_steps,
        y_ref=y_ref,
        weights=weights,
        nonlinear_solver=nonlinear_solver,
        linear_solver=linear_solver,
        max_wall_time=max_wall_time,
    )
    p = theoretical_order(method)
    return {"study": study, "theoretical_order": p, **_verdict(study, p)}


def _verdict(study: OrderStudy, p: int) -> dict:
    if study.slope is None:
        return {
            "convergent": None,
            "order_confirmed": None,
            "reasoning_de": "Konvergenzordnung nicht messbar: zu wenige verwertbare Messpunkte ("
            + "; ".join(study.reason_excluded)
            + ").",
        }
    convergent = study.slope >= 0.7
    confirmed = abs(study.slope - p) <= 0.3
    if confirmed:
        text = f"Gemessene Ordnung {study.slope:.2f} bestätigt die theoretische Ordnung {p}."
    elif convergent and study.slope < p:
        text = (
            f"Gemessene Ordnung {study.slope:.2f} liegt unter der theoretischen Ordnung {p} "
            "(Ordnungsreduktion, z.B. durch Steifheit oder vorasymptotischen Bereich)."
        )
    elif convergent:
        text = f"Gemessene Ordnung {study.slope:.2f} liegt über der theoretischen Ordnung {p} (Superkonvergenz)."
    else:
        text = f"Gemessene Ordnung {study.slope:.2f} < 0.7: keine Konvergenz erkennbar."
    return {
        "convergent": bool(convergent),
        "order_confirmed": bool(confirmed),
        "reasoning_de": text,
    }


def tolerance_proportionality(
    system,
    t_span,
    y0,
    method,
    rtols=(1e-4, 1e-5, 1e-6, 1e-7),
    *,
    atol_ratio=1e-6,
    y_ref=None,
    weights=None,
    max_wall_time: float | None = None,
    **solver_options,
) -> dict:
    """Runs the adaptive method for each rtol (atol = atol_ratio * rtol) and fits
    log(error) over log(rtol). Ideal tolerance proportionality: slope ≈ 1."""
    y0 = np.asarray(y0, dtype=float)
    handle = as_system(system, y0.size)
    if y_ref is None:
        from autokinetics.numerics.consistency import reference_solution

        y_ref = reference_solution(handle, t_span, y0).y[-1]
    w = np.ones_like(y0) if weights is None else np.asarray(weights, dtype=float)
    errors, costs, steps, times = [], [], [], []
    for rtol in rtols:
        r = integrate(
            handle,
            t_span,
            y0,
            method,
            rtol=rtol,
            atol=atol_ratio * rtol,
            max_wall_time=max_wall_time,
            **solver_options,
        )
        ok = r.success
        errors.append(float(np.max(np.abs(r.y[-1] - y_ref) * w)) if ok else np.inf)
        costs.append(r.stats.get("n_rhs"))
        steps.append(r.stats.get("n_accepted"))
        times.append(r.wall_time)
    scale = max(float(np.max(np.abs(y_ref) * w)), 1e-300)
    study = fit_order(rtols, errors, floor=1e-10 * scale, ceiling=10.0 * scale, kind="tolerance")
    slope = study.slope
    if slope is None:
        text = "Toleranzproportionalität nicht messbar (zu wenige erfolgreiche Läufe)."
        proportional = None
    else:
        proportional = abs(slope - 1.0) <= 0.35
        if proportional:
            text = f"Globaler Fehler ∝ TOL^{slope:.2f}: toleranzproportional."
        elif slope > 1.0:
            text = (
                f"Globaler Fehler ∝ TOL^{slope:.2f}: fällt schneller als die Toleranz "
                "(Fehlerschätzer konservativ, z.B. eingebettete Lösung niedrigerer Ordnung)."
            )
        else:
            text = (
                f"Globaler Fehler ∝ TOL^{slope:.2f}: fällt langsamer als die Toleranz "
                "(Fehlerakkumulation oder Ordnungsreduktion)."
            )
    return {
        "rtol": list(map(float, rtols)),
        "errors": errors,
        "n_rhs": costs,
        "n_steps": steps,
        "wall_time": times,
        "slope": slope,
        "proportional": proportional,
        "study": study,
        "reasoning_de": text,
    }


def newton_statistics(result: IntegrationResult) -> dict:
    """Iteration counts and contraction factors of the nonlinear solver in a run."""
    st = result.stats
    acc = max(st.get("n_accepted") or 0, 1)
    iters = st.get("n_newton_iterations", 0)
    lin_solves = st.get("n_linear_solves", 0)
    theta_mean = st.get("theta_mean", 0.0)
    out = {
        "iterations": iters,
        "iterations_per_step": iters / acc,
        "failures": st.get("n_newton_failures", 0),
        "theta_mean": theta_mean,
        "theta_max": st.get("theta_max", 0.0),
        "linear_iterations_per_solve": (st.get("n_linear_iterations", 0) / lin_solves)
        if lin_solves
        else 0.0,
        "linear_failures": st.get("n_linear_failures", 0),
        "factorizations_per_step": st.get("n_factorizations", 0) / acc,
    }
    if iters == 0:
        out["reasoning_de"] = (
            "Keine nichtlinearen Iterationen (explizites oder linear-implizites Verfahren)."
        )
    else:
        out["reasoning_de"] = (
            f"Ø {out['iterations_per_step']:.2f} Iterationen pro Schritt, mittlerer Kontraktionsfaktor "
            f"θ = {theta_mean:.3g} (max {out['theta_max']:.3g}), {out['failures']} Fehlschläge "
            "(führen zur Schrittverwerfung mit halbierter Schrittweite)."
        )
    return out


def stage_iteration_history(
    system,
    t: float,
    y,
    h: float,
    method: str,
    kind: str = "full_newton",
    max_iter: int = 30,
    tol: float = 1e-14,
) -> dict:
    """Solves the stage equations Z = h (A ⊗ I) F(t + c h, y + Z) of one implicit step in
    Python and records ||ΔZ_k||, θ_k = ||ΔZ_k|| / ||ΔZ_{k-1}|| and the observed order
    q_k = log(||ΔZ_{k+1}|| / ||ΔZ_k||) / log(||ΔZ_k|| / ||ΔZ_{k-1}||)
    (≈ 2 for full Newton, ≈ 1 for simplified Newton and fixed-point iteration)."""
    y = np.asarray(y, dtype=float)
    handle = as_system(system, y.size)
    info = METHODS.get(method)
    if info is None or info.tableau is None or not info.implicit:
        raise ValueError("stage_iteration_history requires an implicit Runge-Kutta method")
    tab = ButcherTableau.builtin(info.tableau)
    A, c = tab.A, tab.c
    s, n = A.shape[0], y.size
    Z = np.zeros(s * n)
    J0 = handle.jacobian(t, y)
    norms = []
    for _ in range(max_iter):
        F = np.concatenate([handle.rhs(t + c[i] * h, y + Z[i * n : (i + 1) * n]) for i in range(s)])
        G = Z - h * np.kron(A, np.eye(n)) @ F
        if kind == "fixed_point":
            dZ = -G
        else:
            if kind == "full_newton":
                Js = [handle.jacobian(t + c[j] * h, y + Z[j * n : (j + 1) * n]) for j in range(s)]
            elif kind == "simplified_newton":
                Js = [J0] * s
            else:
                raise ValueError(f"unknown iteration kind {kind!r}")
            M = np.eye(s * n) - h * np.block(
                [[A[i, j] * Js[j] for j in range(s)] for i in range(s)]
            )
            dZ = solve(M, -G)
        Z = Z + dZ
        nrm = float(np.linalg.norm(dZ))
        norms.append(nrm)
        if not np.isfinite(nrm) or nrm <= tol * max(1.0, float(np.linalg.norm(Z))):
            break
    norms_a = np.array(norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = norms_a[1:] / norms_a[:-1]
        lr = np.log(norms_a)
        q = (lr[2:] - lr[1:-1]) / (lr[1:-1] - lr[:-2])
    return {
        "norms": norms_a,
        "theta": theta,
        "observed_order": q,
        "converged": bool(norms and norms[-1] <= tol * max(1.0, float(np.linalg.norm(Z)))),
    }


def invariant_drift(
    result: IntegrationResult, invariant: Callable[[np.ndarray], np.ndarray]
) -> dict:
    """max_k |I(y_k) - I(y_0)| for a conserved quantity I (e.g. total mass)."""
    vals = np.array([np.atleast_1d(invariant(yk)) for yk in result.y], dtype=float)
    drift = np.abs(vals - vals[0])
    scale = np.maximum(np.abs(vals[0]), 1e-300)
    return {
        "max_abs_drift": float(drift.max()),
        "max_rel_drift": float((drift / scale).max()),
        "values_start": vals[0],
        "values_end": vals[-1],
    }


def nonnegativity_violation(result: IntegrationResult, mask=None) -> dict:
    """Most negative value among the (masked) components."""
    Y = result.y if mask is None else result.y[:, np.asarray(mask, dtype=bool)]
    if Y.size == 0:
        return {"min_value": None, "n_negative": 0}
    return {"min_value": float(Y.min()), "n_negative": int(np.sum(Y < 0))}
