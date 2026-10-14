"""Consistency analysis (ODE script Kap. 2.2, 3.5 and 7.4).

* Algebraic order conditions of Runge–Kutta methods from rooted trees
  (Butcher, Satz 3.4): b^T Φ(t) = 1/γ(t) for every tree t with |t| <= p.
* Order and error constant of linear multistep methods (Satz 7.12) and the
  root condition (Def. 7.17).
* Empirical orders: local error of one step from exact data (consistency
  order p: local error = O(h^{p+1}), Def. 2.8) and global error at a fixed
  end time under step refinement (convergence order, Def. 2.12), both as
  least-squares slopes in a log-log plot against a tight-tolerance reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np

from autokinetics.numerics.integrator import as_system, integrate
from autokinetics.numerics.methods import METHODS
from autokinetics.numerics.tableaux import ButcherTableau

__all__ = [
    "rooted_trees",
    "tree_to_string",
    "order_conditions",
    "algebraic_order",
    "lmm_order",
    "root_condition",
    "OrderStudy",
    "fit_order",
    "reference_solution",
    "local_error_study",
    "global_error_study",
    "theoretical_order",
]

Tree = tuple  # a rooted tree = sorted tuple of its subtrees; () is the single node

# Relative accuracy assumed for the tight-tolerance reference (rtol 1e-12): errors
# below this level (relative to the solution scale) are excluded from order fits.
REFERENCE_NOISE = 1e-10


@cache
def rooted_trees(order: int) -> tuple[Tree, ...]:
    """All rooted trees with `order` nodes (1, 1, 2, 4, 9, 20, 48, ... trees).

    A tree is the multiset of its root's subtrees; multisets are enumerated as
    index-non-decreasing selections from the (deterministically ordered) pool of
    smaller trees, so every tree appears exactly once in canonical form."""
    if order < 1:
        return ()
    if order == 1:
        return ((),)
    pool = [t for k in range(1, order) for t in rooted_trees(k)]
    sizes = [_tree_order(t) for t in pool]
    result: list[Tree] = []

    def choose(start: int, remaining: int, children: tuple) -> None:
        if remaining == 0:
            result.append(children)
            return
        for i in range(start, len(pool)):
            if sizes[i] <= remaining:
                choose(i, remaining - sizes[i], children + (pool[i],))

    choose(0, order - 1, ())
    return tuple(result)


def _tree_order(t: Tree) -> int:
    return 1 + sum(_tree_order(c) for c in t)


def tree_to_string(t: Tree) -> str:
    """Bracket notation: '•' for the single node, '[t1,t2]' for a root with children."""
    if t == ():
        return "•"
    return "[" + ",".join(tree_to_string(c) for c in t) + "]"


def _density(t: Tree) -> int:
    g = _tree_order(t)
    for c in t:
        g *= _density(c)
    return g


def _weights(t: Tree, A: np.ndarray) -> np.ndarray:
    """Stage vector Φ(t): product over children of A Φ(child) (Φ(•) = 1)."""
    v = np.ones(A.shape[0])
    for c in t:
        v = v * (A @ _weights(c, A))
    return v


def order_conditions(A, b, max_order: int = 6, tol: float = 1e-10) -> list[dict]:
    """Evaluates b^T Φ(t) = 1/γ(t) for all trees with up to max_order nodes."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    out = []
    for p in range(1, max_order + 1):
        for t in rooted_trees(p):
            lhs = float(b @ _weights(t, A))
            rhs = 1.0 / _density(t)
            out.append(
                {
                    "tree": tree_to_string(t),
                    "order": p,
                    "gamma": _density(t),
                    "lhs": lhs,
                    "rhs": rhs,
                    "residual": lhs - rhs,
                    "satisfied": bool(abs(lhs - rhs) <= tol * max(1.0, abs(rhs))),
                }
            )
    return out


def algebraic_order(A, b, c=None, max_order: int = 6, tol: float = 1e-10) -> dict:
    """Theoretical consistency order from the order conditions.

    Returns order (largest p with all conditions up to p satisfied), whether the
    search was capped at max_order, the first violated conditions and the row-sum
    condition c = A 1 (invariance against autonomisation, Lemma 3.3)."""
    conds = order_conditions(A, b, max_order, tol)
    order = 0
    for p in range(1, max_order + 1):
        if all(cd["satisfied"] for cd in conds if cd["order"] == p):
            order = p
        else:
            break
    failed = [cd for cd in conds if cd["order"] == order + 1 and not cd["satisfied"]]
    row_sum_ok = None
    if c is not None:
        row_sum_ok = bool(
            np.allclose(
                np.asarray(A, dtype=float).sum(axis=1), np.asarray(c, dtype=float), atol=1e-12
            )
        )
    return {
        "order": order,
        "capped": order == max_order,
        "max_order_checked": max_order,
        "first_violated": failed,
        "n_conditions": len(conds),
        "row_sum_condition": row_sum_ok,
        "conditions": conds,
    }


def lmm_order(alpha, beta, max_order: int = 10, tol: float = 1e-10) -> dict:
    """Order of sum alpha_j y_{n+j} = h sum beta_j f_{n+j}: C_q = sum alpha_j j^q - q sum
    beta_j j^{q-1} = 0 for q = 0..p (Satz 7.12); error constant C_{p+1}/((p+1)! sigma(1))."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    j = np.arange(alpha.size, dtype=float)
    Cs = []
    order = -1
    for q in range(max_order + 2):
        Cq = float(np.sum(alpha * j**q) - (q * np.sum(beta * j ** (q - 1)) if q > 0 else 0.0))
        Cs.append(Cq)
        if abs(Cq) <= tol and order == q - 1:
            order = q
    p = order
    from math import factorial

    err_const = Cs[p + 1] / (factorial(p + 1) * beta.sum()) if p + 1 < len(Cs) else None
    return {"order": p, "C": Cs, "error_constant": err_const}


def root_condition(alpha) -> dict:
    """Zero stability (Def. 7.17): roots of rho inside the closed unit disc, roots of
    modulus one simple."""
    roots = np.roots(np.asarray(alpha, dtype=float)[::-1])
    mod = np.abs(roots)
    ok = bool(np.all(mod <= 1 + 1e-12))
    on_circle = roots[np.abs(mod - 1) <= 1e-9]
    for i, r in enumerate(on_circle):
        if np.any(np.abs(np.delete(on_circle, i) - r) <= 1e-7):
            ok = False
    return {"satisfied": ok, "roots_re": roots.real, "roots_im": roots.imag}


def theoretical_order(method) -> int:
    if isinstance(method, ButcherTableau):
        return method.order
    return METHODS[method].order


@dataclass
class OrderStudy:
    """Errors for a sequence of step sizes and the fitted log-log slope."""

    h: np.ndarray
    errors: np.ndarray
    used: np.ndarray  # boolean mask of points used for the fit
    slope: float | None
    intercept: float | None
    kind: str  # "local" | "global"
    reason_excluded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        fit = None
        if self.slope is not None:
            fit = (np.exp(self.intercept) * self.h**self.slope).tolist()
        return {
            "h": self.h.tolist(),
            "errors": self.errors.tolist(),
            "used": self.used.tolist(),
            "slope": self.slope,
            "intercept": self.intercept,
            "kind": self.kind,
            "fitted": fit,
            "reason_excluded": self.reason_excluded,
        }


def fit_order(
    h, errors, floor: float = 0.0, ceiling: float = np.inf, kind: str = "global"
) -> OrderStudy:
    """Least-squares slope of log(error) over log(h), ignoring non-finite errors,
    errors at the noise floor and errors above the ceiling (instability /
    pre-asymptotic range)."""
    h = np.asarray(h, dtype=float)
    e = np.asarray(errors, dtype=float)
    used = np.isfinite(e) & (e > floor) & (e < ceiling)
    reasons = []
    for hi, ei, ok in zip(h, e, used, strict=True):
        if ok:
            continue
        if not np.isfinite(ei):
            reasons.append(f"h={hi:.3g}: nicht-endlicher Fehler (instabil/gescheitert)")
        elif ei <= floor:
            reasons.append(f"h={hi:.3g}: Fehler im Bereich der Referenz-/Rundungsgenauigkeit")
        else:
            reasons.append(f"h={hi:.3g}: Fehler zu groß (instabil oder vorasymptotisch)")
    if used.sum() >= 2:
        slope, intercept = np.polyfit(np.log(h[used]), np.log(e[used]), 1)
        return OrderStudy(h, e, used, float(slope), float(intercept), kind, reasons)
    return OrderStudy(h, e, used, None, None, kind, reasons)


def reference_solution(
    system,
    t_span,
    y0,
    t_eval=None,
    rtol: float = 1e-12,
    atol: float = 1e-16,
    max_wall_time: float | None = None,
):
    """High-accuracy reference by adaptive Radau IIA (s=3) with tight tolerances."""
    res = integrate(
        system,
        t_span,
        y0,
        "radau5",
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
        max_steps=2_000_000,
        max_wall_time=max_wall_time,
        newton_tol=1e-3,
    )
    if not res.success:
        raise RuntimeError(f"reference solution failed: {res.message} ({res.detail})")
    return res


def _scaled_error(y, y_ref, weights) -> float:
    with np.errstate(invalid="ignore", over="ignore"):
        return float(np.max(np.abs(np.asarray(y) - y_ref) * weights))


def _fixed_run(handle, t_span, y0, method, h, nonlinear_solver, linear_solver, max_wall_time=None):
    return integrate(
        handle,
        t_span,
        y0,
        method,
        fixed_step=h,
        rtol=1e-13,
        atol=1e-15,
        newton_tol=1e-4,
        nonlinear_solver=nonlinear_solver,
        linear_solver=linear_solver,
        max_newton_iter=25,
        max_steps=10_000_000,
        max_wall_time=max_wall_time,
    )


def local_error_study(
    system,
    t0: float,
    y0,
    method,
    hs,
    *,
    weights=None,
    exact=None,
    nonlinear_solver="simplified_newton",
    linear_solver="lu_pivot",
    noise_floor: float | None = None,
) -> OrderStudy:
    """One step of size h from exact data (t0, y0): δ(h) = ||Ψ_h y0 - y(t0+h)||.
    For a method of consistency order p, δ(h) = O(h^{p+1}); the slope minus one
    estimates p. ``exact(t)`` may supply the exact solution; otherwise a
    tight-tolerance Radau IIA reference is computed."""
    y0 = np.asarray(y0, dtype=float)
    handle = as_system(system, y0.size)
    w = np.ones_like(y0) if weights is None else np.asarray(weights, dtype=float)
    hs = np.asarray(hs, dtype=float)
    if exact is not None:
        y_ref = np.array([np.asarray(exact(t0 + h), dtype=float) for h in hs])
        default_floor = 1e3 * np.finfo(float).eps
    else:
        order = np.argsort(hs)
        ref = reference_solution(handle, (t0, t0 + hs.max()), y0, t_eval=t0 + hs[order])
        y_ref = np.empty((hs.size, y0.size))
        y_ref[order] = ref.y
        default_floor = REFERENCE_NOISE
    errs = []
    for h, yr in zip(hs, y_ref, strict=True):
        r = _fixed_run(handle, (t0, t0 + h), y0, method, h, nonlinear_solver, linear_solver)
        errs.append(_scaled_error(r.y[-1], yr, w) if r.success else np.inf)
    scale = max(np.max(np.abs(y0) * w), 1e-300)
    floor = noise_floor if noise_floor is not None else default_floor * scale
    return fit_order(hs, errs, floor=floor, ceiling=0.5 * scale + 10.0 * floor, kind="local")


def global_error_study(
    system,
    t_span,
    y0,
    method,
    n_steps=(8, 16, 32, 64),
    *,
    weights=None,
    y_ref=None,
    nonlinear_solver="simplified_newton",
    linear_solver="lu_pivot",
    noise_floor: float | None = None,
    max_wall_time: float | None = None,
) -> OrderStudy:
    """Global error at t_end for fixed steps h = (t_end - t0)/N; slope ≈ convergence order."""
    y0 = np.asarray(y0, dtype=float)
    handle = as_system(system, y0.size)
    t0, tf = map(float, t_span)
    w = np.ones_like(y0) if weights is None else np.asarray(weights, dtype=float)
    default_floor = 1e3 * np.finfo(float).eps
    if y_ref is None:
        y_ref = reference_solution(handle, (t0, tf), y0).y[-1]
        default_floor = REFERENCE_NOISE
    hs = (tf - t0) / np.asarray(n_steps, dtype=float)
    errs = []
    for h in hs:
        r = _fixed_run(
            handle, (t0, tf), y0, method, h, nonlinear_solver, linear_solver, max_wall_time
        )
        errs.append(_scaled_error(r.y[-1], y_ref, w) if r.success else np.inf)
    scale = max(np.max(np.abs(y_ref) * w), np.max(np.abs(y0) * w), 1e-300)
    floor = noise_floor if noise_floor is not None else default_floor * scale
    return fit_order(hs, errs, floor=floor, ceiling=0.5 * scale + 10.0 * floor, kind="global")
