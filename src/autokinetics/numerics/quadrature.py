"""Numerical integration (Numerik I, Kap. 8).

* Newton–Cotes formulas (interpolatory quadrature on equidistant nodes) and
  composite trapezoidal / Simpson rules,
* Gauß–Legendre, Gauß–Lobatto and (right) Gauß–Radau nodes and weights,
  computed by Newton's method on the Legendre polynomials,
* composite Gauß rules,
* integration of sampled simulation results (trapezoid, Simpson for
  non-uniform grids, cubic spline).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np

__all__ = [
    "legendre",
    "gauss_legendre",
    "gauss_lobatto",
    "gauss_radau",
    "newton_cotes_weights",
    "newton_cotes",
    "composite_trapezoid",
    "composite_simpson",
    "gauss_legendre_rule",
    "gauss_lobatto_rule",
    "composite_gauss",
    "integrate_samples",
    "cumulative_trapezoid",
]


def legendre(n: int, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Legendre polynomial P_n(x) and its derivative via the three-term recurrence."""
    x = np.asarray(x, dtype=float)
    p_prev, p = np.ones_like(x), x.copy()
    if n == 0:
        return p_prev, np.zeros_like(x)
    for k in range(2, n + 1):
        p_prev, p = p, ((2 * k - 1) * x * p - (k - 1) * p_prev) / k
    # (1 - x^2) P_n' = n (P_{n-1} - x P_n); evaluated away from x = +-1
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = n * (p_prev - x * p) / (1.0 - x * x)
    at_end = np.isclose(np.abs(x), 1.0)
    if np.any(at_end):
        dp = np.where(at_end, np.sign(x) ** (n + 1) * n * (n + 1) / 2.0, dp)
    return p, dp


def _newton_roots(
    f: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], x0: np.ndarray
) -> np.ndarray:
    x = np.array(x0, dtype=float)
    for _ in range(100):
        val, der = f(x)
        dx = val / der
        x -= dx
        if np.max(np.abs(dx)) < 1e-15:
            break
    return x


@lru_cache(maxsize=64)
def _gauss_legendre_cached(n: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    k = np.arange(1, n + 1)
    x0 = np.cos(np.pi * (k - 0.25) / (n + 0.5))
    x = _newton_roots(lambda z: legendre(n, z), x0)
    _, dp = legendre(n, x)
    w = 2.0 / ((1.0 - x * x) * dp * dp)
    order = np.argsort(x)
    return tuple(x[order]), tuple(w[order])


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the n-point Gauß–Legendre rule on [-1, 1] (exact for degree 2n-1)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    x, w = _gauss_legendre_cached(int(n))
    return np.array(x), np.array(w)


def gauss_lobatto(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the n-point Gauß–Lobatto rule on [-1, 1] (endpoints included,
    exact for degree 2n-3)."""
    if n < 2:
        raise ValueError("Gauss-Lobatto requires n >= 2")
    m = n - 1
    interior = np.empty(0)
    if n > 2:
        # interior nodes: roots of P_m'; Newton on g = (1 - x^2) P_m' = m (P_{m-1} - x P_m)
        def g(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            p_m, dp_m = legendre(m, z)
            p_m1, _ = legendre(m - 1, z)
            val = m * (p_m1 - z * p_m)
            der = -m * (m + 1) * p_m  # d/dx[(1-x^2) P_m'] = -m(m+1) P_m (Legendre ODE)
            return val, der

        x0 = -np.cos(np.pi * np.arange(1, m) / m)
        interior = _newton_roots(g, x0)
    x = np.concatenate([[-1.0], np.sort(interior), [1.0]])
    p, _ = legendre(m, x)
    w = 2.0 / (n * m * p * p)
    return x, w


def gauss_radau(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of the n-point right Gauß–Radau rule on [0, 1] (node 1 included,
    exact for degree 2n-2). These are the collocation nodes of Radau IIA."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([1.0]), np.array([1.0])
    # nodes: roots of P_n(2x-1) - P_{n-1}(2x-1)
    coeffs = np.zeros(n + 1)
    coeffs[n], coeffs[n - 1] = 1.0, -1.0
    xi = np.sort(np.real(np.polynomial.legendre.legroots(coeffs)))

    def q(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pn, dpn = legendre(n, z)
        pm, dpm = legendre(n - 1, z)
        return pn - pm, dpn - dpm

    inner = _newton_roots(q, xi[:-1])
    x = np.concatenate([(inner + 1.0) / 2.0, [1.0]])
    w = _interpolatory_weights(x, 0.0, 1.0)
    return x, w


def _interpolatory_weights(nodes: np.ndarray, a: float, b: float) -> np.ndarray:
    """Weights of the interpolatory rule on [a, b]: integrals of the Lagrange basis."""
    nodes = np.asarray(nodes, dtype=float)
    m = len(nodes)
    # moments of the shifted monomials (x - mid)^k / scale^k for conditioning
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    t = (nodes - mid) / half
    V = np.vander(t, m, increasing=True).T
    k = np.arange(m)
    moments = half * (1.0 - (-1.0) ** (k + 1)) / (k + 1)
    from autokinetics.numerics.linalg import solve

    return solve(V, moments)


def newton_cotes_weights(n: int) -> np.ndarray:
    """Weights of the closed Newton–Cotes formula with n+1 equidistant nodes on [0, 1]
    (n=1 trapezoid, n=2 Simpson, n=3 3/8-rule, n=4 Milne/Boole)."""
    if n < 1 or n > 10:
        raise ValueError(
            "Newton-Cotes order n must be in 1..10 (higher orders have negative weights)"
        )
    return _interpolatory_weights(np.linspace(0.0, 1.0, n + 1), 0.0, 1.0)


def newton_cotes(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """Closed Newton–Cotes formula of degree n on [a, b]."""
    x = np.linspace(a, b, n + 1)
    return float((b - a) * np.dot(newton_cotes_weights(n), f(x)))


def composite_trapezoid(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, m: int) -> float:
    """Composite trapezoidal rule with m subintervals (error O(h^2))."""
    x = np.linspace(a, b, m + 1)
    y = f(x)
    h = (b - a) / m
    return float(h * (0.5 * y[0] + y[1:-1].sum() + 0.5 * y[-1]))


def composite_simpson(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, m: int) -> float:
    """Composite Simpson rule with m (even) subintervals (error O(h^4))."""
    if m % 2:
        raise ValueError("composite Simpson requires an even number of subintervals")
    x = np.linspace(a, b, m + 1)
    y = f(x)
    h = (b - a) / m
    return float(h / 3.0 * (y[0] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum() + y[-1]))


def gauss_legendre_rule(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """n-point Gauß–Legendre quadrature on [a, b]."""
    x, w = gauss_legendre(n)
    return float(0.5 * (b - a) * np.dot(w, f(0.5 * (b - a) * x + 0.5 * (a + b))))


def gauss_lobatto_rule(f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int) -> float:
    """n-point Gauß–Lobatto quadrature on [a, b]."""
    x, w = gauss_lobatto(n)
    return float(0.5 * (b - a) * np.dot(w, f(0.5 * (b - a) * x + 0.5 * (a + b))))


def composite_gauss(
    f: Callable[[np.ndarray], np.ndarray], a: float, b: float, n: int, m: int
) -> float:
    """Composite n-point Gauß–Legendre rule on m equal subintervals (error O(h^{2n}))."""
    x, w = gauss_legendre(n)
    edges = np.linspace(a, b, m + 1)
    half = 0.5 * np.diff(edges)
    mids = 0.5 * (edges[:-1] + edges[1:])
    pts = (mids[:, None] + half[:, None] * x[None, :]).ravel()
    vals = np.asarray(f(pts)).reshape(m, n)
    return float(np.sum(half[:, None] * w[None, :] * vals))


def integrate_samples(t: np.ndarray, y: np.ndarray, method: str = "simpson") -> np.ndarray:
    """Integral of sampled data y(t) over [t_0, t_end] along axis 0.

    method: "trapezoid" (O(h^2)), "simpson" (piecewise quadratic, valid for
    non-uniform grids, O(h^4) on smooth data) or "spline" (natural cubic spline).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.shape[0] != t.shape[0]:
        raise ValueError("y must have the same length as t along axis 0")
    if t.shape[0] < 2:
        raise ValueError("at least two samples required")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    if method == "trapezoid":
        return _trapezoid(t, y)
    if method == "simpson":
        return _simpson_nonuniform(t, y)
    if method == "spline":
        from autokinetics.numerics.interpolation import CubicSpline

        return CubicSpline(t, y, bc="natural").integrate(t[0], t[-1])
    raise ValueError(f"unknown method {method!r}")


def _trapezoid(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.diff(t).reshape((-1,) + (1,) * (y.ndim - 1))
    return np.sum(0.5 * h * (y[1:] + y[:-1]), axis=0)


def _simpson_nonuniform(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = t.shape[0] - 1  # intervals
    if n < 2:
        return _trapezoid(t, y)
    total = np.zeros(y.shape[1:])
    last = n if n % 2 == 0 else n - 1
    for i in range(0, last, 2):
        h0, h1 = t[i + 1] - t[i], t[i + 2] - t[i + 1]
        hs = h0 + h1
        total = total + hs / 6.0 * (
            (2.0 - h1 / h0) * y[i] + hs * hs / (h0 * h1) * y[i + 1] + (2.0 - h0 / h1) * y[i + 2]
        )
    if (
        last < n
    ):  # odd number of intervals: last interval with the quadratic through the last 3 points
        h0, h1 = t[n - 1] - t[n - 2], t[n] - t[n - 1]
        a = (2 * h1 * h1 + 3 * h0 * h1) / (6 * (h0 + h1))
        b = (h1 * h1 + 3 * h0 * h1) / (6 * h0)
        c = h1**3 / (6 * h0 * (h0 + h1))
        total = total + a * y[n] + b * y[n - 1] - c * y[n - 2]
    return total


def cumulative_trapezoid(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Running integral int_{t_0}^{t_k} y dt (trapezoid), same shape as y, starting at 0."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    h = np.diff(t).reshape((-1,) + (1,) * (y.ndim - 1))
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * h * (y[1:] + y[:-1]), axis=0)
    return out
