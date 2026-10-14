"""Interpolation (Numerik I, Kap. 7): Lagrange (barycentric form), Newton's
divided differences, cubic splines (natural and not-a-knot) and resampling of
simulation results."""

from __future__ import annotations

import numpy as np

__all__ = [
    "lagrange_interpolate",
    "divided_differences",
    "newton_evaluate",
    "CubicSpline",
    "resample",
]


def _nodes(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("nodes must be a non-empty 1-D array")
    if np.unique(x).size != x.size:
        raise ValueError("interpolation nodes must be pairwise distinct")
    return x


def lagrange_interpolate(x_nodes, y_nodes, x) -> np.ndarray:
    """Evaluates the interpolation polynomial through (x_i, y_i) at x
    (second barycentric form, numerically stable)."""
    xn = _nodes(x_nodes)
    yn = np.asarray(y_nodes, dtype=float)
    if yn.shape[0] != xn.size:
        raise ValueError("y_nodes must have the same length as x_nodes")
    diff = xn[:, None] - xn[None, :]
    np.fill_diagonal(diff, 1.0)
    w = 1.0 / np.prod(diff, axis=1)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    d = x[:, None] - xn[None, :]
    exact = d == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = w[None, :] / d
        num = np.tensordot(k, yn, axes=(1, 0))
        den = k.sum(axis=1).reshape((-1,) + (1,) * (yn.ndim - 1))
        out = num / den
    rows, cols = np.nonzero(exact)
    out[rows] = yn[cols]
    return out


def divided_differences(x_nodes, y_nodes) -> np.ndarray:
    """Coefficients [y_0], [y_0, y_1], ..., [y_0..y_n] of the Newton form."""
    xn = _nodes(x_nodes)
    c = np.array(y_nodes, dtype=float, copy=True)
    if c.shape[0] != xn.size:
        raise ValueError("y_nodes must have the same length as x_nodes")
    n = xn.size
    for k in range(1, n):
        c[k:] = (c[k:] - c[k - 1 : -1]) / (xn[k:] - xn[: n - k]).reshape(
            (-1,) + (1,) * (c.ndim - 1)
        )
    return c


def newton_evaluate(coeffs, x_nodes, x) -> np.ndarray:
    """Horner-type evaluation of the Newton form."""
    xn = _nodes(x_nodes)
    c = np.asarray(coeffs, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    shape = (-1,) + (1,) * (c.ndim - 1)
    p = np.broadcast_to(c[-1], (x.size,) + c.shape[1:]).copy()
    for k in range(c.shape[0] - 2, -1, -1):
        p = p * (x - xn[k]).reshape(shape) + c[k]
    return p


class CubicSpline:
    """Interpolating cubic spline s in C^2 on the nodes x (strictly increasing).

    bc = "natural" (s'' = 0 at both ends) or "not-a-knot" (s''' continuous at
    x_1 and x_{n-2}). y may be 1-D or 2-D (values along axis 0). The moments
    M_i = s''(x_i) are obtained from a tridiagonal system (Thomas algorithm).
    """

    def __init__(self, x, y, bc: str = "not-a-knot"):
        x = _nodes(x)
        if np.any(np.diff(x) <= 0):
            raise ValueError("spline nodes must be strictly increasing")
        y = np.asarray(y, dtype=float)
        if y.shape[0] != x.size:
            raise ValueError("y must have the same length as x along axis 0")
        if x.size < 2:
            raise ValueError("at least two nodes required")
        if bc not in ("natural", "not-a-knot"):
            raise ValueError("bc must be 'natural' or 'not-a-knot'")
        self.x, self.y, self.bc = x, y, bc
        self.M = self._moments()

    def _moments(self) -> np.ndarray:
        x, y = self.x, self.y
        n = x.size
        h = np.diff(x)
        shape = (-1,) + (1,) * (y.ndim - 1)
        M = np.zeros_like(y)
        if n == 2:
            return M
        if self.bc == "not-a-knot" and n < 4:
            # not-a-knot with 3 nodes: the single interpolating parabola
            c = divided_differences(x, y)
            M[:] = 2.0 * c[2]
            return M
        d = np.diff(y, axis=0) / h.reshape(shape)
        rhs = 6.0 * (d[1:] - d[:-1])  # equations i = 1..n-2
        m = n - 2
        lower = h[:-1].copy()  # coefficient of M_{i-1}
        diag = 2.0 * (h[:-1] + h[1:])
        upper = h[1:].copy()  # coefficient of M_{i+1}
        if self.bc == "not-a-knot":
            # M_0 = (1 + h0/h1) M_1 - (h0/h1) M_2,  M_{n-1} analogous
            r0 = h[0] / h[1]
            diag[0] += h[0] * (1.0 + r0)
            upper[0] -= h[0] * r0
            r1 = h[-1] / h[-2]
            diag[-1] += h[-1] * (1.0 + r1)
            lower[-1] -= h[-1] * r1
        inner = _thomas(lower, diag, upper, rhs, m)
        M[1:-1] = inner
        if self.bc == "not-a-knot":
            M[0] = (1.0 + h[0] / h[1]) * M[1] - (h[0] / h[1]) * M[2]
            M[-1] = (1.0 + h[-1] / h[-2]) * M[-2] - (h[-1] / h[-2]) * M[-3]
        return M

    def _locate(self, xq: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.x, xq, side="right") - 1, 0, self.x.size - 2)

    def __call__(self, xq, nu: int = 0) -> np.ndarray:
        """Spline (nu = 0) or its derivatives (nu = 1, 2) at xq; linear/cubic extrapolation outside."""
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        i = self._locate(xq)
        x, y, M = self.x, self.y, self.M
        h = x[i + 1] - x[i]
        shape = (-1,) + (1,) * (y.ndim - 1)
        a = ((x[i + 1] - xq) / h).reshape(shape)
        b = ((xq - x[i]) / h).reshape(shape)
        hh = h.reshape(shape)
        if nu == 0:
            return (
                a * y[i]
                + b * y[i + 1]
                + ((a**3 - a) * M[i] + (b**3 - b) * M[i + 1]) * hh * hh / 6.0
            )
        if nu == 1:
            return (y[i + 1] - y[i]) / hh + (
                (1 - 3 * a * a) * M[i] + (3 * b * b - 1) * M[i + 1]
            ) * hh / 6.0
        if nu == 2:
            return a * M[i] + b * M[i + 1]
        raise ValueError("nu must be 0, 1 or 2")

    def integrate(self, a: float, b: float) -> np.ndarray:
        """Exact integral of the spline over [a, b] (within the node range)."""
        if (
            a < self.x[0] - 1e-14 * abs(self.x[0])
            or b > self.x[-1] + 1e-14 * abs(self.x[-1])
            or a > b
        ):
            raise ValueError("integration bounds must satisfy x_0 <= a <= b <= x_end")
        x, y, M = self.x, self.y, self.M
        h = np.diff(x)
        shape = (-1,) + (1,) * (y.ndim - 1)
        hh = h.reshape(shape)
        # integral over each full interval
        full = 0.5 * hh * (y[:-1] + y[1:]) - hh**3 / 24.0 * (M[:-1] + M[1:])
        cum = np.concatenate([np.zeros((1,) + y.shape[1:]), np.cumsum(full, axis=0)], axis=0)
        return self._antiderivative(b, cum) - self._antiderivative(a, cum)

    def _antiderivative(self, xq: float, cum: np.ndarray) -> np.ndarray:
        i = int(self._locate(np.array([xq]))[0])
        x, y, M = self.x, self.y, self.M
        h = x[i + 1] - x[i]
        s = xq - x[i]
        # int_{x_i}^{x_i+s} of the cubic in moment form, with a = (x_{i+1}-x)/h, b = (x-x_i)/h:
        # int a = s - s^2/(2h), int b = s^2/(2h), int a^3 = (h - (h-s)^4/h^3)/4, int b^3 = s^4/(4h^3)
        a_int = s - s * s / (2 * h)
        b_int = s * s / (2 * h)
        ia3 = (h - (h - s) ** 4 / h**3) / 4.0
        ib3 = s**4 / (4.0 * h**3)
        part = (
            a_int * y[i]
            + b_int * y[i + 1]
            + ((ia3 - a_int) * M[i] + (ib3 - b_int) * M[i + 1]) * h * h / 6.0
        )
        return cum[i] + part


def _thomas(lower, diag, upper, rhs, m) -> np.ndarray:
    """Tridiagonal solve; lower[k] multiplies x_{k-1} in row k, upper[k] x_{k+1}."""
    c = np.zeros(m)
    d = np.zeros_like(rhs)
    beta = diag[0]
    if beta == 0.0:
        raise ZeroDivisionError("singular tridiagonal system")
    c[0] = upper[0] / beta if m > 1 else 0.0
    d[0] = rhs[0] / beta
    for k in range(1, m):
        beta = diag[k] - lower[k] * c[k - 1]
        if beta == 0.0:
            raise ZeroDivisionError("singular tridiagonal system")
        if k < m - 1:
            c[k] = upper[k] / beta
        d[k] = (rhs[k] - lower[k] * d[k - 1]) / beta
    x = np.zeros_like(rhs)
    x[m - 1] = d[m - 1]
    for k in range(m - 2, -1, -1):
        x[k] = d[k] - c[k] * x[k + 1]
    return x


def resample(t, y, t_new, method: str = "cubic") -> np.ndarray:
    """Resamples data y(t) (values along axis 0) onto t_new.

    method: "linear", "cubic" (not-a-knot spline) or "natural" (natural spline).
    """
    t = _nodes(t)
    y = np.asarray(y, dtype=float)
    t_new = np.asarray(t_new, dtype=float)
    if method == "linear":
        if y.ndim == 1:
            return np.interp(t_new, t, y)
        return np.column_stack([np.interp(t_new, t, y[:, j]) for j in range(y.shape[1])])
    if method in ("cubic", "natural"):
        return CubicSpline(t, y, bc="not-a-knot" if method == "cubic" else "natural")(t_new)
    raise ValueError("method must be 'linear', 'cubic' or 'natural'")
