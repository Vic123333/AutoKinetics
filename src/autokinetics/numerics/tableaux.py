"""Butcher tableaux: the built-in C++ tableaux and collocation methods
generated from quadrature nodes (ODE script Kap. 5.4, Lemma 5.7–5.9).

A collocation method with nodes c_1..c_s has
    a_ij = int_0^{c_i} l_j(tau) dtau,   b_j = int_0^1 l_j(tau) dtau,
with the Lagrange basis l_j of the nodes. Gauß–Legendre nodes give the Gauß
methods (order 2s), right Radau nodes Radau IIA (order 2s-1), Lobatto nodes
Lobatto IIIA (order 2s-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autokinetics._ext import _core
from autokinetics.numerics.quadrature import gauss_legendre, gauss_lobatto, gauss_radau

__all__ = [
    "ButcherTableau",
    "builtin_tableau_names",
    "collocation_tableau",
    "gauss_legendre_tableau",
    "radau_iia_tableau",
    "lobatto_iiia_tableau",
]


@dataclass(frozen=True)
class ButcherTableau:
    """Runge–Kutta method (A, b, c) with optional embedded weights bhat."""

    name: str
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    order: int
    bhat: np.ndarray | None = None
    order_hat: int | None = None
    fsal: bool = False
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        A = np.asarray(self.A, dtype=float)
        s = A.shape[0]
        if A.shape != (s, s):
            raise ValueError("A must be square")
        if np.shape(self.b) != (s,) or np.shape(self.c) != (s,):
            raise ValueError("b and c must have length s")
        if self.bhat is not None and np.shape(self.bhat) != (s,):
            raise ValueError("bhat must have length s")
        if self.bhat is not None and not self.order_hat:
            raise ValueError("order_hat required together with bhat")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", np.asarray(self.b, dtype=float))
        object.__setattr__(self, "c", np.asarray(self.c, dtype=float))
        if self.bhat is not None:
            object.__setattr__(self, "bhat", np.asarray(self.bhat, dtype=float))

    @property
    def stages(self) -> int:
        return self.A.shape[0]

    @property
    def is_explicit(self) -> bool:
        return bool(np.all(np.triu(self.A) == 0.0))

    @property
    def is_diagonally_implicit(self) -> bool:
        return bool(np.all(np.triu(self.A, 1) == 0.0)) and not self.is_explicit

    @property
    def stiffly_accurate(self) -> bool:
        return bool(self.c[-1] == 1.0 and np.allclose(self.A[-1], self.b, rtol=0, atol=1e-15))

    @classmethod
    def builtin(cls, name: str) -> ButcherTableau:
        d = _core.num_tableau(name)
        return cls(
            name=d["name"],
            A=d["A"],
            b=d["b"],
            c=d["c"],
            order=d["order"],
            bhat=d["bhat"],
            order_hat=d["order_hat"],
            fsal=d["fsal"],
        )

    def to_cpp_dict(self) -> dict:
        return {
            "name": self.name,
            "A": self.A,
            "b": self.b,
            "c": self.c,
            "order": self.order,
            "bhat": self.bhat,
            "order_hat": self.order_hat,
            "fsal": self.fsal,
        }

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        return {
            "name": self.name,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "c": self.c.tolist(),
            "order": self.order,
            "bhat": None if self.bhat is None else self.bhat.tolist(),
            "order_hat": self.order_hat,
            "fsal": self.fsal,
            "stages": self.stages,
            "explicit": self.is_explicit,
        }


def builtin_tableau_names() -> list[str]:
    return list(_core.num_tableau_names())


def collocation_tableau(c, name: str, order: int) -> ButcherTableau:
    """Collocation Runge–Kutta tableau for the nodes c (exact polynomial integration
    of the Lagrange basis)."""
    c = np.asarray(c, dtype=float)
    s = c.size
    A = np.zeros((s, s))
    b = np.zeros(s)
    for j in range(s):
        others = np.delete(c, j)
        lj = np.polynomial.Polynomial([1.0])
        for xk in others:
            lj = lj * np.polynomial.Polynomial([-xk, 1.0]) / (c[j] - xk)
        Lj = lj.integ()  # antiderivative with Lj(0) = 0
        b[j] = Lj(1.0)
        A[:, j] = Lj(c)
    return ButcherTableau(
        name=name,
        A=A,
        b=b,
        c=c,
        order=order,
        metadata={"construction": "collocation", "nodes": c.tolist()},
    )


def gauss_legendre_tableau(s: int) -> ButcherTableau:
    """Gauß method with s stages (order 2s; s=1 is the implicit midpoint rule)."""
    x, _ = gauss_legendre(s)
    return collocation_tableau((x + 1.0) / 2.0, f"gauss_legendre_s{s}", 2 * s)


def radau_iia_tableau(s: int) -> ButcherTableau:
    """Radau IIA with s stages (order 2s-1; s=1 is implicit Euler)."""
    x, _ = gauss_radau(s)
    return collocation_tableau(x, f"radau_iia_s{s}", 2 * s - 1)


def lobatto_iiia_tableau(s: int) -> ButcherTableau:
    """Lobatto IIIA with s >= 2 stages (order 2s-2; s=2 is the trapezoidal rule)."""
    x, _ = gauss_lobatto(s)
    return collocation_tableau((x + 1.0) / 2.0, f"lobatto_iiia_s{s}", 2 * s - 2)
