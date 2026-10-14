"""Chemical equilibrium by Gibbs/Helmholtz minimisation (element-potential method).

At fixed T and V the ideal-gas / ideal-dilute equilibrium solves

    max_λ  D(λ) = bᵀλ − Σ_i K_i exp(a_iᵀλ),     K_i = V c° exp(−g°_i / RT)

which is the (unconstrained, strictly concave) dual of the Helmholtz minimisation
with conservation constraints A n = b; then n_i = K_i exp(a_iᵀλ). Newton's method
with backtracking converges globally. Constant pressure is handled by an outer
1-D root search on ln V, adiabatic modes (HP/UV) by a root search on T.

Conservation constraints are either the element balances or — when compositions
are unknown — the conservation laws of the reaction network (left null space of
the stoichiometric matrix). In the latter case the equilibrium is exactly the
stationary state that reversible, thermodynamically consistent kinetics reach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .compiled import compile_mechanism
from .constants import GAS_CONSTANT as R
from .constants import STANDARD_CONCENTRATION_LIQUID, STANDARD_PRESSURE
from .errors import SimulationError
from .mechanism import Mechanism


@dataclass
class EquilibriumResult:
    T: float
    P: float
    V: float  # m³ (for the given initial amounts)
    amounts: np.ndarray  # mol
    concentrations: np.ndarray  # mol/m³
    mole_fractions: np.ndarray
    basis: str
    iterations: int
    residual: float
    species: list[str]
    notes: list[str]

    def to_dict(self, concentration_unit: str = "mol/L") -> dict:
        f = 1e-3 if concentration_unit == "mol/L" else 1.0
        return {"T": self.T, "P": self.P, "V": self.V, "species": self.species,
                "concentrations": (self.concentrations * f).tolist(), "concentration_unit": concentration_unit,
                "mole_fractions": self.mole_fractions.tolist(), "basis": self.basis,
                "iterations": self.iterations, "residual": self.residual, "notes": self.notes}


def conservation_matrix(nu: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Basis W (k × n_species) of conservation laws: W @ nu.T = 0 (nu: n_rxn × n_species)."""
    ns = nu.shape[1]
    if nu.shape[0] == 0:
        return np.eye(ns)
    _, s, vt = np.linalg.svd(nu)
    rank = int(np.sum(s > tol * max(s.max(), 1.0)))
    W = vt[rank:]
    # Make the basis as "element-like" as possible (reduced row echelon form, small integers if possible)
    W = _rref(W)
    return W


def _rref(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    A = M.copy()
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        p = r + int(np.argmax(np.abs(A[r:, c])))
        if abs(A[p, c]) < tol:
            continue
        A[[r, p]] = A[[p, r]]
        A[r] /= A[r, c]
        for i in range(rows):
            if i != r:
                A[i] -= A[i, c] * A[r]
        r += 1
    A[np.abs(A) < tol] = 0.0
    return A[:r]


class _Problem:
    """Pre-processed conservation system for one mechanism and initial state."""

    def __init__(self, mech: Mechanism, n0: np.ndarray, basis: str):
        self.mech = mech
        self.notes: list[str] = []
        ns = mech.n_species
        if basis == "auto":
            basis = "elements" if mech.elements and all(s.composition for s in mech.species) else "reactions"
        if basis == "elements":
            A = mech.element_matrix()
        elif basis == "reactions":
            nr, np_ = mech.stoichiometric_matrices()
            A = conservation_matrix(np_ - nr)
        else:
            raise SimulationError(f"Unbekannte Bilanzbasis '{basis}'")
        self.basis = basis
        b = A @ n0
        # Constraints whose rows are non-negative and b == 0 force the species involved to zero.
        active = np.ones(ns, bool)
        keep_rows = np.ones(A.shape[0], bool)
        for k in range(A.shape[0]):
            if abs(b[k]) <= 1e-14 * max(np.abs(n0).sum(), 1e-300) and np.all(A[k] >= -1e-12):
                active &= ~(A[k] > 1e-12)
                keep_rows[k] = False
        A = A[keep_rows][:, active]
        b = b[keep_rows]
        # Remove linearly dependent rows
        if A.shape[0]:
            q, r_, piv = _qr_pivot(A.T)
            rank = int(np.sum(np.abs(np.diag(r_)) > 1e-10 * max(np.abs(r_).max(), 1.0))) if r_.size else 0
            rows = sorted(piv[:rank])
            A, b = A[rows], b[rows]
        self.A, self.b, self.active = A, b, active
        if not np.any(active):
            raise SimulationError("Gleichgewicht: keine Spezies mit positiver Stoffmenge möglich")

    def initial_lambda(self, lnK: np.ndarray) -> np.ndarray:
        """Start value from the linear programme min Σ n_i(−ln K_i) s.t. A n = b, n ≥ 0
        (the T → 0 limit of the Gibbs minimisation): solve A_Bᵀ λ = ln n_B − ln K_B on
        the species of the optimal vertex, with a small floor so all species exist."""
        from scipy.optimize import linprog

        A, b = self.A, self.b
        lp = linprog(-lnK, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if lp.status != 0:
            raise SimulationError("Gleichgewicht: Bilanzgleichungen nicht erfüllbar (keine zulässige Zusammensetzung)")
        n = np.asarray(lp.x)
        floor = 1e-6 * max(float(n.sum()), 1e-300)
        basis = n > floor
        if basis.sum() < A.shape[0]:
            order = np.argsort(-(lnK + 0.0))  # add the most stable remaining species
            for i in order:
                if not basis[i]:
                    basis[i] = True
                    if np.linalg.matrix_rank(A[:, basis]) >= A.shape[0]:
                        break
        target = np.log(np.maximum(n[basis], floor))
        lam, *_ = np.linalg.lstsq(A[:, basis].T, target - lnK[basis], rcond=None)
        return lam

    def solve_TV(self, lnK: np.ndarray, lam0: np.ndarray | None = None, tol: float = 1e-11,
                 max_iter: int = 300) -> tuple[np.ndarray, np.ndarray, int, float]:
        A, b = self.A, self.b
        lnK = lnK[self.active]
        m = A.shape[0]
        if m == 0:
            return np.exp(np.minimum(lnK, 700)), np.zeros(0), 0, 0.0
        if lam0 is not None and len(lam0) == m:
            lam = lam0.copy()
            if np.max(lnK + A.T @ lam) > 600:
                lam = self.initial_lambda(lnK)
        else:
            lam = self.initial_lambda(lnK)
        bscale = np.maximum(np.abs(b), 1e-300)

        def amounts(lam_: np.ndarray) -> np.ndarray:
            return np.exp(np.minimum(lnK + A.T @ lam_, 700.0))

        def dual(lam_: np.ndarray) -> float:
            return float(b @ lam_ - amounts(lam_).sum())

        n = amounts(lam)
        res = np.inf
        best = np.inf
        stall = 0
        for it in range(1, max_iter + 1):
            g = b - A @ n
            res = float(np.max(np.abs(g) / bscale))
            if res < tol:
                return n, lam, it, res
            if res < 1e-8 and res >= 0.5 * best:
                stall += 1
                if stall >= 3:  # converged to round-off level
                    return n, lam, it, res
            best = min(best, res)
            H = (A * n) @ A.T
            try:
                step = np.linalg.solve(H + 1e-300 * np.eye(m), g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            # limit the step to avoid exp overflow, then backtrack on the concave dual
            mx = np.max(np.abs(A.T @ step))
            if mx > 20.0:
                step *= 20.0 / mx
            d0 = dual(lam)
            slope = float(g @ step)
            alpha = 1.0
            while alpha > 1e-12:
                trial = lam + alpha * step
                if dual(trial) >= d0 + 1e-4 * alpha * slope:
                    break
                alpha *= 0.5
            lam = lam + alpha * step
            n = amounts(lam)
        return n, lam, max_iter, res


def _qr_pivot(M: np.ndarray):
    from scipy.linalg import qr

    return qr(M, pivoting=True, mode="economic")


def _standard_g_RT(mech: Mechanism, T: float, km, notes: list[str]) -> np.ndarray:
    """Dimensionless standard chemical potentials g°/RT for each species.

    Uses species thermo when complete, otherwise derives pseudo-potentials from the
    equilibrium constants of the (reversible) reactions (min-norm least squares) and
    checks the Wegscheider consistency conditions.
    """
    if mech.has_thermo:
        _, h, s = km.species_thermo(T)
        return (h - T * s) / (R * T)
    irreversible = [j for j, r in enumerate(mech.reactions) if not r.reversible]
    if irreversible:
        raise SimulationError("Gleichgewicht nicht definiert: Thermodaten fehlen und Reaktionen "
                              f"{[j + 1 for j in irreversible[:5]]} sind irreversibel")
    c0 = STANDARD_PRESSURE / (R * T) if mech.phase == "ideal-gas" else STANDARD_CONCENTRATION_LIQUID
    nr, np_ = mech.stoichiometric_matrices()
    nu = np_ - nr
    kf, kr, _ = km.rate_constants(T, 1e5, np.ones(mech.n_species))
    Kc = np.where(kr > 0, kf / np.where(kr > 0, kr, 1.0), np.nan)
    if np.any(~np.isfinite(Kc)) or np.any(Kc <= 0):
        raise SimulationError("Gleichgewicht: Gleichgewichtskonstanten nicht bestimmbar")
    rhs = -(np.log(Kc) - nu.sum(axis=1) * math.log(c0))
    g, *_ = np.linalg.lstsq(nu, rhs, rcond=None)
    resid = float(np.max(np.abs(nu @ g - rhs))) if len(rhs) else 0.0
    if resid > 1e-6 * max(1.0, float(np.max(np.abs(rhs)))):
        raise SimulationError("Gleichgewichtskonstanten sind thermodynamisch inkonsistent "
                              f"(Wegscheider-Bedingung verletzt, Residuum {resid:.2e})")
    notes.append("Standardpotentiale aus Gleichgewichtskonstanten der Reaktionen abgeleitet")
    return g


def equilibrate(mech: Mechanism, n0: np.ndarray, T: float, *, P: float | None = None,
                V: float | None = None, mode: str = "TP", basis: str = "auto",
                T_bounds: tuple[float, float] = (200.0, 6000.0)) -> EquilibriumResult:
    """Equilibrium composition.

    ``n0``: initial amounts [mol] (any consistent scale). ``mode``: TP, TV, HP, UV.
    For HP/UV, ``T`` is the initial temperature; the result holds the equilibrium T.
    Liquid phases are treated at constant volume (TV / UV).
    """
    from ._ext._core import KineticsModel

    n0 = np.asarray(n0, float)
    if np.any(n0 < 0) or n0.sum() <= 0:
        raise SimulationError("Gleichgewicht: Startstoffmengen müssen ≥ 0 und nicht alle 0 sein")
    gas = mech.phase == "ideal-gas"
    if not gas and mode in ("TP", "HP"):
        mode = {"TP": "TV", "HP": "UV"}[mode]
    if mode in ("TP", "HP") and not (P and P > 0):
        raise SimulationError("Gleichgewicht bei konstantem Druck benötigt P > 0")
    if mode in ("TV", "UV") and V is None:
        if gas:
            if not P:
                raise SimulationError("Gleichgewicht: V oder P angeben")
            V = n0.sum() * R * T / P
        else:
            raise SimulationError("Gleichgewicht (Flüssigphase): Volumen V angeben")
    km = KineticsModel(compile_mechanism(mech))
    prob = _Problem(mech, n0, basis)
    notes = prob.notes
    c_std = (lambda T_: STANDARD_PRESSURE / (R * T_)) if gas else (lambda T_: STANDARD_CONCENTRATION_LIQUID)

    state = {"lam": None, "it": 0, "res": 0.0}

    def solve_at(T_: float, V_: float) -> np.ndarray:
        g = _standard_g_RT(mech, T_, km, notes if not notes else [])
        lnK = math.log(V_ * c_std(T_)) - g
        n_act, lam, it, res = prob.solve_TV(lnK, state["lam"])
        state.update(lam=lam, it=state["it"] + it, res=res)
        n = np.zeros(mech.n_species)
        n[prob.active] = n_act
        return n

    def solve_TP(T_: float, P_: float) -> tuple[np.ndarray, float]:
        lnV = math.log(n0.sum() * R * T_ / P_)

        def f(lv: float) -> float:
            n = solve_at(T_, math.exp(lv))
            return math.log(n.sum() * R * T_ / math.exp(lv)) - math.log(P_)

        # Newton-secant on ln V (ln P is monotonically decreasing in ln V)
        x0, f0 = lnV, f(lnV)
        x1 = x0 - f0  # d lnP / d lnV ≈ -1
        for _ in range(60):
            f1 = f(x1)
            if abs(f1) < 1e-13:
                break
            denom = (f1 - f0) if abs(f1 - f0) > 1e-300 else -1.0
            x0, f0, x1 = x1, f1, x1 - f1 * (x1 - x0) / denom
        Vv = math.exp(x1)
        return solve_at(T_, Vv), Vv

    def energy(n: np.ndarray, T_: float, kind: str, Vv: float) -> float:
        _, h, _ = km.species_thermo(T_)
        if kind == "H":
            return float(n @ h)
        if gas:
            return float(n @ (h - R * T_))
        return float(n @ h)  # liquids: U ≈ H

    if mode in ("HP", "UV"):
        if not mech.has_thermo:
            raise SimulationError("Adiabates Gleichgewicht benötigt Thermodaten für alle Spezies")
        kind = "H" if mode == "HP" else "U"
        V_fixed = V
        E0 = energy(n0, T, kind, V_fixed or 0.0)

        def resid(T_: float) -> float:
            if mode == "HP":
                n, Vv = solve_TP(T_, P)  # type: ignore[arg-type]
            else:
                n, Vv = solve_at(T_, V_fixed), V_fixed  # type: ignore[assignment]
            return energy(n, T_, kind, Vv) - E0

        lo, hi = T_bounds
        try:
            T_eq = brentq(resid, max(lo, 200.0), hi, xtol=1e-9, rtol=1e-13, maxiter=200)
        except ValueError:
            raise SimulationError("Adiabate Gleichgewichtstemperatur liegt außerhalb "
                                  f"{lo:.0f}–{hi:.0f} K") from None
        T = T_eq
        mode = "TP" if mode == "HP" else "TV"
        V = V_fixed

    if mode == "TP":
        n, Vv = solve_TP(T, P)  # type: ignore[arg-type]
    elif mode == "TV":
        n, Vv = solve_at(T, V), V  # type: ignore[assignment]
    else:
        raise SimulationError(f"Unbekannter Gleichgewichtsmodus '{mode}'")
    c = n / Vv
    P_out = float(n.sum() * R * T / Vv) if gas else float(P or 1e5)
    return EquilibriumResult(T=float(T), P=P_out, V=float(Vv), amounts=n, concentrations=c,
                             mole_fractions=n / n.sum(), basis=prob.basis, iterations=state["it"],
                             residual=float(state["res"]), species=mech.species_names, notes=notes)
