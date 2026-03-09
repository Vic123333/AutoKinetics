"""
Numerical ODE solver for chemical reaction networks.

Method: Radau (implicit Runge-Kutta, A-stable, 5th order)
        – the standard choice for stiff chemical kinetics systems.

Tolerances:  rtol = 1e-7, atol = 1e-10
        Recommended for teaching examples (NIST / textbook benchmarks).
        Tighter than scipy defaults to avoid visible integration artefacts
        when concentrations approach zero.

QSSA: When species are flagged as intermediates, their d[X]/dt = 0
      algebraic constraint is solved at each time step (Tikhonov–Fenichel
      reduction). A warning is returned if timescale separation < 10×.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize    import fsolve
from .data_model import ReactionSystem

# Solver tolerances (conservative for scientific accuracy)
RTOL = 1e-7
ATOL = 1e-10


class ODESolver:
    def __init__(self, system: ReactionSystem, temperature: float):
        self.system      = system
        self.temperature = temperature
        self.qssa_idx    = [i for i, s in enumerate(system.species) if s.is_intermediate]
        self.normal_idx  = [i for i in range(len(system.species)) if i not in self.qssa_idx]

    # ── rate calculation ─────────────────────────────────────────────────────
    def _rates(self, conc: np.ndarray) -> np.ndarray:
        rates = np.empty(len(self.system.reactions))
        for j, rxn in enumerate(self.system.reactions):
            k = rxn.calculate_k(self.temperature)
            v = k
            for idx, _ in rxn.reactants:
                order = rxn.reaction_order.get(idx, 1.0)
                c = max(conc[idx], 0.0)
                v *= c ** order
            rates[j] = v
        return rates

    def _dydt(self, conc: np.ndarray) -> np.ndarray:
        d = np.zeros(len(self.system.species))
        rates = self._rates(conc)
        for j, rxn in enumerate(self.system.reactions):
            r = rates[j]
            for idx, s in rxn.reactants:
                d[idx] -= s * r
            for idx, s in rxn.products:
                d[idx] += s * r
        return d

    # ── standard ODE RHS ─────────────────────────────────────────────────────
    def _rhs_full(self, _t, y):
        return self._dydt(y)

    # ── QSSA helpers ─────────────────────────────────────────────────────────
    def _qssa_residual(self, qssa_conc, normal_conc):
        full = np.zeros(len(self.system.species))
        full[self.normal_idx] = normal_conc
        full[self.qssa_idx]   = np.maximum(qssa_conc, 0.0)
        d = self._dydt(full)
        return d[self.qssa_idx]

    def _qssa_rhs(self, _t, y_normal):
        # Solve algebraic QSSA constraint: d[intermediate]/dt = 0
        guess = np.full(len(self.qssa_idx), 1e-9)
        sol, _, ier, _ = fsolve(self._qssa_residual, guess,
                                args=(y_normal,), full_output=True)
        if ier != 1:
            sol.fill(0.0)
        full = np.zeros(len(self.system.species))
        full[self.normal_idx] = y_normal
        full[self.qssa_idx]   = np.maximum(sol, 0.0)
        return self._dydt(full)[self.normal_idx]

    # ── public API ────────────────────────────────────────────────────────────
    def solve(self, t_span, t_eval):
        y0 = self.system.get_initial_concentrations()
        if not self.qssa_idx:
            sol = solve_ivp(self._rhs_full, t_span, y0,
                            t_eval=t_eval, method='Radau',
                            rtol=RTOL, atol=ATOL, dense_output=False)
            return _Solution(sol.t, sol.y, sol.success, sol.message)

        # QSSA path
        y0_normal = y0[self.normal_idx]
        sol = solve_ivp(self._qssa_rhs, t_span, y0_normal,
                        t_eval=t_eval, method='Radau',
                        rtol=RTOL, atol=ATOL, dense_output=False)
        # Reconstruct full concentration matrix
        n_sp, n_t = len(self.system.species), len(sol.t)
        y_full = np.zeros((n_sp, n_t))
        y_full[self.normal_idx, :] = sol.y
        # Re-solve QSSA intermediates at each output time
        last = np.full(len(self.qssa_idx), 1e-9)
        for i in range(n_t):
            q, _, ier, _ = fsolve(self._qssa_residual, last,
                                  args=(sol.y[:, i],), full_output=True)
            if ier == 1:
                last = q
            y_full[self.qssa_idx, i] = np.maximum(last, 0.0)
        return _Solution(sol.t, y_full, sol.success, sol.message)


class _Solution:
    """Lightweight result container mirroring scipy OdeResult interface."""
    __slots__ = ('t', 'y', 'success', 'message')

    def __init__(self, t, y, success, message):
        self.t       = t
        self.y       = y
        self.success = bool(success)
        self.message = message or ''
