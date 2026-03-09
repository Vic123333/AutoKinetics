"""
ODE solver for chemical reaction kinetics.
Adapted from the desktop app for the web backend (import paths fixed).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from data_model import ReactionSystem


class ODESolver:
    def __init__(self, system: ReactionSystem, temperature: float):
        self.system      = system
        self.temperature = temperature

        self.qssa_indices  = [i for i, s in enumerate(system.species)
                               if getattr(s, 'is_intermediate', False)]
        self.normal_indices = [i for i in range(len(system.species))
                                if i not in self.qssa_indices]

    def _calculate_rates(self, concentrations):
        rates = []
        for reaction in self.system.reactions:
            k    = reaction.calculate_k(self.temperature)
            rate = k
            for reactant_idx, _ in reaction.reactants:
                conc  = max(concentrations[reactant_idx], 0)
                order = reaction.reaction_order.get(reactant_idx, 1.0)
                rate *= conc ** order
            rates.append(rate)
        return np.array(rates)

    def model_standard(self, t, y):
        dydt  = np.zeros_like(y)
        rates = self._calculate_rates(y)
        for r_idx, reaction in enumerate(self.system.reactions):
            rate = rates[r_idx]
            for reactant_idx, stoich in reaction.reactants:
                dydt[reactant_idx] -= stoich * rate
            for product_idx, stoich in reaction.products:
                dydt[product_idx]  += stoich * rate
        return dydt

    def _qssa_equations(self, qssa_concs, normal_concs_array):
        full_concs = np.zeros(len(self.system.species))
        full_concs[self.normal_indices] = normal_concs_array
        full_concs[self.qssa_indices]   = qssa_concs

        rates      = self._calculate_rates(full_concs)
        d_qssa_dt  = np.zeros_like(qssa_concs)

        for i, qssa_idx in enumerate(self.qssa_indices):
            net_rate = 0
            for r_idx, reaction in enumerate(self.system.reactions):
                for product_idx, stoich in reaction.products:
                    if product_idx == qssa_idx:
                        net_rate += stoich * rates[r_idx]
                for reactant_idx, stoich in reaction.reactants:
                    if reactant_idx == qssa_idx:
                        net_rate -= stoich * rates[r_idx]
            d_qssa_dt[i] = net_rate
        return d_qssa_dt

    def model_qssa(self, t, y_normal):
        initial_guess = np.full(len(self.qssa_indices), 1e-9)
        qssa_concs, _, ier, _ = fsolve(
            self._qssa_equations, initial_guess,
            args=(y_normal,), full_output=True
        )
        if ier != 1:
            qssa_concs.fill(1e-12)
        qssa_concs[qssa_concs < 0] = 0

        concentrations = np.zeros(len(self.system.species))
        concentrations[self.normal_indices] = y_normal
        concentrations[self.qssa_indices]   = qssa_concs

        rates      = self._calculate_rates(concentrations)
        dydt_full  = np.zeros_like(concentrations)
        for r_idx, reaction in enumerate(self.system.reactions):
            rate = rates[r_idx]
            for reactant_idx, stoich in reaction.reactants:
                dydt_full[reactant_idx] -= stoich * rate
            for product_idx, stoich in reaction.products:
                dydt_full[product_idx]  += stoich * rate
        return dydt_full[self.normal_indices]

    def solve(self, t_span, t_eval):
        if not self.qssa_indices:
            y0 = self.system.get_initial_concentrations()
            return solve_ivp(
                fun=self.model_standard, t_span=t_span, y0=y0,
                t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-10
            )

        y0_full   = self.system.get_initial_concentrations()
        y0_normal = y0_full[self.normal_indices]

        sol_normal = solve_ivp(
            fun=self.model_qssa, t_span=t_span, y0=y0_normal,
            t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-10
        )

        y_full = np.zeros((len(self.system.species), len(sol_normal.t)))
        y_full[self.normal_indices, :] = sol_normal.y

        last_qssa = np.full(len(self.qssa_indices), 1e-9)
        for i in range(len(sol_normal.t)):
            qssa_t, _, ier, _ = fsolve(
                self._qssa_equations, last_qssa,
                args=(sol_normal.y[:, i],), full_output=True
            )
            if ier == 1:
                last_qssa = qssa_t
            else:
                qssa_t.fill(1e-12)
            qssa_t[qssa_t < 0] = 0
            y_full[self.qssa_indices, i] = qssa_t

        class _Sol:
            def __init__(self, t, y):
                self.t = t
                self.y = y
        return _Sol(sol_normal.t, y_full)
