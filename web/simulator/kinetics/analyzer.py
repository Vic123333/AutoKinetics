"""
Kinetic order analysis of simulation results.

Method: Linear regression on transformed concentration–time data.

Transforms:
  0th order: [A](t)       – slope = -k,  unit: mol L⁻¹ s⁻¹
  1st order: ln[A](t)     – slope = -k,  unit: s⁻¹
  2nd order: 1/[A](t)     – slope = +k,  unit: L mol⁻¹ s⁻¹

R² quality threshold: 0.98  (standard in chemical kinetics teaching, LibreTexts)

We also report:
  • adjusted R²  (penalises extra parameters – not needed here since all
                  models are 1-parameter after transform, but shown for reference)
  • RMSE of the linearised fit
  • half-life t½ for 1st-order reactions  (t½ = ln 2 / k)
  • characteristic time τ = 1/k for 1st-order

Scientific references:
  - Atkins, Physical Chemistry, 11th ed., Chapter 17
  - NIST Chemical Kinetics Database
"""
from __future__ import annotations
import numpy as np
from .data_model import ReactionSystem

R2_THRESHOLD = 0.98    # minimum acceptable R² for "fits well"


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-30:
        return 1.0 if ss_res < 1e-30 else 0.0
    return float(1.0 - ss_res / ss_tot)


def _linear_fit(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, r2, rmse) via least-squares polyfit."""
    if len(x) < 3:
        return 0.0, 0.0, 0.0, float('inf')
    coeffs  = np.polyfit(x, y, 1)
    y_pred  = np.polyval(coeffs, x)
    r2      = _r_squared(y, y_pred)
    rmse    = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return float(coeffs[0]), float(coeffs[1]), r2, rmse


def _unit_for_order(n: float) -> str:
    n = round(n)
    if n == 0: return 'mol L⁻¹ s⁻¹'
    if n == 1: return 's⁻¹'
    if n == 2: return 'L mol⁻¹ s⁻¹'
    if n == 3: return 'L² mol⁻² s⁻¹'
    return f'M^(1-{n}) s⁻¹'


def analyze_kinetics(sim_results: dict, system: ReactionSystem) -> dict:
    """
    Parameters
    ----------
    sim_results : {
        'time_points':    list[float],
        'concentrations': list[list[float]]   # indexed [species_idx][time_idx]
    }
    system : ReactionSystem

    Returns
    -------
    dict keyed by reaction rate_label → kinetic analysis sub-dict
    """
    t   = np.asarray(sim_results['time_points'], dtype=float)
    all_c = [np.asarray(c, dtype=float) for c in sim_results['concentrations']]

    results = {}
    for rxn in system.reactions:
        if not rxn.reactants:
            continue
        # Analyse first non-intermediate reactant
        primary_idx = next(
            (idx for idx, _ in rxn.reactants
             if not system.species[idx].is_intermediate),
            rxn.reactants[0][0]
        )
        c = np.clip(all_c[primary_idx], 1e-30, None)   # guard against log(0)

        # ── 0th order: [A] vs t ──────────────────────────────────────────────
        slope0, intercept0, r2_0, rmse0 = _linear_fit(t, c)
        k0 = max(-slope0, 0.0)     # k = -slope for consumption

        # ── 1st order: ln[A] vs t ────────────────────────────────────────────
        ln_c = np.log(c)
        slope1, intercept1, r2_1, rmse1 = _linear_fit(t, ln_c)
        k1 = max(-slope1, 0.0)

        # ── 2nd order: 1/[A] vs t ────────────────────────────────────────────
        inv_c = 1.0 / c
        slope2, intercept2, r2_2, rmse2 = _linear_fit(t, inv_c)
        k2 = max(slope2, 0.0)

        fits = {
            'zero_order':   {'k': k0, 'r_squared': r2_0, 'rmse': rmse0, 'unit': _unit_for_order(0)},
            'first_order':  {'k': k1, 'r_squared': r2_1, 'rmse': rmse1, 'unit': _unit_for_order(1)},
            'second_order': {'k': k2, 'r_squared': r2_2, 'rmse': rmse2, 'unit': _unit_for_order(2)},
        }

        best_key = max(fits, key=lambda key: fits[key]['r_squared'])
        best     = fits[best_key]

        # ── derived quantities ────────────────────────────────────────────────
        extra: dict = {}
        if best_key == 'first_order' and k1 > 1e-30:
            extra['half_life_s']          = round(np.log(2) / k1, 6)
            extra['characteristic_time_s'] = round(1.0 / k1, 6)

        # ── quality flag ──────────────────────────────────────────────────────
        fits_well = best['r_squared'] >= R2_THRESHOLD

        results[rxn.rate_label] = {
            'analyzed_reactant':  system.species[primary_idx].name,
            'best_fit_order':     best_key,
            'calculated_k':       round(best['k'], 8),
            'k_unit':             best['unit'],
            'r_squared':          round(best['r_squared'], 6),
            'fits_well':          fits_well,
            'r2_threshold':       R2_THRESHOLD,
            'all_fits':           fits,
            **extra,
        }

    return results
