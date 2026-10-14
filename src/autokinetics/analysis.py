"""Kinetic analysis: rates of production, sensitivities, reaction orders,
Arrhenius parameters and parameter estimation from experimental data."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import numpy as np

from .compiled import CompiledMechanism, compile_mechanism
from .constants import GAS_CONSTANT as R
from .errors import AutoKineticsError, SimulationError
from .mechanism import Mechanism
from .reactor import ReactorSpec, build_reactor
from .simulation import SimulationOptions, SimulationResult, simulate


# ---------------------------------------------------------------------------- rates of production
def rates_of_production(result: SimulationResult, species: str, top: int = 10) -> dict:
    """Contribution ν_ij q_j(t) of every reaction to d[species]/dt along the solution,
    plus time-integrated contributions (production / consumption pathways)."""
    from ._ext._core import KineticsModel

    mech = result.mechanism
    i = mech.species_index(species)
    if result.options.t_eval is None:
        # integrate the contributions on a refined grid (the solver's own steps can be coarse)
        opts = copy.copy(result.options)
        opts.t_eval = np.unique(np.concatenate([result.t, np.linspace(0.0, opts.t_end, 801)]))
        opts.max_points = len(opts.t_eval)
        result = simulate(mech, result.spec, opts)
    km = KineticsModel(compile_mechanism(mech))
    nr, np_ = mech.stoichiometric_matrices()
    nu_i = (np_ - nr)[:, i]
    involved = np.nonzero(nu_i)[0]
    contrib = np.zeros((len(result.t), len(involved)))
    for k in range(len(result.t)):
        _, _, q = km.rates_of_progress(result.T[k], result.P[k], result.concentrations[k])
        contrib[k] = nu_i[involved] * q[involved]
    integral = np.trapezoid(contrib * result.V[:, None], result.t, axis=0) if len(result.t) > 1 else contrib[0]
    order = np.argsort(-np.abs(integral))[:top]
    return {
        "species": species,
        "t": result.t.tolist(),
        "reactions": [{"index": int(involved[o]), "equation": mech.reactions[involved[o]].equation,
                       "integral": float(integral[o]), "rate": contrib[:, o].tolist()} for o in order],
        "unit": "mol/(m³·s)",
    }


# ---------------------------------------------------------------------------- sensitivity
def _scaled_compiled(cm: CompiledMechanism, j: int, factor: float) -> CompiledMechanism:
    c = copy.copy(cm)
    c.A = cm.A.copy()
    c.A[j] *= factor
    c.rev_A = cm.rev_A.copy()
    c.rev_A[j] *= factor  # keep K = kf/kr for explicit reverse rates
    if j in set(cm.fo_rxn.tolist()):
        c.fo_A = cm.fo_A.copy()
        c.fo_A[list(cm.fo_rxn).index(j)] *= factor
    if j in set(cm.plog_rxn.tolist()):
        k = list(cm.plog_rxn).index(j)
        c.plog_A = cm.plog_A.copy()
        c.plog_A[cm.plog_ptr[k]:cm.plog_ptr[k + 1]] *= factor
    return c


def evaluate_target(result: SimulationResult, target: str) -> float:
    """Scalar target from a simulation: ignition_delay | T_final | T_max |
    final:<sp> | max:<sp> | conversion:<sp> | half_life:<sp>."""
    kind, _, sp = target.partition(":")
    if kind == "ignition_delay":
        v = result.metrics.get("ignition_delay")
    elif kind == "T_final":
        v = float(result.T[-1])
    elif kind == "T_max":
        v = float(result.T.max())
    elif kind == "final":
        v = float(result.species(sp)[-1])
    elif kind == "max":
        v = float(result.species(sp).max())
    elif kind == "conversion":
        v = result.metrics.get("conversion", {}).get(sp)
    elif kind == "half_life":
        v = result.metrics.get("half_life", {}).get(sp)
    else:
        raise AutoKineticsError(f"Unbekannte Zielgröße '{target}'")
    if v is None or not np.isfinite(v):
        raise SimulationError(f"Zielgröße '{target}' ist für diese Simulation nicht definiert")
    return float(v)


def sensitivity(mech: Mechanism, spec: ReactorSpec, options: SimulationOptions, target: str,
                reactions: list[int] | None = None, rel_perturbation: float = 0.05,
                max_reactions: int = 40, budget_s: float = 120.0) -> dict:
    """Normalised sensitivities S_j = ∂ln(target)/∂ln(A_j) by central differences.

    Without an explicit reaction list the ``max_reactions`` reactions with the largest
    time-integrated |rate of progress| are analysed (pre-screening)."""
    cm = compile_mechanism(mech)
    base = simulate(mech, spec, options)
    y0 = evaluate_target(base, target)
    if reactions is None:
        from ._ext._core import KineticsModel

        km = KineticsModel(cm)
        qint = np.zeros(mech.n_reactions)
        for k in range(len(base.t) - 1):
            _, _, q = km.rates_of_progress(base.T[k], base.P[k], base.concentrations[k])
            qint += np.abs(q) * (base.t[k + 1] - base.t[k])
        reactions = list(np.argsort(-qint)[:max_reactions])
    deadline = time.perf_counter() + budget_s
    out = []
    skipped = 0
    for j in reactions:
        if time.perf_counter() > deadline:
            skipped += 1
            continue
        vals = []
        for f in (1 + rel_perturbation, 1 / (1 + rel_perturbation)):
            built = build_reactor(mech, spec, options.rtol, options.atol_rel, compiled=_scaled_compiled(cm, j, f))
            vals.append(evaluate_target(simulate(mech, spec, options, built=built), target))
        s = (math.log(vals[0]) - math.log(vals[1])) / (2 * math.log(1 + rel_perturbation)) \
            if min(vals) > 0 else float("nan")
        out.append({"index": int(j), "equation": mech.reactions[j].equation, "sensitivity": s})
    out.sort(key=lambda d: -abs(d["sensitivity"]) if np.isfinite(d["sensitivity"]) else 0)
    return {"target": target, "value": y0, "perturbation": rel_perturbation, "results": out,
            "skipped_due_to_budget": skipped}


# ---------------------------------------------------------------------------- reaction order
@dataclass
class OrderFit:
    order: float
    k: float
    r_squared: float
    aic: float
    k_unit: str
    method: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _integrated(c0: float, k: float, n: float, t: np.ndarray) -> np.ndarray:
    if abs(n - 1.0) < 1e-9:
        return c0 * np.exp(-k * t)
    base = c0 ** (1 - n) - (1 - n) * k * t
    out = np.where(base > 0, np.abs(base) ** (1 / (1 - n)), 0.0)
    return out


def fit_reaction_order(t, c, concentration_unit: str = "mol/L") -> dict:
    """Determine the reaction order of a decaying species from c(t).

    Linearised integrated rate laws for n = 0, 1, 2 plus a nonlinear least-squares fit
    of (k, n, c0); the model is selected by AIC."""
    from scipy.optimize import least_squares

    t = np.asarray(t, float)
    c = np.asarray(c, float)
    mask = np.isfinite(t) & np.isfinite(c) & (c > 0)
    t, c = t[mask], c[mask]
    if len(t) < 4:
        raise AutoKineticsError("Mindestens 4 positive Messpunkte benötigt")
    units = {0: f"{concentration_unit}/s", 1: "1/s", 2: f"1/({concentration_unit}·s)"}
    fits: list[OrderFit] = []
    ss_tot = float(np.sum((c - c.mean()) ** 2)) or 1e-300

    def aic(rss: float, p: int) -> float:
        return len(c) * math.log(max(rss, 1e-300) / len(c)) + 2 * p

    for n, ylin in ((0, c), (1, np.log(c)), (2, 1 / c)):
        slope, intercept = np.polyfit(t, ylin, 1)
        k = -slope if n < 2 else slope
        c0 = intercept if n == 0 else (math.exp(intercept) if n == 1 else 1 / intercept if intercept > 0 else c[0])
        model = _integrated(c0, k, n, t)
        rss = float(np.sum((c - model) ** 2))
        fits.append(OrderFit(n, float(k), 1 - rss / ss_tot, aic(rss, 2), units[n], "linearisiert"))

    def resid(p):
        lk, n, c0 = p
        return _integrated(c0, math.exp(lk), n, t) - c

    best_lin = min(fits, key=lambda f: f.aic)
    x0 = [math.log(max(best_lin.k, 1e-30)), float(best_lin.order) + 0.01, float(c[0])]
    try:
        sol = least_squares(resid, x0, bounds=([-80, -1.0, 0.0], [80, 4.0, np.inf]), x_scale="jac")
        rss = float(np.sum(sol.fun ** 2))
        n = float(sol.x[1])
        fits.append(OrderFit(n, float(math.exp(sol.x[0])), 1 - rss / ss_tot, aic(rss, 3),
                             f"({concentration_unit})^{1 - n:.2f}/s", "nichtlinear (n frei)"))
    except (ValueError, FloatingPointError):
        pass
    best = min(fits, key=lambda f: f.aic)
    return {"best": best.to_dict(), "fits": [f.to_dict() for f in fits]}


# ---------------------------------------------------------------------------- Arrhenius
def fit_arrhenius(T, k, modified: bool = False) -> dict:
    """Weighted linear least squares (Householder QR) of ln k = ln A + b ln T − Ea/(R T)."""
    T = np.asarray(T, float)
    k = np.asarray(k, float)
    if np.any(T <= 0) or np.any(k <= 0):
        raise AutoKineticsError("Temperaturen und Geschwindigkeitskonstanten müssen > 0 sein")
    cols = [np.ones_like(T), -1.0 / (R * T)] + ([np.log(T)] if modified else [])
    X = np.column_stack(cols)
    if len(T) <= X.shape[1]:
        raise AutoKineticsError(f"Mindestens {X.shape[1] + 1} Messpunkte benötigt")
    y = np.log(k)
    Q, Rm = np.linalg.qr(X)
    beta = np.linalg.solve(Rm, Q.T @ y)
    res = y - X @ beta
    dof = len(T) - X.shape[1]
    s2 = float(res @ res) / dof
    Rinv = np.linalg.inv(Rm)
    cov = s2 * Rinv @ Rinv.T
    se = np.sqrt(np.diag(cov))
    from scipy.stats import t as student_t

    tq = float(student_t.ppf(0.975, dof))
    out = {"A": float(math.exp(beta[0])), "A_rel_error": float(se[0]), "Ea": float(beta[1]),
           "Ea_error": float(se[1]), "Ea_ci95": [float(beta[1] - tq * se[1]), float(beta[1] + tq * se[1])],
           "A_ci95": [float(math.exp(beta[0] - tq * se[0])), float(math.exp(beta[0] + tq * se[0]))],
           "dof": dof, "r_squared": float(1 - res @ res / np.sum((y - y.mean()) ** 2)),
           "residuals_lnk": res.tolist(), "condition_number": float(np.linalg.cond(X))}
    if modified:
        out.update(b=float(beta[2]), b_error=float(se[2]))
    return out


# ---------------------------------------------------------------------------- parameter estimation
def estimate_parameters(mech: Mechanism, spec: ReactorSpec, options: SimulationOptions,
                        data: list[dict], parameters: list[dict], budget_s: float = 300.0) -> dict:
    """Fit Arrhenius parameters to measured concentrations.

    ``data``: [{"species": name, "t": [...], "c": [...] (mol/m³), "sigma": optional}]
    ``parameters``: [{"reaction": j, "param": "A" | "Ea"}]; A is fitted in log space.
    Returns estimates, standard errors (Gauss–Newton covariance), correlations and fit curves.
    """
    from scipy.optimize import least_squares

    if not parameters:
        raise AutoKineticsError("Keine Parameter zum Anpassen ausgewählt")
    for p in parameters:
        if not (0 <= int(p["reaction"]) < mech.n_reactions) or p["param"] not in ("A", "Ea"):
            raise AutoKineticsError(f"Ungültiger Parameter {p}")
    series = []
    for d in data:
        mech.species_index(d["species"])
        t = np.asarray(d["t"], float)
        c = np.asarray(d["c"], float)
        if t.shape != c.shape or len(t) == 0:
            raise AutoKineticsError(f"Messreihe {d['species']}: t und c müssen gleich lang sein")
        sigma = np.asarray(d.get("sigma") or np.full_like(c, max(np.abs(c).max() * 0.01, 1e-12)), float)
        series.append((d["species"], t, c, sigma))
    t_all = np.unique(np.concatenate([s[1] for s in series]))
    if t_all.max() > options.t_end:
        options = copy.copy(options)
        options.t_end = float(t_all.max())
    cm0 = compile_mechanism(mech)
    x0 = []
    for p in parameters:
        j = int(p["reaction"])
        x0.append(math.log(cm0.A[j]) if p["param"] == "A" else cm0.Ea_R[j] * R / 1e3)
    deadline = time.perf_counter() + budget_s
    n_eval = [0]

    def model(x) -> list[np.ndarray]:
        cm = copy.copy(cm0)
        cm.A, cm.Ea_R = cm0.A.copy(), cm0.Ea_R.copy()
        for p, v in zip(parameters, x, strict=True):
            j = int(p["reaction"])
            if p["param"] == "A":
                cm.A[j] = math.exp(v)
            else:
                cm.Ea_R[j] = v * 1e3 / R
        opts = copy.copy(options)
        opts.t_eval = np.concatenate([[0.0], t_all[t_all > 0]])
        built = build_reactor(mech, spec, opts.rtol, opts.atol_rel, compiled=cm)
        res = simulate(mech, spec, opts, built=built)
        n_eval[0] += 1
        return [np.interp(t, res.t, res.species(sp)) for sp, t, _, _ in series]

    def residuals(x):
        if time.perf_counter() > deadline:
            raise TimeoutError
        pred = model(x)
        return np.concatenate([(p - c) / s for p, (_, _, c, s) in zip(pred, series, strict=True)])

    try:
        sol = least_squares(residuals, x0, method="trf", x_scale="jac", diff_step=1e-4)
    except TimeoutError:
        raise SimulationError(f"Parameterschätzung: Zeitbudget von {budget_s:.0f} s überschritten") from None
    J = sol.jac
    dof = max(len(sol.fun) - len(x0), 1)
    s2 = float(sol.fun @ sol.fun) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        corr = cov / np.outer(se, se)
    except np.linalg.LinAlgError:
        se = np.full(len(x0), np.nan)
        corr = np.full((len(x0), len(x0)), np.nan)
    estimates = []
    for p, v, e in zip(parameters, sol.x, se, strict=True):
        if p["param"] == "A":
            estimates.append({**p, "value": math.exp(v), "rel_std_error": float(e), "unit": "SI"})
        else:
            estimates.append({**p, "value": float(v), "std_error": float(e), "unit": "kJ/mol"})
    pred = model(sol.x)
    return {
        "success": bool(sol.success), "message": str(sol.message), "estimates": estimates,
        "correlation": np.nan_to_num(corr).tolist(), "chi2_reduced": s2, "n_simulations": n_eval[0],
        "fit": [{"species": sp, "t": t.tolist(), "measured": c.tolist(), "model": m.tolist()}
                for (sp, t, c, _), m in zip(series, pred, strict=True)],
    }
