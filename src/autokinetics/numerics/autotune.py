"""Automatic solver selection based on stability, consistency and convergence
analysis plus a cost benchmark (see docs/NUMERICS.md for the full description
and the JSON schema of :class:`AutoTuneReport`).

Pipeline
--------
1. Reference solution (Radau IIA s=3, tolerance 1e-3 x requested) on the full
   interval, or on a shortened pilot interval if the reference does not finish
   within 30 % of the budget.
2. Stability: Jacobian spectra along the reference trajectory (own QR
   eigenvalue solver), stiffness ratio/index, maximal stable step of every
   explicit candidate (explicit methods whose stability limit would require
   more than MAX_EXPLICIT_STEPS steps are not benchmarked), stability
   properties and region grids of all candidates.
3. Consistency: algebraic order (rooted-tree conditions, LMM conditions) and
   empirical local-error order on a short initial interval.
4. Convergence: global error under step refinement on that interval.
5. Benchmark of every remaining candidate at the requested tolerances with
   simplified Newton + LU with pivoting; error measured against the reference.
6. Solver variants (full Newton, fixed point, other linear solvers) for the
   best implicit candidates, pre-screened by the Banach contraction bound and
   the spectral radii of the Jacobi/Gauß-Seidel iteration matrices.
7. Tolerance proportionality of the best configurations.
8. Selection: cheapest (wall time) valid configuration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autokinetics.numerics._json import jsonable
from autokinetics.numerics.consistency import (
    algebraic_order,
    fit_order,
    lmm_order,
    reference_solution,
    root_condition,
)
from autokinetics.numerics.convergence import newton_statistics, tolerance_proportionality
from autokinetics.numerics.integrator import IntegrationResult, as_system, integrate
from autokinetics.numerics.methods import LINEAR_SOLVERS, METHODS, NONLINEAR_SOLVERS
from autokinetics.numerics.stability import (
    BDF2_ALPHA,
    BDF2_BETA,
    fixed_point_contraction,
    jacobian_spectra,
    linear_iteration_radius,
    max_stable_step,
    stability_properties,
    stability_region,
)
from autokinetics.numerics.tableaux import ButcherTableau

__all__ = ["AutoTuneReport", "auto_optimize", "DEFAULT_CANDIDATES", "ERROR_TARGET_FACTOR"]

DEFAULT_CANDIDATES = (
    "dopri5",
    "rk4",
    "heun",
    "radau5",
    "radau3",
    "sdirk2",
    "ros2",
    "bdf2",
    "implicit_euler",
    "gauss4",
)
ERROR_TARGET_FACTOR = 10.0  # accepted: max_i |y_i - y_ref,i| / (atol_i + rtol |y_ref,i|) <= 10
MAX_EXPLICIT_STEPS = 200_000  # explicit candidates needing more steps for stability are skipped
STIFF_RATIO = 1e3
STIFF_INDEX = 1e3
ORDER_TOLERANCE = 0.5  # |p_emp - p| <= 0.5: order confirmed
MIN_CONSISTENT_ORDER = 0.7  # p_emp < 0.7: not consistent / not convergent


@dataclass
class AutoTuneReport:
    """Result of :func:`auto_optimize`. All fields are JSON-serialisable via :meth:`to_dict`."""

    problem: dict
    budget: dict
    reference: dict
    stability: dict
    consistency: dict
    convergence: dict
    benchmark: list[dict]
    recommendation: dict
    reasoning_de: list[str]
    skipped: list[dict] = field(default_factory=list)

    @property
    def recommended_options(self) -> dict:
        """Options for ``integrate(system, t_span, y0, **options)``."""
        return dict(self.recommendation["options"])

    def to_dict(self) -> dict:
        return jsonable(
            {
                "problem": self.problem,
                "budget": self.budget,
                "reference": self.reference,
                "stability": self.stability,
                "consistency": self.consistency,
                "convergence": self.convergence,
                "benchmark": self.benchmark,
                "recommendation": self.recommendation,
                "reasoning_de": self.reasoning_de,
                "skipped": self.skipped,
            }
        )


class _Budget:
    def __init__(self, total: float):
        self.total = float(total)
        self.start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    @property
    def remaining(self) -> float:
        return max(self.total - self.elapsed, 0.0)

    def has(self, seconds: float) -> bool:
        return self.remaining > seconds


def _fmt(x: float) -> str:
    return f"{x:.3g}"


class _Tuner:
    def __init__(
        self,
        system,
        t_span,
        y0,
        rtol,
        atol,
        budget_s,
        invariants,
        candidates,
        linear_solvers,
        nonlinear_solvers,
    ):
        self.y0 = np.atleast_1d(np.asarray(y0, dtype=float))
        self.n = self.y0.size
        self.handle = as_system(system, self.n)
        self.t0, self.tf = (float(v) for v in t_span)
        if not self.tf > self.t0:
            raise ValueError("t_span must satisfy t0 < tf")
        self.rtol = float(rtol)
        self.atol = atol
        self.atol_vec = np.broadcast_to(np.asarray(atol, dtype=float), (self.n,)).astype(float)
        self.budget = _Budget(budget_s)
        self.invariants = invariants
        cands = list(candidates) if candidates is not None else list(DEFAULT_CANDIDATES)
        for c in cands:
            if c not in METHODS:
                raise ValueError(f"unknown candidate method {c!r}")
        self.candidates = cands
        self.linear_solvers = (
            list(linear_solvers) if linear_solvers is not None else list(LINEAR_SOLVERS)
        )
        self.nonlinear_solvers = (
            list(nonlinear_solvers) if nonlinear_solvers is not None else list(NONLINEAR_SOLVERS)
        )
        for s in self.linear_solvers:
            if s not in LINEAR_SOLVERS:
                raise ValueError(f"unknown linear solver {s!r}")
        for s in self.nonlinear_solvers:
            if s not in NONLINEAR_SOLVERS:
                raise ValueError(f"unknown nonlinear solver {s!r}")
        self.skipped: list[dict] = []
        self.reasoning: list[str] = []
        self.mask = self.handle.nonnegative_mask

    def skip(self, stage: str, item: str, reason_de: str) -> None:
        self.skipped.append({"stage": stage, "item": item, "reason_de": reason_de})

    # ------------------------------------------------------------------ reference
    def reference(self) -> dict:
        self.rtol_ref = float(min(max(self.rtol * 1e-4, 1e-12), 1e-8))
        self.atol_ref = self.atol_vec * 1e-3
        cap = max(0.3 * self.budget.total, 0.05)
        ref = integrate(
            self.handle,
            (self.t0, self.tf),
            self.y0,
            "radau5",
            rtol=self.rtol_ref,
            atol=self.atol_ref,
            max_wall_time=cap,
            max_steps=2_000_000,
            newton_tol=0.01,
        )
        info = {
            "method": "radau5",
            "rtol": self.rtol_ref,
            "atol": self.atol_ref,
            "success": ref.success,
            "status": ref.status,
            "message": ref.message,
            "n_steps": ref.stats["n_accepted"],
            "wall_time": ref.wall_time,
            "t_end": float(ref.t[-1]),
        }
        self.ref = ref
        if ref.success:
            self.pilot_tf = self.tf
            info["reasoning_de"] = (
                f"Referenzlösung mit Radau IIA (s=3) bei rtol={_fmt(self.rtol_ref)} auf dem "
                f"gesamten Intervall berechnet ({ref.stats['n_accepted']} Schritte)."
            )
        elif ref.status in ("wall_time", "max_steps") and ref.t.size > 4:
            self.pilot_tf = float(ref.t[-1])
            info["reasoning_de"] = (
                f"Referenzlösung erreichte im Zeitbudget nur t = {_fmt(self.pilot_tf)}; "
                "die Analyse verwendet das verkürzte Pilotintervall "
                f"[{_fmt(self.t0)}, {_fmt(self.pilot_tf)}]."
            )
            self.skip("reference", "full_interval", info["reasoning_de"])
        else:
            self.pilot_tf = None
            info["reasoning_de"] = (
                "Referenzlösung konnte nicht berechnet werden: " + ref.message + f" ({ref.detail})."
            )
        info["pilot_t_end"] = self.pilot_tf
        return info

    def _setup_checks(self) -> None:
        T, Y = self.ref.t, self.ref.y
        last = int(np.searchsorted(T, self.pilot_tf, side="right")) - 1
        idx = np.unique(np.linspace(1, last, 12).round().astype(int))
        self.t_check = T[idx]
        self.y_check = Y[idx]
        # error weights per component: 1 / (atol_i + rtol * max_t |y_ref,i(t)|); an error of 1 equals
        # the requested tolerance (component scale = largest magnitude reached on the pilot interval)
        self.y_scale = np.max(np.abs(Y[: last + 1]), axis=0)
        self.w = 1.0 / (self.atol_vec + self.rtol * self.y_scale)
        self.y_final_ref = Y[last]

    def scaled_error(self, run: IntegrationResult) -> float:
        if not run.success or run.y.shape[0] != self.t_check.size:
            return float("inf")
        with np.errstate(invalid="ignore", over="ignore"):
            e = float(np.max(np.abs(run.y - self.y_check) * self.w[None, :]))
        return e if np.isfinite(e) else float("inf")

    # ------------------------------------------------------------------ stability
    def stability(self) -> dict:
        T, Y = self.ref.t, self.ref.y
        last = int(np.searchsorted(T, self.pilot_tf, side="right")) - 1
        idx = np.unique(np.linspace(0, last, 10).round().astype(int))
        self.sample_t = T[idx]
        self.sample_y = Y[idx]
        self.jacobians = [
            self.handle.jacobian(float(t), y)
            for t, y in zip(self.sample_t, self.sample_y, strict=True)
        ]
        spectra = jacobian_spectra(self.handle, self.sample_t, self.sample_y)
        max_decay = max(s["max_decay_rate"] for s in spectra)
        ratio = max(s["stiffness_ratio"] for s in spectra)
        max_growth = max(s["max_growth_rate"] for s in spectra)
        length = self.pilot_tf - self.t0
        index = max_decay * length
        self.stiff = bool(ratio >= STIFF_RATIO and index >= STIFF_INDEX)
        eig_samples = [s["re"] + 1j * s["im"] for s in spectra]
        # time weights for the explicit step estimate: midpoint cells around the samples
        edges = np.concatenate(
            [[self.t0], 0.5 * (self.sample_t[1:] + self.sample_t[:-1]), [self.pilot_tf]]
        )
        dts = np.diff(edges)
        limits = []
        self.explicit_blocked: dict[str, str] = {}
        props, regions = {}, {}
        for m in self.candidates:
            model = "dopri5" if m == "scipy:RK45" else m
            info = METHODS[m]
            if info.backend == "cpp" or m == "scipy:RK45":
                props[m] = stability_properties(model)
                regions[m] = {
                    **stability_region(model, n=51, digits=4),
                    **{k: props[m][k] for k in ("a_stable", "l_stable", "r_infinity")},
                }
            if info.implicit:
                continue
            hs = np.array([max_stable_step(model, ev) for ev in eig_samples])
            with np.errstate(divide="ignore"):
                est = float(np.sum(dts / hs))
            h_min = float(hs.min())
            limits.append(
                {
                    "method": m,
                    "h_max_stable": h_min,
                    "h_max_stable_per_sample": hs,
                    "estimated_steps": est,
                }
            )
            if not np.isfinite(est) or est > MAX_EXPLICIT_STEPS:
                if h_min == 0.0:
                    why = (
                        "Eigenwerte auf/nahe der imaginären Achse liegen für keine Schrittweite h > 0 im "
                        "Stabilitätsgebiet"
                    )
                else:
                    why = (
                        f"maximal stabile Schrittweite h ≤ {_fmt(h_min)} erfordert ≈ {_fmt(est)} Schritte "
                        f"(> {MAX_EXPLICIT_STEPS})"
                    )
                self.explicit_blocked[m] = why
                self.skip(
                    "benchmark",
                    m,
                    f"{info.label_de}: {why} – Verfahren für dieses Problem ungeeignet.",
                )
        if self.stiff:
            text = (
                f"Problem ist steif: Steifheitsquotient max|Re λ|/min|Re λ| = {_fmt(ratio)}, schnellste "
                f"Abklingrate {_fmt(max_decay)} bei Intervalllänge {_fmt(length)} "
                f"(Steifheitsindex {_fmt(index)}). Explizite Verfahren sind durch ihr beschränktes "
                "Stabilitätsgebiet auf sehr kleine Schritte angewiesen; A-/L-stabile Verfahren bevorzugt."
            )
        else:
            text = (
                f"Problem ist nicht steif (Steifheitsquotient {_fmt(ratio)}, Steifheitsindex {_fmt(index)}): "
                "die Schrittweite wird durch die Genauigkeit, nicht durch die Stabilität begrenzt – "
                "explizite Verfahren sind in der Regel am günstigsten."
            )
        if max_growth > 0:
            text += (
                f" Hinweis: Eigenwerte mit positivem Realteil (bis {_fmt(max_growth)}) – physikalisch "
                "wachsende Moden, keine Schrittweitenrestriktion."
            )
        self.reasoning.append(text)
        return {
            "stiff": self.stiff,
            "stiffness_ratio_max": ratio,
            "max_decay_rate": max_decay,
            "max_growth_rate": max_growth,
            "stiffness_index": index,
            "spectra": spectra,
            "explicit_limits": limits,
            "method_properties": props,
            "stability_regions": regions,
            "reasoning_de": text,
        }

    # ------------------------------------------------------------------ consistency + convergence
    def _short_interval(self) -> float:
        T = self.ref.t
        k = min(T.size - 1, 10)
        return float(T[k])

    def consistency_and_convergence(self) -> tuple[dict, dict]:
        consistency, convergence = {}, {}
        t_c = self._short_interval()
        delta = t_c - self.t0
        self.consistency_verdict: dict[str, bool | None] = {}
        hs_local = delta * 0.5 ** np.arange(1, 6)
        n_global = np.array([4, 8, 16, 32, 64])
        try:
            ref_local = reference_solution(
                self.handle,
                (self.t0, self.t0 + hs_local.max()),
                self.y0,
                t_eval=self.t0 + np.sort(hs_local),
                max_wall_time=max(0.05 * self.budget.remaining, 0.05),
            )
            ref_short = reference_solution(
                self.handle,
                (self.t0, t_c),
                self.y0,
                max_wall_time=max(0.05 * self.budget.remaining, 0.05),
            )
        except RuntimeError as exc:
            self.skip("consistency", "reference", f"Kurzintervall-Referenz fehlgeschlagen: {exc}")
            return consistency, convergence
        y_loc = ref_local.y[np.argsort(np.argsort(hs_local))]
        y_short = ref_short.y[-1]
        w = self.w
        scale_loc = float(np.max(np.maximum(np.abs(self.y0), np.abs(y_loc).max(axis=0)) * w))
        scale_short = float(np.max(np.maximum(np.abs(self.y0), np.abs(y_short)) * w))
        cpp_cands = [m for m in self.candidates if METHODS[m].backend == "cpp"]
        per_method = max(self.budget.remaining * 0.25 / max(len(cpp_cands), 1), 0.02)
        for m in cpp_cands:
            if not self.budget.has(0.1 * self.budget.total):
                self.skip(
                    "consistency", m, "Zeitbudget: Konsistenz-/Konvergenzanalyse übersprungen."
                )
                continue
            info = METHODS[m]
            entry: dict[str, Any] = {"theoretical_order": info.order}
            if info.tableau:
                tab = ButcherTableau.builtin(info.tableau)
                alg = algebraic_order(tab.A, tab.b, tab.c, max_order=6)
                entry["algebraic_order"] = alg["order"]
                entry["algebraic_capped"] = alg["capped"]
                entry["order_conditions_checked"] = alg["n_conditions"]
                entry["first_violated_conditions"] = [c["tree"] for c in alg["first_violated"]]
                entry["row_sum_condition"] = alg["row_sum_condition"]
                if info.embedded_order and tab.bhat is not None:
                    entry["embedded_algebraic_order"] = algebraic_order(
                        tab.A, tab.bhat, max_order=6
                    )["order"]
                alg_text = (
                    f"Butcher-Bedingungen (Satz 3.4) bis Ordnung {alg['order']} erfüllt"
                    + (
                        " (Prüfung bis Ordnung 6)"
                        if alg["capped"]
                        else f", verletzt ab Ordnung {alg['order'] + 1}"
                    )
                    + "."
                )
            elif m == "bdf2":
                lo = lmm_order(BDF2_ALPHA, BDF2_BETA)
                rc = root_condition(BDF2_ALPHA)
                entry["algebraic_order"] = lo["order"]
                entry["error_constant"] = lo["error_constant"]
                entry["root_condition"] = rc["satisfied"]
                alg_text = (
                    f"Mehrschritt-Ordnungsbedingungen (Satz 7.12) liefern Ordnung {lo['order']}, "
                    f"Fehlerkonstante {_fmt(lo['error_constant'])}; Wurzelbedingung "
                    + ("erfüllt (nullstabil)." if rc["satisfied"] else "verletzt!")
                )
            else:
                entry["algebraic_order"] = info.order
                alg_text = (
                    f"Theoretische Ordnung {info.order} (Rosenbrock-Ordnungsbedingungen aus der "
                    "Literatur; empirisch geprüft)."
                )
            deadline = time.perf_counter() + per_method
            local_err, global_err = [], []
            for h, yr in zip(hs_local, y_loc, strict=True):
                if time.perf_counter() > deadline:
                    local_err.append(np.nan)
                    continue
                r = integrate(
                    self.handle,
                    (self.t0, self.t0 + h),
                    self.y0,
                    m,
                    fixed_step=h,
                    rtol=1e-13,
                    atol=1e-15,
                    newton_tol=1e-4,
                    max_newton_iter=25,
                    max_wall_time=per_method,
                )
                local_err.append(float(np.max(np.abs(r.y[-1] - yr) * w)) if r.success else np.inf)
            for N in n_global:
                if time.perf_counter() > deadline:
                    global_err.append(np.nan)
                    continue
                h = delta / N
                r = integrate(
                    self.handle,
                    (self.t0, t_c),
                    self.y0,
                    m,
                    fixed_step=h,
                    rtol=1e-13,
                    atol=1e-15,
                    newton_tol=1e-4,
                    max_newton_iter=25,
                    max_wall_time=per_method,
                )
                global_err.append(
                    float(np.max(np.abs(r.y[-1] - y_short) * w)) if r.success else np.inf
                )
            # errors are in units of the requested tolerance; the reference (rtol 1e-12) is
            # accurate to ~1e-12 relative, i.e. ~1e-12 * scale in these units -> floor 1e-10 * scale
            local = fit_order(
                hs_local, local_err, floor=1e-10 * scale_loc, ceiling=0.5 * scale_loc, kind="local"
            )
            glob = fit_order(
                delta / n_global,
                global_err,
                floor=1e-10 * scale_short,
                ceiling=0.5 * scale_short,
                kind="global",
            )
            p = info.order
            local_order = None if local.slope is None else local.slope - 1.0
            entry["local"] = local
            entry["empirical_order"] = local_order
            if local_order is None:
                verdict = None
                emp_text = (
                    "Empirische Konsistenzordnung nicht messbar ("
                    + "; ".join(local.reason_excluded)
                    + ")."
                )
            elif abs(local_order - p) <= ORDER_TOLERANCE:
                verdict = True
                emp_text = (
                    f"Lokaler Fehler ∝ h^{local.slope:.2f} ⇒ empirische Konsistenzordnung "
                    f"{local_order:.2f} ≈ {p} (Def. 2.8): konsistent."
                )
            elif local_order >= MIN_CONSISTENT_ORDER:
                verdict = True
                emp_text = (
                    f"Empirische Konsistenzordnung {local_order:.2f} weicht von {p} ab "
                    "(Ordnungsreduktion bei steifen Problemen bzw. vorasymptotischer Bereich), "
                    "Verfahren aber konsistent."
                )
            elif (entry.get("algebraic_order") or 0) >= 1:
                # Algebraically consistent (order conditions hold), so a small/negative empirical
                # order only means the step sizes are not in the asymptotic range: for stiff
                # problems with h·|λ| ≫ 1 the local error is not O(h^(p+1)) (order reduction).
                verdict = None
                emp_text = (
                    f"Empirische Ordnung {local_order:.2f} nicht aussagekräftig: Schrittweiten außerhalb des "
                    "asymptotischen Bereichs (h·|λ| ≫ 1, Ordnungsreduktion bei steifen Problemen). "
                    "Konsistenz folgt aus den erfüllten Ordnungsbedingungen."
                )
            else:
                verdict = False
                emp_text = f"Empirische Konsistenzordnung {local_order:.2f} < {MIN_CONSISTENT_ORDER}: nicht konsistent."
            entry["consistent"] = verdict
            entry["reasoning_de"] = alg_text + " " + emp_text
            consistency[m] = entry
            # convergence (global error)
            conv: dict[str, Any] = {
                "global": glob,
                "theoretical_order": p,
                "empirical_order": glob.slope,
            }
            if glob.slope is None:
                conv["convergent"] = None
                conv["reasoning_de"] = (
                    "Globale Konvergenzordnung nicht messbar ("
                    + "; ".join(glob.reason_excluded)
                    + ")."
                )
            else:
                conv["convergent"] = bool(glob.slope >= MIN_CONSISTENT_ORDER)
                conv["order_confirmed"] = bool(abs(glob.slope - p) <= ORDER_TOLERANCE)
                conv["reasoning_de"] = (
                    f"Globaler Fehler bei Schrittweitenhalbierung ∝ h^{glob.slope:.2f} "
                    f"(theoretisch {p}, Satz 2.13: Konsistenz + Stabilität ⇒ Konvergenz)."
                )
                if any(not np.isfinite(e) for e in global_err):
                    conv["reasoning_de"] += (
                        " Große Schrittweiten instabil (Fehler explodiert) – wie von der Stabilitätsanalyse vorhergesagt."
                    )
            convergence[m] = conv
            self.consistency_verdict[m] = verdict if conv.get("convergent") is not False else False
        return consistency, convergence

    # ------------------------------------------------------------------ benchmark
    def _run(self, method: str, nonlinear: str, linear: str, cap: float,
             rtol: float | None = None, atol=None) -> dict:
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        opts = {
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "linear_solver": linear,
            "nonlinear_solver": nonlinear,
        }
        run = integrate(
            self.handle,
            (self.t0, self.pilot_tf),
            self.y0,
            method,
            rtol=rtol,
            atol=atol,
            linear_solver=linear,
            nonlinear_solver=nonlinear,
            t_eval=self.t_check,
            t_eval_exact=True,
            max_wall_time=cap,
            max_steps=5_000_000,
        )
        wall = run.wall_time
        if run.success and wall < 0.02:
            for _ in range(2):  # repeat short runs for a robust timing
                rep = integrate(
                    self.handle,
                    (self.t0, self.pilot_tf),
                    self.y0,
                    method,
                    rtol=rtol,
                    atol=atol,
                    linear_solver=linear,
                    nonlinear_solver=nonlinear,
                    t_eval=self.t_check,
                    t_eval_exact=True,
                    max_wall_time=cap,
                    max_steps=5_000_000,
                )
                wall = min(wall, rep.wall_time)
        err = self.scaled_error(run)
        row: dict[str, Any] = {
            "method": method,
            "label_de": METHODS[method].label_de,
            "nonlinear_solver": nonlinear,
            "linear_solver": linear,
            "success": run.success,
            "status": run.status,
            "wall_time": wall,
            "error": err,
            "meets_target": bool(err <= ERROR_TARGET_FACTOR),
            "stats": run.stats,
            "options": opts,
        }
        if METHODS[method].backend == "cpp":
            row["newton"] = newton_statistics(run)
        row["invariants_ok"], row["invariant_drift"] = self._check_invariants(run)
        row["nonnegative_ok"], row["min_value"] = self._check_nonneg(run)
        consistent = self.consistency_verdict.get(method)
        row["consistent"] = consistent
        row["valid"] = bool(
            run.success
            and row["meets_target"]
            and row["invariants_ok"]
            and row["nonnegative_ok"]
            and consistent is not False
        )
        row["reasoning_de"] = self._row_text(row)
        return row

    def _check_invariants(self, run: IntegrationResult) -> tuple[bool, float | None]:
        if self.invariants is None or not run.success:
            return True, None
        i0 = np.atleast_1d(np.asarray(self.invariants(self.y0), dtype=float))
        vals = np.array([np.atleast_1d(np.asarray(self.invariants(y), dtype=float)) for y in run.y])
        drift = float(np.max(np.abs(vals - i0)))
        tol = ERROR_TARGET_FACTOR * (
            float(np.max(self.atol_vec)) + self.rtol * float(np.max(np.abs(i0)))
        )
        return bool(drift <= tol), drift

    def _check_nonneg(self, run: IntegrationResult) -> tuple[bool, float | None]:
        if self.mask is None or not run.success:
            return True, None
        Y = run.y[:, self.mask]
        if Y.size == 0:
            return True, None
        ymin = float(Y.min())
        limit = -ERROR_TARGET_FACTOR * (
            float(np.max(self.atol_vec[self.mask]))
            + self.rtol * float(np.max(np.abs(self.y_check[:, self.mask])))
        )
        return bool(ymin >= limit), ymin

    def _row_text(self, row: dict) -> str:
        if not row["success"]:
            return f"Gescheitert: {STATUS_TEXT.get(row['status'], row['status'])}."
        parts = [
            f"Laufzeit {row['wall_time'] * 1e3:.2f} ms, {row['stats'].get('n_rhs')} f-Auswertungen, "
            f"skalierter Fehler {_fmt(row['error'])} (Ziel ≤ {ERROR_TARGET_FACTOR:g})"
        ]
        if not row["meets_target"]:
            parts.append("Genauigkeitsziel verfehlt")
        if not row["invariants_ok"]:
            parts.append(f"Invariante driftet um {_fmt(row['invariant_drift'])}")
        if not row["nonnegative_ok"]:
            parts.append(f"negative Konzentrationen bis {_fmt(row['min_value'])}")
        if row["consistent"] is False:
            parts.append("Konsistenz/Konvergenz nicht bestätigt")
        return "; ".join(parts) + "."

    def benchmark(self) -> list[dict]:
        rows = []
        runnable = [m for m in self.candidates if m not in self.explicit_blocked]
        runnable.sort(key=lambda m: (METHODS[m].implicit != self.stiff, self.candidates.index(m)))
        for i, m in enumerate(runnable):
            left = len(runnable) - i
            if not self.budget.has(0.05):
                self.skip("benchmark", m, "Zeitbudget ausgeschöpft – Kandidat nicht getestet.")
                continue
            # each run may use up to twice its fair share of the remaining budget (cheap runs leave more)
            cap = max(min(0.5 * self.budget.remaining, 2.0 * self.budget.remaining / left), 0.05)
            info = METHODS[m]
            nl = (
                "simplified_newton"
                if "simplified_newton" in self.nonlinear_solvers
                else self.nonlinear_solvers[0]
            )
            ls = "lu_pivot" if "lu_pivot" in self.linear_solvers else self.linear_solvers[0]
            if not info.uses_nonlinear_solver:
                nl = "simplified_newton"
            if not info.uses_linear_solver:
                ls = "lu_pivot"
            rows.append(self._run(m, nl, ls, cap))
        return rows

    def tolerance_adjustment(self, rows: list[dict]) -> list[dict]:
        """Global errors may exceed the local tolerance by orders of magnitude (error
        amplification, e.g. in ignition or oscillations). For the fastest configurations
        that are correct in every other respect but miss the accuracy target, the
        tolerance is tightened using tolerance proportionality (error ∝ tol) and the run
        repeated – the optimisation is over (method, tolerance) pairs."""
        out: list[dict] = []
        fixable = sorted(
            (r for r in rows if r["success"] and not r["meets_target"] and r["invariants_ok"]
             and r["nonnegative_ok"] and r["consistent"] is not False and r["error"] is not None
             and np.isfinite(r["error"])),
            key=lambda r: r["wall_time"],
        )[:4]
        floor = 3.0 * self.rtol_ref
        for r in fixable:
            row = r
            for _ in range(3):
                if not self.budget.has(max(4 * row["wall_time"], 0.05)):
                    self.skip("toleranzanpassung", r["method"], "Zeitbudget ausgeschöpft.")
                    break
                factor = min(1.0, 0.3 * ERROR_TARGET_FACTOR / row["error"])
                rtol_new = max(row["options"]["rtol"] * factor, floor)
                if rtol_new >= row["options"]["rtol"] * 0.999:
                    self.skip("toleranzanpassung", r["method"],
                              "Toleranz bereits an der Genauigkeitsgrenze der Referenzlösung.")
                    break
                atol_new = np.asarray(row["options"]["atol"], dtype=float) * (rtol_new / row["options"]["rtol"])
                cap = max(min(0.4 * self.budget.remaining, 50 * max(row["wall_time"], 1e-3)), 0.05)
                row = self._run(r["method"], r["nonlinear_solver"], r["linear_solver"], cap,
                                rtol=rtol_new, atol=atol_new)
                row["tolerance_adjusted"] = True
                row["reasoning_de"] = (f"Toleranz auf rtol = {_fmt(rtol_new)} verschärft (globaler Fehler bei "
                                       f"rtol = {_fmt(self.rtol)} zu groß). " + row["reasoning_de"])
                out.append(row)
                if row["valid"] or not row["success"]:
                    break
        if out:
            self.reasoning.append(
                f"Toleranzanpassung: {len(out)} Läufe mit verschärfter Toleranz, davon "
                f"{sum(r['valid'] for r in out)} gültig – das Verfahren wird zusammen mit der nötigen "
                "Toleranz gewählt (globale Fehlerverstärkung).")
        return out

    def solver_variants(self, rows: list[dict]) -> list[dict]:
        out = []
        good = sorted(
            (
                r
                for r in rows
                if r["valid"]
                and METHODS[r["method"]].backend == "cpp"
                and METHODS[r["method"]].uses_linear_solver
            ),
            key=lambda r: r["wall_time"],
        )
        top = []
        for r in good:
            if r["method"] not in [t["method"] for t in top]:
                top.append(r)
            if len(top) == 2:
                break
        for base in top:
            m = base["method"]
            info = METHODS[m]
            h_typ = base["stats"].get("h_mean") or (self.pilot_tf - self.t0) / 100
            variants: list[tuple[str, str]] = []
            if info.uses_nonlinear_solver:
                for nl in self.nonlinear_solvers:
                    if nl == base["nonlinear_solver"]:
                        continue
                    if nl == "fixed_point":
                        q = max(
                            fixed_point_contraction(m, J, h_typ)["q_spectral"]
                            for J in self.jacobians
                        )
                        if q >= 1.0:
                            self.skip(
                                "solver_variants",
                                f"{m}/fixed_point",
                                f"Fixpunktiteration für {info.label_de} verworfen: Kontraktionsfaktor "
                                f"q = h·ρ(A)·ρ(J) = {_fmt(q)} ≥ 1 bei typischer Schrittweite h = {_fmt(h_typ)} "
                                "(Banachscher Fixpunktsatz nicht anwendbar).",
                            )
                            continue
                    variants.append((nl, base["linear_solver"]))
            for ls in self.linear_solvers:
                if ls == base["linear_solver"]:
                    continue
                if ls in ("jacobi", "gauss_seidel"):
                    rho = max(linear_iteration_radius(m, J, h_typ, ls) for J in self.jacobians)
                    if not rho < 1.0:
                        self.skip(
                            "solver_variants",
                            f"{m}/{ls}",
                            f"{LINEAR_SOLVERS[ls].label_de} für {info.label_de} verworfen: Spektralradius der "
                            f"Iterationsmatrix ρ = {_fmt(rho)} ≥ 1 für die Newton-Matrix bei h = {_fmt(h_typ)}.",
                        )
                        continue
                variants.append((base["nonlinear_solver"], ls))
            for nl, ls in variants:
                if not self.budget.has(0.1 * self.budget.total):
                    self.skip(
                        "solver_variants",
                        f"{m}/{nl}/{ls}",
                        "Zeitbudget: Löservariante nicht getestet.",
                    )
                    continue
                cap = max(min(self.budget.remaining * 0.25, 10 * base["wall_time"] + 0.05), 0.05)
                out.append(self._run(m, nl, ls, cap, rtol=base["options"]["rtol"], atol=base["options"]["atol"]))
        return out

    def tolerance_sweeps(self, rows: list[dict]) -> dict:
        res = {}
        valid = sorted((r for r in rows if r["valid"]), key=lambda r: r["wall_time"])
        seen = set()
        for r in valid:
            if r["method"] in seen:
                continue
            seen.add(r["method"])
            if len(seen) > 3:
                break
            if not self.budget.has(0.15 * self.budget.total):
                self.skip(
                    "convergence",
                    f"{r['method']}/tolerance",
                    "Zeitbudget: Toleranzstudie übersprungen.",
                )
                continue
            cap = max(min(self.budget.remaining * 0.1, 20 * r["wall_time"] + 0.05), 0.05)
            sweep = tolerance_proportionality(
                self.handle,
                (self.t0, self.pilot_tf),
                self.y0,
                r["method"],
                rtols=tuple(self.rtol * np.array([10.0, 1.0, 0.1])),
                atol_ratio=float(np.max(self.atol_vec)) / self.rtol,
                y_ref=self.y_final_ref,
                weights=self.w,
                max_wall_time=cap,
                linear_solver=r["linear_solver"],
                nonlinear_solver=r["nonlinear_solver"],
            )
            res[r["method"]] = sweep
        return res

    # ------------------------------------------------------------------ selection
    def select(self, rows: list[dict]) -> dict:
        valid = [r for r in rows if r["valid"]]
        if not valid:
            fallback = {
                "method": "radau5",
                "rtol": self.rtol,
                "atol": self.atol,
                "linear_solver": "lu_pivot",
                "nonlinear_solver": "simplified_newton",
            }
            text = (
                "Keine getestete Konfiguration erfüllte alle Kriterien (Erfolg, Genauigkeit, Invarianten, "
                "Nichtnegativität, Konsistenz) im Zeitbudget. Empfehlung: robustes L-stabiles Radau IIA "
                "(s=3) mit vereinfachtem Newton und LU – ggf. Budget erhöhen."
            )
            self.reasoning.append(text)
            return {
                "options": fallback,
                "label_de": METHODS["radau5"].label_de,
                "reasoning_de": text,
                "fallback": True,
            }
        best_time = min(r["wall_time"] for r in valid)
        close = [r for r in valid if r["wall_time"] <= 1.1 * best_time]
        best = min(
            close,
            key=lambda r: (r["stats"].get("n_rhs") or 0) + self.n * (r["stats"].get("n_jac") or 0),
        )
        info = METHODS[best["method"]]
        text = (
            f"Empfehlung: {info.label_de}"
            + (
                f" mit {NONLINEAR_SOLVERS[best['nonlinear_solver']].label_de}"
                if info.uses_nonlinear_solver
                else ""
            )
            + (
                f" und {LINEAR_SOLVERS[best['linear_solver']].label_de}"
                if info.uses_linear_solver
                else ""
            )
            + f" – günstigste gültige Konfiguration ({best['wall_time'] * 1e3:.2f} ms, skalierter Fehler "
            f"{_fmt(best['error'])}, {len(valid)} von {len(rows)} Konfigurationen gültig)."
        )
        if self.stiff and not info.implicit:
            text += " Trotz Steifheit ist ein explizites Verfahren hier am günstigsten (kurzes Intervall)."
        if not self.stiff and info.implicit:
            text += " Implizites Verfahren gewinnt trotz fehlender Steifheit (hohe Ordnung / wenige Schritte)."
        self.reasoning.append(text)
        options = {
            k: best["options"][k]
            for k in ("method", "rtol", "atol", "linear_solver", "nonlinear_solver")
        }
        return {
            "options": options,
            "label_de": info.label_de,
            "reasoning_de": text,
            "fallback": False,
            "wall_time": best["wall_time"],
            "error": best["error"],
        }


STATUS_TEXT = {
    "max_steps": "maximale Schrittzahl erreicht",
    "wall_time": "Laufzeitlimit überschritten",
    "step_size_too_small": "Schrittweite zu klein geworden",
    "nonlinear_solver_failure": "nichtlinearer Löser gescheitert",
    "non_finite": "Lösung nicht endlich (instabil)",
    "scipy_failure": "SciPy-Fehler",
}


def auto_optimize(
    system,
    t_span,
    y0,
    rtol: float = 1e-6,
    atol=1e-12,
    budget_s: float = 20.0,
    invariants=None,
    candidates=None,
    linear_solvers=None,
    nonlinear_solvers=None,
) -> AutoTuneReport:
    """Analyses the problem and selects the cheapest integrator configuration that
    is stable, consistent, convergent and meets the error target.

    Parameters
    ----------
    system, t_span, y0 : as for :func:`integrate`.
    rtol, atol : requested accuracy (atol scalar or per component).
    budget_s : wall-clock budget in seconds for the whole analysis; stages that
        do not fit are skipped and recorded in ``report.skipped``.
    invariants : optional callable ``invariant(y) -> array`` of conserved
        quantities (e.g. element balances); configurations violating them by
        more than 10 x tolerance are rejected.
    candidates : method ids to consider (default :data:`DEFAULT_CANDIDATES`).
    linear_solvers, nonlinear_solvers : solver ids to consider for implicit methods.
    """
    tuner = _Tuner(
        system,
        t_span,
        y0,
        rtol,
        atol,
        budget_s,
        invariants,
        candidates,
        linear_solvers,
        nonlinear_solvers,
    )
    problem = {
        "n": tuner.n,
        "t_span": [tuner.t0, tuner.tf],
        "y0": tuner.y0,
        "rtol": tuner.rtol,
        "atol": tuner.atol_vec,
        "candidates": tuner.candidates,
        "error_target_factor": ERROR_TARGET_FACTOR,
    }
    reference = tuner.reference()
    stability: dict = {}
    consistency: dict = {}
    convergence: dict = {}
    rows: list[dict] = []
    if tuner.pilot_tf is None:
        tuner.reasoning.append(reference["reasoning_de"])
        recommendation = tuner.select([])
    else:
        problem["pilot_t_span"] = [tuner.t0, tuner.pilot_tf]
        tuner._setup_checks()
        stability = tuner.stability()
        consistency, convergence = tuner.consistency_and_convergence()
        rows = tuner.benchmark()
        rows += tuner.tolerance_adjustment(rows)
        rows += tuner.solver_variants(rows)
        sweeps = tuner.tolerance_sweeps(rows)
        for m, sweep in sweeps.items():
            convergence.setdefault(m, {})["tolerance"] = sweep
        for r in rows:
            if (
                "newton" in r
                and r["method"] in convergence
                and r["nonlinear_solver"] == "simplified_newton"
                and r["linear_solver"] == "lu_pivot"
            ):
                convergence[r["method"]]["newton"] = r["newton"]
        recommendation = tuner.select(rows)
    budget = {
        "budget_s": tuner.budget.total,
        "elapsed_s": tuner.budget.elapsed,
        "exhausted": tuner.budget.remaining <= 0.0
        or any("Zeitbudget" in s["reason_de"] for s in tuner.skipped),
    }
    return AutoTuneReport(
        problem=problem,
        budget=budget,
        reference=reference,
        stability=stability,
        consistency=consistency,
        convergence=convergence,
        benchmark=rows,
        recommendation=recommendation,
        reasoning_de=tuner.reasoning,
        skipped=tuner.skipped,
    )
