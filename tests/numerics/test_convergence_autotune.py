import json
import time

import numpy as np
import pytest
from num_problems import Logistic, Oscillator

from autokinetics.numerics import (
    METHODS,
    LinearOdeSystem,
    RobertsonSystem,
    auto_optimize,
    convergence,
    integrate,
)


def test_global_convergence_verdicts():
    sys = Logistic()
    res = convergence.global_convergence(
        sys, (0, 2), [sys.y0], "rk4", n_steps=(5, 10, 20, 40), y_ref=sys.exact(2.0)
    )
    assert res["convergent"] and res["order_confirmed"]
    assert res["study"].slope == pytest.approx(4.0, abs=0.3)
    d = res["study"].to_dict()
    assert len(d["fitted"]) == 4 and d["kind"] == "global"


def test_unstable_step_sizes_are_excluded_from_fit():
    sys = LinearOdeSystem([[-50.0]])
    res = convergence.global_convergence(
        sys,
        (0, 0.2),
        [1.0],
        "euler",
        n_steps=(2, 80, 160, 320, 640),
        y_ref=np.array([np.exp(-10.0)]),
    )
    study = res["study"]
    assert not study.used[0]  # h = 0.1 > 2/50: unstable, error explodes
    assert study.slope == pytest.approx(1.0, abs=0.3)


def test_tolerance_proportionality_dopri5():
    sys = Oscillator()
    res = convergence.tolerance_proportionality(
        sys,
        (0, 5),
        [1.0, 0.0],
        "dopri5",
        rtols=(1e-4, 1e-6, 1e-8),
        atol_ratio=1e-3,
        y_ref=np.array([np.cos(5), -np.sin(5)]),
    )
    assert res["slope"] == pytest.approx(1.0, abs=0.35)
    assert res["proportional"]
    assert res["n_rhs"][0] < res["n_rhs"][-1]


@pytest.mark.parametrize(
    ("kind", "order"), [("full_newton", 2.0), ("simplified_newton", 1.0), ("fixed_point", 1.0)]
)
def test_stage_iteration_convergence_orders(kind, order):
    class Cubic:
        def rhs(self, t, y):
            return np.array([-(y[0] ** 3) + y[1], -y[1] - y[0] ** 2])

        def jacobian(self, t, y):
            return np.array([[-3 * y[0] ** 2, 1.0], [-2 * y[0], -1.0]])

    hist = convergence.stage_iteration_history(Cubic(), 0.0, [1.5, 0.5], 0.3, "radau5", kind)
    assert hist["converged"]
    q = hist["observed_order"]
    q = q[np.isfinite(q)]
    if kind == "full_newton":
        assert np.max(q) > 1.6  # quadratic convergence
        assert len(hist["norms"]) < 10
    else:
        assert np.median(q) == pytest.approx(order, abs=0.3)
        assert np.all(hist["theta"][1:-1] < 1)


def test_invariant_and_nonnegativity_diagnostics():
    r = integrate(RobertsonSystem(), (0, 100), [1.0, 0.0, 0.0], "radau5", rtol=1e-6, atol=1e-10)
    drift = convergence.invariant_drift(r, lambda y: y.sum())
    assert drift["max_abs_drift"] < 1e-12
    neg = convergence.nonnegativity_violation(r, [True, True, True])
    assert neg["min_value"] >= -1e-10
    stats = convergence.newton_statistics(r)
    assert stats["iterations_per_step"] > 0 and "Iterationen" in stats["reasoning_de"]


def test_auto_optimize_robertson_picks_stiff_method():
    budget = 8.0
    start = time.perf_counter()
    rep = auto_optimize(
        RobertsonSystem(),
        (0.0, 1e5),
        [1.0, 0.0, 0.0],
        rtol=1e-6,
        atol=1e-10,
        budget_s=budget,
        invariants=lambda y: [y.sum()],
    )
    elapsed = time.perf_counter() - start
    assert elapsed <= 1.25 * budget + 1.0
    assert rep.stability["stiff"]
    opts = rep.recommended_options
    info = METHODS[opts["method"]]
    assert info.implicit and info.a_stable
    # explicit methods were excluded by the stability analysis, with a German reason
    blocked = {s["item"] for s in rep.skipped if s["stage"] == "benchmark"}
    assert {"dopri5", "rk4", "heun"} <= blocked
    assert all(isinstance(s["reason_de"], str) for s in rep.skipped)
    # the recommendation is valid and directly usable
    res = integrate(RobertsonSystem(), (0.0, 1e5), [1.0, 0.0, 0.0], **opts)
    assert res.success
    d = rep.to_dict()
    text = json.dumps(d, allow_nan=False)
    assert len(text) < 2_000_000
    assert d["recommendation"]["options"]["method"] == opts["method"]
    assert d["consistency"]["radau5"]["algebraic_order"] == 5
    assert "spectra" in d["stability"] and len(d["stability"]["spectra"][0]["re"]) == 3
    assert any(row["valid"] for row in d["benchmark"])


@pytest.mark.parametrize("problem", ["decay", "oscillator"])
def test_auto_optimize_nonstiff_picks_explicit_method(problem):
    if problem == "decay":
        system, y0, t_span = LinearOdeSystem([[-1.0]]), [1.0], (0.0, 10.0)
    else:
        system, y0, t_span = Oscillator(), [1.0, 0.0], (0.0, 10.0)
    rep = auto_optimize(system, t_span, y0, rtol=1e-6, atol=1e-9, budget_s=6.0)
    assert not rep.stability["stiff"]
    assert not METHODS[rep.recommended_options["method"]].implicit
    assert rep.budget["elapsed_s"] <= 6.0 * 1.25 + 1.0
    json.dumps(rep.to_dict(), allow_nan=False)
    assert rep.reasoning_de and all(isinstance(s, str) for s in rep.reasoning_de)
    conv = rep.to_dict()["convergence"]["dopri5"]
    assert conv["global"]["slope"] is not None


def test_auto_optimize_tiny_budget_degrades_gracefully():
    start = time.perf_counter()
    rep = auto_optimize(
        RobertsonSystem(), (0.0, 1e5), [1.0, 0.0, 0.0], rtol=1e-6, atol=1e-10, budget_s=0.3
    )
    assert time.perf_counter() - start < 3.0
    assert rep.recommended_options["method"] in METHODS
    json.dumps(rep.to_dict(), allow_nan=False)


def test_auto_optimize_rejects_unknown_candidates():
    with pytest.raises(ValueError):
        auto_optimize(LinearOdeSystem([[-1.0]]), (0, 1), [1.0], candidates=["nope"])
