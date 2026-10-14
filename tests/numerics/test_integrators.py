import threading

import numpy as np
import pytest
from num_problems import Forced, Logistic, Oscillator
from scipy.integrate import solve_ivp

from autokinetics.numerics import (
    METHODS,
    ButcherTableau,
    LinearOdeSystem,
    PythonOdeSystem,
    RobertsonSystem,
    VanDerPolSystem,
    integrate,
)
from autokinetics.numerics.tableaux import radau_iia_tableau

CPP_METHODS = [m for m, info in METHODS.items() if info.backend == "cpp"]


def steps_for(order):
    if order <= 2:
        return np.array([20, 40, 80, 160])
    if order == 3:
        return np.array([10, 20, 40, 80])
    return np.array([5, 10, 20, 40])


def observed_order(system, y0, T, exact, method):
    ns = steps_for(METHODS[method].order)
    errs = []
    for n in ns:
        r = integrate(
            system, (0.0, T), y0, method, fixed_step=T / n, rtol=1e-13, atol=1e-15, newton_tol=1e-4
        )
        assert r.success, r.detail
        assert r.stats["n_accepted"] == n
        errs.append(np.max(np.abs(r.y[-1] - exact)))
    errs = np.array(errs)
    keep = errs > 1e-13
    assert keep.sum() >= 3, errs
    return -np.polyfit(np.log(ns[keep]), np.log(errs[keep]), 1)[0]


@pytest.mark.parametrize("method", CPP_METHODS)
def test_order_dahlquist(method):
    lam = -1.5
    sys = LinearOdeSystem([[lam]])
    p = observed_order(sys, [1.0], 1.0, np.exp(lam), method)
    assert p == pytest.approx(METHODS[method].order, abs=0.3)


@pytest.mark.parametrize("method", CPP_METHODS)
def test_order_nonlinear_logistic(method):
    sys = Logistic()
    p = observed_order(sys, [sys.y0], 2.0, sys.exact(2.0), method)
    assert p == pytest.approx(METHODS[method].order, abs=0.3)


@pytest.mark.parametrize("method", ["heun", "rk4", "dopri5", "radau5", "sdirk3", "ros2", "bdf2"])
def test_order_non_autonomous(method):
    sys = Forced()
    p = observed_order(sys, [1.0], 2.0, sys.exact(2.0), method)
    assert p == pytest.approx(METHODS[method].order, abs=0.3)


@pytest.mark.parametrize("method", CPP_METHODS)
def test_adaptive_accuracy_oscillator(method):
    sys = LinearOdeSystem([[0.0, 1.0], [-1.0, 0.0]])
    rtol = 1e-6 if METHODS[method].order >= 2 else 1e-4
    r = integrate(sys, (0, 5), [1.0, 0.0], method, rtol=rtol, atol=rtol * 1e-3, max_steps=200_000)
    assert r.success, r.detail
    err = np.max(np.abs(r.y[-1] - [np.cos(5), -np.sin(5)]))
    assert err < 1e3 * rtol
    assert r.stats["n_accepted"] == len(r.t) - 1


def test_tolerance_reduces_error_and_dense_output():
    sys = LinearOdeSystem([[0.0, 1.0], [-1.0, 0.0]])
    errs = []
    for rtol in (1e-4, 1e-6, 1e-8):
        r = integrate(sys, (0, 5), [1.0, 0.0], "dopri5", rtol=rtol, atol=1e-12)
        errs.append(np.max(np.abs(r.y[-1] - [np.cos(5), -np.sin(5)])))
    assert errs[0] > errs[1] > errs[2]
    t_eval = np.linspace(0, 5, 51)
    r = integrate(sys, (0, 5), [1.0, 0.0], "dopri5", rtol=1e-9, atol=1e-12, t_eval=t_eval)
    np.testing.assert_array_equal(r.t, t_eval)
    np.testing.assert_allclose(r.y[:, 0], np.cos(t_eval), atol=1e-6)  # Hermite cubic interpolation
    r2 = integrate(
        sys, (0, 5), [1.0, 0.0], "radau5", rtol=1e-8, atol=1e-12, t_eval=t_eval, t_eval_exact=True
    )
    np.testing.assert_allclose(r2.y[:, 0], np.cos(t_eval), atol=1e-7)


@pytest.fixture(scope="module")
def robertson_reference():
    sys = RobertsonSystem()
    t_eval = np.array([1e-3, 1.0, 40.0, 1e3, 1e5])
    ref = solve_ivp(
        lambda t, y: sys.rhs(t, y),
        (0, 1e5),
        [1.0, 0.0, 0.0],
        method="Radau",
        rtol=1e-12,
        atol=1e-20,
        jac=lambda t, y: sys.jacobian(t, y),
        t_eval=t_eval,
    )
    return t_eval, ref.y.T


@pytest.mark.parametrize("method", ["radau5", "radau3", "ros2", "bdf2", "sdirk2", "implicit_euler"])
def test_robertson_stiff_methods(method, robertson_reference):
    t_eval, y_ref = robertson_reference
    sys = RobertsonSystem()
    rtol = 1e-9 if METHODS[method].order >= 2 else 1e-9
    r = integrate(
        sys,
        (0, 1e5),
        [1.0, 0.0, 0.0],
        method,
        rtol=rtol,
        atol=1e-16,
        t_eval=t_eval,
        t_eval_exact=True,
        max_steps=2_000_000,
    )
    assert r.success, r.detail
    rel = np.abs(r.y - y_ref) / np.abs(y_ref)
    assert rel.max() < 1e-4, (method, rel.max())
    # linear invariant y1 + y2 + y3 = 1 is preserved by RK, Rosenbrock (exact J) and BDF methods
    assert np.max(np.abs(r.y.sum(axis=1) - 1.0)) < 1e-10


def test_robertson_explicit_method_is_impractical():
    r = integrate(
        RobertsonSystem(),
        (0, 1e3),
        [1.0, 0.0, 0.0],
        "dopri5",
        rtol=1e-6,
        atol=1e-10,
        max_steps=20_000,
    )
    assert not r.success and r.status == "max_steps"
    assert r.t[-1] < 1e3  # partial solution is returned


def test_van_der_pol_stiff_against_scipy():
    mu = 1000.0
    sys = VanDerPolSystem(mu)
    # Hairer-Wanner scaling y2' = ((1 - y1^2) y2 - y1)/eps with eps = 1/mu: one relaxation cycle in [0, 2]
    ref = solve_ivp(
        lambda t, y: sys.rhs(t, y),
        (0, 2),
        [2.0, 0.0],
        method="Radau",
        rtol=1e-11,
        atol=1e-11,
        jac=lambda t, y: sys.jacobian(t, y),
    )
    for method in ("radau5", "ros2", "bdf2"):
        r = integrate(sys, (0, 2), [2.0, 0.0], method, rtol=1e-7, atol=1e-7, max_steps=1_000_000)
        assert r.success, method
        assert r.y[-1, 0] == pytest.approx(ref.y[0, -1], rel=1e-4), method


def test_python_callback_system_with_and_without_jacobian():
    class WithJac(Logistic):
        calls = 0

        def jacobian(self, t, y):
            WithJac.calls += 1
            return super().jacobian(t, y)

    class NoJac:
        def rhs(self, t, y):
            return y * (1.0 - y)

    for system, expect_fd in ((WithJac(), False), (NoJac(), True)):
        r = integrate(system, (0, 2), [0.1], "radau5", rtol=1e-8, atol=1e-12)
        assert r.success
        assert r.y[-1, 0] == pytest.approx(Logistic().exact(2.0)[0], rel=1e-7)
        assert (r.stats["n_jac_fd"] > 0) == expect_fd
    assert WithJac.calls > 0
    # plain callables are accepted too
    r = integrate(lambda t, y: -y, (0, 1), [1.0], "rk4", fixed_step=0.01)
    assert r.y[-1, 0] == pytest.approx(np.exp(-1), rel=1e-9)
    h = PythonOdeSystem(NoJac(), 1)
    np.testing.assert_allclose(h.jacobian(0.0, [0.25]), [[0.5]], rtol=1e-6)  # finite differences
    assert not h.has_jacobian and h.size == 1


class Boom(RuntimeError):
    pass


def test_python_callback_exceptions_propagate():
    class Failing:
        n = 0

        def rhs(self, t, y):
            Failing.n += 1
            if t > 0.5:
                raise Boom("rhs failed at t > 0.5")
            return -y

    for method in ("dopri5", "radau5", "bdf2", "scipy:RK45"):
        with pytest.raises(Boom, match="rhs failed"):
            integrate(Failing(), (0, 1), [1.0], method)

    class BadJac(Logistic):
        def jacobian(self, t, y):
            raise KeyError("jacobian broken")

    with pytest.raises(KeyError, match="jacobian broken"):
        integrate(BadJac(), (0, 1), [0.1], "radau5")

    class WrongShape:
        def rhs(self, t, y):
            return np.zeros(3)

    with pytest.raises(ValueError, match="shape"):
        integrate(WrongShape(), (0, 1), [1.0, 2.0], "rk4")
    with pytest.raises(Boom):
        PythonOdeSystem(Failing(), 1).rhs(1.0, [1.0])


def test_invalid_input_raises():
    sys = LinearOdeSystem([[-1.0]])
    with pytest.raises(ValueError):
        integrate(sys, (0, 1), [1.0, 2.0])
    with pytest.raises(ValueError):
        integrate(sys, (1, 0), [1.0])
    with pytest.raises(ValueError, match="unknown method"):
        integrate(sys, (0, 1), [1.0], "nope")
    with pytest.raises(ValueError, match="unknown linear solver"):
        integrate(sys, (0, 1), [1.0], "radau5", linear_solver="magic")
    with pytest.raises(ValueError, match="t_eval"):
        integrate(sys, (0, 1), [1.0], t_eval=[0.5, 2.0])


def test_budgets_are_respected():
    sys = RobertsonSystem()
    r = integrate(sys, (0, 1e5), [1.0, 0.0, 0.0], "rk4", max_steps=50)
    assert r.status == "max_steps" and not r.success and r.stats["n_steps"] == 50
    assert "Schrittzahl" in r.message
    r = integrate(
        sys, (0, 1e5), [1.0, 0.0, 0.0], "euler", rtol=1e-8, max_wall_time=0.05, max_steps=10**9
    )
    assert r.status == "wall_time" and r.wall_time < 1.0


def test_fixed_step_solver_failures_are_reported_not_hidden():
    sys = RobertsonSystem()
    # fixed-point iteration cannot converge for stiff problems with large h (Banach: q >> 1)
    r = integrate(
        sys, (0, 10), [1.0, 0.0, 0.0], "radau5", nonlinear_solver="fixed_point", fixed_step=0.1
    )
    assert not r.success and r.status == "nonlinear_solver_failure"
    assert "fixed_point" in r.detail
    # adaptive mode instead rejects steps and reduces h (and here gives up at h_min or budget)
    r = integrate(
        sys,
        (0, 1e-2),
        [1.0, 0.0, 0.0],
        "radau5",
        nonlinear_solver="fixed_point",
        h0=1e-2,
        max_steps=20_000,
    )
    assert r.stats["n_solver_rejections"] > 0 and r.stats["n_newton_failures"] > 0
    # LU without pivoting meets an exactly zero pivot in I - hJ
    A = np.array([[10.0, 1.0], [1.0, 0.0]])
    r = integrate(
        LinearOdeSystem(A),
        (0, 0.5),
        [1.0, 1.0],
        "implicit_euler",
        linear_solver="lu_nopivot",
        fixed_step=0.1,
    )
    assert not r.success and "pivot" in r.detail
    r = integrate(
        LinearOdeSystem(A),
        (0, 0.5),
        [1.0, 1.0],
        "implicit_euler",
        linear_solver="lu_pivot",
        fixed_step=0.1,
    )
    assert r.success


@pytest.mark.parametrize(
    "linear_solver", ["lu_pivot", "lu_nopivot", "qr_householder", "gauss_seidel", "jacobi", "cgnr"]
)
@pytest.mark.parametrize("nonlinear_solver", ["simplified_newton", "full_newton"])
def test_solver_combinations_on_mildly_stiff_problem(linear_solver, nonlinear_solver):
    # Newton matrices are strictly diagonally dominant for this Jacobian -> all linear solvers converge
    A = np.array([[-5.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, -6.0]])
    sys = LinearOdeSystem(A)
    from scipy.linalg import expm

    y0 = np.array([1.0, 2.0, 3.0])
    exact = expm(A) @ y0
    for method in ("radau5", "bdf2", "sdirk2", "ros2"):
        r = integrate(
            sys,
            (0, 1),
            y0,
            method,
            rtol=1e-8,
            atol=1e-12,
            linear_solver=linear_solver,
            nonlinear_solver=nonlinear_solver,
        )
        assert r.success, (method, r.detail)
        assert np.max(np.abs(r.y[-1] - exact)) < 1e-5
        if linear_solver in ("gauss_seidel", "jacobi", "cgnr"):
            assert r.stats["n_linear_iterations"] > 0


def test_iterative_linear_solver_failure_triggers_step_rejection():
    # Jacobi diverges on I - hJ when h is large (off-diagonal coupling dominates)
    A = np.array([[-1.0, 50.0], [-50.0, -1.0]])
    r = integrate(
        LinearOdeSystem(A),
        (0, 1),
        [1.0, 0.0],
        "implicit_euler",
        linear_solver="jacobi",
        rtol=1e-4,
        atol=1e-8,
        h0=0.5,
    )
    assert r.stats["n_linear_failures"] > 0
    assert r.success  # recovered by reducing h until rho(Jacobi) < 1


def test_nonnegative_handling():
    class Bimolecular:
        """A + B -> C with excess B: A decays to 0, BDF2 undershoots slightly below 0."""

        nonnegative_mask = np.array([True, True])

        def rhs(self, t, y):
            r = 1e3 * y[0] * y[1]
            return np.array([-r, -r])

    free = integrate(Bimolecular(), (0, 10), [1.0, 1.1], "bdf2", rtol=1e-2, atol=1e-10)
    assert free.success and free.y.min() < 0
    r = integrate(
        Bimolecular(), (0, 10), [1.0, 1.1], "bdf2", rtol=1e-2, atol=1e-10, nonnegative=True
    )
    assert r.success and r.y.min() >= 0.0
    np.testing.assert_allclose(r.y[-1], free.y[-1], atol=1e-9)
    assert RobertsonSystem().nonnegative_mask.tolist() == [True, True, True]


def test_custom_tableau_and_scipy_backends():
    sys = Logistic()
    gen = radau_iia_tableau(2)
    a = integrate(sys, (0, 2), [0.1], gen, fixed_step=0.1, rtol=1e-12, atol=1e-14)
    b = integrate(sys, (0, 2), [0.1], "radau3", fixed_step=0.1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(a.y, b.y, rtol=1e-12)
    custom = ButcherTableau(
        "ralston",
        np.array([[0, 0], [2 / 3, 0]]),
        np.array([0.25, 0.75]),
        np.array([0, 2 / 3]),
        order=2,
    )
    r = integrate(sys, (0, 2), [0.1], custom, fixed_step=0.05)
    assert r.y[-1, 0] == pytest.approx(sys.exact(2.0)[0], rel=1e-3)
    for m in ("scipy:RK45", "scipy:BDF", "scipy:Radau", "scipy:LSODA"):
        r = integrate(
            RobertsonSystem() if m != "scipy:RK45" else sys,
            (0, 2),
            [1.0, 0, 0] if m != "scipy:RK45" else [0.1],
            m,
            rtol=1e-8,
            atol=1e-12,
        )
        assert r.success, m
        assert r.stats["n_rhs"] > 0
    with pytest.raises(ValueError):
        integrate(sys, (0, 1), [0.1], "scipy:RK45", fixed_step=0.1)


def test_gil_released_for_compiled_systems():
    sys = RobertsonSystem()
    results = []

    def work():
        results.append(
            integrate(sys, (0, 1e5), [1.0, 0.0, 0.0], "radau5", rtol=1e-8, atol=1e-14).success
        )

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [True] * 4


def test_statistics_are_consistent():
    r = integrate(RobertsonSystem(), (0, 100), [1.0, 0.0, 0.0], "radau5", rtol=1e-6, atol=1e-10)
    st = r.stats
    assert st["n_steps"] == st["n_accepted"] + st["n_rejected"]
    assert (
        st["n_rejected"]
        == st["n_error_rejections"] + st["n_solver_rejections"] + st["n_nonneg_rejections"]
    )
    assert st["n_factorizations"] > 0 and st["n_linear_solves"] >= st["n_newton_iterations"]
    assert 0 <= st["theta_mean"] <= st["theta_max"]  # includes diverging (rejected) attempts
    assert st["h_min"] <= st["h_mean"] <= st["h_max"]
    full = integrate(
        RobertsonSystem(),
        (0, 100),
        [1.0, 0.0, 0.0],
        "radau5",
        rtol=1e-6,
        atol=1e-10,
        nonlinear_solver="full_newton",
    )
    assert full.stats["n_jac"] > st["n_jac"]
    to_json = r.to_dict()
    assert to_json["status"] == "success" and isinstance(to_json["y"], list)


def test_oscillator_methods_with_python_system():
    r = integrate(Oscillator(), (0, 2 * np.pi), [1.0, 0.0], "gauss4", rtol=1e-8, atol=1e-10)
    assert r.success
    np.testing.assert_allclose(r.y[-1], [1.0, 0.0], atol=1e-6)
    energy = r.y[:, 0] ** 2 + r.y[:, 1] ** 2
    assert np.max(np.abs(energy - 1.0)) < 1e-6  # Gauss methods conserve quadratic invariants
