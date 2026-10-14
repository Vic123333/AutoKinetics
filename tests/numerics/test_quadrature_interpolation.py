import numpy as np
import pytest
from scipy.interpolate import CubicSpline as ScipySpline

from autokinetics.numerics import interpolation as ip
from autokinetics.numerics import quadrature as q


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 20])
def test_gauss_legendre_matches_numpy(n):
    x, w = q.gauss_legendre(n)
    xr, wr = np.polynomial.legendre.leggauss(n)
    np.testing.assert_allclose(x, xr, atol=1e-14)
    np.testing.assert_allclose(w, wr, atol=1e-14)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 9])
def test_gauss_lobatto_exactness(n):
    x, w = q.gauss_lobatto(n)
    assert x[0] == -1.0 and x[-1] == 1.0
    for k in range(2 * n - 2):  # exact up to degree 2n-3
        exact = (1.0 - (-1.0) ** (k + 1)) / (k + 1)
        assert np.dot(w, x**k) == pytest.approx(exact, abs=1e-13)
    k = 2 * n - 2
    assert abs(np.dot(w, x**k) - 2.0 / (k + 1)) > 1e-8


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_gauss_radau_exactness(n):
    x, w = q.gauss_radau(n)
    assert x[-1] == 1.0
    for k in range(2 * n - 1):  # exact up to degree 2n-2 on [0, 1]
        assert np.dot(w, x**k) == pytest.approx(1.0 / (k + 1), abs=1e-13)


def test_newton_cotes_weights():
    np.testing.assert_allclose(q.newton_cotes_weights(1), [0.5, 0.5])
    np.testing.assert_allclose(q.newton_cotes_weights(2), [1 / 6, 4 / 6, 1 / 6])
    np.testing.assert_allclose(q.newton_cotes_weights(3), [1 / 8, 3 / 8, 3 / 8, 1 / 8])
    np.testing.assert_allclose(q.newton_cotes_weights(4), np.array([7, 32, 12, 32, 7]) / 90)
    assert q.newton_cotes(lambda x: x**3, 0.0, 2.0, 2) == pytest.approx(
        4.0
    )  # Simpson exact for cubics


def test_composite_rules_convergence_rates():
    f, exact = np.exp, np.e - 1.0
    errs_t = [abs(q.composite_trapezoid(f, 0, 1, m) - exact) for m in (8, 16, 32)]
    errs_s = [abs(q.composite_simpson(f, 0, 1, m) - exact) for m in (8, 16, 32)]
    errs_g = [abs(q.composite_gauss(f, 0, 1, 2, m) - exact) for m in (2, 4, 8)]
    assert np.log2(errs_t[0] / errs_t[1]) == pytest.approx(2.0, abs=0.05)
    assert np.log2(errs_s[0] / errs_s[1]) == pytest.approx(4.0, abs=0.1)
    assert np.log2(errs_g[0] / errs_g[1]) == pytest.approx(4.0, abs=0.1)
    assert q.gauss_legendre_rule(lambda x: x**9, -1, 2, 5) == pytest.approx((2**10 - 1) / 10)
    assert q.gauss_lobatto_rule(lambda x: x**7, 0, 1, 5) == pytest.approx(1 / 8)
    with pytest.raises(ValueError):
        q.composite_simpson(f, 0, 1, 3)


@pytest.mark.parametrize("npts", [7, 8])  # even and odd number of intervals
def test_integrate_samples_nonuniform(rng, npts):
    t = np.sort(np.concatenate([[0.0, 2.0], rng.uniform(0, 2, npts - 2)]))
    y = np.column_stack([3 * t**2 - t + 1, np.ones_like(t)])
    exact = np.array([8 - 2 + 2, 2.0])
    np.testing.assert_allclose(q.integrate_samples(t, y, "simpson"), exact, rtol=1e-12)
    tt = np.linspace(0, np.pi, 41)
    for method, tol in (("trapezoid", 2e-3), ("simpson", 1e-6), ("spline", 1e-5)):
        assert q.integrate_samples(tt, np.sin(tt), method) == pytest.approx(2.0, abs=tol)
    cum = q.cumulative_trapezoid(tt, np.sin(tt))
    assert cum[0] == 0.0 and cum[-1] == pytest.approx(2.0, abs=2e-3)


def test_lagrange_and_newton_agree(rng):
    x = np.linspace(-1, 1, 9)
    y = np.cos(3 * x)
    xq = rng.uniform(-1, 1, 50)
    lag = ip.lagrange_interpolate(x, y, xq)
    newt = ip.newton_evaluate(ip.divided_differences(x, y), x, xq)
    np.testing.assert_allclose(lag, newt, atol=1e-12)
    np.testing.assert_allclose(ip.lagrange_interpolate(x, y, x), y)  # interpolation property
    p = np.polynomial.Polynomial([1, -2, 0.5, 3])
    np.testing.assert_allclose(ip.lagrange_interpolate(x[:4], p(x[:4]), xq), p(xq), atol=1e-12)
    with pytest.raises(ValueError):
        ip.divided_differences([0, 0, 1], [1, 2, 3])


@pytest.mark.parametrize("bc", ["natural", "not-a-knot"])
def test_cubic_spline_matches_scipy(rng, bc):
    x = np.sort(rng.uniform(0, 5, 12))
    y = np.column_stack([np.sin(x), x**2])
    s = ip.CubicSpline(x, y, bc=bc)
    ref = ScipySpline(x, y, bc_type=bc)
    xq = np.linspace(x[0], x[-1], 101)
    for nu in (0, 1, 2):
        np.testing.assert_allclose(s(xq, nu), ref(xq, nu), atol=1e-10)
    np.testing.assert_allclose(s.integrate(x[1], x[-2]), ref.integrate(x[1], x[-2]), atol=1e-10)


def test_spline_reproduces_cubics_with_not_a_knot():
    x = np.linspace(0, 1, 6)
    s = ip.CubicSpline(x, x**3 - x, bc="not-a-knot")
    xq = np.linspace(0, 1, 37)
    np.testing.assert_allclose(s(xq), xq**3 - xq, atol=1e-13)
    np.testing.assert_allclose(ip.resample(x, x**3 - x, xq), xq**3 - xq, atol=1e-13)
    np.testing.assert_allclose(ip.resample(x, 2 * x, xq, "linear"), 2 * xq)
