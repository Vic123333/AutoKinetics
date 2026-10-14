from math import factorial

import numpy as np
import pytest

from autokinetics.numerics import METHODS, LinearOdeSystem, integrate, stability

Z = np.array([-3.0 + 0j, -0.5 + 2j, 0.3 - 1.1j, 1e-3 + 0j, -40.0 + 7j])


def R(name, z):
    tab = stability._tableau(name)
    return stability.rk_stability_function(tab.A, tab.b)(z)


def test_stability_functions_closed_forms():
    np.testing.assert_allclose(R("euler", Z), 1 + Z)
    np.testing.assert_allclose(R("implicit_euler", Z), 1 / (1 - Z))
    trap = (1 + Z / 2) / (1 - Z / 2)
    np.testing.assert_allclose(R("trapezoidal", Z), trap)
    np.testing.assert_allclose(R("implicit_midpoint", Z), trap)
    np.testing.assert_allclose(R("heun", Z), 1 + Z + Z**2 / 2)
    np.testing.assert_allclose(R("rk4", Z), sum(Z**k / factorial(k) for k in range(5)))
    # DOPRI5: Taylor polynomial to degree 5 plus z^6/600
    np.testing.assert_allclose(
        R("dopri5", Z), sum(Z**k / factorial(k) for k in range(6)) + Z**6 / 600, rtol=1e-10
    )
    # Gauss s=2: (1 + z/2 + z^2/12) / (1 - z/2 + z^2/12) (diagonal Padé)
    np.testing.assert_allclose(R("gauss4", Z), (1 + Z / 2 + Z**2 / 12) / (1 - Z / 2 + Z**2 / 12))
    # Radau IIA s=2: (1 + z/3) / (1 - 2z/3 + z^2/6)
    np.testing.assert_allclose(R("radau3", Z), (1 + Z / 3) / (1 - 2 * Z / 3 + Z**2 / 6))
    # R(z) ≈ e^z to the order of the method
    for name in ("rk4", "radau5", "gauss4", "sdirk2", "sdirk3"):
        z = 1e-2
        p = METHODS[name].order
        assert abs(R(name, np.array([z]))[0] - np.exp(z)) < 10 * z ** (p + 1)


def test_ros2_stability_function():
    g = stability.ROS2_GAMMA
    z = Z
    expected = (1 + (1 - 2 * g) * z + (g * g - 2 * g + 0.5) * z * z) / (1 - g * z) ** 2
    np.testing.assert_allclose(stability.ros2_stability_function(z), expected, rtol=1e-12)


@pytest.mark.parametrize("name", [m for m, info in METHODS.items() if info.backend == "cpp"])
def test_a_and_l_stability_flags_match_registry(name):
    props = stability.stability_properties(name)
    info = METHODS[name]
    assert props["a_stable"] == info.a_stable
    assert props["l_stable"] == info.l_stable


def test_stability_intervals():
    assert stability.stability_properties("euler")["real_interval"] == pytest.approx(2.0, rel=1e-6)
    assert stability.stability_properties("heun")["real_interval"] == pytest.approx(2.0, rel=1e-6)
    assert stability.stability_properties("rk4")["real_interval"] == pytest.approx(
        2.785293, rel=1e-5
    )
    assert stability.stability_properties("rk4")["imag_interval"] == pytest.approx(
        2 * np.sqrt(2), rel=1e-6
    )
    assert stability.stability_properties("dopri5")["real_interval"] == pytest.approx(
        3.3066, rel=1e-3
    )
    assert stability.stability_properties("radau5")["real_interval"] == np.inf
    assert stability.stability_properties("trapezoidal")["r_infinity"] == pytest.approx(
        1.0, abs=1e-9
    )
    assert stability.stability_properties("sdirk3")["r_infinity"] == pytest.approx(
        np.sqrt(3) - 1, rel=1e-6
    )


def test_bdf2_boundary_locus_and_region():
    z = stability.lmm_boundary_locus(stability.BDF2_ALPHA, stability.BDF2_BETA)
    finite = np.isfinite(z)
    assert np.all(z[finite].real >= -1e-12)  # Re z(θ) = (1 - cos θ)^2 >= 0: A-stable
    reg = stability.stability_region("bdf2", n=41)
    assert np.asarray(reg["abs_R"]).shape == (41, 41)
    assert len(reg["boundary_re"]) > 0


def test_stability_region_grid_and_boundary():
    reg = stability.stability_region("euler", re_range=(-3, 1), im_range=(-2, 2), n=81)
    bre, bim = np.asarray(reg["boundary_re"]), np.asarray(reg["boundary_im"])
    np.testing.assert_allclose(np.abs((bre + 1j * bim) + 1), 1.0, atol=0.02)  # circle |1+z| = 1
    amp = np.asarray(reg["abs_R"])
    assert amp[40, 20] <= 1.0  # z = -2 + 0i ... edge, z = -1 inside
    assert amp[40, 30] < 1.0


@pytest.mark.parametrize("name", ["euler", "heun", "rk4", "dopri5"])
def test_dahlquist_explicit_methods_blow_up_beyond_stability_limit(name):
    lam = -20.0
    h_star = stability.max_stable_step(name, [lam])
    assert h_star == pytest.approx(
        stability.stability_properties(name)["real_interval"] / 20.0, rel=1e-6
    )
    sys = LinearOdeSystem([[lam]])
    n_steps = 400
    stable = integrate(sys, (0, 0.97 * h_star * n_steps), [1.0], name, fixed_step=0.97 * h_star)
    unstable = integrate(sys, (0, 1.05 * h_star * n_steps), [1.0], name, fixed_step=1.05 * h_star)
    assert abs(stable.y[-1, 0]) <= 1.0
    assert np.max(np.abs(stable.y)) <= 1.0 + 1e-12
    assert not unstable.success or abs(unstable.y[-1, 0]) > 1e3


def test_max_stable_step_for_systems_and_special_cases():
    ev = np.array([-1.0, -1000.0, -0.5 + 3j, -0.5 - 3j])
    h = stability.max_stable_step("rk4", ev)
    assert np.all(np.abs(R("rk4", h * ev)) <= 1 + 1e-9)
    assert np.any(np.abs(R("rk4", 1.01 * h * ev)) > 1)
    assert stability.max_stable_step("radau5", ev) == np.inf
    assert stability.max_stable_step("euler", [2j, -2j]) == 0.0  # imaginary axis not in S
    assert stability.max_stable_step("rk4", [0.5, -1.0]) == pytest.approx(
        2.785293, rel=1e-5
    )  # growth ignored
    assert stability.stiffness_ratio([0.0, -0.04, -1e4]) == pytest.approx(2.5e5)


def test_jacobian_spectra_and_stiffness_on_robertson():
    from autokinetics.numerics import RobertsonSystem

    sys = RobertsonSystem()
    res = integrate(sys, (0, 40.0), [1.0, 0.0, 0.0], "radau5", rtol=1e-6, atol=1e-10)
    spectra = stability.jacobian_spectra(sys, res.t[[5, -1]], res.y[[5, -1]])
    for s in spectra:
        assert s["stiffness_ratio"] > 1e3
        J = sys.jacobian(s["t"], res.y[list(res.t).index(s["t"])])
        np.testing.assert_allclose(
            np.sort(s["re"]), np.sort(np.linalg.eigvals(J).real), atol=1e-8 * s["spectral_radius"]
        )


def test_fixed_point_contraction_and_iterative_solver_radius():
    J = np.array([[-2.0, 1.0], [0.5, -3.0]])
    fp = stability.fixed_point_contraction("implicit_euler", J, 0.1)
    assert fp["q_spectral"] == pytest.approx(0.1 * np.max(np.abs(np.linalg.eigvals(J))))
    assert fp["q_norm"] == pytest.approx(0.1 * 3.5)
    assert fp["converges"] and fp["guaranteed"]
    assert not stability.fixed_point_contraction("radau5", 1000 * J, 0.1)["converges"]
    M = stability.newton_matrix("radau3", J, 0.2)
    assert M.shape == (4, 4)
    rho_gs = stability.linear_iteration_radius("radau3", J, 0.2, "gauss_seidel")
    rho_j = stability.linear_iteration_radius("radau3", J, 0.2, "jacobi")
    T = np.eye(4) - M / np.diag(M)[:, None]
    assert rho_j == pytest.approx(np.max(np.abs(np.linalg.eigvals(T))), rel=1e-10)
    assert rho_gs < 1 and rho_j < 1
