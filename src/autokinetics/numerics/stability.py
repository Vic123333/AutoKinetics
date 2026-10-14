"""Linear stability analysis (ODE script Kap. 5.6 and 7.6; Numerik I Kap. 5, 6, 9).

* Stability function R(z) of Runge–Kutta methods, R(z) = 1 + z b^T (I - zA)^{-1} 1
  (Lemma 5.15), evaluated as det(I - zA + z 1 b^T) / det(I - zA); Rosenbrock
  ROS2 analogously from its linear-test-equation recursion.
* BDF2 via the roots of rho(zeta) - z sigma(zeta) (Def. 7.22) and the
  boundary locus z(theta) = rho(e^{i theta}) / sigma(e^{i theta}).
* A-/L-stability checks, stability intervals and region grids for plotting.
* Spectra of the Jacobian along a trajectory (own QR eigenvalue solver),
  stiffness ratio, maximal stable step of explicit methods.
* Banach contraction bound of the fixed-point iteration for the stage
  equations, spectral radii of the Jacobi/Gauß–Seidel iteration matrices of
  the Newton matrix (convergence of the iterative linear solvers).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from autokinetics.numerics import linalg
from autokinetics.numerics.methods import METHODS
from autokinetics.numerics.tableaux import ButcherTableau

__all__ = [
    "ROS2_GAMMA",
    "BDF2_ALPHA",
    "BDF2_BETA",
    "rk_stability_function",
    "ros2_stability_function",
    "lmm_root_modulus",
    "lmm_boundary_locus",
    "amplification",
    "stability_properties",
    "stability_region",
    "max_stable_step",
    "stiffness_ratio",
    "jacobian_spectra",
    "fixed_point_contraction",
    "newton_matrix",
    "linear_iteration_radius",
]

ROS2_GAMMA = 1.0 + 1.0 / np.sqrt(2.0)
# BDF2 with constant step:  3/2 y_{n+2} - 2 y_{n+1} + 1/2 y_n = h f_{n+2}
BDF2_ALPHA = np.array([0.5, -2.0, 1.5])
BDF2_BETA = np.array([0.0, 0.0, 1.0])
_TOL = 1e-9

MethodLike = str | ButcherTableau


def _tableau(method: MethodLike) -> ButcherTableau | None:
    if isinstance(method, ButcherTableau):
        return method
    info = METHODS.get(method)
    if info is None:
        raise ValueError(f"unknown method {method!r}")
    return ButcherTableau.builtin(info.tableau) if info.tableau else None


def rk_stability_function(A, b) -> Callable[[np.ndarray], np.ndarray]:
    """R(z) = det(I - zA + z 1 b^T) / det(I - zA) (equivalent to Lemma 5.15)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    s = A.shape[0]
    E = np.ones((s, 1)) @ b[None, :]
    eye = np.eye(s)

    def R(z):
        z = np.asarray(z, dtype=complex)
        zz = z.reshape(-1, 1, 1)
        num = np.linalg.det(eye - zz * A + zz * E)
        den = np.linalg.det(eye - zz * A)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = num / den
        out = np.where(den == 0, np.inf, out)
        return out.reshape(z.shape)

    return R


def ros2_stability_function(z):
    """R(z) of ROS2 from (1-γz)K1 = z, (1-γz)K2 = z(1+K1) - 2K1, R = 1 + 3/2 K1 + 1/2 K2."""
    z = np.asarray(z, dtype=complex)
    g = ROS2_GAMMA
    with np.errstate(divide="ignore", invalid="ignore"):
        k1 = z / (1 - g * z)
        k2 = (z * (1 + k1) - 2 * k1) / (1 - g * z)
    return 1 + 1.5 * k1 + 0.5 * k2


def lmm_root_modulus(alpha, beta, z) -> np.ndarray:
    """max |zeta| over the roots of rho(zeta) - z sigma(zeta) (linear multistep method
    sum alpha_j y_{n+j} = h sum beta_j f_{n+j})."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    z = np.asarray(z, dtype=complex)
    flat = z.ravel()
    if alpha.size == 3:  # closed form for 2-step methods
        a = alpha[2] - flat * beta[2]
        b = alpha[1] - flat * beta[1]
        c = alpha[0] - flat * beta[0]
        disc = np.sqrt(b * b - 4 * a * c + 0j)
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = (-b + disc) / (2 * a)
            r2 = (-b - disc) / (2 * a)
        out = np.maximum(np.abs(r1), np.abs(r2))
        out = np.where(a == 0, np.abs(c / np.where(b == 0, 1, b)), out)
    else:
        out = np.array([np.max(np.abs(np.roots((alpha - zi * beta)[::-1]))) for zi in flat])
    return out.reshape(z.shape)


def lmm_boundary_locus(alpha, beta, n: int = 721) -> np.ndarray:
    """z(theta) = rho(e^{i theta}) / sigma(e^{i theta}) for theta in [0, 2 pi]."""
    zeta = np.exp(1j * np.linspace(0.0, 2 * np.pi, n))
    rho = np.polyval(np.asarray(alpha, dtype=float)[::-1], zeta)
    sigma = np.polyval(np.asarray(beta, dtype=float)[::-1], zeta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return rho / sigma


def amplification(method: MethodLike, z) -> np.ndarray:
    """|R(z)| for one-step methods, max root modulus for BDF2. Stable iff <= 1."""
    if method == "bdf2":
        return lmm_root_modulus(BDF2_ALPHA, BDF2_BETA, z)
    if method == "ros2":
        return np.abs(ros2_stability_function(z))
    tab = _tableau(method)
    if tab is None:
        raise ValueError(f"no linear stability model for method {method!r}")
    return np.abs(rk_stability_function(tab.A, tab.b)(z))


def _scan_interval(method: MethodLike, direction: complex) -> float:
    """Largest x such that the method is stable on the segment direction * [0, x]."""
    xs = np.concatenate([[0.0], np.logspace(-4, 6, 2001)])
    amp = amplification(method, direction * xs)
    bad = np.nonzero(~(amp <= 1 + _TOL))[0]
    if bad.size == 0:
        return np.inf
    k = bad[0]
    if k == 0:
        return 0.0
    lo, hi = xs[k - 1], xs[k]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if amplification(method, direction * mid) <= 1 + _TOL:
            lo = mid
        else:
            hi = mid
    return lo


def stability_properties(method: MethodLike) -> dict:
    """A-/L-stability, |R(∞)| and stability intervals on the real and imaginary axes.

    |R(∞)| is evaluated at z = -1e12 (BDF2's root modulus decays only like |z|^{-1/2});
    L-stable means A-stable with |R(∞)| < 1e-4."""
    with np.errstate(over="ignore", invalid="ignore"):
        r_inf = float(np.asarray(amplification(method, np.array([-1e12])))[0])
    tab = None if method in ("bdf2", "ros2") else _tableau(method)
    explicit = tab is not None and tab.is_explicit
    a_stable = False
    if not explicit:
        y = np.concatenate([[0.0], np.logspace(-4, 6, 3001)])
        on_axis = np.all(amplification(method, 1j * y) <= 1 + _TOL) and np.all(
            amplification(method, -1j * y) <= 1 + _TOL
        )
        x = -np.logspace(-3, 4, 60)
        grid = (
            x[:, None]
            + 1j * np.concatenate([-np.logspace(-3, 4, 60), [0.0], np.logspace(-3, 4, 60)])[None, :]
        )
        in_half_plane = np.all(amplification(method, grid) <= 1 + _TOL)
        poles_ok = True
        if tab is not None:
            lam = np.linalg.eigvals(tab.A)
            lam = lam[np.abs(lam) > 1e-14]
            poles_ok = bool(np.all((1.0 / lam).real > 0))
        a_stable = bool(on_axis and in_half_plane and poles_ok and r_inf <= 1 + _TOL)
    return {
        "a_stable": a_stable,
        "l_stable": bool(a_stable and r_inf < 1e-4),
        "r_infinity": r_inf if np.isfinite(r_inf) else None,
        "real_interval": float(_scan_interval(method, -1.0 + 0j)),
        "imag_interval": float(_scan_interval(method, 1j)),
    }


def _default_window(method: MethodLike) -> tuple[tuple[float, float], tuple[float, float]]:
    tab = None if method in ("bdf2", "ros2") else _tableau(method)
    if tab is not None and tab.is_explicit:
        return (-5.0, 1.0), (-3.5, 3.5)
    return (-6.0, 12.0), (-9.0, 9.0)


def stability_region(
    method: MethodLike, re_range=None, im_range=None, n: int = 81, digits: int | None = None
) -> dict:
    """Grid data for plotting the stability region S = {z : amplification(z) <= 1}.

    Returns x (n), y (n), abs_R (n rows of n values, row j at y[j], clipped to
    [0, 1e3]), and boundary points (re, im) where the amplification crosses 1
    between neighbouring grid points (linear interpolation). For BDF2 the
    boundary is the exact boundary locus. ``digits`` rounds all values (smaller JSON).
    """
    dre, dim = _default_window(method)
    re_range = re_range or dre
    im_range = im_range or dim
    x = np.linspace(re_range[0], re_range[1], n)
    y = np.linspace(im_range[0], im_range[1], n)
    Z = x[None, :] + 1j * y[:, None]
    amp = np.minimum(np.nan_to_num(amplification(method, Z), nan=1e3, posinf=1e3), 1e3)
    if method == "bdf2":
        loc = lmm_boundary_locus(BDF2_ALPHA, BDF2_BETA, 361)
        keep = (
            (loc.real >= re_range[0])
            & (loc.real <= re_range[1])
            & (loc.imag >= im_range[0])
            & (loc.imag <= im_range[1])
        )
        bre, bim = loc.real[keep], loc.imag[keep]
    else:
        bre, bim = _crossings(x, y, amp)
    out = {"x": x, "y": y, "abs_R": amp, "boundary_re": bre, "boundary_im": bim}
    if digits is not None:
        out = {k: np.round(v, digits) for k, v in out.items()}
    return out


def _crossings(x, y, amp) -> tuple[np.ndarray, np.ndarray]:
    f = amp - 1.0
    pts_re, pts_im = [], []
    # horizontal neighbours
    s = f[:, :-1] * f[:, 1:] < 0
    jj, ii = np.nonzero(s)
    w = f[jj, ii] / (f[jj, ii] - f[jj, ii + 1])
    pts_re.append(x[ii] + w * (x[ii + 1] - x[ii]))
    pts_im.append(y[jj])
    # vertical neighbours
    s = f[:-1, :] * f[1:, :] < 0
    jj, ii = np.nonzero(s)
    w = f[jj, ii] / (f[jj, ii] - f[jj + 1, ii])
    pts_re.append(x[ii])
    pts_im.append(y[jj] + w * (y[jj + 1] - y[jj]))
    return np.concatenate(pts_re), np.concatenate(pts_im)


def _relevant_eigenvalues(eigenvalues) -> np.ndarray:
    lam = np.asarray(eigenvalues, dtype=complex).ravel()
    if lam.size == 0:
        return lam
    rho = np.max(np.abs(lam))
    if rho == 0:
        return lam[:0]
    # growing modes (Re > 0) are physical instabilities, not a step-size restriction
    return lam[(lam.real <= 1e-10 * rho) & (np.abs(lam) > 1e-14 * rho)]


def max_stable_step(method: MethodLike, eigenvalues) -> float:
    """Largest h such that amplification(h λ) <= 1 for all (non-growing)
    eigenvalues λ and all step sizes in (0, h] (first boundary crossing along
    the rays h λ). inf if no restriction (e.g. A-stable methods), 0 if no
    practically usable step is stable, i.e. h ρ < 1e-3 with the spectral
    radius ρ (e.g. explicit Euler on purely imaginary λ, where only the
    round-off tolerance would admit tiny steps)."""
    lam = _relevant_eigenvalues(eigenvalues)
    if lam.size == 0:
        return np.inf
    rho = np.max(np.abs(lam))

    def stable(h: float) -> bool:
        return bool(np.all(amplification(method, h * lam) <= 1 + _TOL))

    hs = np.logspace(-8, 8, 801) / rho
    # coarse scan for the first unstable step size, then bisection
    k = next((i for i, h in enumerate(hs) if not stable(h)), None)
    if k is None:
        return np.inf
    if k == 0:
        return 0.0
    lo, hi = hs[k - 1], hs[k]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if stable(mid):
            lo = mid
        else:
            hi = mid
    return float(lo) if lo * rho >= 1e-3 else 0.0


def stiffness_ratio(eigenvalues) -> float:
    """max |Re λ| / min |Re λ| over decaying modes (1 if fewer than two)."""
    lam = _relevant_eigenvalues(eigenvalues)
    if lam.size == 0:
        return 1.0
    rho = np.max(np.abs(lam))
    decay = -lam.real[lam.real < -1e-10 * rho]
    if decay.size < 2:
        return 1.0
    return float(decay.max() / decay.min())


def jacobian_spectra(system, t: Sequence[float], Y: np.ndarray) -> list[dict]:
    """Eigenvalues of J(t_k, y_k) (own Hessenberg/QR solver) for each sample."""
    out = []
    for tk, yk in zip(np.asarray(t, dtype=float), np.asarray(Y, dtype=float), strict=True):
        J = system.jacobian(float(tk), yk)
        ev = linalg.eigvals(J)
        rel = _relevant_eigenvalues(ev)
        out.append(
            {
                "t": float(tk),
                "re": ev.real,
                "im": ev.imag,
                "spectral_radius": float(np.max(np.abs(ev))) if ev.size else 0.0,
                "max_decay_rate": float(max(-rel.real.min(), 0.0)) if rel.size else 0.0,
                "max_growth_rate": float(max(ev.real.max(), 0.0)) if ev.size else 0.0,
                "stiffness_ratio": stiffness_ratio(ev),
                "jacobian_norm_inf": float(np.max(np.sum(np.abs(J), axis=1))),
            }
        )
    return out


def _method_matrices(method: MethodLike) -> np.ndarray:
    """Coefficient matrix of the implicit stage equations: A (RK), [[2/3]] (BDF2, constant
    step), [[γ]] (ROS2 linear systems)."""
    if method == "bdf2":
        return np.array([[2.0 / 3.0]])
    if method == "ros2":
        return np.array([[ROS2_GAMMA]])
    tab = _tableau(method)
    if tab is None:
        raise ValueError(f"no stage-equation model for method {method!r}")
    return tab.A


def fixed_point_contraction(method: MethodLike, J, h: float) -> dict:
    """Banach contraction of Z -> h (A ⊗ I) F(Z): sufficient bound
    q_norm = h ||A||_inf ||J||_inf and the asymptotic rate q_spec = h ρ(A) ρ(J)
    (eigenvalues of A ⊗ J are the products λ_i(A) μ_j(J))."""
    A = _method_matrices(method)
    J = np.asarray(J, dtype=float)
    q_norm = h * linalg.norm(A, np.inf) * linalg.norm(J, np.inf)
    q_spec = h * linalg.spectral_radius(A) * linalg.spectral_radius(J)
    return {
        "q_norm": float(q_norm),
        "q_spectral": float(q_spec),
        "converges": bool(q_spec < 1.0),
        "guaranteed": bool(q_norm < 1.0),
    }


def newton_matrix(method: MethodLike, J, h: float) -> np.ndarray:
    """Iteration matrix of the linear systems: I - h (A ⊗ J) (RK), I - (2/3) h J
    (BDF2), I - γ h J (ROS2)."""
    A = _method_matrices(method)
    J = np.asarray(J, dtype=float)
    return np.eye(A.shape[0] * J.shape[0]) - h * np.kron(A, J)


def linear_iteration_radius(method: MethodLike, J, h: float, kind: str) -> float:
    """Spectral radius of the Jacobi or Gauß–Seidel iteration matrix for the
    Newton matrix (< 1 <=> the iterative linear solver converges); inf if the
    splitting is not applicable (zero diagonal)."""
    M = newton_matrix(method, J, h)
    try:
        return linalg.spectral_radius(linalg.iteration_matrix(M, kind))
    except linalg.LinAlgError:
        return np.inf
