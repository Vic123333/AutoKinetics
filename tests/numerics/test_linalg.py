import numpy as np
import pytest

from autokinetics.numerics import linalg


def well_conditioned(rng, n):
    return rng.standard_normal((n, n)) + n * np.eye(n)


def hilbert(n):
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1.0)


@pytest.mark.parametrize("n", [1, 5, 40, 120])
def test_lu_pivot_matches_numpy(rng, n):
    A = rng.standard_normal((n, n))
    b = rng.standard_normal(n)
    f = linalg.lu(A)
    np.testing.assert_allclose(f.P @ A, f.L @ f.U, atol=1e-12 * np.abs(A).max() * n)
    np.testing.assert_allclose(linalg.solve(A, b), np.linalg.solve(A, b), rtol=1e-9, atol=1e-11)
    assert linalg.det(A) == pytest.approx(np.linalg.det(A), rel=1e-9)
    np.testing.assert_allclose(linalg.inv(A), np.linalg.inv(A), rtol=1e-8, atol=1e-10)


def test_lu_nopivot_works_for_diagonally_dominant(rng):
    A = well_conditioned(rng, 30)
    b = rng.standard_normal(30)
    np.testing.assert_allclose(linalg.solve(A, b, "lu_nopivot"), np.linalg.solve(A, b), rtol=1e-10)


def test_pivoting_necessity():
    # Without pivoting the 1e-20 pivot either breaks down or destroys the solution.
    A = np.array([[1e-20, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(linalg.LinAlgError, match="pivot"):
        linalg.lu(A, pivoting=False)
    with pytest.raises(linalg.LinAlgError):
        linalg.solve(A, b, "lu_nopivot")
    np.testing.assert_allclose(linalg.solve(A, b, "lu_pivot"), [1.0, 1.0], rtol=1e-14)
    with pytest.raises(linalg.LinAlgError, match="pivot"):  # exact zero pivot
        linalg.lu(np.array([[0.0, 1.0], [1.0, 0.0]]), pivoting=False)


def test_singular_matrix_reported():
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(linalg.LinAlgError):
        linalg.solve(A, [1.0, 1.0])
    with pytest.raises(linalg.LinAlgError):
        linalg.inv(A)
    assert linalg.det(A) == 0.0
    assert linalg.cond(A) == np.inf


@pytest.mark.parametrize("n", [6, 9])
def test_ill_conditioned_hilbert(n):
    A = hilbert(n)
    x_true = np.ones(n)
    b = A @ x_true
    for method in ("lu_pivot", "qr_householder", "cholesky"):
        x = linalg.solve(A, b, method)
        assert np.linalg.norm(A @ x - b) <= 1e-12 * np.linalg.norm(b)
        # forward error bounded by kappa * eps (Kap. 3)
        assert np.max(np.abs(x - x_true)) <= 100 * linalg.cond(A) * np.finfo(float).eps
    for p in (1, np.inf):
        assert linalg.cond(A, p) == pytest.approx(np.linalg.cond(A, p), rel=1e-6)
    assert linalg.cond(A, 2) == pytest.approx(np.linalg.cond(A, 2), rel=1e-3)


def test_cholesky(rng):
    B = rng.standard_normal((20, 20))
    A = B @ B.T + 20 * np.eye(20)
    L = linalg.cholesky(A)
    np.testing.assert_allclose(L @ L.T, A, rtol=1e-12, atol=1e-10)
    assert np.allclose(L, np.tril(L))
    with pytest.raises(linalg.LinAlgError, match="positive definite"):
        linalg.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(linalg.LinAlgError, match="symmetric"):
        linalg.cholesky(np.array([[2.0, 1.0], [0.0, 2.0]]))


def test_qr_householder_and_givens(rng):
    A = rng.standard_normal((12, 7))
    Q, R = linalg.qr(A)
    np.testing.assert_allclose(Q @ R, A, atol=1e-12)
    np.testing.assert_allclose(Q.T @ Q, np.eye(7), atol=1e-12)
    assert np.allclose(R, np.triu(R))
    Qg, Rg = linalg.qr_givens(A)
    np.testing.assert_allclose(Qg @ Rg, A, atol=1e-12)
    np.testing.assert_allclose(Qg.T @ Qg, np.eye(12), atol=1e-12)
    assert np.allclose(Rg, np.triu(Rg), atol=1e-14)


@pytest.mark.parametrize("method", ["householder", "givens"])
def test_least_squares(rng, method):
    A = rng.standard_normal((30, 5))
    b = rng.standard_normal(30)
    x, res = linalg.lstsq(A, b, method)
    x_np, res_np, *_ = np.linalg.lstsq(A, b, rcond=None)
    np.testing.assert_allclose(x, x_np, rtol=1e-10)
    assert res == pytest.approx(np.sqrt(res_np[0]), rel=1e-10)
    A_def = np.column_stack([A[:, 0], A[:, 0], A[:, 1]])
    with pytest.raises(linalg.LinAlgError, match="rank"):
        linalg.lstsq(A_def, b, method)


def test_iterative_solvers_converge_for_diagonally_dominant(rng):
    n = 40
    A = rng.uniform(-1, 1, (n, n))
    A += np.diag(np.abs(A).sum(axis=1) + 1.0)  # strictly diagonally dominant
    b = rng.standard_normal(n)
    x_ref = np.linalg.solve(A, b)
    for solver in (linalg.jacobi, linalg.gauss_seidel, linalg.cgnr):
        r = solver(A, b, tol=1e-12)
        assert r.converged, r.message
        np.testing.assert_allclose(r.x, x_ref, rtol=1e-9, atol=1e-11)
        assert r.history[-1] <= 1e-12
    gs, jac = linalg.gauss_seidel(A, b, tol=1e-12), linalg.jacobi(A, b, tol=1e-12)
    assert gs.iterations <= jac.iterations
    S = A @ A.T  # SPD
    r = linalg.cg(S, b, tol=1e-12)
    assert r.converged and r.iterations <= n + 5
    np.testing.assert_allclose(r.x, np.linalg.solve(S, b), rtol=1e-8)


def test_iterative_solvers_report_non_convergence():
    A = np.array([[1.0, 3.0], [3.0, 1.0]])  # rho(Jacobi iteration matrix) = 3
    b = np.array([1.0, 2.0])
    for solver in (linalg.jacobi, linalg.gauss_seidel):
        r = solver(A, b, tol=1e-10, max_iter=500)
        assert not r.converged
        assert "diverge" in r.message
        with pytest.raises(linalg.LinAlgError):
            linalg.solve(A, b, "jacobi")
    assert linalg.spectral_radius(linalg.iteration_matrix(A, "jacobi")) == pytest.approx(3.0)
    assert linalg.spectral_radius(linalg.iteration_matrix(A, "gauss_seidel")) == pytest.approx(9.0)
    slow = linalg.jacobi(np.array([[1.0, 0.99], [0.99, 1.0]]), b, tol=1e-12, max_iter=20)
    assert not slow.converged and "no convergence" in slow.message
    r = linalg.cg(np.array([[2.0, 1.0], [0.0, 2.0]]), b)
    assert not r.converged and "symmetric" in r.message
    assert linalg.cgnr(A, b, tol=1e-12).converged  # CGNR works for any regular matrix


def test_norms(rng):
    A = rng.standard_normal((6, 6))
    x = rng.standard_normal(6)
    for p in (1, 2, np.inf):
        assert linalg.norm(x, p) == pytest.approx(np.linalg.norm(x, p))
    for p in (1, np.inf, "fro", 2):
        assert linalg.norm(A, p) == pytest.approx(np.linalg.norm(A, p), rel=1e-10)


def _match_eigenvalues(ev, ref):
    ref = list(ref)
    worst = 0.0
    for e in ev:
        k = int(np.argmin([abs(e - r) for r in ref]))
        worst = max(worst, abs(e - ref[k]) / max(1.0, abs(ref[k])))
        ref.pop(k)
    return worst


@pytest.mark.parametrize("n", [2, 7, 25, 80])
def test_eigenvalues_nonsymmetric(rng, n):
    A = rng.standard_normal((n, n))
    ev = linalg.eigvals(A)
    ref = np.linalg.eigvals(A)
    assert np.sum(np.abs(ev.imag) > 1e-8) == np.sum(np.abs(ref.imag) > 1e-8)
    assert _match_eigenvalues(ev, ref) < 1e-9


def test_eigenvalues_complex_pairs_and_badly_scaled():
    rot = np.array([[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    ev = np.sort_complex(linalg.eigvals(rot))
    np.testing.assert_allclose(ev, np.sort_complex(np.array([-1.0, 2j, -2j])), atol=1e-12)
    # Robertson-like scaling (entries from 1e-2 to 1e8): balancing keeps it accurate
    J = np.array(
        [
            [-0.04, 1e4 * 0.5, 1e4 * 3e-5],
            [0.04, -1e4 * 0.5 - 6e7 * 3e-5, -1e4 * 3e-5],
            [0.0, 6e7 * 3e-5, 0.0],
        ]
    )
    assert _match_eigenvalues(linalg.eigvals(J), np.linalg.eigvals(J)) < 1e-9


def test_hessenberg_and_balance_are_similarity_transforms(rng):
    A = rng.standard_normal((10, 10)) * np.logspace(-3, 3, 10)[:, None]
    H = linalg.hessenberg(A)
    assert np.allclose(np.tril(H, -2), 0.0)
    assert _match_eigenvalues(np.linalg.eigvals(H), np.linalg.eigvals(A)) < 1e-9
    B = linalg.balance(A)
    assert _match_eigenvalues(np.linalg.eigvals(B), np.linalg.eigvals(A)) < 1e-9
    assert np.linalg.norm(B) <= np.linalg.norm(A)


def test_power_and_inverse_iteration(rng):
    Q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    lam = np.array([10.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.5, -0.2])
    A = Q @ np.diag(lam) @ Q.T
    p = linalg.power_iteration(A, tol=1e-12, max_iter=5000)
    assert p["converged"] and p["eigenvalue"] == pytest.approx(10.0, rel=1e-10)
    i = linalg.inverse_iteration(A, sigma=2.9, tol=1e-12)
    assert i["converged"] and i["eigenvalue"] == pytest.approx(3.0, rel=1e-10)
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])  # dominant eigenvalues ±i: no convergence
    r = linalg.power_iteration(rot, max_iter=200)
    assert not r["converged"] and r["message"]


def test_gershgorin_discs_contain_eigenvalues(rng):
    A = rng.standard_normal((12, 12)) + np.diag(np.arange(12) * 3.0)
    discs = linalg.gershgorin_discs(A)
    for ev in linalg.eigvals(A):
        assert any(abs(ev - c) <= r + 1e-9 for c, r in discs)
