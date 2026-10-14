"""Dense linear algebra backed by the C++ library (Numerik I, Kap. 3, 4, 6 and 9).

All routines are self-implemented in C++ (no LAPACK). Failures (singular
matrices, zero pivots without pivoting, non-SPD input, non-convergent
iterations) raise :class:`LinAlgError` or are reported in the result objects —
never silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from autokinetics._ext import _core

__all__ = [
    "LinAlgError",
    "LUResult",
    "IterativeSolveResult",
    "lu",
    "solve",
    "det",
    "inv",
    "cholesky",
    "qr",
    "qr_givens",
    "lstsq",
    "jacobi",
    "gauss_seidel",
    "cg",
    "cgnr",
    "cond",
    "norm",
    "eigvals",
    "hessenberg",
    "balance",
    "power_iteration",
    "inverse_iteration",
    "gershgorin_discs",
    "spectral_radius",
    "iteration_matrix",
]

DIRECT_SOLVERS = ("lu_pivot", "lu_nopivot", "qr_householder", "cholesky")
ITERATIVE_SOLVERS = ("jacobi", "gauss_seidel", "cg", "cgnr")


class LinAlgError(ValueError):
    """Raised when a linear-algebra routine cannot produce a valid result."""


@dataclass
class LUResult:
    """P A = L U (P given as permutation vector: row i of PA is row perm[i] of A)."""

    L: np.ndarray
    U: np.ndarray
    perm: np.ndarray
    min_pivot: float
    growth: float  # element growth factor max|U| / max|A|

    @property
    def P(self) -> np.ndarray:
        n = len(self.perm)
        P = np.zeros((n, n))
        P[np.arange(n), self.perm] = 1.0
        return P


@dataclass
class IterativeSolveResult:
    x: np.ndarray
    iterations: int
    residual: float
    converged: bool
    message: str
    history: np.ndarray = field(default_factory=lambda: np.empty(0))


def _square(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError(f"expected a non-empty square matrix, got shape {A.shape}")
    return A


def lu(A, pivoting: bool = True, pivot_tol: float = 1e-12) -> LUResult:
    """LU decomposition by Gauß elimination, with or without column pivoting
    (Spaltenpivotsuche). Without pivoting, pivots |u_kk| <= pivot_tol * max|a_ij|
    raise :class:`LinAlgError`."""
    L, U, perm, ok, msg, min_pivot, growth = _core.num_lu_factor(_square(A), pivoting, pivot_tol)
    if not ok:
        raise LinAlgError(msg)
    return LUResult(L, U, perm, min_pivot, growth)


def solve(
    A, b, method: str = "lu_pivot", *, tol: float = 1e-10, max_iter: int | None = None
) -> np.ndarray:
    """Solves A x = b with the chosen method.

    Direct: ``lu_pivot``, ``lu_nopivot``, ``qr_householder``, ``cholesky``.
    Iterative: ``jacobi``, ``gauss_seidel``, ``cg``, ``cgnr`` (raise if not converged).
    """
    A = _square(A)
    b = np.asarray(b, dtype=float)
    if method in ("lu_pivot", "lu_nopivot"):
        x, msg = _core.num_lu_solve(
            A, b, method == "lu_pivot", 1e-12 if method == "lu_nopivot" else 1e-14
        )
        if x is None:
            raise LinAlgError(msg)
        return x
    if method == "qr_householder":
        if b.ndim == 2:
            return np.column_stack([solve(A, b[:, j], method) for j in range(b.shape[1])])
        x, _, ok, msg = _core.num_lstsq(A, b, "householder", 1e-14)
        if not ok:
            raise LinAlgError(msg)
        return x
    if method == "cholesky":
        L = cholesky(A)
        y = solve_triangular(L, b, lower=True)
        return solve_triangular(L.T, y, lower=False)
    if method in ITERATIVE_SOLVERS:
        res = _iterative(method, A, b, None, tol, max_iter)
        if not res.converged:
            raise LinAlgError(f"{method}: {res.message}")
        return res.x
    raise ValueError(f"unknown method {method!r}; choose from {DIRECT_SOLVERS + ITERATIVE_SOLVERS}")


def solve_triangular(T, b, lower: bool) -> np.ndarray:
    """Forward (lower) or backward (upper) substitution."""
    T = _square(T)
    x = np.array(b, dtype=float, copy=True)
    n = T.shape[0]
    rng = range(n) if lower else range(n - 1, -1, -1)
    for i in rng:
        s = T[i, :i] @ x[:i] if lower else T[i, i + 1 :] @ x[i + 1 :]
        if T[i, i] == 0.0:
            raise LinAlgError(f"zero diagonal entry in triangular matrix (row {i})")
        x[i] = (x[i] - s) / T[i, i]
    return x


def det(A) -> float:
    """Determinant via LU with column pivoting."""
    return float(_core.num_det(_square(A)))


def inv(A) -> np.ndarray:
    """Inverse via LU with column pivoting (column-wise solves)."""
    X, msg = _core.num_inv(_square(A))
    if X is None:
        raise LinAlgError(msg)
    return X


def cholesky(A) -> np.ndarray:
    """Lower triangular L with A = L L^T (A symmetric positive definite)."""
    L, ok, msg = _core.num_cholesky(_square(A))
    if not ok:
        raise LinAlgError(msg)
    return L


def qr(A) -> tuple[np.ndarray, np.ndarray]:
    """Thin QR decomposition by Householder reflections (m >= n): Q (m x n), R (n x n)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] < A.shape[1]:
        raise ValueError("qr requires a matrix with rows >= cols")
    return _core.num_qr_householder(A)


def qr_givens(A) -> tuple[np.ndarray, np.ndarray]:
    """Full QR decomposition by Givens rotations: Q (m x m), R (m x n)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("qr_givens requires a matrix")
    return _core.num_qr_givens(A)


def lstsq(A, b, method: str = "householder", rank_tol: float = 1e-12) -> tuple[np.ndarray, float]:
    """Linear least squares min ||A x - b||_2 via QR (``householder`` or ``givens``).
    Returns (x, residual norm); raises on rank deficiency."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] < A.shape[1]:
        raise ValueError("lstsq requires a matrix with rows >= cols")
    x, res, ok, msg = _core.num_lstsq(A, np.asarray(b, dtype=float), method, rank_tol)
    if not ok:
        raise LinAlgError(msg)
    return x, res


def _iterative(kind, A, b, x0, tol, max_iter) -> IterativeSolveResult:
    n = A.shape[0]
    r = _core.num_iterative(
        kind, A, np.asarray(b, dtype=float), x0, tol, max_iter or max(1000, 10 * n)
    )
    return IterativeSolveResult(**r)


def jacobi(A, b, x0=None, tol: float = 1e-10, max_iter: int | None = None) -> IterativeSolveResult:
    """Jacobi (Gesamtschrittverfahren); converges iff rho(D^{-1}(L+U)) < 1."""
    return _iterative("jacobi", _square(A), b, x0, tol, max_iter)


def gauss_seidel(
    A, b, x0=None, tol: float = 1e-10, max_iter: int | None = None
) -> IterativeSolveResult:
    """Gauß–Seidel (Einzelschrittverfahren); converges iff rho((D+L)^{-1} U) < 1."""
    return _iterative("gauss_seidel", _square(A), b, x0, tol, max_iter)


def cg(A, b, x0=None, tol: float = 1e-10, max_iter: int | None = None) -> IterativeSolveResult:
    """Conjugate gradients for symmetric positive definite A."""
    return _iterative("cg", _square(A), b, x0, tol, max_iter)


def cgnr(A, b, x0=None, tol: float = 1e-10, max_iter: int | None = None) -> IterativeSolveResult:
    """CG applied to the normal equations A^T A x = A^T b (any regular A)."""
    return _iterative("cgnr", _square(A), b, x0, tol, max_iter)


def cond(A, p=np.inf) -> float:
    """Condition number kappa_p(A) = ||A||_p ||A^{-1}||_p for p in {1, 2, inf}
    (inf if A is singular)."""
    return float(_core.num_cond(_square(A), p))


def norm(x, p=2) -> float:
    """Vector norms (p = 1, 2, inf) and matrix norms (p = 1, inf, 'fro'; 2 = spectral norm)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        if p == 1:
            return float(np.sum(np.abs(x)))
        if p == 2:
            return float(np.sqrt(np.sum(x * x)))
        if p in (np.inf, "inf"):
            return float(np.max(np.abs(x))) if x.size else 0.0
        raise ValueError("vector norm p must be 1, 2 or inf")
    if x.ndim == 2:
        if p == 2:
            ev = eigvals(x.T @ x)
            return float(np.sqrt(np.max(np.abs(ev))))
        return float(_core.num_matrix_norm(x, p))
    raise ValueError("norm expects a vector or a matrix")


def eigvals(A, balance: bool = True) -> np.ndarray:
    """All eigenvalues (complex array) via balancing, Householder–Hessenberg
    reduction and the Francis double-shift QR algorithm."""
    ev, ok, msg, _ = _core.num_eigvals(_square(A), balance)
    if not ok:
        raise LinAlgError(msg)
    return ev


def hessenberg(A) -> np.ndarray:
    """Upper Hessenberg matrix orthogonally similar to A (Householder)."""
    return _core.num_hessenberg(_square(A))


def balance(A) -> np.ndarray:
    """Diagonally similar matrix with balanced row/column norms."""
    return _core.num_balance(_square(A))


def power_iteration(A, x0=None, tol: float = 1e-10, max_iter: int = 1000) -> dict:
    """Vector iteration (von Mises) for the dominant eigenvalue."""
    return _core.num_power_iteration(_square(A), x0, tol, max_iter)


def inverse_iteration(A, sigma: float, x0=None, tol: float = 1e-10, max_iter: int = 1000) -> dict:
    """Inverse vector iteration with shift sigma: eigenvalue closest to sigma."""
    return _core.num_inverse_iteration(_square(A), float(sigma), x0, tol, max_iter)


def gershgorin_discs(A) -> list[tuple[float, float]]:
    """Gershgorin row discs as (center, radius) pairs."""
    centers, radii = _core.num_gershgorin(_square(A))
    return list(zip(centers.tolist(), radii.tolist(), strict=True))


def spectral_radius(A) -> float:
    """rho(A) = max |lambda_i| (QR algorithm)."""
    return float(np.max(np.abs(eigvals(A))))


def iteration_matrix(A, kind: str) -> np.ndarray:
    """Iteration matrix T of the splitting method x <- T x + c:
    Jacobi T = I - D^{-1} A, Gauß–Seidel T = -(D + L)^{-1} U."""
    A = _square(A)
    d = np.diag(A)
    if np.any(d == 0.0):
        raise LinAlgError("zero diagonal entry: splitting method not applicable")
    if kind == "jacobi":
        return np.eye(A.shape[0]) - A / d[:, None]
    if kind == "gauss_seidel":
        DL = np.tril(A)
        U = np.triu(A, 1)
        return -solve(DL, U, "lu_pivot")
    raise ValueError("kind must be 'jacobi' or 'gauss_seidel'")
