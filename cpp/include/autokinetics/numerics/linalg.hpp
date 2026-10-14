// Direct and iterative solvers for dense linear systems (Numerik I, Kap. 4 and 6).
#pragma once

#include <string>
#include <vector>

#include "autokinetics/numerics/matrix.hpp"

namespace ak::num {

// ---------------------------------------------------------------------------
// LU decomposition (Gauß elimination), Kap. 4.1
// ---------------------------------------------------------------------------
struct LUFactorization {
    Matrix lu;                     // L (unit lower, below diagonal) and U (upper) packed
    std::vector<std::size_t> perm; // row i of P A is row perm[i] of A
    bool pivoting = true;
    int swaps = 0;
    bool ok = false;
    std::string message;           // reason for failure (empty if ok)
    double min_pivot = 0.0;        // smallest |u_kk|
    double growth = 0.0;           // max|U| / max|A| (element growth factor)
};

// Without pivoting the factorisation fails if a pivot satisfies
// |u_kk| <= pivot_tol * max|a_ij|; with column pivoting only exact singularity
// (relative to the same threshold) is reported.
LUFactorization lu_factor(const Matrix& A, bool pivoting, double pivot_tol = 1e-12);
void lu_solve_inplace(const LUFactorization& f, double* b);  // b <- A^{-1} b
Vec lu_solve(const LUFactorization& f, const Vec& b);
double lu_determinant(const LUFactorization& f);
Matrix lu_inverse(const LUFactorization& f);
Matrix inverse(const Matrix& A);  // throws std::runtime_error if singular

// Triangular solves (in place).
void forward_substitution(const Matrix& L, double* b, bool unit_diagonal);
void back_substitution(const Matrix& U, double* b);

// ---------------------------------------------------------------------------
// Cholesky decomposition A = L L^T, Kap. 4.2
// ---------------------------------------------------------------------------
struct CholeskyFactorization {
    Matrix L;
    bool ok = false;
    std::string message;
};
CholeskyFactorization cholesky(const Matrix& A);
Vec cholesky_solve(const CholeskyFactorization& f, const Vec& b);

// ---------------------------------------------------------------------------
// QR decomposition and least squares, Kap. 4.3
// ---------------------------------------------------------------------------
struct QRFactorization {
    std::size_t m = 0, n = 0;
    Matrix V;  // m x n, column k holds the Householder vector v_k (unit norm, zeros above k)
    Matrix R;  // n x n upper triangular
};
QRFactorization qr_householder(const Matrix& A);  // requires m >= n
void qr_apply_qt(const QRFactorization& f, double* b);  // b (length m) <- Q^T b
Matrix qr_q(const QRFactorization& f);  // explicit thin Q (m x n)

struct LeastSquaresResult {
    Vec x;
    double residual_norm = 0.0;  // ||A x - b||_2
    bool ok = false;
    std::string message;
};
// Solves min ||A x - b||_2 via Householder QR; reports rank deficiency.
LeastSquaresResult lstsq_householder(const Matrix& A, const Vec& b, double rank_tol = 1e-12);

// Givens rotation G with [c s; -s c]^T-convention: [c s; -s c] [a; b] = [r; 0].
void givens(double a, double b, double& c, double& s, double& r);
// QR by Givens rotations: returns Q (m x m) and R (m x n).
void qr_givens(const Matrix& A, Matrix& Q, Matrix& R);
LeastSquaresResult lstsq_givens(const Matrix& A, const Vec& b, double rank_tol = 1e-12);

// ---------------------------------------------------------------------------
// Iterative solvers, Kap. 6
// ---------------------------------------------------------------------------
struct IterativeResult {
    Vec x;
    int iterations = 0;
    double residual = 0.0;      // final ||b - A x||_2 / ||b||_2
    bool converged = false;
    std::string message;
    std::vector<double> history;  // relative residual per iteration
};

// Stopping rule for all solvers: ||b - A x||_2 <= tol * ||b||_2.
// Divergence (residual growth by 1e8 or non-finite values) is reported.
IterativeResult jacobi(const Matrix& A, const Vec& b, const Vec& x0, double tol, int max_iter);
IterativeResult gauss_seidel(const Matrix& A, const Vec& b, const Vec& x0, double tol, int max_iter);
IterativeResult conjugate_gradient(const Matrix& A, const Vec& b, const Vec& x0, double tol, int max_iter);
// CG on the normal equations A^T A x = A^T b (works for any regular A).
IterativeResult cgnr(const Matrix& A, const Vec& b, const Vec& x0, double tol, int max_iter);

// ---------------------------------------------------------------------------
// Condition numbers, Kap. 3
// ---------------------------------------------------------------------------
double cond_1(const Matrix& A);    // ||A||_1 ||A^{-1}||_1 (inf if singular)
double cond_inf(const Matrix& A);  // ||A||_inf ||A^{-1}||_inf
double cond_2(const Matrix& A);    // sigma_max / sigma_min via eigenvalues of A^T A

}  // namespace ak::num
