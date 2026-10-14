// LU (with/without column pivoting), Cholesky, Householder/Givens QR,
// condition numbers (Numerik I, Kap. 3 and 4).
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "autokinetics/numerics/eigen.hpp"
#include "autokinetics/numerics/linalg.hpp"

namespace ak::num {

LUFactorization lu_factor(const Matrix& A, bool pivoting, double pivot_tol) {
    require_square(A, "lu_factor");
    const std::size_t n = A.rows;
    LUFactorization f;
    f.lu = A;
    f.pivoting = pivoting;
    f.perm.resize(n);
    std::iota(f.perm.begin(), f.perm.end(), std::size_t{0});
    const double amax = matrix_max_abs(A);
    if (!std::isfinite(amax)) {
        f.message = "matrix contains non-finite entries";
        return f;
    }
    if (amax == 0.0) {
        f.message = "matrix is zero (singular)";
        return f;
    }
    const double threshold = pivot_tol * amax;
    double umax = 0.0;
    f.min_pivot = std::numeric_limits<double>::infinity();
    Matrix& M = f.lu;
    for (std::size_t k = 0; k < n; ++k) {
        if (pivoting) {
            std::size_t p = k;
            double best = std::abs(M(k, k));
            for (std::size_t i = k + 1; i < n; ++i) {
                if (std::abs(M(i, k)) > best) {
                    best = std::abs(M(i, k));
                    p = i;
                }
            }
            if (p != k) {
                std::swap_ranges(M.row(k), M.row(k) + n, M.row(p));
                std::swap(f.perm[k], f.perm[p]);
                ++f.swaps;
            }
        }
        const double pivot = M(k, k);
        f.min_pivot = std::min(f.min_pivot, std::abs(pivot));
        if (!(std::abs(pivot) > threshold)) {
            f.message = pivoting ? "matrix is singular to working precision (pivot " + std::to_string(pivot) +
                                       " in column " + std::to_string(k) + ")"
                                 : "zero or tiny pivot " + std::to_string(pivot) + " in elimination step " +
                                       std::to_string(k) + " (Gauss elimination without pivoting breaks down)";
            return f;
        }
        const double* rk = M.row(k);
        for (std::size_t i = k + 1; i < n; ++i) {
            double* ri = M.row(i);
            const double l = ri[k] / pivot;
            ri[k] = l;
            if (l == 0.0) continue;
            for (std::size_t j = k + 1; j < n; ++j) ri[j] -= l * rk[j];
        }
        for (std::size_t j = k; j < n; ++j) umax = std::max(umax, std::abs(rk[j]));
    }
    f.growth = umax / amax;
    if (!std::isfinite(f.growth)) {
        f.message = "non-finite values during elimination";
        return f;
    }
    f.ok = true;
    return f;
}

void forward_substitution(const Matrix& L, double* b, bool unit_diagonal) {
    const std::size_t n = L.rows;
    for (std::size_t i = 0; i < n; ++i) {
        const double* r = L.row(i);
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j) s -= r[j] * b[j];
        b[i] = unit_diagonal ? s : s / r[i];
    }
}

void back_substitution(const Matrix& U, double* b) {
    const std::size_t n = U.rows;
    for (std::size_t ii = n; ii-- > 0;) {
        const double* r = U.row(ii);
        double s = b[ii];
        for (std::size_t j = ii + 1; j < n; ++j) s -= r[j] * b[j];
        b[ii] = s / r[ii];
    }
}

void lu_solve_inplace(const LUFactorization& f, double* b) {
    if (!f.ok) throw std::runtime_error("lu_solve: factorisation failed: " + f.message);
    const std::size_t n = f.lu.rows;
    Vec tmp(n);
    for (std::size_t i = 0; i < n; ++i) tmp[i] = b[f.perm[i]];
    forward_substitution(f.lu, tmp.data(), true);
    back_substitution(f.lu, tmp.data());
    std::copy(tmp.begin(), tmp.end(), b);
}

Vec lu_solve(const LUFactorization& f, const Vec& b) {
    if (b.size() != f.lu.rows) throw std::invalid_argument("lu_solve: dimension mismatch");
    Vec x = b;
    lu_solve_inplace(f, x.data());
    return x;
}

double lu_determinant(const LUFactorization& f) {
    if (!f.ok) throw std::runtime_error("lu_determinant: factorisation failed: " + f.message);
    double d = (f.swaps % 2 == 0) ? 1.0 : -1.0;
    for (std::size_t i = 0; i < f.lu.rows; ++i) d *= f.lu(i, i);
    return d;
}

Matrix lu_inverse(const LUFactorization& f) {
    const std::size_t n = f.lu.rows;
    Matrix inv(n, n);
    Vec e(n);
    for (std::size_t j = 0; j < n; ++j) {
        std::fill(e.begin(), e.end(), 0.0);
        e[j] = 1.0;
        lu_solve_inplace(f, e.data());
        for (std::size_t i = 0; i < n; ++i) inv(i, j) = e[i];
    }
    return inv;
}

Matrix inverse(const Matrix& A) {
    LUFactorization f = lu_factor(A, true);
    if (!f.ok) throw std::runtime_error("inverse: " + f.message);
    return lu_inverse(f);
}

CholeskyFactorization cholesky(const Matrix& A) {
    require_square(A, "cholesky");
    const std::size_t n = A.rows;
    CholeskyFactorization f;
    f.L = Matrix(n, n);
    const double scale = matrix_max_abs(A);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (std::abs(A(i, j) - A(j, i)) > 1e-12 * scale) {
                f.message = "matrix is not symmetric";
                return f;
            }
    Matrix& L = f.L;
    for (std::size_t j = 0; j < n; ++j) {
        const double d = A(j, j) - dot(L.row(j), L.row(j), j);
        if (!(d > 0.0)) {
            f.message = "matrix is not positive definite (d_" + std::to_string(j) + " = " + std::to_string(d) + ")";
            return f;
        }
        const double ljj = std::sqrt(d);
        L(j, j) = ljj;
        for (std::size_t i = j + 1; i < n; ++i) L(i, j) = (A(i, j) - dot(L.row(i), L.row(j), j)) / ljj;
    }
    f.ok = true;
    return f;
}

Vec cholesky_solve(const CholeskyFactorization& f, const Vec& b) {
    if (!f.ok) throw std::runtime_error("cholesky_solve: factorisation failed: " + f.message);
    const std::size_t n = f.L.rows;
    if (b.size() != n) throw std::invalid_argument("cholesky_solve: dimension mismatch");
    Vec x = b;
    forward_substitution(f.L, x.data(), false);
    for (std::size_t ii = n; ii-- > 0;) {  // L^T x = y
        double s = x[ii];
        for (std::size_t j = ii + 1; j < n; ++j) s -= f.L(j, ii) * x[j];
        x[ii] = s / f.L(ii, ii);
    }
    return x;
}

QRFactorization qr_householder(const Matrix& A) {
    const std::size_t m = A.rows, n = A.cols;
    if (m < n || n == 0) throw std::invalid_argument("qr_householder: requires rows >= cols > 0");
    QRFactorization f;
    f.m = m;
    f.n = n;
    f.V = Matrix(m, n);
    f.R = Matrix(n, n);
    Matrix W = A;
    Vec v(m);
    for (std::size_t k = 0; k < n; ++k) {
        std::fill(v.begin(), v.end(), 0.0);
        for (std::size_t i = k; i < m; ++i) v[i] = W(i, k);
        const double nx = norm2(v);
        const double alpha = (W(k, k) >= 0.0) ? -nx : nx;
        v[k] -= alpha;
        const double nv = norm2(v);
        if (nv > 0.0) {
            for (std::size_t i = k; i < m; ++i) v[i] /= nv;
            for (std::size_t j = k; j < n; ++j) {  // W <- (I - 2 v v^T) W
                double s = 0.0;
                for (std::size_t i = k; i < m; ++i) s += v[i] * W(i, j);
                s *= 2.0;
                for (std::size_t i = k; i < m; ++i) W(i, j) -= s * v[i];
            }
        }
        for (std::size_t i = k; i < m; ++i) f.V(i, k) = v[i];
        for (std::size_t j = k; j < n; ++j) f.R(k, j) = W(k, j);
    }
    return f;
}

void qr_apply_qt(const QRFactorization& f, double* b) {
    for (std::size_t k = 0; k < f.n; ++k) {
        double s = 0.0;
        for (std::size_t i = k; i < f.m; ++i) s += f.V(i, k) * b[i];
        s *= 2.0;
        for (std::size_t i = k; i < f.m; ++i) b[i] -= s * f.V(i, k);
    }
}

Matrix qr_q(const QRFactorization& f) {
    Matrix Q(f.m, f.n);  // Q = H_0 H_1 ... H_{n-1} applied to e_j
    Vec e(f.m);
    for (std::size_t j = 0; j < f.n; ++j) {
        std::fill(e.begin(), e.end(), 0.0);
        e[j] = 1.0;
        for (std::size_t kk = f.n; kk-- > 0;) {
            double s = 0.0;
            for (std::size_t i = kk; i < f.m; ++i) s += f.V(i, kk) * e[i];
            s *= 2.0;
            for (std::size_t i = kk; i < f.m; ++i) e[i] -= s * f.V(i, kk);
        }
        for (std::size_t i = 0; i < f.m; ++i) Q(i, j) = e[i];
    }
    return Q;
}

namespace {

LeastSquaresResult solve_triangular_ls(const Matrix& R, const Vec& qtb, std::size_t n, double rank_tol) {
    LeastSquaresResult res;
    double rmax = 0.0;
    for (std::size_t i = 0; i < n; ++i) rmax = std::max(rmax, std::abs(R(i, i)));
    for (std::size_t i = 0; i < n; ++i) {
        if (rmax == 0.0 || !(std::abs(R(i, i)) > rank_tol * rmax)) {
            res.message = "matrix is rank deficient (|r_ii| below tolerance in column " + std::to_string(i) +
                          "); solution is not unique";
            return res;
        }
    }
    Matrix Rn(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i; j < n; ++j) Rn(i, j) = R(i, j);
    res.x.assign(qtb.begin(), qtb.begin() + static_cast<long>(n));
    back_substitution(Rn, res.x.data());
    res.residual_norm = norm2(Vec(qtb.begin() + static_cast<long>(n), qtb.end()));
    res.ok = true;
    return res;
}

}  // namespace

LeastSquaresResult lstsq_householder(const Matrix& A, const Vec& b, double rank_tol) {
    if (b.size() != A.rows) throw std::invalid_argument("lstsq: dimension mismatch");
    QRFactorization f = qr_householder(A);
    Vec qtb = b;
    qr_apply_qt(f, qtb.data());
    return solve_triangular_ls(f.R, qtb, f.n, rank_tol);
}

void givens(double a, double b, double& c, double& s, double& r) {
    if (b == 0.0) {
        c = (a >= 0.0) ? 1.0 : -1.0;
        s = 0.0;
        r = std::abs(a);
        return;
    }
    r = std::hypot(a, b);
    c = a / r;
    s = b / r;
}

void qr_givens(const Matrix& A, Matrix& Q, Matrix& R) {
    const std::size_t m = A.rows, n = A.cols;
    R = A;
    Q = Matrix::identity(m);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = m; i-- > j + 1;) {
            const double a = R(i - 1, j), b = R(i, j);
            if (b == 0.0) continue;
            double c, s, r;
            givens(a, b, c, s, r);
            for (std::size_t k = 0; k < n; ++k) {  // rows i-1, i of R
                const double x = R(i - 1, k), y = R(i, k);
                R(i - 1, k) = c * x + s * y;
                R(i, k) = -s * x + c * y;
            }
            R(i, j) = 0.0;
            for (std::size_t k = 0; k < m; ++k) {  // Q <- Q G^T
                const double x = Q(k, i - 1), y = Q(k, i);
                Q(k, i - 1) = c * x + s * y;
                Q(k, i) = -s * x + c * y;
            }
        }
    }
}

LeastSquaresResult lstsq_givens(const Matrix& A, const Vec& b, double rank_tol) {
    if (b.size() != A.rows) throw std::invalid_argument("lstsq: dimension mismatch");
    if (A.rows < A.cols || A.cols == 0) throw std::invalid_argument("lstsq_givens: requires rows >= cols > 0");
    Matrix Q, R;
    qr_givens(A, Q, R);
    return solve_triangular_ls(R, matTvec(Q, b), A.cols, rank_tol);
}

double cond_1(const Matrix& A) {
    LUFactorization f = lu_factor(A, true, 0.0);
    if (!f.ok) return std::numeric_limits<double>::infinity();
    return matrix_norm_1(A) * matrix_norm_1(lu_inverse(f));
}

double cond_inf(const Matrix& A) {
    LUFactorization f = lu_factor(A, true, 0.0);
    if (!f.ok) return std::numeric_limits<double>::infinity();
    return matrix_norm_inf(A) * matrix_norm_inf(lu_inverse(f));
}

double cond_2(const Matrix& A) {
    require_square(A, "cond_2");
    const Matrix AtA = matmul(transpose(A), A);
    EigenResult e = eigenvalues(AtA, true);
    if (!e.ok) throw std::runtime_error("cond_2: eigenvalue iteration failed: " + e.message);
    double lmax = 0.0, lmin = std::numeric_limits<double>::infinity();
    for (double v : e.re) {
        lmax = std::max(lmax, std::abs(v));
        lmin = std::min(lmin, std::abs(v));
    }
    if (lmin == 0.0) return std::numeric_limits<double>::infinity();
    return std::sqrt(lmax / lmin);
}

}  // namespace ak::num
