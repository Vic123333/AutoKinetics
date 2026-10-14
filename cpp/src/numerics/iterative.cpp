// Iterative solvers for linear systems (Numerik I, Kap. 6): Jacobi, Gauß-Seidel,
// conjugate gradients (SPD) and CGNR (CG on the normal equations).
#include <cmath>

#include "autokinetics/numerics/linalg.hpp"

namespace ak::num {

namespace {

void check_system(const Matrix& A, const Vec& b, const Vec& x0, const char* what) {
    require_square(A, what);
    if (b.size() != A.rows) throw std::invalid_argument(std::string(what) + ": rhs has wrong length");
    if (!x0.empty() && x0.size() != A.rows) throw std::invalid_argument(std::string(what) + ": x0 has wrong length");
}

double residual_norm(const Matrix& A, const Vec& b, const Vec& x) {
    Vec r = matvec(A, x);
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
    return norm2(r);
}

// Shared bookkeeping: records the relative residual and decides whether to stop.
// Returns true when the iteration must terminate (converged or diverged).
bool record(IterativeResult& res, double rel, double rel0, double tol) {
    res.history.push_back(rel);
    res.residual = rel;
    if (!std::isfinite(rel)) {
        res.message = "iteration diverged (non-finite residual)";
        return true;
    }
    if (rel <= tol) {
        res.converged = true;
        return true;
    }
    if (rel > 1e8 * std::max(rel0, 1.0)) {
        res.message = "iteration diverged (residual grew by more than 1e8)";
        return true;
    }
    return false;
}

void finish(IterativeResult& res, int max_iter) {
    if (!res.converged && res.message.empty()) {
        res.message = "no convergence within " + std::to_string(max_iter) + " iterations (relative residual " +
                      std::to_string(res.residual) + ")";
    }
}

}  // namespace

IterativeResult jacobi(const Matrix& A, const Vec& b, const Vec& x0, double tol, int max_iter) {
    check_system(A, b, x0, "jacobi");
    const std::size_t n = A.rows;
    IterativeResult res;
    res.x = x0.empty() ? Vec(n, 0.0) : x0;
    for (std::size_t i = 0; i < n; ++i) {
        if (A(i, i) == 0.0) {
            res.message = "zero diagonal entry a_" + std::to_string(i) + std::to_string(i) + "; Jacobi not applicable";
            return res;
        }
    }
    const double bn = norm2(b);
    if (bn == 0.0) {
        res.x.assign(n, 0.0);
        res.converged = true;
        return res;
    }
    const double rel0 = residual_norm(A, b, res.x) / bn;
    if (record(res, rel0, rel0, tol)) return res;
    Vec xn(n);
    for (int it = 1; it <= max_iter; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            const double* r = A.row(i);
            double s = b[i];
            for (std::size_t j = 0; j < n; ++j)
                if (j != i) s -= r[j] * res.x[j];
            xn[i] = s / r[i];
        }
        res.x.swap(xn);
        res.iterations = it;
        if (record(res, residual_norm(A, b, res.x) / bn, rel0, tol)) break;
    }
    finish(res, max_iter);
    return res;
}

IterativeResult gauss_seidel(const Matrix& A, const Vec& b, const Vec& x0, double tol, int max_iter) {
    check_system(A, b, x0, "gauss_seidel");
    const std::size_t n = A.rows;
    IterativeResult res;
    res.x = x0.empty() ? Vec(n, 0.0) : x0;
    for (std::size_t i = 0; i < n; ++i) {
        if (A(i, i) == 0.0) {
            res.message =
                "zero diagonal entry a_" + std::to_string(i) + std::to_string(i) + "; Gauss-Seidel not applicable";
            return res;
        }
    }
    const double bn = norm2(b);
    if (bn == 0.0) {
        res.x.assign(n, 0.0);
        res.converged = true;
        return res;
    }
    const double rel0 = residual_norm(A, b, res.x) / bn;
    if (record(res, rel0, rel0, tol)) return res;
    for (int it = 1; it <= max_iter; ++it) {
        for (std::size_t i = 0; i < n; ++i) {
            const double* r = A.row(i);
            double s = b[i];
            for (std::size_t j = 0; j < n; ++j)
                if (j != i) s -= r[j] * res.x[j];
            res.x[i] = s / r[i];
        }
        res.iterations = it;
        if (record(res, residual_norm(A, b, res.x) / bn, rel0, tol)) break;
    }
    finish(res, max_iter);
    return res;
}

namespace {

// CG on the operator x -> op(x) (symmetric positive definite), rhs rhs.
// The relative residual reported is measured by `true_residual`.
template <class Op, class TrueResidual>
IterativeResult cg_core(std::size_t n, Op op, const Vec& rhs, Vec x, double tol, int max_iter,
                        TrueResidual true_residual) {
    IterativeResult res;
    Vec r = op(x);
    for (std::size_t i = 0; i < n; ++i) r[i] = rhs[i] - r[i];
    Vec p = r;
    double rr = dot(r.data(), r.data(), n);
    res.x = x;
    const double rel0 = true_residual(res.x);
    if (record(res, rel0, rel0, tol)) return res;
    for (int it = 1; it <= max_iter; ++it) {
        Vec Ap = op(p);
        const double pAp = dot(p.data(), Ap.data(), n);
        if (!(pAp > 0.0)) {
            res.message = "operator is not positive definite (p^T A p = " + std::to_string(pAp) + ")";
            res.iterations = it;
            return res;
        }
        const double alpha = rr / pAp;
        for (std::size_t i = 0; i < n; ++i) {
            res.x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        const double rr_new = dot(r.data(), r.data(), n);
        res.iterations = it;
        if (record(res, true_residual(res.x), rel0, tol)) break;
        const double beta = rr_new / rr;
        rr = rr_new;
        for (std::size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
    }
    finish(res, max_iter);
    return res;
}

}  // namespace

IterativeResult conjugate_gradient(const Matrix& A, const Vec& b, const Vec& x0, double tol, int max_iter) {
    check_system(A, b, x0, "conjugate_gradient");
    const std::size_t n = A.rows;
    const double scale = matrix_max_abs(A);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (std::abs(A(i, j) - A(j, i)) > 1e-12 * scale) {
                IterativeResult res;
                res.x = x0.empty() ? Vec(n, 0.0) : x0;
                res.message = "matrix is not symmetric; CG requires an SPD matrix (use CGNR)";
                return res;
            }
    const double bn = norm2(b);
    if (bn == 0.0) {
        IterativeResult res;
        res.x.assign(n, 0.0);
        res.converged = true;
        return res;
    }
    return cg_core(
        n, [&](const Vec& v) { return matvec(A, v); }, b, x0.empty() ? Vec(n, 0.0) : x0, tol, max_iter,
        [&](const Vec& x) { return residual_norm(A, b, x) / bn; });
}

IterativeResult cgnr(const Matrix& A, const Vec& b, const Vec& x0, double tol, int max_iter) {
    check_system(A, b, x0, "cgnr");
    const std::size_t n = A.rows;
    const double bn = norm2(b);
    if (bn == 0.0) {
        IterativeResult res;
        res.x.assign(n, 0.0);
        res.converged = true;
        return res;
    }
    const Vec Atb = matTvec(A, b);
    return cg_core(
        n, [&](const Vec& v) { return matTvec(A, matvec(A, v)); }, Atb, x0.empty() ? Vec(n, 0.0) : x0, tol,
        max_iter, [&](const Vec& x) { return residual_norm(A, b, x) / bn; });
}

}  // namespace ak::num
