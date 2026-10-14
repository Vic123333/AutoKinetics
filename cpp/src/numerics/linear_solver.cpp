#include "autokinetics/numerics/linear_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ak::num {

LinearSolverKind linear_solver_from_string(const std::string& name) {
    if (name == "lu_pivot") return LinearSolverKind::LUPivot;
    if (name == "lu_nopivot") return LinearSolverKind::LUNoPivot;
    if (name == "qr_householder") return LinearSolverKind::QRHouseholder;
    if (name == "gauss_seidel") return LinearSolverKind::GaussSeidel;
    if (name == "jacobi") return LinearSolverKind::Jacobi;
    if (name == "cgnr") return LinearSolverKind::CGNR;
    throw std::invalid_argument("unknown linear solver '" + name + "'");
}

std::string to_string(LinearSolverKind kind) {
    switch (kind) {
        case LinearSolverKind::LUPivot: return "lu_pivot";
        case LinearSolverKind::LUNoPivot: return "lu_nopivot";
        case LinearSolverKind::QRHouseholder: return "qr_householder";
        case LinearSolverKind::GaussSeidel: return "gauss_seidel";
        case LinearSolverKind::Jacobi: return "jacobi";
        case LinearSolverKind::CGNR: return "cgnr";
    }
    return "unknown";
}

namespace {

class LUSolver final : public LinearSolver {
public:
    explicit LUSolver(bool pivoting) : pivoting_(pivoting) {}
    bool factor(const Matrix& M) override {
        f_ = lu_factor(M, pivoting_);
        if (!f_.ok) error_ = f_.message;
        return f_.ok;
    }
    bool solve(double* b) override {
        lu_solve_inplace(f_, b);
        last_iterations_ = 0;
        return true;
    }
    bool is_direct() const override { return true; }

private:
    bool pivoting_;
    LUFactorization f_;
};

class QRSolver final : public LinearSolver {
public:
    bool factor(const Matrix& M) override {
        f_ = qr_householder(M);
        double rmax = 0.0;
        for (std::size_t i = 0; i < f_.n; ++i) rmax = std::max(rmax, std::abs(f_.R(i, i)));
        for (std::size_t i = 0; i < f_.n; ++i) {
            if (!(std::abs(f_.R(i, i)) > 1e-14 * rmax)) {
                error_ = "matrix is singular to working precision (QR: tiny r_ii)";
                return false;
            }
        }
        return true;
    }
    bool solve(double* b) override {
        qr_apply_qt(f_, b);
        back_substitution(f_.R, b);
        last_iterations_ = 0;
        return true;
    }
    bool is_direct() const override { return true; }

private:
    QRFactorization f_;
};

class IterativeSolver final : public LinearSolver {
public:
    IterativeSolver(LinearSolverKind kind, double tol, int max_iter) : kind_(kind), tol_(tol), max_iter_(max_iter) {}
    bool factor(const Matrix& M) override {
        M_ = M;
        return true;
    }
    bool solve(double* b) override {
        const std::size_t n = M_.rows;
        const int max_iter = max_iter_ > 0 ? max_iter_ : static_cast<int>(std::max<std::size_t>(200, 20 * n));
        Vec rhs(b, b + n);
        IterativeResult r;
        switch (kind_) {
            case LinearSolverKind::Jacobi: r = jacobi(M_, rhs, {}, tol_, max_iter); break;
            case LinearSolverKind::GaussSeidel: r = gauss_seidel(M_, rhs, {}, tol_, max_iter); break;
            default: r = cgnr(M_, rhs, {}, tol_, max_iter); break;
        }
        last_iterations_ = r.iterations;
        if (!r.converged) {
            error_ = to_string(kind_) + ": " + r.message;
            return false;
        }
        std::copy(r.x.begin(), r.x.end(), b);
        return true;
    }
    bool is_direct() const override { return false; }

private:
    LinearSolverKind kind_;
    double tol_;
    int max_iter_;
    Matrix M_;
};

}  // namespace

std::unique_ptr<LinearSolver> make_linear_solver(LinearSolverKind kind, double tol, int max_iter) {
    switch (kind) {
        case LinearSolverKind::LUPivot: return std::make_unique<LUSolver>(true);
        case LinearSolverKind::LUNoPivot: return std::make_unique<LUSolver>(false);
        case LinearSolverKind::QRHouseholder: return std::make_unique<QRSolver>();
        case LinearSolverKind::GaussSeidel:
        case LinearSolverKind::Jacobi:
        case LinearSolverKind::CGNR: return std::make_unique<IterativeSolver>(kind, tol, max_iter);
    }
    throw std::invalid_argument("make_linear_solver: unknown kind");
}

}  // namespace ak::num
