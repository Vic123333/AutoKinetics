// Strategy objects for the linear systems that arise inside Newton iterations
// of implicit integrators (matrix M = I - h (A (x) J) or I - h gamma J).
#pragma once

#include <memory>
#include <string>

#include "autokinetics/numerics/linalg.hpp"

namespace ak::num {

enum class LinearSolverKind { LUPivot, LUNoPivot, QRHouseholder, GaussSeidel, Jacobi, CGNR };

LinearSolverKind linear_solver_from_string(const std::string& name);
std::string to_string(LinearSolverKind kind);

class LinearSolver {
public:
    virtual ~LinearSolver() = default;

    // Prepares the solver for M (direct: factorisation; iterative: stores M).
    // Returns false (and sets error()) if M cannot be handled.
    virtual bool factor(const Matrix& M) = 0;

    // Solves M x = b in place (b <- x). Returns false (and sets error()) if the
    // solve failed, e.g. an iterative method did not converge.
    virtual bool solve(double* b) = 0;

    virtual bool is_direct() const = 0;

    const std::string& error() const { return error_; }
    int last_iterations() const { return last_iterations_; }

protected:
    std::string error_;
    int last_iterations_ = 0;
};

// tol / max_iter only affect the iterative kinds (relative residual tolerance;
// max_iter <= 0 selects a default proportional to the system size).
std::unique_ptr<LinearSolver> make_linear_solver(LinearSolverKind kind, double tol, int max_iter);

}  // namespace ak::num
