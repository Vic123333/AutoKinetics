// Eigenvalue algorithms for real nonsymmetric matrices (Numerik I, Kap. 9).
#pragma once

#include <string>
#include <vector>

#include "autokinetics/numerics/matrix.hpp"

namespace ak::num {

// Diagonal similarity scaling (power-of-two factors) that reduces the norm of
// badly scaled matrices; does not change the eigenvalues.
Matrix balance(const Matrix& A);

// Orthogonal reduction to upper Hessenberg form H = Q^T A Q by Householder
// reflections (entries below the first subdiagonal are exactly zero).
Matrix hessenberg(const Matrix& A);

struct EigenResult {
    Vec re, im;
    bool ok = false;
    int iterations = 0;  // total QR sweeps
    std::string message;
};

// All eigenvalues via balancing, Hessenberg reduction and the implicitly
// shifted QR algorithm with Francis double shifts (complex pairs as re +- i im).
EigenResult eigenvalues(const Matrix& A, bool do_balance = true);

struct PowerIterationResult {
    double eigenvalue = 0.0;  // Rayleigh quotient estimate
    Vec eigenvector;
    int iterations = 0;
    bool converged = false;
    std::string message;
};

// Vector iteration x_{k+1} = A x_k / ||A x_k||; converges to the dominant real
// eigenvalue if it is simple and strictly dominant in modulus.
PowerIterationResult power_iteration(const Matrix& A, const Vec& x0, double tol, int max_iter);

// Inverse iteration with shift sigma: x_{k+1} = (A - sigma I)^{-1} x_k (LU once).
PowerIterationResult inverse_iteration(const Matrix& A, double sigma, const Vec& x0, double tol, int max_iter);

struct GershgorinDisc {
    double center;
    double radius;
};
// Row discs {z : |z - a_ii| <= sum_{j != i} |a_ij|}.
std::vector<GershgorinDisc> gershgorin_discs(const Matrix& A);

}  // namespace ak::num
