// Dense row-major matrices and vector helpers for the numerics layer.
//
// The numerics library targets small to medium systems (n up to a few hundred)
// as they occur in chemical kinetics; everything is dense and self-contained.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ak::num {

using Vec = std::vector<double>;

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;  // row-major: (i, j) -> data[i * cols + j]

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c, double value = 0.0) : rows(r), cols(c), data(r * c, value) {}
    Matrix(std::size_t r, std::size_t c, const double* src) : rows(r), cols(c), data(src, src + r * c) {}

    static Matrix identity(std::size_t n);

    double& operator()(std::size_t i, std::size_t j) { return data[i * cols + j]; }
    double operator()(std::size_t i, std::size_t j) const { return data[i * cols + j]; }
    double* row(std::size_t i) { return data.data() + i * cols; }
    const double* row(std::size_t i) const { return data.data() + i * cols; }
    bool is_square() const { return rows == cols; }
};

// Throws std::invalid_argument unless A is square.
void require_square(const Matrix& A, const char* what);

double dot(const double* x, const double* y, std::size_t n);
double norm2(const Vec& x);
double norm1(const Vec& x);
double norm_inf(const Vec& x);

double matrix_norm_1(const Matrix& A);          // max column sum
double matrix_norm_inf(const Matrix& A);        // max row sum
double matrix_norm_frobenius(const Matrix& A);
double matrix_max_abs(const Matrix& A);

Vec matvec(const Matrix& A, const Vec& x);      // A x
Vec matTvec(const Matrix& A, const Vec& x);     // A^T x
Matrix matmul(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& A);

}  // namespace ak::num
