#include "autokinetics/numerics/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace ak::num {

Matrix Matrix::identity(std::size_t n) {
    Matrix I(n, n);
    for (std::size_t i = 0; i < n; ++i) I(i, i) = 1.0;
    return I;
}

void require_square(const Matrix& A, const char* what) {
    if (!A.is_square() || A.rows == 0) {
        throw std::invalid_argument(std::string(what) + ": matrix must be square and non-empty");
    }
}

double dot(const double* x, const double* y, std::size_t n) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) s += x[i] * y[i];
    return s;
}

double norm2(const Vec& x) {
    // Scaled to avoid overflow/underflow for extreme magnitudes.
    double scale = 0.0;
    for (double v : x) scale = std::max(scale, std::abs(v));
    if (scale == 0.0 || !std::isfinite(scale)) return scale;
    double s = 0.0;
    for (double v : x) {
        const double r = v / scale;
        s += r * r;
    }
    return scale * std::sqrt(s);
}

double norm1(const Vec& x) {
    double s = 0.0;
    for (double v : x) s += std::abs(v);
    return s;
}

double norm_inf(const Vec& x) {
    double s = 0.0;
    for (double v : x) {
        if (std::isnan(v)) return v;
        s = std::max(s, std::abs(v));
    }
    return s;
}

double matrix_norm_1(const Matrix& A) {
    double best = 0.0;
    for (std::size_t j = 0; j < A.cols; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < A.rows; ++i) s += std::abs(A(i, j));
        best = std::max(best, s);
    }
    return best;
}

double matrix_norm_inf(const Matrix& A) {
    double best = 0.0;
    for (std::size_t i = 0; i < A.rows; ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < A.cols; ++j) s += std::abs(A(i, j));
        best = std::max(best, s);
    }
    return best;
}

double matrix_norm_frobenius(const Matrix& A) { return norm2(A.data); }

double matrix_max_abs(const Matrix& A) { return norm_inf(A.data); }

Vec matvec(const Matrix& A, const Vec& x) {
    Vec y(A.rows, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) y[i] = dot(A.row(i), x.data(), A.cols);
    return y;
}

Vec matTvec(const Matrix& A, const Vec& x) {
    Vec y(A.cols, 0.0);
    for (std::size_t i = 0; i < A.rows; ++i) {
        const double xi = x[i];
        const double* r = A.row(i);
        for (std::size_t j = 0; j < A.cols; ++j) y[j] += r[j] * xi;
    }
    return y;
}

Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) throw std::invalid_argument("matmul: inner dimensions differ");
    Matrix C(A.rows, B.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        double* c = C.row(i);
        for (std::size_t k = 0; k < A.cols; ++k) {
            const double a = A(i, k);
            if (a == 0.0) continue;
            const double* b = B.row(k);
            for (std::size_t j = 0; j < B.cols; ++j) c[j] += a * b[j];
        }
    }
    return C;
}

Matrix transpose(const Matrix& A) {
    Matrix T(A.cols, A.rows);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < A.cols; ++j) T(j, i) = A(i, j);
    return T;
}

}  // namespace ak::num
