// Eigenvalues of real nonsymmetric matrices (Numerik I, Kap. 9): balancing,
// Householder-Hessenberg reduction, Francis double-shift QR algorithm, vector
// iteration, inverse iteration, Gershgorin discs.
#include <algorithm>
#include <cmath>

#include "autokinetics/numerics/eigen.hpp"
#include "autokinetics/numerics/linalg.hpp"

namespace ak::num {

Matrix balance(const Matrix& A) {
    require_square(A, "balance");
    const std::size_t n = A.rows;
    Matrix B = A;
    const double radix = 2.0, sqrdx = radix * radix;
    bool done = false;
    int sweeps = 0;
    while (!done && sweeps < 100) {
        done = true;
        ++sweeps;
        for (std::size_t i = 0; i < n; ++i) {
            double r = 0.0, c = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                c += std::abs(B(j, i));
                r += std::abs(B(i, j));
            }
            if (c == 0.0 || r == 0.0) continue;
            double g = r / radix, f = 1.0;
            const double s = c + r;
            while (c < g) {
                f *= radix;
                c *= sqrdx;
            }
            g = r * radix;
            while (c > g) {
                f /= radix;
                c /= sqrdx;
            }
            if ((c + r) / f < 0.95 * s) {
                done = false;
                const double ginv = 1.0 / f;
                for (std::size_t j = 0; j < n; ++j) B(i, j) *= ginv;
                for (std::size_t j = 0; j < n; ++j) B(j, i) *= f;
            }
        }
    }
    return B;
}

Matrix hessenberg(const Matrix& A) {
    require_square(A, "hessenberg");
    const std::size_t n = A.rows;
    Matrix H = A;
    Vec v(n);
    for (std::size_t k = 0; k + 2 < n; ++k) {
        std::fill(v.begin(), v.end(), 0.0);
        for (std::size_t i = k + 1; i < n; ++i) v[i] = H(i, k);
        const double nx = norm2(v);
        if (nx == 0.0) continue;
        const double alpha = (H(k + 1, k) >= 0.0) ? -nx : nx;
        v[k + 1] -= alpha;
        const double nv = norm2(v);
        if (nv == 0.0) continue;
        for (std::size_t i = k + 1; i < n; ++i) v[i] /= nv;
        // H <- P H, P = I - 2 v v^T (rows k+1..n-1)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t i = k + 1; i < n; ++i) s += v[i] * H(i, j);
            s *= 2.0;
            for (std::size_t i = k + 1; i < n; ++i) H(i, j) -= s * v[i];
        }
        // H <- H P (columns k+1..n-1)
        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = k + 1; j < n; ++j) s += H(i, j) * v[j];
            s *= 2.0;
            for (std::size_t j = k + 1; j < n; ++j) H(i, j) -= s * v[j];
        }
        for (std::size_t i = k + 2; i < n; ++i) H(i, k) = 0.0;
    }
    return H;
}

EigenResult eigenvalues(const Matrix& A, bool do_balance) {
    require_square(A, "eigenvalues");
    const int n = static_cast<int>(A.rows);
    EigenResult res;
    res.re.assign(n, 0.0);
    res.im.assign(n, 0.0);
    if (!std::isfinite(matrix_max_abs(A))) {
        res.message = "matrix contains non-finite entries";
        return res;
    }
    Matrix H = hessenberg(do_balance ? balance(A) : A);
    // 1-based accessor for the classical EISPACK/hqr formulation.
    auto a = [&](int i, int j) -> double& { return H(static_cast<std::size_t>(i - 1), static_cast<std::size_t>(j - 1)); };
    double* wr = res.re.data() - 1;
    double* wi = res.im.data() - 1;

    double anorm = 0.0;
    for (int i = 1; i <= n; ++i)
        for (int j = std::max(i - 1, 1); j <= n; ++j) anorm += std::abs(a(i, j));
    int nn = n, l = 1;
    double t = 0.0, p = 0.0, q = 0.0, r = 0.0, s = 0.0, w = 0.0, x = 0.0, y = 0.0, z = 0.0;
    const int max_its = 60;
    while (nn >= 1) {
        int its = 0;
        do {
            for (l = nn; l >= 2; --l) {
                s = std::abs(a(l - 1, l - 1)) + std::abs(a(l, l));
                if (s == 0.0) s = anorm;
                if (std::abs(a(l, l - 1)) + s == s) {
                    a(l, l - 1) = 0.0;
                    break;
                }
            }
            x = a(nn, nn);
            if (l == nn) {  // one root found
                wr[nn] = x + t;
                wi[nn--] = 0.0;
            } else {
                y = a(nn - 1, nn - 1);
                w = a(nn, nn - 1) * a(nn - 1, nn);
                if (l == nn - 1) {  // two roots found
                    p = 0.5 * (y - x);
                    q = p * p + w;
                    z = std::sqrt(std::abs(q));
                    x += t;
                    if (q >= 0.0) {
                        z = p + std::copysign(z, p);
                        wr[nn - 1] = wr[nn] = x + z;
                        if (z != 0.0) wr[nn] = x - w / z;
                        wi[nn - 1] = wi[nn] = 0.0;
                    } else {
                        wr[nn - 1] = wr[nn] = x + p;
                        wi[nn - 1] = -(wi[nn] = z);
                    }
                    nn -= 2;
                } else {
                    if (its == max_its) {
                        res.message = "QR algorithm did not converge within " + std::to_string(max_its) +
                                      " iterations for an eigenvalue";
                        return res;
                    }
                    if (its == 10 || its == 20 || its == 40) {  // exceptional shift
                        t += x;
                        for (int i = 1; i <= nn; ++i) a(i, i) -= x;
                        s = std::abs(a(nn, nn - 1)) + std::abs(a(nn - 1, nn - 2));
                        y = x = 0.75 * s;
                        w = -0.4375 * s * s;
                    }
                    ++its;
                    ++res.iterations;
                    int m = nn - 2;
                    for (; m >= l; --m) {
                        z = a(m, m);
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / a(m + 1, m) + a(m, m + 1);
                        q = a(m + 1, m + 1) - z - r - s;
                        r = a(m + 2, m + 1);
                        s = std::abs(p) + std::abs(q) + std::abs(r);
                        p /= s;
                        q /= s;
                        r /= s;
                        if (m == l) break;
                        const double u = std::abs(a(m, m - 1)) * (std::abs(q) + std::abs(r));
                        const double v = std::abs(p) * (std::abs(a(m - 1, m - 1)) + std::abs(z) + std::abs(a(m + 1, m + 1)));
                        if (u + v == v) break;
                    }
                    for (int i = m + 2; i <= nn; ++i) {
                        a(i, i - 2) = 0.0;
                        if (i != m + 2) a(i, i - 3) = 0.0;
                    }
                    for (int k = m; k <= nn - 1; ++k) {  // double-shift QR sweep (bulge chase)
                        if (k != m) {
                            p = a(k, k - 1);
                            q = a(k + 1, k - 1);
                            r = 0.0;
                            if (k != nn - 1) r = a(k + 2, k - 1);
                            if ((x = std::abs(p) + std::abs(q) + std::abs(r)) != 0.0) {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        if ((s = std::copysign(std::sqrt(p * p + q * q + r * r), p)) != 0.0) {
                            if (k == m) {
                                if (l != m) a(k, k - 1) = -a(k, k - 1);
                            } else {
                                a(k, k - 1) = -s * x;
                            }
                            p += s;
                            x = p / s;
                            y = q / s;
                            z = r / s;
                            q /= p;
                            r /= p;
                            for (int j = k; j <= nn; ++j) {
                                p = a(k, j) + q * a(k + 1, j);
                                if (k != nn - 1) {
                                    p += r * a(k + 2, j);
                                    a(k + 2, j) -= p * z;
                                }
                                a(k + 1, j) -= p * y;
                                a(k, j) -= p * x;
                            }
                            const int mmin = nn < k + 3 ? nn : k + 3;
                            for (int i = l; i <= mmin; ++i) {
                                p = x * a(i, k) + y * a(i, k + 1);
                                if (k != nn - 1) {
                                    p += z * a(i, k + 2);
                                    a(i, k + 2) -= p * r;
                                }
                                a(i, k + 1) -= p * q;
                                a(i, k) -= p;
                            }
                        }
                    }
                }
            }
        } while (l < nn - 1);
    }
    res.ok = true;
    return res;
}

namespace {

double normalize(Vec& x) {
    const double nx = norm2(x);
    if (nx > 0.0)
        for (double& v : x) v /= nx;
    return nx;
}

double rayleigh(const Matrix& A, const Vec& x) {
    const Vec Ax = matvec(A, x);
    return dot(x.data(), Ax.data(), x.size()) / dot(x.data(), x.data(), x.size());
}

Vec start_vector(const Vec& x0, std::size_t n) {
    if (!x0.empty()) {
        if (x0.size() != n) throw std::invalid_argument("start vector has wrong length");
        return x0;
    }
    Vec x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = 1.0 + 0.1 * static_cast<double>(i % 7);  // generic, non-symmetric start
    return x;
}

}  // namespace

PowerIterationResult power_iteration(const Matrix& A, const Vec& x0, double tol, int max_iter) {
    require_square(A, "power_iteration");
    PowerIterationResult res;
    Vec x = start_vector(x0, A.rows);
    if (normalize(x) == 0.0) throw std::invalid_argument("power_iteration: start vector is zero");
    double lambda = rayleigh(A, x);
    for (int it = 1; it <= max_iter; ++it) {
        Vec y = matvec(A, x);
        if (normalize(y) == 0.0) {
            res.eigenvalue = 0.0;
            res.eigenvector = x;
            res.iterations = it;
            res.converged = true;  // x lies in the null space: A x = 0
            return res;
        }
        const double lambda_new = rayleigh(A, y);
        // residual ||A y - lambda y|| measures convergence of the eigenpair
        Vec r = matvec(A, y);
        for (std::size_t i = 0; i < r.size(); ++i) r[i] -= lambda_new * y[i];
        x = y;
        lambda = lambda_new;
        res.iterations = it;
        if (norm2(r) <= tol * std::max(std::abs(lambda), 1e-300)) {
            res.converged = true;
            break;
        }
    }
    res.eigenvalue = lambda;
    res.eigenvector = x;
    if (!res.converged)
        res.message = "no convergence within " + std::to_string(max_iter) +
                      " iterations (dominant eigenvalue not unique in modulus, e.g. a complex pair?)";
    return res;
}

PowerIterationResult inverse_iteration(const Matrix& A, double sigma, const Vec& x0, double tol, int max_iter) {
    require_square(A, "inverse_iteration");
    const std::size_t n = A.rows;
    Matrix S = A;
    for (std::size_t i = 0; i < n; ++i) S(i, i) -= sigma;
    LUFactorization f = lu_factor(S, true, 0.0);
    PowerIterationResult res;
    Vec x = start_vector(x0, n);
    normalize(x);
    if (!f.ok) {
        // sigma is (numerically) an eigenvalue: return it with a null vector of A - sigma I
        res.eigenvalue = sigma;
        res.eigenvector = x;
        res.converged = false;
        res.message = "shift is an eigenvalue to working precision (A - sigma I singular)";
        return res;
    }
    double lambda = sigma;
    for (int it = 1; it <= max_iter; ++it) {
        lu_solve_inplace(f, x.data());
        if (!std::isfinite(normalize(x))) {
            res.message = "non-finite values in inverse iteration";
            res.iterations = it;
            return res;
        }
        lambda = rayleigh(A, x);
        Vec r = matvec(A, x);
        for (std::size_t i = 0; i < n; ++i) r[i] -= lambda * x[i];
        res.iterations = it;
        if (norm2(r) <= tol * std::max(std::abs(lambda), 1.0)) {
            res.converged = true;
            break;
        }
    }
    res.eigenvalue = lambda;
    res.eigenvector = x;
    if (!res.converged)
        res.message = "no convergence within " + std::to_string(max_iter) + " iterations";
    return res;
}

std::vector<GershgorinDisc> gershgorin_discs(const Matrix& A) {
    require_square(A, "gershgorin_discs");
    std::vector<GershgorinDisc> discs(A.rows);
    for (std::size_t i = 0; i < A.rows; ++i) {
        double r = 0.0;
        for (std::size_t j = 0; j < A.cols; ++j)
            if (j != i) r += std::abs(A(i, j));
        discs[i] = {A(i, i), r};
    }
    return discs;
}

}  // namespace ak::num
