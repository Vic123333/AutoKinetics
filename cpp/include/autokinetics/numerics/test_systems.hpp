// Built-in ODE systems for testing and analysis, and an adapter that calls
// back into Python (used for user-supplied right-hand sides).
#pragma once

#include <stdexcept>
#include <vector>

#include "autokinetics/numerics/matrix.hpp"
#include "autokinetics/ode_system.hpp"

namespace ak::num {

// y' = A y
class LinearSystem final : public OdeSystem {
public:
    explicit LinearSystem(Matrix A);
    std::size_t size() const override { return A_.rows; }
    void rhs(double t, const double* y, double* dydt) override;
    bool has_jacobian() const override { return true; }
    void jacobian(double t, const double* y, double* J) override;

private:
    Matrix A_;
};

// Robertson (1966) stiff chemical kinetics:
//   A -> B (k1),  B + B -> C + B (k3),  B + C -> A + C (k2)
//   y1' = -k1 y1 + k2 y2 y3
//   y2' =  k1 y1 - k2 y2 y3 - k3 y2^2
//   y3' =  k3 y2^2
class RobertsonSystem final : public OdeSystem {
public:
    RobertsonSystem(double k1 = 0.04, double k2 = 1e4, double k3 = 3e7);
    std::size_t size() const override { return 3; }
    void rhs(double t, const double* y, double* dydt) override;
    bool has_jacobian() const override { return true; }
    void jacobian(double t, const double* y, double* J) override;
    const unsigned char* nonnegative_mask() const override { return mask_; }

private:
    double k1_, k2_, k3_;
    unsigned char mask_[3] = {1, 1, 1};
};

// Van der Pol oscillator: y1' = y2, y2' = mu ((1 - y1^2) y2 - y1).
class VanDerPolSystem final : public OdeSystem {
public:
    explicit VanDerPolSystem(double mu);
    std::size_t size() const override { return 2; }
    void rhs(double t, const double* y, double* dydt) override;
    bool has_jacobian() const override { return true; }
    void jacobian(double t, const double* y, double* J) override;

private:
    double mu_;
};

// Thrown when a callback reports an error; the binding layer re-raises the
// original (stored) Python exception.
class CallbackError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Callback signature: returns 0 on success, non-zero if an exception occurred.
// For the Jacobian, `out` has n*n entries (row-major).
typedef int (*ode_callback_fn)(void* ctx, double t, const double* y, double* out, std::size_t n);

class CallbackSystem final : public OdeSystem {
public:
    CallbackSystem(std::size_t n, void* ctx, ode_callback_fn rhs_fn, ode_callback_fn jac_fn);
    std::size_t size() const override { return n_; }
    void rhs(double t, const double* y, double* dydt) override;
    bool has_jacobian() const override { return jac_fn_ != nullptr; }
    void jacobian(double t, const double* y, double* J) override;
    const unsigned char* nonnegative_mask() const override { return mask_.empty() ? nullptr : mask_.data(); }
    void set_nonnegative_mask(const std::vector<unsigned char>& mask);

private:
    std::size_t n_;
    void* ctx_;
    ode_callback_fn rhs_fn_;
    ode_callback_fn jac_fn_;
    std::vector<unsigned char> mask_;
};

}  // namespace ak::num
