#include "autokinetics/numerics/test_systems.hpp"

#include <algorithm>

namespace ak::num {

LinearSystem::LinearSystem(Matrix A) : A_(std::move(A)) { require_square(A_, "LinearSystem"); }

void LinearSystem::rhs(double /*t*/, const double* y, double* dydt) {
    const std::size_t n = A_.rows;
    for (std::size_t i = 0; i < n; ++i) dydt[i] = dot(A_.row(i), y, n);
}

void LinearSystem::jacobian(double /*t*/, const double* /*y*/, double* J) {
    std::copy(A_.data.begin(), A_.data.end(), J);
}

RobertsonSystem::RobertsonSystem(double k1, double k2, double k3) : k1_(k1), k2_(k2), k3_(k3) {}

void RobertsonSystem::rhs(double /*t*/, const double* y, double* d) {
    const double r1 = k1_ * y[0], r2 = k2_ * y[1] * y[2], r3 = k3_ * y[1] * y[1];
    d[0] = -r1 + r2;
    d[1] = r1 - r2 - r3;
    d[2] = r3;
}

void RobertsonSystem::jacobian(double /*t*/, const double* y, double* J) {
    J[0] = -k1_;
    J[1] = k2_ * y[2];
    J[2] = k2_ * y[1];
    J[3] = k1_;
    J[4] = -k2_ * y[2] - 2.0 * k3_ * y[1];
    J[5] = -k2_ * y[1];
    J[6] = 0.0;
    J[7] = 2.0 * k3_ * y[1];
    J[8] = 0.0;
}

VanDerPolSystem::VanDerPolSystem(double mu) : mu_(mu) {}

void VanDerPolSystem::rhs(double /*t*/, const double* y, double* d) {
    d[0] = y[1];
    d[1] = mu_ * ((1.0 - y[0] * y[0]) * y[1] - y[0]);
}

void VanDerPolSystem::jacobian(double /*t*/, const double* y, double* J) {
    J[0] = 0.0;
    J[1] = 1.0;
    J[2] = mu_ * (-2.0 * y[0] * y[1] - 1.0);
    J[3] = mu_ * (1.0 - y[0] * y[0]);
}

CallbackSystem::CallbackSystem(std::size_t n, void* ctx, ode_callback_fn rhs_fn, ode_callback_fn jac_fn)
    : n_(n), ctx_(ctx), rhs_fn_(rhs_fn), jac_fn_(jac_fn) {
    if (n == 0) throw std::invalid_argument("CallbackSystem: dimension must be positive");
    if (rhs_fn == nullptr) throw std::invalid_argument("CallbackSystem: rhs callback required");
}

void CallbackSystem::rhs(double t, const double* y, double* dydt) {
    if (rhs_fn_(ctx_, t, y, dydt, n_) != 0) throw CallbackError("exception in Python rhs callback");
}

void CallbackSystem::jacobian(double t, const double* y, double* J) {
    if (jac_fn_ == nullptr) throw std::logic_error("CallbackSystem: no Jacobian callback");
    if (jac_fn_(ctx_, t, y, J, n_) != 0) throw CallbackError("exception in Python jacobian callback");
}

void CallbackSystem::set_nonnegative_mask(const std::vector<unsigned char>& mask) {
    if (!mask.empty() && mask.size() != n_) throw std::invalid_argument("nonnegative mask has wrong length");
    mask_ = mask;
}

}  // namespace ak::num
