// Contract between the kinetics layer (reactor models) and the numerics layer
// (integrators, Newton solvers, linear algebra, stability analysis).
//
// Every right-hand side y' = f(t, y) that the C++ integrators can solve
// implements this interface. Reactor models (cpp/src/kinetics) and test
// problems / Python callbacks (cpp/src/numerics) both derive from it.
#pragma once

#include <cstddef>

namespace ak {

class OdeSystem {
public:
    virtual ~OdeSystem() = default;

    // Dimension n of the state vector.
    virtual std::size_t size() const = 0;

    // dydt = f(t, y); both arrays have length size().
    virtual void rhs(double t, const double* y, double* dydt) = 0;

    // True if jacobian() returns an analytic (or system-provided) Jacobian.
    // If false, integrators must build J by finite differences of rhs().
    virtual bool has_jacobian() const { return false; }

    // J = df/dy at (t, y), dense, row-major: J[i * n + j] = d f_i / d y_j.
    virtual void jacobian(double /*t*/, const double* /*y*/, double* /*J*/) {}

    // Components that are physically non-negative (concentrations, amounts).
    // Integrators may use this to reject steps or clip tiny negative round-off.
    // Default: none. Returns nullptr or an array of length size() with 0/1.
    virtual const unsigned char* nonnegative_mask() const { return nullptr; }
};

}  // namespace ak
