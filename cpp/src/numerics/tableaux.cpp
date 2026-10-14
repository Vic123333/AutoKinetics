// Butcher tableaux in closed form (ODE script Kap. 3 and 5; the collocation
// tableaux are cross-checked in Python against Gauß/Radau/Lobatto quadrature).
#include <cmath>
#include <stdexcept>

#include "autokinetics/numerics/integrator.hpp"

namespace ak::num {

bool ButcherTableau::is_explicit() const {
    for (std::size_t i = 0; i < s; ++i)
        for (std::size_t j = i; j < s; ++j)
            if (A[i * s + j] != 0.0) return false;
    return true;
}

void ButcherTableau::validate() const {
    if (s == 0) throw std::invalid_argument("tableau '" + name + "': no stages");
    if (A.size() != s * s || b.size() != s || c.size() != s)
        throw std::invalid_argument("tableau '" + name + "': A must be s x s, b and c of length s");
    if (!bhat.empty() && bhat.size() != s)
        throw std::invalid_argument("tableau '" + name + "': bhat must have length s");
    if (order < 1) throw std::invalid_argument("tableau '" + name + "': order must be >= 1");
    if (!bhat.empty() && order_hat < 1)
        throw std::invalid_argument("tableau '" + name + "': embedded order must be >= 1");
    for (double v : A)
        if (!std::isfinite(v)) throw std::invalid_argument("tableau '" + name + "': non-finite entry in A");
}

namespace {

ButcherTableau make(const std::string& name, std::size_t s, std::vector<double> A, std::vector<double> b,
                    std::vector<double> c, int order, std::vector<double> bhat = {}, int order_hat = 0,
                    bool fsal = false) {
    ButcherTableau t;
    t.name = name;
    t.s = s;
    t.A = std::move(A);
    t.b = std::move(b);
    t.c = std::move(c);
    t.order = order;
    t.bhat = std::move(bhat);
    t.order_hat = order_hat;
    t.fsal = fsal;
    t.validate();
    return t;
}

}  // namespace

std::vector<std::string> builtin_tableau_names() {
    return {"euler",   "heun",   "rk4",    "dopri5", "implicit_euler", "implicit_midpoint", "trapezoidal",
            "gauss4", "radau3", "radau5", "sdirk2", "sdirk3"};
}

ButcherTableau builtin_tableau(const std::string& name) {
    if (name == "euler") return make(name, 1, {0.0}, {1.0}, {0.0}, 1);
    if (name == "heun")  // Heun-Euler 2(1), explicit Euler embedded
        return make(name, 2, {0.0, 0.0, 1.0, 0.0}, {0.5, 0.5}, {0.0, 1.0}, 2, {1.0, 0.0}, 1);
    if (name == "rk4")
        return make(name, 4,
                    {0, 0, 0, 0,  //
                     0.5, 0, 0, 0,  //
                     0, 0.5, 0, 0,  //
                     0, 0, 1, 0},
                    {1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6}, {0.0, 0.5, 0.5, 1.0}, 4);
    if (name == "dopri5")
        return make(name, 7,
                    {0, 0, 0, 0, 0, 0, 0,  //
                     1.0 / 5, 0, 0, 0, 0, 0, 0,  //
                     3.0 / 40, 9.0 / 40, 0, 0, 0, 0, 0,  //
                     44.0 / 45, -56.0 / 15, 32.0 / 9, 0, 0, 0, 0,  //
                     19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729, 0, 0, 0,  //
                     9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656, 0, 0,  //
                     35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0},
                    {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0},
                    {0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1}, 5,
                    {5179.0 / 57600, 0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200, 187.0 / 2100, 1.0 / 40}, 4,
                    true);
    if (name == "implicit_euler") return make(name, 1, {1.0}, {1.0}, {1.0}, 1);
    if (name == "implicit_midpoint") return make(name, 1, {0.5}, {1.0}, {0.5}, 2);  // Gauss-Legendre s=1
    if (name == "trapezoidal")  // Lobatto IIIA s=2
        return make(name, 2, {0.0, 0.0, 0.5, 0.5}, {0.5, 0.5}, {0.0, 1.0}, 2);
    if (name == "gauss4") {  // Gauss-Legendre s=2
        const double r = std::sqrt(3.0) / 6.0;
        return make(name, 2, {0.25, 0.25 - r, 0.25 + r, 0.25}, {0.5, 0.5}, {0.5 - r, 0.5 + r}, 4);
    }
    if (name == "radau3")  // Radau IIA s=2
        return make(name, 2, {5.0 / 12, -1.0 / 12, 0.75, 0.25}, {0.75, 0.25}, {1.0 / 3, 1.0}, 3);
    if (name == "radau5") {  // Radau IIA s=3
        const double q = std::sqrt(6.0);
        return make(name, 3,
                    {(88 - 7 * q) / 360, (296 - 169 * q) / 1800, (-2 + 3 * q) / 225,  //
                     (296 + 169 * q) / 1800, (88 + 7 * q) / 360, (-2 - 3 * q) / 225,  //
                     (16 - q) / 36, (16 + q) / 36, 1.0 / 9},
                    {(16 - q) / 36, (16 + q) / 36, 1.0 / 9}, {(4 - q) / 10, (4 + q) / 10, 1.0}, 5);
    }
    if (name == "sdirk2") {  // Alexander: L-stable, stiffly accurate, order 2
        const double g = 1.0 - std::sqrt(2.0) / 2.0;
        return make(name, 2, {g, 0.0, 1.0 - g, g}, {1.0 - g, g}, {g, 1.0}, 2);
    }
    if (name == "sdirk3") {  // Crouzeix/Norsett: A-stable, order 3
        const double g = (3.0 + std::sqrt(3.0)) / 6.0;
        return make(name, 2, {g, 0.0, 1.0 - 2.0 * g, g}, {0.5, 0.5}, {g, 1.0 - g}, 3);
    }
    throw std::invalid_argument("unknown Butcher tableau '" + name + "'");
}

Method method_from_tableau(const ButcherTableau& tableau) {
    tableau.validate();
    Method m;
    m.kind = tableau.is_explicit() ? MethodKind::ExplicitRK : MethodKind::ImplicitRK;
    m.tableau = tableau;
    m.name = tableau.name;
    m.order = tableau.order;
    return m;
}

Method builtin_method(const std::string& name) {
    if (name == "ros2") {
        Method m;
        m.kind = MethodKind::Rosenbrock2;
        m.name = name;
        m.order = 2;
        return m;
    }
    if (name == "bdf2") {
        Method m;
        m.kind = MethodKind::BDF2;
        m.name = name;
        m.order = 2;
        return m;
    }
    return method_from_tableau(builtin_tableau(name));
}

NonlinearSolverKind nonlinear_solver_from_string(const std::string& name) {
    if (name == "full_newton") return NonlinearSolverKind::FullNewton;
    if (name == "simplified_newton") return NonlinearSolverKind::SimplifiedNewton;
    if (name == "fixed_point") return NonlinearSolverKind::FixedPoint;
    throw std::invalid_argument("unknown nonlinear solver '" + name + "'");
}

std::string to_string(NonlinearSolverKind kind) {
    switch (kind) {
        case NonlinearSolverKind::FullNewton: return "full_newton";
        case NonlinearSolverKind::SimplifiedNewton: return "simplified_newton";
        case NonlinearSolverKind::FixedPoint: return "fixed_point";
    }
    return "unknown";
}

std::string to_string(Status status) {
    switch (status) {
        case Status::Success: return "success";
        case Status::MaxSteps: return "max_steps";
        case Status::WallTime: return "wall_time";
        case Status::StepSizeTooSmall: return "step_size_too_small";
        case Status::NonlinearSolverFailure: return "nonlinear_solver_failure";
        case Status::NonFinite: return "non_finite";
    }
    return "unknown";
}

}  // namespace ak::num
