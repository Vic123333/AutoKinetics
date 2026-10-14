// Time integrators for y' = f(t, y) on ak::OdeSystem:
//  * explicit Runge-Kutta from a Butcher tableau (fixed step, embedded pair or
//    step doubling),
//  * implicit Runge-Kutta from a tableau (stage system in z-formulation solved
//    by full / simplified Newton or fixed-point iteration),
//  * linearly implicit Rosenbrock ROS2,
//  * variable-step BDF2.
#pragma once

#include <limits>
#include <string>
#include <vector>

#include "autokinetics/numerics/linear_solver.hpp"
#include "autokinetics/numerics/matrix.hpp"
#include "autokinetics/ode_system.hpp"

namespace ak::num {

struct ButcherTableau {
    std::string name;
    std::size_t s = 0;
    std::vector<double> A;     // s x s row-major
    std::vector<double> b;     // weights (propagated solution)
    std::vector<double> c;     // nodes
    std::vector<double> bhat;  // embedded weights (empty if none)
    int order = 0;             // classical order of (A, b, c)
    int order_hat = 0;         // order of (A, bhat, c); 0 if no embedded pair
    bool fsal = false;         // last stage equals f(t+h, y_{n+1})

    bool is_explicit() const;  // A strictly lower triangular
    void validate() const;     // throws std::invalid_argument on inconsistent data
};

// Built-in tableaux: euler, heun, rk4, dopri5, implicit_euler,
// implicit_midpoint, trapezoidal, gauss4, radau3, radau5, sdirk2, sdirk3.
ButcherTableau builtin_tableau(const std::string& name);
std::vector<std::string> builtin_tableau_names();

enum class MethodKind { ExplicitRK, ImplicitRK, Rosenbrock2, BDF2 };

struct Method {
    MethodKind kind = MethodKind::ExplicitRK;
    ButcherTableau tableau;  // used by the Runge-Kutta kinds
    std::string name;
    int order = 0;
};

// Built-in methods: every tableau name plus "ros2" and "bdf2".
Method builtin_method(const std::string& name);
// A Runge-Kutta method from a user-supplied tableau (explicit or implicit).
Method method_from_tableau(const ButcherTableau& tableau);

enum class NonlinearSolverKind { FullNewton, SimplifiedNewton, FixedPoint };
NonlinearSolverKind nonlinear_solver_from_string(const std::string& name);
std::string to_string(NonlinearSolverKind kind);

struct Options {
    double rtol = 1e-6;
    std::vector<double> atol{1e-12};  // length 1 (scalar) or n
    double h0 = 0.0;                  // initial step (0: automatic)
    double h_min = 0.0;               // smallest admissible step (0: 16 eps |t|)
    double h_max = std::numeric_limits<double>::infinity();
    double fixed_step = 0.0;          // > 0: constant step size, no error control
    long max_steps = 100000;
    double max_wall_time = 0.0;       // seconds; 0: unlimited
    std::vector<double> t_eval;       // output times (sorted, within [t0, tf]); empty: every step
    bool exact_t_eval = false;        // adaptive mode: steps land exactly on t_eval (no interpolation)
    bool nonnegative = false;         // enforce the system's nonnegative_mask()

    NonlinearSolverKind nonlinear = NonlinearSolverKind::SimplifiedNewton;
    LinearSolverKind linear = LinearSolverKind::LUPivot;
    double newton_tol = 0.03;   // kappa: stop when theta/(1-theta) ||dZ|| <= kappa (scaled norm)
    int max_newton_iter = 0;    // 0: 10 (Newton) / 50 (fixed point)
    double linear_tol = 1e-12;  // relative residual for iterative linear solvers
    int linear_max_iter = 0;    // 0: automatic

    double safety = 0.9;
    double facmin = 0.2;
    double facmax = 5.0;
};

struct Stats {
    long n_steps = 0;            // attempted steps (incl. rejected)
    long n_accepted = 0;
    long n_rejected = 0;         // error test failures + solver failures
    long n_error_rejections = 0; // rejected by the error estimator
    long n_solver_rejections = 0;// rejected because Newton / linear solver failed
    long n_nonneg_rejections = 0;
    long n_rhs = 0;              // f evaluations (incl. finite-difference Jacobians)
    long n_jac = 0;              // Jacobian evaluations (analytic or finite differences)
    long n_jac_fd = 0;           // ... of which by finite differences
    long n_factorizations = 0;
    long n_linear_solves = 0;
    long n_linear_iterations = 0;// iterations of iterative linear solvers
    long n_linear_failures = 0;
    long n_newton_iterations = 0;// nonlinear iterations (Newton or fixed point)
    long n_newton_failures = 0;
    double theta_max = 0.0;      // largest observed contraction factor
    double theta_sum = 0.0;
    long n_theta = 0;
    double h_min_used = 0.0;     // over accepted steps
    double h_max_used = 0.0;
    double h_sum = 0.0;
};

enum class Status {
    Success,
    MaxSteps,
    WallTime,
    StepSizeTooSmall,
    NonlinearSolverFailure,  // fixed-step mode only (adaptive mode retries)
    NonFinite,
};
std::string to_string(Status status);

struct Result {
    std::vector<double> t;
    std::vector<double> y;  // row-major, t.size() x n
    std::size_t n = 0;
    Status status = Status::Success;
    std::string message;
    Stats stats;
    bool success() const { return status == Status::Success; }
};

// Integrates from t0 to tf (> t0). Python/C++ exceptions thrown by the system
// propagate unchanged; invalid input throws std::invalid_argument.
Result integrate(OdeSystem& system, const Method& method, double t0, double tf, const Vec& y0, const Options& opt);

// Dense Jacobian J = df/dy (row-major) at (t, y): analytic if the system has
// one, otherwise forward differences with increments sqrt(eps * max(1e-5, |y_j|)).
// f0 = f(t, y) may be passed to save one evaluation (nullptr: computed).
// Returns the number of extra rhs evaluations.
long evaluate_jacobian(OdeSystem& system, double t, const double* y, const double* f0, double* J);

}  // namespace ak::num
