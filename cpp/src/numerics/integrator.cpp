// Integration driver and steppers (ODE script Kap. 2-5 and 7).
//
// Driver: fixed step or adaptive step-size control with an error estimate
// that comes either from the method (embedded RK pair, Rosenbrock embedded
// solution, BDF2 predictor/corrector difference) or from step doubling
// (Richardson error estimate) for methods without their own estimator.
// Rejected steps (error test, nonlinear/linear solver failure, negativity)
// are retried with a smaller step; in fixed-step mode a solver failure ends
// the integration with a clear status instead of returning garbage.
#include "autokinetics/numerics/integrator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace ak::num {

long evaluate_jacobian(OdeSystem& system, double t, const double* y, const double* f0, double* J) {
    const std::size_t n = system.size();
    if (system.has_jacobian()) {
        system.jacobian(t, y, J);
        return 0;
    }
    long evals = 0;
    Vec base;
    if (f0 == nullptr) {
        base.resize(n);
        system.rhs(t, y, base.data());
        f0 = base.data();
        ++evals;
    }
    Vec yp(y, y + n), fp(n);
    for (std::size_t j = 0; j < n; ++j) {
        const double yj = y[j];
        const double delta0 = std::sqrt(std::numeric_limits<double>::epsilon() * std::max(1e-5, std::abs(yj)));
        yp[j] = yj + delta0;
        const double delta = yp[j] - yj;  // exactly representable increment
        system.rhs(t, yp.data(), fp.data());
        ++evals;
        for (std::size_t i = 0; i < n; ++i) J[i * n + j] = (fp[i] - f0[i]) / delta;
        yp[j] = yj;
    }
    return evals;
}

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

bool all_finite(const Vec& v) {
    for (double x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// Shared state of one integration run: system access with call counting,
// tolerances and statistics.
struct Context {
    OdeSystem& sys;
    const Options& opt;
    Stats& st;
    std::size_t n;
    Vec atol;  // expanded to length n

    void f(double t, const double* y, double* out) {
        ++st.n_rhs;
        sys.rhs(t, y, out);
    }

    void jac(double t, const double* y, const double* f0, double* J) {
        ++st.n_jac;
        if (!sys.has_jacobian()) ++st.n_jac_fd;
        st.n_rhs += evaluate_jacobian(sys, t, y, f0, J);
    }

    double scale(std::size_t i, double ya, double yb) const {
        return atol[i] + opt.rtol * std::max(std::abs(ya), std::abs(yb));
    }

    // Weighted RMS norm used by the local error test.
    double wrms(const Vec& e, const Vec& ya, const Vec& yb) const {
        double s = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double r = e[i] / scale(i, ya[i], yb[i]);
            s += r * r;
        }
        return std::sqrt(s / static_cast<double>(n));
    }

    Vec scale_vector(const Vec& y) const {
        Vec sc(n);
        for (std::size_t i = 0; i < n; ++i) sc[i] = scale(i, y[i], y[i]);
        return sc;
    }
};

// Jacobian at the start point of the current step; reused across rejected
// attempts and step-doubling sub-steps, invalidated when a step is accepted.
struct JacobianCache {
    Matrix J;
    bool valid = false;
    long version = 0;

    void ensure(Context& c, double t, const Vec& y, const Vec& fy) {
        if (valid) return;
        if (J.rows != c.n) J = Matrix(c.n, c.n);
        c.jac(t, y.data(), fy.data(), J.data.data());
        for (double v : J.data)
            if (!std::isfinite(v)) throw std::runtime_error("Jacobian contains non-finite entries");
        valid = true;
        ++version;
    }
    void invalidate() { valid = false; }
};

// A factorisation of an iteration matrix kept for reuse while its defining
// parameters (key) and the Jacobian are unchanged. Two slots allow step
// doubling (h and h/2) without refactorising every sub-step.
struct FactorSlot {
    std::unique_ptr<LinearSolver> solver;
    std::vector<double> key;
    long jac_version = -1;
    bool valid = false;
    long last_used = 0;
};

class FactorCache {
public:
    explicit FactorCache(const Options& opt) {
        for (auto& slot : slots_) slot.solver = make_linear_solver(opt.linear, opt.linear_tol, opt.linear_max_iter);
    }

    // Returns the slot for (key, jac_version); `fresh` tells whether it must be (re)factorised.
    FactorSlot& lookup(const std::vector<double>& key, long jac_version, bool& fresh) {
        ++counter_;
        for (auto& slot : slots_) {
            if (slot.valid && slot.jac_version == jac_version && slot.key == key) {
                slot.last_used = counter_;
                fresh = false;
                return slot;
            }
        }
        FactorSlot& victim = slots_[0].last_used <= slots_[1].last_used ? slots_[0] : slots_[1];
        victim.valid = false;
        victim.key = key;
        victim.jac_version = jac_version;
        victim.last_used = counter_;
        fresh = true;
        return victim;
    }

    FactorSlot& scratch() {
        slots_[0].valid = false;
        return slots_[0];
    }

private:
    FactorSlot slots_[2];
    long counter_ = 0;
};

// Factorises M into slot; returns false and sets failure on breakdown.
bool factor_into(Context& ctx, FactorSlot& slot, const Matrix& M, std::string& failure) {
    ++ctx.st.n_factorizations;
    if (!slot.solver->factor(M)) {
        ++ctx.st.n_linear_failures;
        failure = "linear solver " + to_string(ctx.opt.linear) + ": " + slot.solver->error();
        return false;
    }
    slot.valid = true;
    return true;
}

bool solve_with(Context& ctx, FactorSlot& slot, double* b, std::string& failure) {
    ++ctx.st.n_linear_solves;
    const bool ok = slot.solver->solve(b);
    ctx.st.n_linear_iterations += slot.solver->last_iterations();
    if (!ok) {
        ++ctx.st.n_linear_failures;
        failure = "linear solver " + to_string(ctx.opt.linear) + ": " + slot.solver->error();
    }
    return ok;
}

// M = I_{sn} - (hA (x) I_n) diag(J_1..J_s).
void build_stage_matrix(std::size_t s, std::size_t n, const double* hA, const std::vector<const Matrix*>& Js,
                        Matrix& M) {
    const std::size_t N = s * n;
    if (M.rows != N) M = Matrix(N, N);
    std::fill(M.data.begin(), M.data.end(), 0.0);
    for (std::size_t i = 0; i < s; ++i) {
        for (std::size_t j = 0; j < s; ++j) {
            const double a = hA[i * s + j];
            if (a == 0.0) continue;
            const Matrix& Jj = *Js[j];
            for (std::size_t p = 0; p < n; ++p) {
                double* row = M.row(i * n + p) + j * n;
                const double* jr = Jj.row(p);
                for (std::size_t q = 0; q < n; ++q) row[q] = -a * jr[q];
            }
        }
        for (std::size_t p = 0; p < n; ++p) M(i * n + p, i * n + p) += 1.0;
    }
}

// Solves the stage equations Z = (hA (x) I) F(t + hc, X + Z) (z-formulation,
// ODE script Lemma 5.5 / eq. (5.10)) by full Newton, simplified Newton
// (frozen Jacobian and factorisation) or fixed-point iteration (Banach).
// Termination as in Numerik I, Alg. 5.10: theta_k = ||dZ_k|| / ||dZ_{k-1}||,
// stop if theta_k / (1 - theta_k) ||dZ_k|| <= kappa, divergence if theta_k >= 1.
class StageSolver {
public:
    StageSolver(Context& ctx, std::size_t s, JacobianCache& jac)
        : ctx_(ctx), s_(s), n_(ctx.n), jac_(jac), cache_(ctx.opt) {
        kmax_ = ctx.opt.max_newton_iter > 0
                    ? ctx.opt.max_newton_iter
                    : (ctx.opt.nonlinear == NonlinearSolverKind::FixedPoint ? 50 : 10);
    }

    // Z: initial guess on entry, solution on success.
    bool solve(double t, const Vec& X, const Vec& scale, const std::vector<double>& hA, const std::vector<double>& hc,
               Vec& Z, std::string& failure) {
        const std::size_t N = s_ * n_;
        const auto kind = ctx_.opt.nonlinear;
        Stats& st = ctx_.st;
        F_.resize(N);
        G_.resize(N);
        Xz_.resize(n_);
        FactorSlot* slot = nullptr;
        if (kind == NonlinearSolverKind::SimplifiedNewton) {
            bool fresh = false;
            slot = &cache_.lookup(hA, jac_.version, fresh);
            if (fresh) {
                std::vector<const Matrix*> Js(s_, &jac_.J);
                build_stage_matrix(s_, n_, hA.data(), Js, M_);
                if (!factor_into(ctx_, *slot, M_, failure)) return fail(failure);
            }
        }
        double nrm_prev = 0.0, theta = 0.0, eta_used = 1.0;
        for (int k = 0; k < kmax_; ++k) {
            ++st.n_newton_iterations;
            for (std::size_t i = 0; i < s_; ++i) {
                for (std::size_t p = 0; p < n_; ++p) Xz_[p] = X[p] + Z[i * n_ + p];
                ctx_.f(t + hc[i], Xz_.data(), F_.data() + i * n_);
            }
            if (!all_finite(F_)) {
                failure = "non-finite right-hand side in a stage";
                return fail(failure);
            }
            for (std::size_t i = 0; i < s_; ++i) {
                for (std::size_t p = 0; p < n_; ++p) {
                    double acc = 0.0;
                    for (std::size_t j = 0; j < s_; ++j) acc += hA[i * s_ + j] * F_[j * n_ + p];
                    G_[i * n_ + p] = acc - Z[i * n_ + p];  // -G(Z)
                }
            }
            if (kind == NonlinearSolverKind::FullNewton) {
                stage_jacobians(t, X, Z, hc);
                std::vector<const Matrix*> Js(s_);
                for (std::size_t j = 0; j < s_; ++j) Js[j] = &stage_J_[j];
                build_stage_matrix(s_, n_, hA.data(), Js, M_);
                slot = &cache_.scratch();
                if (!factor_into(ctx_, *slot, M_, failure)) return fail(failure);
            }
            if (kind != NonlinearSolverKind::FixedPoint) {
                if (!solve_with(ctx_, *slot, G_.data(), failure)) return fail(failure);
            }
            // G_ now holds the increment dZ.
            double s2 = 0.0, floor2 = 0.0;
            for (std::size_t i = 0; i < s_; ++i) {
                for (std::size_t p = 0; p < n_; ++p) {
                    const std::size_t idx = i * n_ + p;
                    Z[idx] += G_[idx];
                    const double r = G_[idx] / scale[p];
                    s2 += r * r;
                    const double fl = 10.0 * kEps * (std::abs(X[p]) + std::abs(Z[idx])) / scale[p];
                    floor2 += fl * fl;
                }
            }
            const double nrm = std::sqrt(s2 / static_cast<double>(N));
            const double nrm_floor = std::sqrt(floor2 / static_cast<double>(N));
            if (!std::isfinite(nrm)) {
                failure = "non-finite iterate";
                return fail(failure);
            }
            bool converged = nrm <= nrm_floor;
            if (k == 0) {
                eta_used = std::pow(std::max(eta_, kEps), 0.8);
                converged = converged || eta_used * nrm <= ctx_.opt.newton_tol;
            } else {
                theta = nrm / nrm_prev;
                st.theta_max = std::max(st.theta_max, theta);
                st.theta_sum += theta;
                ++st.n_theta;
                if (!converged) {
                    if (theta >= 0.99) {
                        failure = "iteration diverges (contraction factor theta = " + std::to_string(theta) + ")";
                        return fail(failure);
                    }
                    eta_used = theta / (1.0 - theta);
                    converged = eta_used * nrm <= ctx_.opt.newton_tol;
                    if (!converged && k + 1 < kmax_ &&
                        std::pow(theta, kmax_ - 1 - k) / (1.0 - theta) * nrm > 1e3 * ctx_.opt.newton_tol) {
                        failure = "iteration converges too slowly (theta = " + std::to_string(theta) + ")";
                        return fail(failure);
                    }
                }
            }
            if (converged) {
                eta_ = (k == 0) ? eta_used : std::max(theta / (1.0 - theta), kEps);
                return true;
            }
            nrm_prev = nrm;
        }
        failure = "no convergence within " + std::to_string(kmax_) + " iterations";
        return fail(failure);
    }

private:
    bool fail(std::string& failure) {
        ++ctx_.st.n_newton_failures;
        eta_ = 1.0;
        failure = to_string(ctx_.opt.nonlinear) + ": " + failure;
        return false;
    }

    void stage_jacobians(double t, const Vec& X, const Vec& Z, const std::vector<double>& hc) {
        if (stage_J_.size() != s_) stage_J_.assign(s_, Matrix(n_, n_));
        for (std::size_t j = 0; j < s_; ++j) {
            for (std::size_t p = 0; p < n_; ++p) Xz_[p] = X[p] + Z[j * n_ + p];
            ctx_.jac(t + hc[j], Xz_.data(), F_.data() + j * n_, stage_J_[j].data.data());
        }
    }

    Context& ctx_;
    std::size_t s_, n_;
    JacobianCache& jac_;
    FactorCache cache_;
    int kmax_;
    double eta_ = 1.0;
    Vec F_, G_, Xz_;
    Matrix M_;
    std::vector<Matrix> stage_J_;
};

// ---------------------------------------------------------------------------
// Steppers
// ---------------------------------------------------------------------------
class Stepper {
public:
    explicit Stepper(Context& ctx) : ctx_(ctx) {}
    virtual ~Stepper() = default;

    // One step of size h from (t, y) with fy = f(t, y). On success y_new is
    // set and, if err != nullptr and the method has an estimator, the local
    // error estimate. On failure returns false and sets failure().
    virtual bool attempt(double t, const Vec& y, const Vec& fy, double h, Vec& y_new, Vec* err) = 0;
    virtual bool has_estimator() const = 0;
    virtual int estimator_order() const = 0;  // q with estimate = O(h^{q+1})
    virtual int order() const = 0;
    virtual void on_accept(double /*t_old*/, const Vec& /*y_old*/, double /*h*/) {}
    virtual const Vec* fsal() const { return nullptr; }  // f(t+h, y_new) of the last attempt
    virtual double max_ratio() const { return ctx_.opt.facmax; }

    const std::string& failure() const { return failure_; }

protected:
    Context& ctx_;
    std::string failure_;
};

class ExplicitRKStepper final : public Stepper {
public:
    ExplicitRKStepper(Context& ctx, const ButcherTableau& tab) : Stepper(ctx), tab_(tab), k_(tab.s, Vec(ctx.n)) {
        reuse_first_ = tab_.c[0] == 0.0;
        fsal_ok_ = tab_.fsal && tab_.c[tab_.s - 1] == 1.0;
        for (std::size_t j = 0; fsal_ok_ && j < tab_.s; ++j)
            if (tab_.A[(tab_.s - 1) * tab_.s + j] != tab_.b[j]) fsal_ok_ = false;
    }

    bool attempt(double t, const Vec& y, const Vec& fy, double h, Vec& y_new, Vec* err) override {
        const std::size_t s = tab_.s, n = ctx_.n;
        Vec yi(n);
        for (std::size_t i = 0; i < s; ++i) {
            if (i == 0 && reuse_first_) {
                k_[0] = fy;
                continue;
            }
            for (std::size_t p = 0; p < n; ++p) {
                double acc = 0.0;
                for (std::size_t j = 0; j < i; ++j) acc += tab_.A[i * s + j] * k_[j][p];
                yi[p] = y[p] + h * acc;
            }
            ctx_.f(t + tab_.c[i] * h, yi.data(), k_[i].data());
        }
        y_new.assign(n, 0.0);
        for (std::size_t p = 0; p < n; ++p) {
            double acc = 0.0;
            for (std::size_t i = 0; i < s; ++i) acc += tab_.b[i] * k_[i][p];
            y_new[p] = y[p] + h * acc;
        }
        if (err != nullptr && has_estimator()) {
            err->assign(n, 0.0);
            for (std::size_t p = 0; p < n; ++p) {
                double acc = 0.0;
                for (std::size_t i = 0; i < s; ++i) acc += (tab_.b[i] - tab_.bhat[i]) * k_[i][p];
                (*err)[p] = h * acc;
            }
        }
        return true;
    }
    bool has_estimator() const override { return !tab_.bhat.empty(); }
    int estimator_order() const override { return has_estimator() ? std::min(tab_.order, tab_.order_hat) : tab_.order; }
    int order() const override { return tab_.order; }
    const Vec* fsal() const override { return fsal_ok_ ? &k_[tab_.s - 1] : nullptr; }

private:
    ButcherTableau tab_;
    std::vector<Vec> k_;
    bool reuse_first_ = false;
    bool fsal_ok_ = false;
};

class ImplicitRKStepper final : public Stepper {
public:
    ImplicitRKStepper(Context& ctx, const ButcherTableau& tab)
        : Stepper(ctx), tab_(tab), solver_(ctx, tab.s, jac_) {
        const std::size_t s = tab_.s;
        // How to form y_{n+1} without extra f evaluations (Lemma 5.5 (iii)):
        stiffly_accurate_ = tab_.c[s - 1] == 1.0;
        for (std::size_t j = 0; stiffly_accurate_ && j < s; ++j)
            if (tab_.A[(s - 1) * s + j] != tab_.b[j]) stiffly_accurate_ = false;
        if (!stiffly_accurate_) {
            Matrix At(s, s);
            for (std::size_t i = 0; i < s; ++i)
                for (std::size_t j = 0; j < s; ++j) At(i, j) = tab_.A[j * s + i];
            LUFactorization f = lu_factor(At, true);
            if (f.ok) w_ = lu_solve(f, tab_.b);  // w = A^{-T} b
        }
    }

    bool attempt(double t, const Vec& y, const Vec& fy, double h, Vec& y_new, Vec* /*err*/) override {
        const std::size_t s = tab_.s, n = ctx_.n;
        if (ctx_.opt.nonlinear == NonlinearSolverKind::SimplifiedNewton) jac_.ensure(ctx_, t, y, fy);
        std::vector<double> hA(s * s), hc(s);
        for (std::size_t i = 0; i < s * s; ++i) hA[i] = h * tab_.A[i];
        for (std::size_t i = 0; i < s; ++i) hc[i] = h * tab_.c[i];
        Z_.assign(s * n, 0.0);
        if (!solver_.solve(t, y, ctx_.scale_vector(y), hA, hc, Z_, failure_)) return false;
        y_new = y;
        if (stiffly_accurate_) {
            for (std::size_t p = 0; p < n; ++p) y_new[p] += Z_[(s - 1) * n + p];
        } else if (!w_.empty()) {
            for (std::size_t i = 0; i < s; ++i)
                for (std::size_t p = 0; p < n; ++p) y_new[p] += w_[i] * Z_[i * n + p];
        } else {  // singular A, not stiffly accurate: y + h sum b_i f(t + c_i h, y + z_i)
            Vec yi(n), fi(n);
            for (std::size_t i = 0; i < s; ++i) {
                for (std::size_t p = 0; p < n; ++p) yi[p] = y[p] + Z_[i * n + p];
                ctx_.f(t + hc[i], yi.data(), fi.data());
                for (std::size_t p = 0; p < n; ++p) y_new[p] += h * tab_.b[i] * fi[p];
            }
        }
        return true;
    }
    bool has_estimator() const override { return false; }
    int estimator_order() const override { return tab_.order; }
    int order() const override { return tab_.order; }
    void on_accept(double, const Vec&, double) override { jac_.invalidate(); }

private:
    ButcherTableau tab_;
    JacobianCache jac_;
    StageSolver solver_;
    bool stiffly_accurate_ = false;
    Vec w_;
    Vec Z_;
};

// ROS2 (Verwer, Spee, Blom, Hundsdorfer 1999), gamma = 1 + 1/sqrt(2):
//   (I - gamma h J) k1 = f(t, y) + gamma h f_t
//   (I - gamma h J) k2 = f(t + h, y + h k1) - 2 k1 - gamma h f_t
//   y_{n+1} = y + 3/2 h k1 + 1/2 h k2,   embedded order 1: y + h k1.
class Rosenbrock2Stepper final : public Stepper {
public:
    explicit Rosenbrock2Stepper(Context& ctx) : Stepper(ctx), cache_(ctx.opt) {}

    bool attempt(double t, const Vec& y, const Vec& fy, double h, Vec& y_new, Vec* err) override {
        const std::size_t n = ctx_.n;
        const double gamma = 1.0 + 1.0 / std::sqrt(2.0);
        jac_.ensure(ctx_, t, y, fy);
        if (!ft_valid_) {  // time derivative by forward differences (zero for autonomous systems)
            ft_.assign(n, 0.0);
            const double delta = std::sqrt(kEps) * std::max({std::abs(t), std::abs(h), 1e-300});
            Vec f1(n);
            ctx_.f(t + delta, y.data(), f1.data());
            for (std::size_t p = 0; p < n; ++p) ft_[p] = (f1[p] - fy[p]) / delta;
            ft_valid_ = true;
        }
        bool fresh = false;
        FactorSlot& slot = cache_.lookup({gamma * h}, jac_.version, fresh);
        if (fresh) {
            Matrix W(n, n);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < n; ++j) W(i, j) = -gamma * h * jac_.J(i, j);
                W(i, i) += 1.0;
            }
            if (!factor_into(ctx_, slot, W, failure_)) return false;
        }
        Vec k1(n), k2(n), y2(n), f2(n);
        for (std::size_t p = 0; p < n; ++p) k1[p] = fy[p] + gamma * h * ft_[p];
        if (!solve_with(ctx_, slot, k1.data(), failure_)) return false;
        for (std::size_t p = 0; p < n; ++p) y2[p] = y[p] + h * k1[p];
        ctx_.f(t + h, y2.data(), f2.data());
        for (std::size_t p = 0; p < n; ++p) k2[p] = f2[p] - 2.0 * k1[p] - gamma * h * ft_[p];
        if (!solve_with(ctx_, slot, k2.data(), failure_)) return false;
        y_new.assign(n, 0.0);
        for (std::size_t p = 0; p < n; ++p) y_new[p] = y[p] + h * (1.5 * k1[p] + 0.5 * k2[p]);
        if (err != nullptr) {
            err->assign(n, 0.0);
            for (std::size_t p = 0; p < n; ++p) (*err)[p] = 0.5 * h * (k1[p] + k2[p]);
        }
        return true;
    }
    bool has_estimator() const override { return true; }
    int estimator_order() const override { return 1; }
    int order() const override { return 2; }
    void on_accept(double, const Vec&, double) override {
        jac_.invalidate();
        ft_valid_ = false;
    }

private:
    JacobianCache jac_;
    FactorCache cache_;
    Vec ft_;
    bool ft_valid_ = false;
};

// Variable-step BDF2 (ODE script Kap. 7.3), started with one implicit Euler step:
//   y_{n+1} - (1+w)^2/(1+2w) y_n + w^2/(1+2w) y_{n-1} = h (1+w)/(1+2w) f(t_{n+1}, y_{n+1}),  w = h / h_1.
// Local error estimate (Milne's device) from an explicit extrapolation
// predictor through past solution values only. (A predictor using f_n is
// unsuitable for stiff problems: h f_n amplifies tiny deviations from the
// slow manifold in stiff directions and pollutes the estimate.)
//   * step 1 (implicit Euler), predictor y_n + h f_n:   err = (y - y_pred) / 2
//   * step 2, linear extrapolation through y_{n-1}, y_n: err = h/(h+h_1) (y - y_pred) / 2,
//     a first-order estimate (conservative, the BDF2 error is O(h^3))
//   * later, quadratic extrapolation through y_n, y_{n-1}, y_{n-2}:
//       err = Q/(Q+P) (y - y_pred),  Q = h^2 (h+h_1)^2 / (2h+h_1),  P = h (h+h_1)(h+h_1+h_2)
//     (corrector and predictor errors are Q y'''/6 and -P y'''/6).
class BDF2Stepper final : public Stepper {
public:
    explicit BDF2Stepper(Context& ctx) : Stepper(ctx), solver_(ctx, 1, jac_) {}

    bool attempt(double t, const Vec& y, const Vec& fy, double h, Vec& y_new, Vec* err) override {
        const std::size_t n = ctx_.n;
        if (ctx_.opt.nonlinear == NonlinearSolverKind::SimplifiedNewton) jac_.ensure(ctx_, t, y, fy);
        Vec psi(n), ypred(n);
        double beta, cerr;
        if (history_ == 0) {  // implicit Euler, predictor explicit Euler
            psi = y;
            beta = 1.0;
            for (std::size_t p = 0; p < n; ++p) ypred[p] = y[p] + h * fy[p];
            cerr = 0.5;
        } else {
            const double w = h / h1_;
            const double a1 = (1 + w) * (1 + w) / (1 + 2 * w), a2 = w * w / (1 + 2 * w);
            beta = (1 + w) / (1 + 2 * w);
            for (std::size_t p = 0; p < n; ++p) psi[p] = a1 * y[p] - a2 * y1_[p];
            if (history_ == 1) {
                for (std::size_t p = 0; p < n; ++p) ypred[p] = y[p] + h * (y[p] - y1_[p]) / h1_;
                cerr = 0.5 * h / (h + h1_);
            } else {
                for (std::size_t p = 0; p < n; ++p) {
                    const double d1 = (y[p] - y1_[p]) / h1_;
                    const double d2 = (d1 - (y1_[p] - y2_[p]) / h2_) / (h1_ + h2_);
                    ypred[p] = y[p] + h * d1 + h * (h + h1_) * d2;
                }
                const double Q = h * h * (h + h1_) * (h + h1_) / (2 * h + h1_);
                const double P = h * (h + h1_) * (h + h1_ + h2_);
                cerr = Q / (Q + P);
            }
        }
        Z_.resize(n);
        for (std::size_t p = 0; p < n; ++p) Z_[p] = ypred[p] - psi[p];
        if (!solver_.solve(t, psi, ctx_.scale_vector(y), {beta * h}, {h}, Z_, failure_)) return false;
        y_new.resize(n);
        for (std::size_t p = 0; p < n; ++p) y_new[p] = psi[p] + Z_[p];
        if (err != nullptr) {
            err->resize(n);
            for (std::size_t p = 0; p < n; ++p) (*err)[p] = cerr * (y_new[p] - ypred[p]);
        }
        return true;
    }
    bool has_estimator() const override { return true; }
    int estimator_order() const override { return history_ >= 2 ? 2 : 1; }
    int order() const override { return 2; }
    void on_accept(double /*t_old*/, const Vec& y_old, double h) override {
        jac_.invalidate();
        y2_.swap(y1_);
        h2_ = h1_;
        y1_ = y_old;
        h1_ = h;
        history_ = std::min(history_ + 1, 2);
    }
    // Zero-stability of variable-step BDF2 requires w < 1 + sqrt(2).
    double max_ratio() const override { return std::min(ctx_.opt.facmax, 2.0); }

private:
    JacobianCache jac_;
    StageSolver solver_;
    int history_ = 0;  // past points available (capped at 2)
    Vec y1_, y2_;      // y_{n-1}, y_{n-2}
    double h1_ = 0.0, h2_ = 0.0;
    Vec Z_;
};

std::unique_ptr<Stepper> make_stepper(const Method& m, Context& ctx) {
    switch (m.kind) {
        case MethodKind::ExplicitRK: return std::make_unique<ExplicitRKStepper>(ctx, m.tableau);
        case MethodKind::ImplicitRK: return std::make_unique<ImplicitRKStepper>(ctx, m.tableau);
        case MethodKind::Rosenbrock2: return std::make_unique<Rosenbrock2Stepper>(ctx);
        case MethodKind::BDF2: return std::make_unique<BDF2Stepper>(ctx);
    }
    throw std::invalid_argument("unknown method kind");
}

void validate(const OdeSystem& sys, const Method& method, double t0, double tf, const Vec& y0, const Options& opt) {
    const std::size_t n = sys.size();
    if (n == 0) throw std::invalid_argument("system has dimension 0");
    if (y0.size() != n)
        throw std::invalid_argument("y0 has length " + std::to_string(y0.size()) + ", system has dimension " +
                                    std::to_string(n));
    if (!all_finite(y0)) throw std::invalid_argument("y0 contains non-finite values");
    if (!(std::isfinite(t0) && std::isfinite(tf) && tf > t0))
        throw std::invalid_argument("require finite t0 < tf");
    if (!(opt.rtol >= 0.0)) throw std::invalid_argument("rtol must be >= 0");
    if (opt.atol.size() != 1 && opt.atol.size() != n)
        throw std::invalid_argument("atol must be a scalar or have length n");
    for (double a : opt.atol)
        if (!(a >= 0.0)) throw std::invalid_argument("atol must be >= 0");
    if (opt.fixed_step <= 0.0) {
        double amin = *std::min_element(opt.atol.begin(), opt.atol.end());
        if (opt.rtol == 0.0 && amin == 0.0) throw std::invalid_argument("rtol and atol must not both be zero");
    }
    if (opt.fixed_step < 0.0 || !std::isfinite(opt.fixed_step)) throw std::invalid_argument("fixed_step must be >= 0");
    if (!(opt.h_max > 0.0)) throw std::invalid_argument("h_max must be > 0");
    if (opt.max_steps <= 0) throw std::invalid_argument("max_steps must be > 0");
    if (!(opt.newton_tol > 0.0)) throw std::invalid_argument("newton_tol must be > 0");
    if (!(opt.facmin > 0.0 && opt.facmin < 1.0 && opt.facmax > 1.0 && opt.safety > 0.0 && opt.safety <= 1.0))
        throw std::invalid_argument("invalid step-size controller parameters");
    for (std::size_t i = 0; i < opt.t_eval.size(); ++i) {
        const double te = opt.t_eval[i];
        if (te < t0 || te > tf) throw std::invalid_argument("t_eval entries must lie in [t0, tf]");
        if (i > 0 && te < opt.t_eval[i - 1]) throw std::invalid_argument("t_eval must be sorted");
    }
    if (method.kind == MethodKind::ExplicitRK || method.kind == MethodKind::ImplicitRK) method.tableau.validate();
}

// Hairer-Norsett-Wanner initial step size heuristic.
double initial_step(Context& ctx, double t0, double tf, const Vec& y0, const Vec& f0, int order) {
    const std::size_t n = ctx.n;
    double d0 = 0.0, d1 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double sc = ctx.scale(i, y0[i], y0[i]);
        d0 += (y0[i] / sc) * (y0[i] / sc);
        d1 += (f0[i] / sc) * (f0[i] / sc);
    }
    d0 = std::sqrt(d0 / n);
    d1 = std::sqrt(d1 / n);
    double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
    h0 = std::min({h0, tf - t0, ctx.opt.h_max});
    Vec y1(n), f1(n);
    for (std::size_t i = 0; i < n; ++i) y1[i] = y0[i] + h0 * f0[i];
    ctx.f(t0 + h0, y1.data(), f1.data());
    double d2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double sc = ctx.scale(i, y0[i], y0[i]);
        d2 += ((f1[i] - f0[i]) / sc) * ((f1[i] - f0[i]) / sc);
    }
    d2 = std::sqrt(d2 / n) / h0;
    const double dm = std::max(d1, d2);
    const double h1 = (dm <= 1e-15 || !std::isfinite(dm)) ? std::max(1e-6, h0 * 1e-3)
                                                          : std::pow(0.01 / dm, 1.0 / (order + 1));
    return std::min({100.0 * h0, h1, tf - t0, ctx.opt.h_max});
}

void hermite(double theta, double h, const Vec& y0, const Vec& f0, const Vec& y1, const Vec& f1, double* out) {
    const double t2 = theta * theta, t3 = t2 * theta;
    const double h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + theta, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
    for (std::size_t i = 0; i < y0.size(); ++i)
        out[i] = h00 * y0[i] + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i];
}

}  // namespace

Result integrate(OdeSystem& system, const Method& method, double t0, double tf, const Vec& y0, const Options& opt) {
    validate(system, method, t0, tf, y0, opt);
    const std::size_t n = system.size();
    Result res;
    res.n = n;
    Stats& st = res.stats;
    Context ctx{system, opt, st, n, Vec(n)};
    for (std::size_t i = 0; i < n; ++i) ctx.atol[i] = opt.atol.size() == 1 ? opt.atol[0] : opt.atol[i];

    std::unique_ptr<Stepper> stepper = make_stepper(method, ctx);
    const bool adaptive = opt.fixed_step <= 0.0;
    const bool doubling = adaptive && !stepper->has_estimator();
    const unsigned char* mask = opt.nonnegative ? system.nonnegative_mask() : nullptr;
    const auto wall_start = std::chrono::steady_clock::now();

    double t = t0;
    Vec y = y0, f(n);
    ctx.f(t, y.data(), f.data());
    std::size_t next_out = 0;
    auto emit = [&](double te, const double* ye) {
        res.t.push_back(te);
        res.y.insert(res.y.end(), ye, ye + n);
    };
    auto finish = [&](Status status, const std::string& msg) {
        res.status = status;
        res.message = msg;
        return res;
    };
    if (opt.t_eval.empty()) {
        emit(t, y.data());
    } else {
        while (next_out < opt.t_eval.size() && opt.t_eval[next_out] == t0) emit(t0, y.data()), ++next_out;
    }
    if (!all_finite(f)) return finish(Status::NonFinite, "right-hand side is not finite at the initial value");

    long n_fixed = 0;
    double h;
    if (adaptive) {
        h = opt.h0 > 0.0 ? std::min(opt.h0, tf - t0) : initial_step(ctx, t0, tf, y, f, stepper->order());
    } else {
        n_fixed = std::max<long>(1, static_cast<long>(std::ceil((tf - t0) / opt.fixed_step - 1e-9)));
        h = (tf - t0) / static_cast<double>(n_fixed);
    }
    double err_prev = 1e-4;
    double facmax_now = opt.facmax;
    const double beta_pi = 0.04;  // PI controller (Gustafsson / Hairer DOPRI5 constants)
    Vec y_new(n), err(n), f_new(n), y_half(n), f_half(n), y_big(n), out(n);

    while (t < tf) {
        if (st.n_steps >= opt.max_steps)
            return finish(Status::MaxSteps, "maximum number of steps (" + std::to_string(opt.max_steps) + ") reached at t = " +
                                                std::to_string(t));
        if (opt.max_wall_time > 0.0) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
            if (elapsed > opt.max_wall_time)
                return finish(Status::WallTime, "wall-clock budget exhausted at t = " + std::to_string(t));
        }
        double t_new;
        if (adaptive) {
            h = std::min(h, opt.h_max);
            const double hmin = std::max(opt.h_min, 16.0 * kEps * std::abs(t));
            if (!(h >= hmin) || h <= 0.0)
                return finish(Status::StepSizeTooSmall,
                              "step size " + std::to_string(h) + " below minimum at t = " + std::to_string(t) +
                                  (stepper->failure().empty() ? "" : " (last failure: " + stepper->failure() + ")"));
            // Land exactly on the end point (and, if requested, on the next output time).
            double t_stop = tf;
            if (opt.exact_t_eval && next_out < opt.t_eval.size()) t_stop = std::min(t_stop, opt.t_eval[next_out]);
            if (t + 1.01 * h >= t_stop) h = t_stop - t;
            t_new = (t + h >= t_stop) ? t_stop : t + h;
        } else {
            t_new = (st.n_accepted + 1 >= n_fixed) ? tf : t0 + static_cast<double>(st.n_accepted + 1) * h;
        }
        const double hs = t_new - t;
        ++st.n_steps;

        bool ok;
        if (doubling) {
            ok = stepper->attempt(t, y, f, hs, y_big, nullptr) && stepper->attempt(t, y, f, 0.5 * hs, y_half, nullptr);
            if (ok) {
                ctx.f(t + 0.5 * hs, y_half.data(), f_half.data());
                ok = stepper->attempt(t + 0.5 * hs, y_half, f_half, 0.5 * hs, y_new, nullptr);
            }
            if (ok) {
                const double denom = std::pow(2.0, stepper->order()) - 1.0;
                for (std::size_t i = 0; i < n; ++i) err[i] = (y_new[i] - y_big[i]) / denom;
            }
        } else {
            ok = stepper->attempt(t, y, f, hs, y_new, adaptive ? &err : nullptr);
        }
        if (!ok) {
            ++st.n_rejected;
            ++st.n_solver_rejections;
            if (!adaptive)
                return finish(Status::NonlinearSolverFailure,
                              "step at t = " + std::to_string(t) + " failed: " + stepper->failure());
            h = 0.5 * hs;
            facmax_now = 1.0;
            continue;
        }
        if (!all_finite(y_new)) {
            if (!adaptive) return finish(Status::NonFinite, "solution became non-finite at t = " + std::to_string(t_new));
            ++st.n_rejected;
            ++st.n_error_rejections;
            h = 0.25 * hs;
            facmax_now = 1.0;
            continue;
        }
        double en = 0.0;
        const int k_exp = stepper->estimator_order() + 1;
        if (adaptive) {
            en = ctx.wrms(err, y, y_new);
            if (!std::isfinite(en)) en = std::numeric_limits<double>::infinity();
            if (en > 1.0) {
                ++st.n_rejected;
                ++st.n_error_rejections;
                const double fac = std::isfinite(en) ? std::max(opt.facmin, opt.safety * std::pow(en, -1.0 / k_exp)) : 0.25;
                h = hs * fac;
                facmax_now = 1.0;
                continue;
            }
        }
        if (mask != nullptr) {
            bool reject = false;
            for (std::size_t i = 0; i < n; ++i) {
                if (!mask[i] || y_new[i] >= 0.0) continue;
                if (adaptive && y_new[i] < -ctx.atol[i]) {
                    reject = true;
                    break;
                }
                y_new[i] = 0.0;  // round-off level negativity is clipped
            }
            if (reject) {
                ++st.n_rejected;
                ++st.n_nonneg_rejections;
                h = 0.5 * hs;
                facmax_now = 1.0;
                continue;
            }
        }
        // Accept.
        const Vec* fs = doubling ? nullptr : stepper->fsal();
        if (fs != nullptr && mask == nullptr) {
            f_new = *fs;
        } else {
            ctx.f(t_new, y_new.data(), f_new.data());
        }
        if (!all_finite(f_new))
            return finish(Status::NonFinite, "right-hand side became non-finite at t = " + std::to_string(t_new));
        if (opt.t_eval.empty()) {
            emit(t_new, y_new.data());
        } else {
            while (next_out < opt.t_eval.size() && opt.t_eval[next_out] <= t_new) {
                const double te = opt.t_eval[next_out];
                if (te == t_new) {
                    emit(te, y_new.data());
                } else {
                    hermite((te - t) / hs, hs, y, f, y_new, f_new, out.data());
                    emit(te, out.data());
                }
                ++next_out;
            }
        }
        stepper->on_accept(t, y, hs);
        ++st.n_accepted;
        st.h_min_used = st.n_accepted == 1 ? hs : std::min(st.h_min_used, hs);
        st.h_max_used = std::max(st.h_max_used, hs);
        st.h_sum += hs;
        y.swap(y_new);
        f.swap(f_new);
        t = t_new;
        if (adaptive) {
            double fac;
            if (en == 0.0) {
                fac = facmax_now;
            } else {
                const double alpha = 1.0 / k_exp - 0.75 * beta_pi;
                fac = opt.safety * std::pow(en, -alpha) * std::pow(err_prev, beta_pi);
            }
            fac = std::clamp(fac, opt.facmin, std::min(facmax_now, stepper->max_ratio()));
            h = hs * fac;
            err_prev = std::max(en, 1e-4);
            facmax_now = opt.facmax;
        }
    }
    return finish(Status::Success, "integration successful");
}

}  // namespace ak::num
