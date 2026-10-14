#include "autokinetics/kinetics/reactor.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ak {

namespace {
constexpr double R = 8.31446261815324;
}

double piecewise_linear(const std::vector<double>& t, const std::vector<double>& v, double x) {
    if (t.empty()) throw std::invalid_argument("empty program");
    if (x <= t.front()) return v.front();
    if (x >= t.back()) return v.back();
    const auto it = std::upper_bound(t.begin(), t.end(), x);
    const std::size_t k = static_cast<std::size_t>(it - t.begin());
    const double w = (x - t[k - 1]) / (t[k] - t[k - 1]);
    return v[k - 1] + w * (v[k] - v[k - 1]);
}

ReactorSystem::ReactorSystem(std::shared_ptr<Kinetics> kin, ReactorConfig cfg)
    : kin_(std::move(kin)), cfg_(std::move(cfg)) {
    if (!kin_) throw std::invalid_argument("ReactorSystem: kinetics is null");
    ns_ = kin_->n_species();
    const bool gas = kin_->phase() == PHASE_GAS;
    if (cfg_.constraint == CONST_PRESSURE && !gas)
        throw std::invalid_argument("constant-pressure reactor requires an ideal-gas mechanism");
    if (cfg_.energy == ENERGY_BALANCE && gas && !kin_->all_thermo())
        throw std::invalid_argument("energy balance requires thermo data for all species");
    if (cfg_.flow == FLOW_CSTR) {
        if (!(cfg_.tau > 0.0)) throw std::invalid_argument("CSTR residence time must be > 0");
        if (cfg_.constraint != CONST_VOLUME) throw std::invalid_argument("CSTR requires constant volume");
        if (cfg_.energy == ENERGY_BALANCE && gas)
            throw std::invalid_argument("CSTR with energy balance is only supported for liquids");
        if (cfg_.c_in.size() != static_cast<std::size_t>(ns_)) throw std::invalid_argument("CSTR feed size");
    }
    if (cfg_.T_prog_t.size() != cfg_.T_prog_T.size() || cfg_.T_env_t.size() != cfg_.T_env_T.size())
        throw std::invalid_argument("temperature program: size mismatch");
    if (cfg_.energy == ENERGY_BALANCE && !gas && !(cfg_.rho_cp > 0.0))
        throw std::invalid_argument("liquid energy balance requires rho*cp > 0");
    n_state_ = static_cast<std::size_t>(ns_) + (cfg_.energy == ENERGY_BALANCE ? 1 : 0);
    mask_.assign(n_state_, 1);
    if (cfg_.energy == ENERGY_BALANCE) mask_[ns_] = 0;
    c_.resize(ns_); wdot_.resize(ns_); q_.resize(kin_->n_reactions());
    cp_.resize(ns_); h_.resize(ns_); s_.resize(ns_); dH_.resize(kin_->n_reactions());
    Jc_.resize(static_cast<std::size_t>(ns_) * ns_);
    f0_.resize(n_state_); f1_.resize(n_state_); y1_.resize(n_state_);
}

void ReactorSystem::observe(double t, const double* y, double* c, double* T, double* P, double* V) const {
    const bool energy = cfg_.energy == ENERGY_BALANCE;
    const double Tv = energy ? y[ns_] : piecewise_linear(cfg_.T_prog_t, cfg_.T_prog_T, t);
    double Vv = 1.0, Pv = cfg_.P0;
    if (cfg_.constraint == CONST_PRESSURE) {
        double ntot = 0.0;
        for (int i = 0; i < ns_; ++i) ntot += y[i];
        Vv = ntot * R * Tv / cfg_.P0;
        for (int i = 0; i < ns_; ++i) c[i] = y[i] / Vv;
    } else {
        double ctot = 0.0;
        for (int i = 0; i < ns_; ++i) { c[i] = y[i]; ctot += y[i]; }
        if (kin_->phase() == PHASE_GAS) Pv = ctot * R * Tv;
    }
    *T = Tv; *P = Pv; *V = Vv;
}

void ReactorSystem::rhs(double t, const double* y, double* dydt) {
    ++n_rhs_;
    eval(t, y, dydt);
}

void ReactorSystem::eval(double t, const double* y, double* dydt) {
    double T, P, V;
    observe(t, y, c_.data(), &T, &P, &V);
    if (!(T > 0.0) || !std::isfinite(T)) throw std::domain_error("non-physical temperature in reactor state");
    if (cfg_.constraint == CONST_PRESSURE && !(V > 0.0)) throw std::domain_error("non-positive reactor volume");
    kin_->production_rates(T, P, c_.data(), wdot_.data(), q_.data());
    const bool gas = kin_->phase() == PHASE_GAS;
    if (cfg_.constraint == CONST_PRESSURE) {
        for (int i = 0; i < ns_; ++i) dydt[i] = wdot_[i] * V;
    } else {
        for (int i = 0; i < ns_; ++i) dydt[i] = wdot_[i];
        if (cfg_.flow == FLOW_CSTR)
            for (int i = 0; i < ns_; ++i) dydt[i] += (cfg_.c_in[i] - c_[i]) / cfg_.tau;
    }
    if (cfg_.energy != ENERGY_BALANCE) return;
    const double T_env = piecewise_linear(cfg_.T_env_t, cfg_.T_env_T, t);
    const double q_ext = cfg_.UA_V * (T_env - T);
    if (gas) {
        kin_->species_thermo(T, cp_.data(), h_.data(), s_.data());
        double num = 0.0, den = 0.0;
        if (cfg_.constraint == CONST_PRESSURE) {
            for (int i = 0; i < ns_; ++i) { num -= h_[i] * wdot_[i]; den += y[i] * cp_[i]; }
            dydt[ns_] = (V * num + V * q_ext) / den;
        } else {
            for (int i = 0; i < ns_; ++i) {
                num -= (h_[i] - R * T) * wdot_[i];
                den += c_[i] * (cp_[i] - R);
            }
            dydt[ns_] = (num + q_ext) / den;
        }
    } else {
        kin_->reaction_enthalpies(T, dH_.data());
        double heat = 0.0;
        for (int j = 0; j < kin_->n_reactions(); ++j) heat -= dH_[j] * q_[j];
        double dT = (heat + q_ext) / cfg_.rho_cp;
        if (cfg_.flow == FLOW_CSTR) dT += (cfg_.T_in - T) / cfg_.tau;
        dydt[ns_] = dT;
    }
}

void ReactorSystem::jacobian(double t, const double* y, double* J) {
    ++n_jac_;
    const std::size_t n = n_state_;
    const bool energy = cfg_.energy == ENERGY_BALANCE;
    const bool gas = kin_->phase() == PHASE_GAS;
    double T, P, V;
    observe(t, y, c_.data(), &T, &P, &V);
    kin_->jacobian_conc(T, P, c_.data(), Jc_.data());
    std::fill(J, J + n * n, 0.0);

    if (cfg_.constraint == CONST_PRESSURE) {
        // y = n, c = n/V, V = R T sum(n)/P.  d(V wdot)/dn = Jc (I - x 1^T) + wdot (V/sum n) 1^T
        kin_->production_rates(T, P, c_.data(), wdot_.data(), q_.data());
        double ntot = 0.0;
        for (int i = 0; i < ns_; ++i) ntot += y[i];
        for (int s = 0; s < ns_; ++s) {
            double row_dot_x = 0.0;
            for (int k = 0; k < ns_; ++k) row_dot_x += Jc_[static_cast<std::size_t>(s) * ns_ + k] * (y[k] / ntot);
            for (int i = 0; i < ns_; ++i)
                J[s * n + i] = Jc_[static_cast<std::size_t>(s) * ns_ + i] - row_dot_x + wdot_[s] * V / ntot;
        }
    } else {
        for (int s = 0; s < ns_; ++s)
            for (int i = 0; i < ns_; ++i) J[s * n + i] = Jc_[static_cast<std::size_t>(s) * ns_ + i];
        if (cfg_.flow == FLOW_CSTR)
            for (int i = 0; i < ns_; ++i) J[i * n + i] -= 1.0 / cfg_.tau;
    }
    if (!energy) return;

    // Temperature row (analytic in the composition variables).
    eval(t, y, f0_.data());
    const double Tdot = f0_[ns_];
    const double* Jsp = J;  // species block already filled (d f_species / d y_species)
    if (gas) {
        kin_->species_thermo(T, cp_.data(), h_.data(), s_.data());
        if (cfg_.constraint == CONST_PRESSURE) {
            double Cp = 0.0;
            for (int i = 0; i < ns_; ++i) Cp += y[i] * cp_[i];
            const double T_env = piecewise_linear(cfg_.T_env_t, cfg_.T_env_T, t);
            const double q_ext = cfg_.UA_V * (T_env - T);
            for (int i = 0; i < ns_; ++i) {
                double v = 0.0;
                for (int s = 0; s < ns_; ++s) v -= h_[s] * Jsp[s * n + i];
                v += q_ext * (R * T / cfg_.P0);
                J[static_cast<std::size_t>(ns_) * n + i] = (v - Tdot * cp_[i]) / Cp;
            }
        } else {
            double Cv = 0.0;
            for (int i = 0; i < ns_; ++i) Cv += c_[i] * (cp_[i] - R);
            for (int i = 0; i < ns_; ++i) {
                double v = 0.0;
                for (int s = 0; s < ns_; ++s) v -= (h_[s] - R * T) * Jc_[static_cast<std::size_t>(s) * ns_ + i];
                J[static_cast<std::size_t>(ns_) * n + i] = (v - Tdot * (cp_[i] - R)) / Cv;
            }
        }
    } else {
        kin_->reaction_enthalpies(T, dH_.data());
        const auto& ptr = kin_->dq_ptr();
        const auto& col = kin_->dq_col();
        const auto& val = kin_->dq_val();
        for (int j = 0; j < kin_->n_reactions(); ++j)
            for (int k = ptr[j]; k < ptr[j + 1]; ++k)
                J[static_cast<std::size_t>(ns_) * n + col[k]] -= dH_[j] * val[k] / cfg_.rho_cp;
    }

    // Temperature column by a forward difference (one extra rhs evaluation).
    const double dT = 1e-7 * std::max(T, 1.0);
    std::copy(y, y + n, y1_.begin());
    y1_[ns_] = T + dT;
    eval(t, y1_.data(), f1_.data());
    for (std::size_t s = 0; s < n; ++s) J[s * n + ns_] = (f1_[s] - f0_[s]) / dT;
}

}  // namespace ak
