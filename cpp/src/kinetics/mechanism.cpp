#include "autokinetics/kinetics/mechanism.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ak {

namespace {
constexpr double R = 8.31446261815324;          // J/(mol K)
constexpr double P_STD = 1.0e5;                 // Pa
constexpr double C_STD_LIQUID = 1000.0;         // mol/m^3
constexpr double SMALL = 1e-300;
constexpr double LN10 = 2.302585092994046;
const double NaN = std::numeric_limits<double>::quiet_NaN();

inline double arrhenius(double A, double b, double Ea_R, double T, double lnT) {
    return A * std::exp(b * lnT - Ea_R / T);
}

// c^o with exact integer powers (also for negative round-off concentrations).
inline double cpow(double c, double o) {
    if (o == 1.0) return c;
    if (o == 2.0) return c * c;
    if (o == 3.0) return c * c * c;
    if (o == std::floor(o) && o > 0.0 && o < 16.0) {
        double r = 1.0;
        for (int i = 0; i < static_cast<int>(o); ++i) r *= c;
        return r;
    }
    return std::pow(std::max(c, 0.0), o);
}

// d(c^o)/dc
inline double dcpow(double c, double o) {
    if (o == 1.0) return 1.0;
    if (o == 2.0) return 2.0 * c;
    if (o == std::floor(o) && o > 0.0 && o < 16.0) return o * cpow(c, o - 1.0);
    if (c <= 0.0) return 0.0;  // one-sided limit; avoids inf for orders < 1
    return o * std::pow(c, o - 1.0);
}

template <class T>
void require_size(const std::vector<T>& v, std::size_t n, const char* what) {
    if (v.size() != n)
        throw std::invalid_argument(std::string("MechanismData: inconsistent size of ") + what);
}
}  // namespace

Kinetics::Kinetics(MechanismData data) : d_(std::move(data)) {
    const int ns = d_.n_species, nr = d_.n_reactions;
    if (ns <= 0) throw std::invalid_argument("Mechanism has no species");
    require_size(d_.fwd_ptr, nr + 1, "fwd_ptr");
    require_size(d_.rev_ptr, nr + 1, "rev_ptr");
    require_size(d_.nu_ptr, nr + 1, "nu_ptr");
    require_size(d_.fwd_idx, d_.fwd_ptr[nr], "fwd_idx");
    require_size(d_.fwd_ord, d_.fwd_ptr[nr], "fwd_ord");
    require_size(d_.rev_idx, d_.rev_ptr[nr], "rev_idx");
    require_size(d_.rev_ord, d_.rev_ptr[nr], "rev_ord");
    require_size(d_.nu_idx, d_.nu_ptr[nr], "nu_idx");
    require_size(d_.nu_val, d_.nu_ptr[nr], "nu_val");
    for (auto* v : {&d_.dnu, &d_.A, &d_.b, &d_.Ea_R, &d_.rev_A, &d_.rev_b, &d_.rev_Ea_R, &d_.K_ref,
                    &d_.K_Tref, &d_.K_dH_R, &d_.dH_rxn})
        require_size(*v, nr, "per-reaction array");
    require_size(d_.rtype, nr, "rtype");
    require_size(d_.rev_mode, nr, "rev_mode");
    require_size(d_.tb_eff, d_.tb_rxn.size() * ns, "tb_eff");
    require_size(d_.fo_params, d_.fo_rxn.size() * 5, "fo_params");
    require_size(d_.plog_ptr, d_.plog_rxn.size() + 1, "plog_ptr");
    require_size(d_.has_thermo, ns, "has_thermo");
    require_size(d_.th_ptr, ns + 1, "th_ptr");
    require_size(d_.th_coef, d_.th_Tlo.size() * 9, "th_coef");
    for (int idx : d_.fwd_idx) if (idx < 0 || idx >= ns) throw std::out_of_range("species index");
    for (int idx : d_.rev_idx) if (idx < 0 || idx >= ns) throw std::out_of_range("species index");
    for (int idx : d_.nu_idx) if (idx < 0 || idx >= ns) throw std::out_of_range("species index");

    tb_pos_.assign(nr, -1);
    fo_pos_.assign(nr, -1);
    plog_pos_.assign(nr, -1);
    for (std::size_t k = 0; k < d_.tb_rxn.size(); ++k) tb_pos_.at(d_.tb_rxn[k]) = static_cast<int>(k);
    for (std::size_t k = 0; k < d_.fo_rxn.size(); ++k) fo_pos_.at(d_.fo_rxn[k]) = static_cast<int>(k);
    for (std::size_t k = 0; k < d_.plog_rxn.size(); ++k) plog_pos_.at(d_.plog_rxn[k]) = static_cast<int>(k);
    for (int j = 0; j < nr; ++j) {
        if ((d_.rtype[j] == RXN_THREE_BODY || d_.rtype[j] == RXN_FALLOFF) && tb_pos_[j] < 0)
            throw std::invalid_argument("three-body/fall-off reaction without efficiencies");
        if (d_.rtype[j] == RXN_FALLOFF && fo_pos_[j] < 0)
            throw std::invalid_argument("fall-off reaction without low-pressure data");
        if (d_.rtype[j] == RXN_PLOG && plog_pos_[j] < 0)
            throw std::invalid_argument("PLOG reaction without rate table");
    }
    for (int i = 0; i < ns; ++i) all_thermo_ = all_thermo_ && d_.has_thermo[i];

    kf_base_.assign(nr, 0.0); Kc_.assign(nr, NaN); krev_arr_.assign(nr, 0.0);
    k0_.assign(d_.fo_rxn.size(), 0.0);
    cp_.assign(ns, NaN); h_.assign(ns, NaN); s_.assign(ns, NaN); g_RT_.assign(ns, NaN);
    kf_.assign(nr, 0.0); kr_.assign(nr, 0.0); M_.assign(nr, 1.0);
    qf_.assign(nr, 0.0); qr_.assign(nr, 0.0); q_.assign(nr, 0.0); dk_dM_.assign(nr, 0.0);
}

void Kinetics::species_thermo(double T, double* cp, double* h, double* s) const {
    const double lnT = std::log(T), T2 = T * T;
    for (int i = 0; i < d_.n_species; ++i) {
        const int lo = d_.th_ptr[i], hi = d_.th_ptr[i + 1];
        if (hi == lo) { cp[i] = h[i] = s[i] = NaN; continue; }
        int k = hi - 1;
        for (int kk = lo; kk < hi; ++kk) if (T <= d_.th_Thi[kk]) { k = kk; break; }
        const double* a = &d_.th_coef[9 * k];
        cp[i] = R * (a[0] / T2 + a[1] / T + a[2] + T * (a[3] + T * (a[4] + T * (a[5] + T * a[6]))));
        h[i] = R * T * (-a[0] / T2 + a[1] * lnT / T + a[2] +
                        T * (a[3] / 2 + T * (a[4] / 3 + T * (a[5] / 4 + T * a[6] / 5))) + a[7] / T);
        s[i] = R * (-a[0] / (2 * T2) - a[1] / T + a[2] * lnT +
                    T * (a[3] + T * (a[4] / 2 + T * (a[5] / 3 + T * a[6] / 4))) + a[8]);
    }
}

void Kinetics::update_T(double T, double P) {
    if (!(T > 0.0) || !std::isfinite(T)) throw std::domain_error("temperature must be positive and finite");
    const bool T_changed = (T != cached_T_);
    const bool P_changed = (P != cached_P_);
    if (!T_changed && !P_changed) return;
    const int nr = d_.n_reactions;
    const double lnT = std::log(T);
    if (T_changed) {
        for (int j = 0; j < nr; ++j)
            if (d_.rtype[j] != RXN_PLOG) kf_base_[j] = arrhenius(d_.A[j], d_.b[j], d_.Ea_R[j], T, lnT);
        for (std::size_t k = 0; k < d_.fo_rxn.size(); ++k)
            k0_[k] = arrhenius(d_.fo_A[k], d_.fo_b[k], d_.fo_Ea_R[k], T, lnT);
        species_thermo(T, cp_.data(), h_.data(), s_.data());
        for (int i = 0; i < d_.n_species; ++i) g_RT_[i] = (h_[i] - T * s_[i]) / (R * T);
        const double c0 = (d_.phase == PHASE_GAS) ? P_STD / (R * T) : C_STD_LIQUID;
        for (int j = 0; j < nr; ++j) {
            switch (d_.rev_mode[j]) {
                case REV_ARRHENIUS:
                    krev_arr_[j] = arrhenius(d_.rev_A[j], d_.rev_b[j], d_.rev_Ea_R[j], T, lnT);
                    Kc_[j] = NaN;
                    break;
                case REV_EQCONST:
                    Kc_[j] = d_.K_ref[j] * std::exp(-d_.K_dH_R[j] * (1.0 / T - 1.0 / d_.K_Tref[j]));
                    break;
                case REV_THERMO: {
                    double dg = 0.0;
                    for (int k = d_.nu_ptr[j]; k < d_.nu_ptr[j + 1]; ++k) dg += d_.nu_val[k] * g_RT_[d_.nu_idx[k]];
                    Kc_[j] = std::exp(std::min(-dg, 700.0)) * std::pow(c0, d_.dnu[j]);
                    break;
                }
                default:
                    Kc_[j] = NaN;
            }
        }
    }
    // PLOG depends on T and P
    for (std::size_t k = 0; k < d_.plog_rxn.size(); ++k) {
        const int j = d_.plog_rxn[k];
        const int lo = d_.plog_ptr[k], hi = d_.plog_ptr[k + 1];
        const double lnP = std::log(std::max(P, SMALL));
        // collect unique pressures (entries are sorted by pressure); sum duplicates
        double lnP_prev = 0.0, k_prev = 0.0;
        bool have_prev = false, done = false;
        int i = lo;
        while (i < hi && !done) {
            const double lp = d_.plog_lnP[i];
            double ksum = 0.0;
            while (i < hi && d_.plog_lnP[i] == lp) {
                ksum += arrhenius(d_.plog_A[i], d_.plog_b[i], d_.plog_Ea_R[i], T, lnT);
                ++i;
            }
            if (lnP <= lp) {
                if (!have_prev) kf_base_[j] = ksum;
                else {
                    const double l1 = std::log(std::max(k_prev, SMALL)), l2 = std::log(std::max(ksum, SMALL));
                    kf_base_[j] = std::exp(l1 + (l2 - l1) * (lnP - lnP_prev) / (lp - lnP_prev));
                }
                done = true;
            }
            lnP_prev = lp; k_prev = ksum; have_prev = true;
        }
        if (!done) kf_base_[j] = k_prev;
    }
    cached_T_ = T;
    cached_P_ = P;
}

void Kinetics::equilibrium_constants(double T, double* Kc) {
    update_T(T, cached_P_ > 0 ? cached_P_ : P_STD);
    std::copy(Kc_.begin(), Kc_.end(), Kc);
}

double Kinetics::falloff_k(int k, double T, double kinf, double conc_m, double* dk_dm) const {
    const double k0 = k0_[k];
    if (kinf <= 0.0) { *dk_dm = 0.0; return 0.0; }
    const double pr = conc_m * k0 / kinf;
    const double* p = &d_.fo_params[5 * k];
    double F = 1.0, dlnF_dlpr = 0.0;  // d ln F / d log10(Pr)
    const double lpr = std::log10(std::max(pr, SMALL));
    switch (d_.fo_kind[k]) {
        case FALLOFF_TROE: {
            const double A = p[0], T3 = p[1], T1 = p[2], T2 = p[3];
            double fcent = (1.0 - A) * (std::fabs(T3) > SMALL ? std::exp(-T / T3) : 0.0) +
                           A * (std::fabs(T1) > SMALL ? std::exp(-T / T1) : 0.0);
            if (p[4] != 0.0) fcent += std::exp(-T2 / T);
            const double log_fc = std::log10(std::max(fcent, SMALL));
            const double cc = -0.4 - 0.67 * log_fc;
            const double nn = 0.75 - 1.27 * log_fc;
            const double u = lpr + cc;
            const double den = nn - 0.14 * u;
            const double f1 = u / den;
            const double log_f = log_fc / (1.0 + f1 * f1);
            F = std::pow(10.0, log_f);
            const double df1 = nn / (den * den);
            dlnF_dlpr = LN10 * (-log_fc * 2.0 * f1 / ((1.0 + f1 * f1) * (1.0 + f1 * f1)) * df1);
            break;
        }
        case FALLOFF_SRI: {
            const double a = p[0], bb = p[1], c = p[2], dd = p[3], e = p[4];
            const double X = 1.0 / (1.0 + lpr * lpr);
            const double base = a * std::exp(-bb / T) + std::exp(-T / c);
            F = dd * std::pow(base, X) * std::pow(T, e);
            dlnF_dlpr = std::log(base) * (-2.0 * lpr * X * X);
            break;
        }
        default:
            break;
    }
    const double frac = pr / (1.0 + pr);
    const double kval = kinf * frac * F;
    // dk/dM = dk/dPr * k0/kinf ; d(frac)/dPr = 1/(1+Pr)^2 ; dF/dPr = F * dlnF/dlpr / (Pr ln10)
    if (pr > SMALL) {
        const double dfrac = 1.0 / ((1.0 + pr) * (1.0 + pr));
        const double dF = F * dlnF_dlpr / (pr * LN10);
        *dk_dm = kinf * (dfrac * F + frac * dF) * (k0 / kinf);
    } else {
        *dk_dm = k0 * F;  // limit Pr -> 0: k ~ k0 [M] F
    }
    return kval;
}

void Kinetics::rate_constants(double T, double P, const double* c, double* kf, double* kr, double* M) {
    update_T(T, P);
    const int nr = d_.n_reactions, ns = d_.n_species;
    for (int j = 0; j < nr; ++j) {
        double kfj = kf_base_[j], m = 1.0, dkdm = 0.0;
        const int tb = tb_pos_[j];
        if (tb >= 0) {
            const double* eff = &d_.tb_eff[static_cast<std::size_t>(tb) * ns];
            double cm = 0.0;
            for (int i = 0; i < ns; ++i) cm += eff[i] * c[i];
            if (d_.rtype[j] == RXN_THREE_BODY) m = cm;
            else kfj = falloff_k(fo_pos_[j], T, kf_base_[j], cm, &dkdm);
        }
        kf[j] = kfj;
        M[j] = m;
        dk_dM_[j] = dkdm;
        switch (d_.rev_mode[j]) {
            case REV_ARRHENIUS: kr[j] = krev_arr_[j]; break;
            case REV_THERMO:
            case REV_EQCONST: kr[j] = kfj / Kc_[j]; break;
            default: kr[j] = 0.0;
        }
    }
}

void Kinetics::rates_of_progress(double T, double P, const double* c, double* qf, double* qr, double* q) {
    rate_constants(T, P, c, kf_.data(), kr_.data(), M_.data());
    for (int j = 0; j < d_.n_reactions; ++j) {
        double pf = kf_[j];
        for (int k = d_.fwd_ptr[j]; k < d_.fwd_ptr[j + 1]; ++k) pf *= cpow(c[d_.fwd_idx[k]], d_.fwd_ord[k]);
        double pr = 0.0;
        if (kr_[j] != 0.0) {
            pr = kr_[j];
            for (int k = d_.rev_ptr[j]; k < d_.rev_ptr[j + 1]; ++k) pr *= cpow(c[d_.rev_idx[k]], d_.rev_ord[k]);
        }
        qf[j] = M_[j] * pf;
        qr[j] = M_[j] * pr;
        q[j] = qf[j] - qr[j];
    }
}

void Kinetics::production_rates(double T, double P, const double* c, double* wdot, double* q) {
    rates_of_progress(T, P, c, qf_.data(), qr_.data(), q_.data());
    std::fill(wdot, wdot + d_.n_species, 0.0);
    for (int j = 0; j < d_.n_reactions; ++j) {
        const double qj = q_[j];
        for (int k = d_.nu_ptr[j]; k < d_.nu_ptr[j + 1]; ++k) wdot[d_.nu_idx[k]] += d_.nu_val[k] * qj;
    }
    if (q) std::copy(q_.begin(), q_.end(), q);
}

void Kinetics::jacobian_conc(double T, double P, const double* c, double* J) {
    const int ns = d_.n_species, nr = d_.n_reactions;
    rate_constants(T, P, c, kf_.data(), kr_.data(), M_.data());
    std::fill(J, J + static_cast<std::size_t>(ns) * ns, 0.0);
    dq_ptr_.assign(1, 0);
    dq_col_.clear();
    dq_val_.clear();

    auto add = [&](int j, int i, double v) {
        if (v == 0.0) return;
        dq_col_.push_back(i);
        dq_val_.push_back(v);
        for (int k = d_.nu_ptr[j]; k < d_.nu_ptr[j + 1]; ++k)
            J[static_cast<std::size_t>(d_.nu_idx[k]) * ns + i] += d_.nu_val[k] * v;
    };

    for (int j = 0; j < nr; ++j) {
        const int f0 = d_.fwd_ptr[j], f1 = d_.fwd_ptr[j + 1];
        const int r0 = d_.rev_ptr[j], r1 = d_.rev_ptr[j + 1];
        double pf = 1.0, pr = 1.0;
        for (int k = f0; k < f1; ++k) pf *= cpow(c[d_.fwd_idx[k]], d_.fwd_ord[k]);
        for (int k = r0; k < r1; ++k) pr *= cpow(c[d_.rev_idx[k]], d_.rev_ord[k]);
        const double m = M_[j];
        // mass-action part: d(prod)/dc_i = dcpow(c_i) * prod_{others}
        for (int k = f0; k < f1; ++k) {
            double others = 1.0;
            for (int l = f0; l < f1; ++l) if (l != k) others *= cpow(c[d_.fwd_idx[l]], d_.fwd_ord[l]);
            add(j, d_.fwd_idx[k], m * kf_[j] * dcpow(c[d_.fwd_idx[k]], d_.fwd_ord[k]) * others);
        }
        if (kr_[j] != 0.0) {
            for (int k = r0; k < r1; ++k) {
                double others = 1.0;
                for (int l = r0; l < r1; ++l) if (l != k) others *= cpow(c[d_.rev_idx[l]], d_.rev_ord[l]);
                add(j, d_.rev_idx[k], -m * kr_[j] * dcpow(c[d_.rev_idx[k]], d_.rev_ord[k]) * others);
            }
        }
        // third-body / fall-off dependence on [M] = sum eff_i c_i
        const int tb = tb_pos_[j];
        if (tb >= 0) {
            double factor;
            if (d_.rtype[j] == RXN_THREE_BODY) {
                factor = kf_[j] * pf - kr_[j] * pr;
            } else {
                const double dkr = (d_.rev_mode[j] == REV_THERMO || d_.rev_mode[j] == REV_EQCONST)
                                       ? dk_dM_[j] / Kc_[j] : 0.0;
                factor = dk_dM_[j] * pf - dkr * pr;
            }
            if (factor != 0.0) {
                const double* eff = &d_.tb_eff[static_cast<std::size_t>(tb) * ns];
                for (int i = 0; i < ns; ++i) if (eff[i] != 0.0) add(j, i, eff[i] * factor);
            }
        }
        dq_ptr_.push_back(static_cast<int>(dq_col_.size()));
    }
}

void Kinetics::reaction_enthalpies(double T, double* dH) {
    update_T(T, cached_P_ > 0 ? cached_P_ : P_STD);
    for (int j = 0; j < d_.n_reactions; ++j) {
        if (!std::isnan(d_.dH_rxn[j])) { dH[j] = d_.dH_rxn[j]; continue; }
        double v = 0.0;
        for (int k = d_.nu_ptr[j]; k < d_.nu_ptr[j + 1]; ++k) v += d_.nu_val[k] * h_[d_.nu_idx[k]];
        dH[j] = v;
    }
}

}  // namespace ak
