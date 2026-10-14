// Compiled reaction mechanism: thermodynamics, rate constants, rates of
// progress, production rates and the analytic Jacobian d(wdot)/dc.
//
// All quantities are SI: mol, m^3, s, J, K, Pa. The data layout mirrors
// autokinetics.compiled.CompiledMechanism (Python), which fills it.
#pragma once

#include <cstddef>
#include <vector>

namespace ak {

enum RxnType : int { RXN_ELEMENTARY = 0, RXN_THREE_BODY = 1, RXN_FALLOFF = 2, RXN_PLOG = 3 };
enum RevMode : int { REV_NONE = 0, REV_THERMO = 1, REV_ARRHENIUS = 2, REV_EQCONST = 3 };
enum FalloffKind : int { FALLOFF_LINDEMANN = 0, FALLOFF_TROE = 1, FALLOFF_SRI = 2 };
enum PhaseKind : int { PHASE_GAS = 0, PHASE_LIQUID = 1 };

struct MechanismData {
    int n_species = 0;
    int n_reactions = 0;
    int phase = PHASE_GAS;

    std::vector<int> fwd_ptr, fwd_idx;  std::vector<double> fwd_ord;
    std::vector<int> rev_ptr, rev_idx;  std::vector<double> rev_ord;
    std::vector<int> nu_ptr, nu_idx;    std::vector<double> nu_val;
    std::vector<double> dnu;
    std::vector<int> rtype;
    std::vector<double> A, b, Ea_R;

    std::vector<int> tb_rxn;      // reactions with third-body efficiencies
    std::vector<double> tb_eff;   // n_tb x n_species, row-major

    std::vector<int> fo_rxn, fo_kind;
    std::vector<double> fo_A, fo_b, fo_Ea_R;
    std::vector<double> fo_params;  // n_fo x 5

    std::vector<int> plog_rxn, plog_ptr;
    std::vector<double> plog_lnP, plog_A, plog_b, plog_Ea_R;

    std::vector<int> rev_mode;
    std::vector<double> rev_A, rev_b, rev_Ea_R;
    std::vector<double> K_ref, K_Tref, K_dH_R;

    std::vector<unsigned char> has_thermo;
    std::vector<int> th_ptr;
    std::vector<double> th_Tlo, th_Thi, th_coef;  // coef: n_ranges x 9

    std::vector<double> dH_rxn;  // NaN -> from thermo
};

class Kinetics {
public:
    explicit Kinetics(MechanismData data);

    int n_species() const { return d_.n_species; }
    int n_reactions() const { return d_.n_reactions; }
    int phase() const { return d_.phase; }
    bool all_thermo() const { return all_thermo_; }
    const MechanismData& data() const { return d_; }

    // Molar cp [J/mol/K], h [J/mol], s [J/mol/K] at p° (NaN without thermo).
    void species_thermo(double T, double* cp, double* h, double* s) const;

    // Concentration-equilibrium constants (NaN when not determinable).
    void equilibrium_constants(double T, double* Kc);

    // kf (incl. fall-off at the given composition), kr, and third-body factor M.
    void rate_constants(double T, double P, const double* c, double* kf, double* kr, double* M);

    // Forward / reverse / net rates of progress [mol/m^3/s].
    void rates_of_progress(double T, double P, const double* c, double* qf, double* qr, double* q);

    // Net molar production rates wdot [mol/m^3/s]; optional q output (may be nullptr).
    void production_rates(double T, double P, const double* c, double* wdot, double* q = nullptr);

    // Analytic Jacobian of wdot w.r.t. concentrations at constant T, P (dense,
    // row-major n x n, J[s*n+i] = d wdot_s / d c_i). If dq_dc_out is given it
    // receives a callback-free sparse representation via dq_row()/dq_col()/dq_val().
    void jacobian_conc(double T, double P, const double* c, double* J);

    // Sparse d q_j / d c_i from the last jacobian_conc() call (CSR over reactions).
    const std::vector<int>& dq_ptr() const { return dq_ptr_; }
    const std::vector<int>& dq_col() const { return dq_col_; }
    const std::vector<double>& dq_val() const { return dq_val_; }

    // Reaction enthalpies [J/mol] at T (user value if given, else from thermo).
    void reaction_enthalpies(double T, double* dH);

private:
    void update_T(double T, double P);
    double falloff_k(int k, double T, double kinf, double conc_m, double* dk_dm) const;

    MechanismData d_;
    bool all_thermo_ = true;
    std::vector<int> tb_pos_, fo_pos_, plog_pos_;  // reaction -> row in tb/fo/plog tables, -1 if none

    // cache of T-dependent quantities
    double cached_T_ = -1.0, cached_P_ = -1.0;
    std::vector<double> kf_base_, Kc_, krev_arr_, k0_;
    std::vector<double> cp_, h_, s_, g_RT_;

    // work arrays
    std::vector<double> kf_, kr_, M_, qf_, qr_, q_, dk_dM_;
    std::vector<int> dq_ptr_, dq_col_;
    std::vector<double> dq_val_;
};

}  // namespace ak
