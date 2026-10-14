// Reactor models as ODE systems (implement ak::OdeSystem).
//
// State vector layout
//   constant volume  : y = [c_0 .. c_{N-1}]          (mol/m^3)   (+ T if energy balance)
//   constant pressure: y = [n_0 .. n_{N-1}]          (mol per m^3 of *initial* volume) (+ T)
//
// Energy balances (q_ext = UA/V * (T_env - T), W/m^3):
//   ideal gas, const V : sum(c_i cv_i) dT/dt = -sum(u_i wdot_i) + q_ext
//   ideal gas, const P : sum(n_i cp_i) dT/dt = -V sum(h_i wdot_i) + V q_ext
//   liquid             : rho cp dT/dt        = -sum_j dH_j q_j + q_ext (+ rho cp (T_in - T)/tau for CSTR)
// CSTR (constant volume): dc/dt = wdot + (c_in - c)/tau
// Buffered ("fixed") species keep their concentration: dc_i/dt = 0 (constant volume only).
#pragma once

#include <memory>
#include <vector>

#include "autokinetics/kinetics/mechanism.hpp"
#include "autokinetics/ode_system.hpp"

namespace ak {

enum EnergyMode : int { ENERGY_ISOTHERMAL = 0, ENERGY_BALANCE = 1 };
enum Constraint : int { CONST_VOLUME = 0, CONST_PRESSURE = 1 };
enum FlowMode : int { FLOW_BATCH = 0, FLOW_CSTR = 1 };

struct ReactorConfig {
    int energy = ENERGY_ISOTHERMAL;
    int constraint = CONST_VOLUME;
    int flow = FLOW_BATCH;
    double P0 = 1.0e5;                    // Pa (constant-pressure reactor; reference for liquids)
    std::vector<double> T_prog_t{0.0};    // isothermal: temperature program (piecewise linear)
    std::vector<double> T_prog_T{300.0};
    double UA_V = 0.0;                    // W/(m^3 K)
    std::vector<double> T_env_t{0.0};     // ambient / coolant temperature program
    std::vector<double> T_env_T{300.0};
    double rho_cp = 4.18e6;               // J/(m^3 K), liquid energy balance
    double tau = 0.0;                     // s, CSTR residence time
    std::vector<double> c_in;             // mol/m^3, CSTR feed
    double T_in = 300.0;                  // K, CSTR feed temperature
    std::vector<unsigned char> fixed;     // buffered species (constant concentration), size N or empty
};

double piecewise_linear(const std::vector<double>& t, const std::vector<double>& v, double x);

class ReactorSystem final : public OdeSystem {
public:
    ReactorSystem(std::shared_ptr<Kinetics> kin, ReactorConfig cfg);

    std::size_t size() const override { return n_state_; }
    void rhs(double t, const double* y, double* dydt) override;
    bool has_jacobian() const override { return true; }
    void jacobian(double t, const double* y, double* J) override;
    const unsigned char* nonnegative_mask() const override { return mask_.data(); }

    // Derived state: concentrations, T, P, V at (t, y).
    void observe(double t, const double* y, double* c, double* T, double* P, double* V) const;

    const ReactorConfig& config() const { return cfg_; }
    Kinetics& kinetics() { return *kin_; }
    long n_rhs() const { return n_rhs_; }
    long n_jac() const { return n_jac_; }

private:
    void eval(double t, const double* y, double* dydt);

    std::shared_ptr<Kinetics> kin_;
    ReactorConfig cfg_;
    int ns_;
    std::size_t n_state_;
    std::vector<unsigned char> mask_;
    std::vector<double> c_, wdot_, q_, cp_, h_, s_, dH_, Jc_, f0_, f1_, y1_;
    long n_rhs_ = 0, n_jac_ = 0;
};

}  // namespace ak
