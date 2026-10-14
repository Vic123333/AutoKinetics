# AutoKinetics 2 — Chemical Reaction Network Simulator

AutoKinetics simulates real chemical processes: gas-phase combustion mechanisms (e.g. the full
GRI-Mech 3.0), liquid-phase kinetics, enzyme kinetics, oscillating reactions and thermal runaway in
cooled reactors. A C++ core (bound via Cython) evaluates thermodynamics, rates and analytic Jacobians;
self-implemented C++ integrators and linear solvers handle stiff systems; a web UI (German) makes it
usable without programming.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg) ![C++17](https://img.shields.io/badge/C++-17-blue.svg) ![License](https://img.shields.io/badge/license-GPLv3-green.svg)

## Features

**Chemistry**
- Cantera-compatible YAML mechanisms (NASA-7/9, Shomate, constant-cp thermo; elementary, three-body,
  fall-off Lindemann/Troe/SRI, PLOG reactions) — import/export, plus legacy AutoKinetics 1.x `.kin` files
- Reversible reactions from thermodynamics (detailed balance via ΔG°), explicit reverse Arrhenius or K(T)
- Validation: element and charge balance, missing thermo, duplicates — with clear (German) messages
- Reactors: batch at constant volume or pressure, isothermal / temperature programme / adiabatic /
  heat exchange (UA), liquid energy balance, CSTR, buffered species
- Chemical equilibrium (Gibbs/Helmholtz minimisation; TP, TV, HP, UV), e.g. adiabatic flame temperatures
- Analysis: rates of production, sensitivities, reaction-order and Arrhenius fits (95 % CIs),
  parameter estimation from measured concentration data

**Open data**
- PubChem (identity, formula, CAS), NIST Chemistry WebBook (ΔfH°, S°, Shomate / cp(T) fits),
  bundled NASA thermo database (McBride et al., 748 species) for offline use; cached and rate-limited

**Numerics** (see [docs/NUMERICS.md](docs/NUMERICS.md))
- C++ integrators: explicit RK (Euler, Heun, RK4, Dormand–Prince), implicit RK (implicit Euler/midpoint,
  trapezoidal, Gauss–Legendre, Radau IIA 3/5, SDIRK2/3), Rosenbrock ROS2, variable-step BDF2;
  Newton / simplified Newton / fixed-point iteration; LU (with/without pivoting), Householder QR,
  Jacobi, Gauß–Seidel, CGNR; eigenvalues via Hessenberg + Francis QR
- **Auto-optimisation**: stability (Jacobian spectra, stiffness, stability regions), consistency
  (order conditions, empirical local order), convergence (global order, tolerance proportionality,
  Newton contraction), then picks method, solvers and tolerance with the least work for a requested
  global accuracy

**Example library** (with literature sources): N₂O₅ decomposition, cyclopropane isomerisation,
N₂O₄ ⇌ 2 NO₂, esterification (K = 4), saponification, Michaelis–Menten, Oregonator, Robertson,
thermal runaway (Semenov), CSTR, H₂/O₂ and methane/air ignition (GRI-Mech 3.0).

## Validation against physical reality

The test-suite (`pytest`, ~280 tests + Playwright end-to-end tests) checks, among others:

| Check | Result |
|---|---|
| Rates/thermo vs. Cantera (H₂/O₂, GRI-Mech 3.0, n-dodecane, fall-off/SRI/PLOG) | rel. deviation ≤ 1e-10 |
| Ignition delay and final temperature vs. Cantera reactors | ≤ 2e-3 / ≤ 1e-6 |
| Equilibrium vs. Cantera (TP/HP/UV/TV) | ΔT ≤ 1e-9 rel., Δx ≤ 1e-10 |
| Adiabatic flame temperature CH₄/air, H₂/air | 2225.5 K, 2380.8 K (literature ≈ 2226 K, ≈ 2390 K) |
| Analytical solutions (0th/½/1st/2nd order, Bateman, reversible, CSTR, T-programme) | ≤ 1e-7 |
| Robertson reference solution (Hairer & Wanner) | ≤ 1e-6 |
| Conservation of elements, mass, enthalpy (const p) and internal energy (const V) | ≤ 1e-12 / 1e-7 |
| Long-time kinetics = Gibbs equilibrium; detailed balance k_f/k_r = K_c | ✓ |
| N₂O₄ K_p(298 K) ≈ 0.15, Le Chatelier, esterification conversion 2/3 | ✓ |
| Every integrator reaches its theoretical order; stability functions match closed forms | ✓ |

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"            # builds the C++/Cython extension
(cd frontend && npm ci && npm run build)
autokinetics serve                 # http://127.0.0.1:8000
```

Docker:

```bash
docker compose up --build          # http://localhost:8000
```

CLI:

```bash
autokinetics examples
autokinetics example methane_air_gri30 --out result.csv
autokinetics run my_mechanism.yaml --reactor reactor.yaml --t-end 10 --method radau5
autokinetics validate my_mechanism.yaml
autokinetics lookup ethanol
```

Python:

```python
from autokinetics.library import load_example
from autokinetics.presets import reactor_spec_from_dict, options_from_dict
from autokinetics.simulation import simulate

mech, meta, _ = load_example("hydrogen_oxygen_ignition")
spec = reactor_spec_from_dict({"constraint": "pressure", "energy": "adiabatic", "T": 1000, "P": "1 atm",
                               "composition": {"H2": 2, "O2": 1, "AR": 7}}, mech)
res = simulate(mech, spec, options_from_dict({"t_end": 2e-3, "method": "radau5"}))
print(res.metrics["ignition_delay"], res.T[-1])
```

## Architecture

```
frontend/ (React + TypeScript + Plotly)  ──REST──►  src/autokinetics/api (FastAPI, process pool)
src/autokinetics/  mechanism model, Cantera YAML I/O, units, thermo, reactors, equilibrium, analysis
src/autokinetics/numerics/  solver registry, stability/consistency/convergence analysis, auto-tuner
src/autokinetics/data/  PubChem, NIST WebBook, NASA database
cpp/src/kinetics/  thermo, rates, analytic Jacobian, reactor models   ┐ one Cython extension,
cpp/src/numerics/  integrators, Newton, linear algebra, eigenvalues   ┘ joined at ak::OdeSystem
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/NUMERICS.md](docs/NUMERICS.md).

## Configuration (environment)

| Variable | Default | Meaning |
|---|---|---|
| `AUTOKINETICS_WORKERS` | CPUs/2 | worker processes for simulations |
| `AUTOKINETICS_MAX_WALL_TIME` | 120 | time limit per job [s] |
| `AUTOKINETICS_MAX_SPECIES` / `_MAX_REACTIONS` | 2000 / 20000 | mechanism size limits |
| `AUTOKINETICS_OFFLINE` | 0 | `1`: no PubChem/NIST requests |
| `AUTOKINETICS_CACHE_DIR` | `~/.cache/autokinetics` | cache for open-data responses |
| `AUTOKINETICS_CORS_ORIGINS` | – | comma-separated origins (only for a separately hosted frontend) |

## Development

```bash
python setup.py build_ext --inplace     # rebuild C++ after changes
pytest -m "not network"                 # all offline tests
(cd frontend && npx vitest run && npx playwright test)
```

## Data sources and licences

GRI-Mech 3.0 (Smith et al.) and the H₂/O₂ submechanism are redistributed from Cantera's data files;
NASA thermodynamic data: McBride, Gordon & Reno, NASA TM-4513 (public domain). Code licensed under
GPL-3.0-or-later (see `LICENSE.md`).
