# AutoKinetics 2 – Architecture

```
frontend/ (React + TS + Vite, Plotly)  ──HTTP/JSON──►  src/autokinetics/api (FastAPI)
                                                           │
             ┌─────────────────────────────────────────────┼──────────────────────────┐
             ▼                                             ▼                          ▼
 src/autokinetics/ (Python)                    src/autokinetics/numerics/   src/autokinetics/data/
  mechanism model, YAML/Cantera import,        solver registry, stability /  PubChem, NIST WebBook
  units, thermo, equilibrium, reactors,        consistency / convergence     (cached open APIs)
  analysis (ROP, sensitivity, fitting)         analysis, auto-tuning
             │                                             │
             └───────────────► _ext/_core (Cython, ONE extension module) ◄────┘
                                   │ _kinetics.pxi        │ _numerics.pxi
                                   ▼                      ▼
                    cpp/src/kinetics (thermo,      cpp/src/numerics (Runge–Kutta,
                    rates, Jacobian, reactors)     Rosenbrock, BDF, Newton, LU/QR,
                                                   iterative solvers, eigenvalues)
                                   └──── both meet at ak::OdeSystem ────┘
                                   (cpp/include/autokinetics/ode_system.hpp)
```

## Units

Internally everything is SI: mol, m³, s, J, K, Pa. Concentrations are mol/m³
(1 mol/L = 1000 mol/m³). Unit conversion happens only at the boundaries
(mechanism loading, API input/output).

## Shared contract: `ak::OdeSystem`

`cpp/include/autokinetics/ode_system.hpp` – `size()`, `rhs(t, y, dydt)`,
optional analytic `jacobian(t, y, J)` (dense, row-major), optional
`nonnegative_mask()`. Reactor models implement it; integrators consume it.

On the Python side every compiled system is a subclass of the Cython class
`OdeSystemHandle` (defined in `_numerics.pxi`), which holds the `OdeSystem*`.

## Numerics public API (`autokinetics.numerics`)

* `integrate(system, t_span, y0, method=..., rtol=..., atol=..., ...) -> IntegrationResult`
  – `system` is an `OdeSystemHandle` (C++) or a Python object with
  `rhs(t, y)` and optional `jacobian(t, y)`.
* `METHODS`, `LINEAR_SOLVERS`, `NONLINEAR_SOLVERS` – registries with metadata.
* `auto_optimize(system, t_span, y0, ...) -> AutoTuneReport` – stability,
  consistency and convergence analysis plus solver selection; JSON-serialisable.
* `autokinetics.numerics.linalg` – LU (with/without pivoting), Householder QR,
  Cholesky, Jacobi, Gauß–Seidel, CG/CGNR, condition numbers, eigenvalues
  (Hessenberg + QR algorithm), power / inverse iteration, Gershgorin discs.

See `docs/NUMERICS.md` for the full description of methods and analyses.

## Build

```
python setup.py build_ext --inplace    # or: pip install .
pytest                                  # pythonpath=src is configured
```
