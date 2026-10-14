# AutoKinetics 2 – Numerics

Package `autokinetics.numerics` (Python) on top of the C++ library `ak::num`
(`cpp/include/autokinetics/numerics`, `cpp/src/numerics`), bound through
`src/autokinetics/_ext/_numerics.pxi` / `num_core.pxd`.

Lecture references:
* **ODE** – *Numerik gewöhnlicher Differentialgleichungen* (Lederer), Kap. 1–8
* **Numerik I** – Vorlesungsskript Numerik I, Kap. 1–9

## 1. Public API

```python
from autokinetics.numerics import (
    integrate, IntegrationResult, as_system,          # integration
    auto_optimize, AutoTuneReport,                    # automatic solver selection
    METHODS, LINEAR_SOLVERS, NONLINEAR_SOLVERS, get_method,
    ButcherTableau,                                   # custom RK methods
    OdeSystemHandle, PythonOdeSystem,                 # system handles
    LinearOdeSystem, RobertsonSystem, VanDerPolSystem,
    linalg, stability, consistency, convergence, interpolation, quadrature,
)

integrate(system, t_span, y0, method="dopri5", *, rtol=1e-6, atol=1e-12,
          linear_solver="lu_pivot", nonlinear_solver="simplified_newton",
          fixed_step=None, t_eval=None, t_eval_exact=False, max_steps=100_000,
          h0=None, h_min=None, h_max=None, max_wall_time=None, nonnegative=False,
          newton_tol=0.03, max_newton_iter=None, linear_tol=1e-12,
          linear_max_iter=None) -> IntegrationResult

auto_optimize(system, t_span, y0, rtol=1e-6, atol=1e-12, budget_s=20.0,
              invariants=None, candidates=None, linear_solvers=None,
              nonlinear_solvers=None) -> AutoTuneReport
```

`system` is an `OdeSystemHandle` (compiled; the GIL is released during
integration), any object with `rhs(t, y)` and optional `jacobian(t, y)` /
`nonnegative_mask`, or a plain callable `f(t, y)`. Python objects are wrapped
in a C++ callback system; exceptions raised in the callbacks abort the
integration and are re-raised unchanged.

`IntegrationResult`: `t (m,)`, `y (m, n)`, `success`, `status`
(`success | max_steps | wall_time | step_size_too_small |
nonlinear_solver_failure | non_finite | scipy_failure`), `message` (German),
`detail` (technical), `method`, `stats`, `wall_time`, `options`, `to_dict()`.

`stats` (C++ methods): `n_steps, n_accepted, n_rejected, n_error_rejections,
n_solver_rejections, n_nonneg_rejections, n_rhs, n_jac, n_jac_fd,
n_factorizations, n_linear_solves, n_linear_iterations, n_linear_failures,
n_newton_iterations, n_newton_failures, theta_max, theta_mean, h_min, h_max,
h_mean`. scipy backends report `n_rhs, n_jac, n_factorizations, n_accepted`.

### Options in detail

| option | meaning |
|---|---|
| `rtol`, `atol` | local error test `‖err / (atol + rtol·max(|y_n|,|y_{n+1}|))‖_RMS ≤ 1`; in fixed-step mode they only scale the Newton termination test |
| `fixed_step` | constant step `h = (tf−t0)/N`, `N = ⌈(tf−t0)/fixed_step⌉`; no error control; a solver failure ends the run with `nonlinear_solver_failure` |
| `t_eval` | output times; values between steps by Hermite cubic interpolation |
| `t_eval_exact` | adaptive mode: steps are shortened to end exactly on `t_eval` (no interpolation error) |
| `max_steps`, `max_wall_time` | budgets; the partial solution is returned with `success=False` |
| `nonnegative` | uses the system's `nonnegative_mask`: steps with components `< −atol` are rejected (h halved), round-off negatives are clipped to 0 |
| `newton_tol` | κ in the stopping rule `θ/(1−θ)·‖ΔZ‖ ≤ κ` (scaled norm, Numerik I Alg. 5.10) |
| `linear_tol`, `linear_max_iter` | relative residual / iterations of iterative linear solvers inside Newton |

## 2. Methods (`METHODS`)

| id | family | order | stages | A-stable | L-stable | error estimate | lecture |
|---|---|---|---|---|---|---|---|
| `euler` | explicit RK | 1 | 1 | – | – | step doubling | ODE 2.1, Kor. 2.9 |
| `heun` | explicit RK | 2 | 2 | – | – | embedded Euler (2(1)) | ODE 2.1/3, 4.3 |
| `rk4` | explicit RK | 4 | 4 | – | – | step doubling | ODE 3 (Satz 3.4, Bem. 3.5) |
| `dopri5` | explicit RK | 5 | 7 (FSAL) | – | – | embedded order 4, PI control | ODE 4.3 |
| `implicit_euler` | implicit RK (Radau IIA s=1) | 1 | 1 | ✓ | ✓ | step doubling | ODE 2.1, 5.6 |
| `implicit_midpoint` | Gauß s=1 | 2 | 1 | ✓ | – | step doubling | ODE 5.4 (Bsp. 5.10) |
| `trapezoidal` | Lobatto IIIA s=2 | 2 | 2 | ✓ | – | step doubling | ODE 5.4 (Bem. 5.12) |
| `gauss4` | Gauß s=2 | 4 | 2 | ✓ (B-stable) | – | step doubling | ODE 5.4, Lemma 5.21 |
| `radau3` | Radau IIA s=2 | 3 | 2 | ✓ | ✓ | step doubling | ODE 5.4 (Bem. 5.12) |
| `radau5` | Radau IIA s=3 | 5 | 3 | ✓ | ✓ | step doubling | ODE 5.4 (Bem. 5.12) |
| `sdirk2` | SDIRK (Alexander, γ=1−√2/2) | 2 | 2 | ✓ | ✓ | step doubling | ODE 5.5 |
| `sdirk3` | SDIRK (Crouzeix, γ=(3+√3)/6) | 3 | 2 | ✓ | – (|R(∞)|=√3−1) | step doubling | ODE 5.5 |
| `ros2` | Rosenbrock (Verwer et al. 1999, γ=1+1/√2) | 2 | 2 | ✓ | ✓ | embedded order 1 | ODE 5.5 |
| `bdf2` | BDF, variable step | 2 | – | ✓ | ✓ (roots→0) | predictor (Milne) | ODE 7.3, 7.5, 7.6 |
| `scipy:RK45`, `scipy:BDF`, `scipy:Radau`, `scipy:LSODA` | scipy reference | – | – | see registry | | internal | |

All entries carry `label_de`, `description_de`, `stability_de`,
`lecture_ref`, `family`, `order`, `implicit`, `adaptive`, `error_estimator`,
`embedded_order`, `uses_nonlinear_solver`, `uses_linear_solver`,
`preserves_linear_invariants`, `supports_fixed_step`, `backend`, `tableau`
(`MethodInfo.to_dict()`).

Implementation notes:

* **Explicit RK** from the Butcher tableau; first stage reuses `f(t_n, y_n)`,
  DOPRI5 uses FSAL.
* **Implicit RK** in z-formulation `Z = h(A⊗I)F(t+ch, y+Z)` (ODE Lemma 5.5,
  eq. 5.10) on the full `s·n` system. `y_{n+1} = y_n + z_s` for stiffly
  accurate methods, `y_n + Σ w_i z_i` with `w = A^{−T} b` otherwise (Lemma 5.5
  iii); singular A without stiff accuracy falls back to `h Σ b_i f_i`.
  Collocation tableaux are hard-coded in closed form and verified against the
  ones generated from Gauß/Radau/Lobatto nodes (`tableaux.collocation_tableau`).
* **Step doubling** (methods without own estimator): one step h and two steps
  h/2, `err = (y_{h/2,h/2} − y_h)/(2^p − 1)`, continue with the two half steps.
* **ROS2**: `(I−γhJ)k₁ = f + γh f_t`, `(I−γhJ)k₂ = f(t+h, y+hk₁) − 2k₁ − γh f_t`,
  `y_{n+1} = y + h(3/2 k₁ + 1/2 k₂)`, error `h/2 (k₁+k₂)`; `f_t` by forward
  differences. Because the estimate is first order, step sizes scale like
  `tol^{1/2}` and the solution is typically far more accurate than requested.
* **BDF2**: variable-coefficient formula with `ω = h_n/h_{n−1}` (ratio limited
  to 2 for zero stability, `ω < 1+√2`), implicit Euler start, Milne-type
  estimate from a polynomial extrapolation predictor through past `y` values
  only (`err = Q/(Q+P)·(y−y_pred)` with `Q = h²(h+h₁)²/(2h+h₁)`,
  `P = h(h+h₁)(h+h₁+h₂)`).
* **Step size control**: PI controller (`β = 0.04`, `α = 1/k − 0.75β`,
  `k = q+1`), safety 0.9, factor in [0.2, 5]; after a rejection no increase.
  Newton/linear failure → reject and halve h. Initial step by the
  Hairer–Nørsett–Wanner heuristic.
* **Jacobian**: analytic if `has_jacobian()`, otherwise forward differences
  with increments `sqrt(eps·max(1e−5, |y_j|))`.

### Nonlinear solvers (`NONLINEAR_SOLVERS`)

| id | description | convergence |
|---|---|---|
| `full_newton` | new Jacobians at all stage points and new factorisation in every iteration | quadratic |
| `simplified_newton` | Jacobian at `(t_n, y_n)` and factorisation frozen per step (reused over rejected attempts; two cached factorisations for step doubling) | linear, rate θ |
| `fixed_point` | `Z ← h(A⊗I)F(Z)` (Banach) | linear if `h·ρ(A)·ρ(J) < 1` |

Stopping rule (Numerik I Alg. 5.10, Hairer's η-extension for the first
iterate): `θ_k = ‖ΔZ_k‖/‖ΔZ_{k−1}‖`, divergence if `θ_k ≥ 0.99`, early abort if
the predicted error after the remaining iterations is too large, converged if
`θ/(1−θ)‖ΔZ‖ ≤ κ` or `‖ΔZ‖` is at round-off level. ROS2 has no nonlinear
iteration (two linear solves per step).

### Linear solvers (`LINEAR_SOLVERS`)

`lu_pivot` (Numerik I 4.1.4), `lu_nopivot` (4.1.3, reports tiny/zero pivots),
`qr_householder` (4.3), `gauss_seidel` (6.1.3), `jacobi` (6.1.2), `cgnr` (6.2).
Iterative solvers report non-convergence/divergence; the step is then
rejected (adaptive) or the run fails with a clear status (fixed step).

## 3. Analyses

### Stability (`stability.py`, ODE 5.6, 7.6)

* `rk_stability_function(A, b)`: `R(z) = 1 + z bᵀ(I−zA)^{−1}𝟙`
  (Lemma 5.15) evaluated as `det(I−zA+z𝟙bᵀ)/det(I−zA)`;
  `ros2_stability_function(z)`; `lmm_root_modulus(α, β, z)` = max |ζ| of
  `ρ(ζ) − zσ(ζ)` (Def. 7.22); `lmm_boundary_locus`.
* `amplification(method, z)`: |R(z)| or max root modulus (stable ⇔ ≤ 1).
* `stability_properties(method)`: `a_stable` (no poles in the closed left
  half-plane, |R(iy)| ≤ 1 on the imaginary axis, check on a left half-plane
  grid), `l_stable` (A-stable and |R(−10¹²)| < 1e−4), `r_infinity`,
  `real_interval`, `imag_interval` (stability intervals on the axes).
* `stability_region(method, re_range, im_range, n)`: grid for plotting.
* `max_stable_step(method, eigenvalues)`: largest h with |R(hλ)| ≤ 1 for all
  non-growing eigenvalues (first crossing along the rays; 0 if only
  `h·ρ < 1e−3` would be stable, ∞ if unrestricted).
* `jacobian_spectra(system, t, Y)`, `stiffness_ratio(eigs)` (max/min |Re λ|
  of decaying modes).
* `fixed_point_contraction(method, J, h)`: `q_norm = h‖A‖∞‖J‖∞` (sufficient),
  `q_spectral = h ρ(A) ρ(J)` (asymptotic rate; eigenvalues of A⊗J are products).
* `newton_matrix(method, J, h)`, `linear_iteration_radius(method, J, h, kind)`:
  ρ of the Jacobi / Gauß–Seidel iteration matrix of `I − h(A⊗J)`.

### Consistency (`consistency.py`, ODE 2.2, 3.5, 7.4)

* `rooted_trees(p)`, `order_conditions(A, b, max_order)`: `bᵀΦ(t) = 1/γ(t)`
  for all rooted trees (Satz 3.4); `algebraic_order(A, b, c)` → order, first
  violated conditions, row-sum condition `c = A𝟙` (Lemma 3.3).
* `lmm_order(α, β)` (Satz 7.12, error constant), `root_condition(α)` (Def. 7.17).
* `local_error_study(...)`: one step from exact data, `δ(h) = O(h^{p+1})`,
  consistency order = slope − 1 (Def. 2.8).
* `global_error_study(...)`, `fit_order(h, err, floor, ceiling)`: least-squares
  slope in log–log; points at the noise floor (reference accuracy / round-off)
  or above the ceiling (instability, pre-asymptotic) are excluded and listed.

### Convergence (`convergence.py`, ODE 2.3, Satz 2.13, Kap. 4)

* `global_convergence(...)`: verdicts `convergent` (slope ≥ 0.7) and
  `order_confirmed` (|p_emp − p| ≤ 0.3).
* `tolerance_proportionality(...)`: error vs. rtol, slope ≈ 1 ideal.
* `newton_statistics(result)`, `stage_iteration_history(...)` (‖ΔZ_k‖, θ_k,
  observed order ≈ 2 / 1).
* `invariant_drift(result, I)`, `nonnegativity_violation(result, mask)`.

## 4. Auto-tuner (`auto_optimize`)

1. **Reference**: `radau5` with `rtol_ref = clip(rtol·10⁻³, 10⁻¹², 10⁻⁸)`,
   `atol_ref = atol·10⁻³`, at most 30 % of the budget. If it does not finish,
   the analysis uses the reached interval (pilot interval).
2. **Error metric**: 12 check points taken from the reference's own step
   times; candidates land exactly on them (`t_eval_exact`). Error =
   `max |y − y_ref| / (atol_i + rtol·max_t |y_ref,i|)` (units of the requested
   tolerance). Target: ≤ `ERROR_TARGET_FACTOR` = 10.
3. **Stability**: spectra of J at 10 points of the reference (own QR solver).
   *Stiff* ⇔ stiffness ratio ≥ 10³ **and** stiffness index
   `max decay rate × interval length` ≥ 10³. For each explicit candidate the
   maximal stable step per sample gives an estimated step count
   `Σ Δt_k / h_stab,k`; above 2·10⁵ (or `h_stab = 0`) it is not benchmarked.
4. **Consistency/convergence** on a short initial interval (first 10
   reference steps): algebraic order; local errors for `h = Δ/2 … Δ/32`
   (consistency order = slope − 1; verdict *consistent* unless the order is
   < 0.7); global errors for `N = 4 … 64` (convergence order).
5. **Benchmark**: every remaining candidate with simplified Newton + LU with
   pivoting at the requested tolerances (wall time = min of up to 3 runs for
   runs < 20 ms). A row is **valid** iff it succeeded, meets the error
   target, keeps the invariants (drift ≤ 10·(atol + rtol·|I₀|)), keeps
   concentrations non-negative (≥ −10·(atol + rtol·max|y|)) and is not
   marked inconsistent.
6. **Solver variants** for the two best valid Newton-type methods: other
   nonlinear solvers (fixed point only if `q_spectral < 1` at the method's
   mean step) and other linear solvers (Jacobi/Gauß–Seidel only if the
   iteration-matrix spectral radius < 1).
7. **Tolerance proportionality** for the three fastest valid methods
   (rtol × {10, 1, 0.1}).
8. **Selection**: fastest valid configuration; candidates within 10 % of the
   best time are ranked by `n_rhs + n·n_jac`. Fallback without valid rows:
   `radau5` + simplified Newton + LU.

Every stage checks the remaining budget; skipped work is listed in
`report.skipped` with a German reason. Each integration gets a wall-clock cap.

### Report JSON schema (`AutoTuneReport.to_dict()`)

Strict JSON (non-finite numbers → `null`; complex → `{re, im}`).

```text
{
  problem: {n: int, t_span: [float, float], pilot_t_span: [float, float], y0: [float],
            rtol: float, atol: [float], candidates: [str], error_target_factor: float},
  budget: {budget_s: float, elapsed_s: float, exhausted: bool},
  reference: {method: str, rtol: float, atol: [float], success: bool, status: str,
              message: str, n_steps: int, wall_time: float, t_end: float,
              pilot_t_end: float|null, reasoning_de: str},
  stability: {
    stiff: bool, stiffness_ratio_max: float, max_decay_rate: float,
    max_growth_rate: float, stiffness_index: float, reasoning_de: str,
    spectra: [{t: float, re: [float], im: [float], spectral_radius: float,
               max_decay_rate: float, max_growth_rate: float,
               stiffness_ratio: float, jacobian_norm_inf: float}],
    explicit_limits: [{method: str, h_max_stable: float,
                       h_max_stable_per_sample: [float|null], estimated_steps: float|null}],
    method_properties: {<method>: {a_stable: bool, l_stable: bool, r_infinity: float|null,
                                   real_interval: float|null, imag_interval: float|null}},
    stability_regions: {<method>: {x: [float], y: [float], abs_R: [[float]] (rows = y),
                                   boundary_re: [float], boundary_im: [float],
                                   a_stable: bool, l_stable: bool, r_infinity: float|null}}
  },
  consistency: {<method>: {
    theoretical_order: int, algebraic_order: int, algebraic_capped?: bool,
    order_conditions_checked?: int, first_violated_conditions?: [str],
    row_sum_condition?: bool, embedded_algebraic_order?: int,
    error_constant?: float, root_condition?: bool,          (bdf2)
    local: OrderStudy, empirical_order: float|null, consistent: bool|null,
    reasoning_de: str}},
  convergence: {<method>: {
    global: OrderStudy, theoretical_order: int, empirical_order: float|null,
    convergent: bool|null, order_confirmed?: bool, reasoning_de: str,
    tolerance?: {rtol: [float], errors: [float|null], n_rhs: [int], n_steps: [int],
                 wall_time: [float], slope: float|null, proportional: bool|null,
                 study: OrderStudy, reasoning_de: str},
    newton?: NewtonStats}},
  benchmark: [{method: str, label_de: str, nonlinear_solver: str, linear_solver: str,
               success: bool, status: str, wall_time: float, error: float|null,
               meets_target: bool, stats: {...}, options: {...},
               newton?: NewtonStats, invariants_ok: bool, invariant_drift: float|null,
               nonnegative_ok: bool, min_value: float|null, consistent: bool|null,
               valid: bool, reasoning_de: str}],
  recommendation: {options: {method, rtol, atol, linear_solver, nonlinear_solver},
                   label_de: str, reasoning_de: str, fallback: bool,
                   wall_time?: float, error?: float},
  reasoning_de: [str],
  skipped: [{stage: str, item: str, reason_de: str}]
}

OrderStudy  = {h: [float], errors: [float|null], used: [bool], slope: float|null,
               intercept: float|null, kind: "local"|"global"|"tolerance",
               fitted: [float]|null, reason_excluded: [str]}
NewtonStats = {iterations: int, iterations_per_step: float, failures: int,
               theta_mean: float, theta_max: float, linear_iterations_per_solve: float,
               linear_failures: int, factorizations_per_step: float, reasoning_de: str}
```

Plot hints: spectra → scatter `re`/`im` per sample; stability regions →
Plotly contour of `abs_R` at level 1 (or scatter `boundary_*`) with the
scaled spectrum `h·λ` overlaid; `OrderStudy` → log–log scatter of
`h`/`errors` with `fitted` as line; tolerance sweep → `rtol` vs `errors`;
benchmark → table/bar chart of `wall_time` coloured by `valid`.
`integrate(system, t_span, y0, **report.recommended_options)` reproduces the
recommended configuration.

## 5. Linear algebra, interpolation, quadrature

* `linalg`: `lu, solve(method), solve_triangular, det, inv, cholesky, qr,
  qr_givens, lstsq(householder|givens), jacobi, gauss_seidel, cg, cgnr,
  cond(p=1|2|inf), norm, eigvals, hessenberg, balance, power_iteration,
  inverse_iteration, gershgorin_discs, spectral_radius, iteration_matrix`;
  errors raise `linalg.LinAlgError`. `cond(·, 2)` uses the eigenvalues of
  `[[0, A], [Aᵀ, 0]]` (no squaring of κ).
* `interpolation`: `lagrange_interpolate` (barycentric), `divided_differences`,
  `newton_evaluate`, `CubicSpline(x, y, bc="natural"|"not-a-knot")` with
  derivatives and exact integrals, `resample(t, y, t_new, method)`.
* `quadrature`: `gauss_legendre`, `gauss_lobatto`, `gauss_radau` (Newton on
  Legendre polynomials), `newton_cotes_weights`, `newton_cotes`,
  `composite_trapezoid`, `composite_simpson`, `gauss_legendre_rule`,
  `gauss_lobatto_rule`, `composite_gauss`, `integrate_samples(t, y,
  "trapezoid"|"simpson"|"spline")` (non-uniform grids), `cumulative_trapezoid`.

## 6. Performance (Robertson, t ∈ [0, 10⁵], compiled system, best of 5)

| method | rtol=1e−6, atol=1e−10 | rtol=1e−8, atol=1e−14 |
|---|---|---|
| radau5 | 0.17 ms (43 steps, rel. err 4e−7) | 0.33 ms (82 steps, 2e−8) |
| bdf2 | 0.32 ms (811 steps, 2e−4) | 1.8 ms (4672 steps, 9e−6) |
| sdirk2 | 0.57 ms (281 steps, 4e−5) | 2.8 ms (1550 steps, 2e−6) |
| ros2 | 3.7 ms (10083 steps, 3e−6) | 53 ms (135271 steps, 3e−8) |
| scipy Radau | 28 ms | 104 ms |
| scipy BDF | 27 ms | 67 ms |
| scipy LSODA | 3.6 ms | 7.3 ms |

(scipy timings include the Python round trip per rhs/Jacobian call.)

## 7. Known limitations

* Dense linear algebra only (n ≲ 500; implicit RK factorises the full
  `s·n × s·n` matrix, as specified).
* Hermite cubic dense output (order 4 locally) can be less accurate than the
  step endpoints for high-order methods with large steps; use
  `t_eval_exact=True` when output accuracy matters.
* Implicit RK methods use step doubling (≈ 3× the cost of an embedded
  estimate); ROS2's first-order estimate leads to many small steps at tight
  tolerances.
* Rosenbrock order conditions are not checked algebraically (theoretical order
  2 from the literature, verified empirically).
* Integration is forward in time only (`t0 < tf`).

## 8. Status

Done: C++ library (dense LA, eigenvalues, all specified integrators, Newton
variants, linear-solver strategies, FD Jacobians, budgets, statistics, test and
callback systems), Cython bindings, Python package (integrate, registries,
stability / consistency / convergence analyses, auto-tuner, linalg,
interpolation, quadrature), 201 tests in `tests/numerics` (~7 s), this document.

Missing / not covered: the adaptive rejection path for `nonnegative=True`
(steps with components `< −atol`) is implemented but not exercised by a test;
no sparse or banded linear algebra; no higher-order dense output.
