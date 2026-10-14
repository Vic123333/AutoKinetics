"""Numerics layer of AutoKinetics: integrators, solver registry, stability /
consistency / convergence analysis and automatic solver selection.

Quick start::

    from autokinetics.numerics import integrate, auto_optimize, RobertsonSystem

    sys = RobertsonSystem()
    res = integrate(sys, (0.0, 1e5), [1.0, 0.0, 0.0], method="radau5", rtol=1e-6, atol=1e-10)
    report = auto_optimize(sys, (0.0, 1e5), [1.0, 0.0, 0.0], rtol=1e-6, atol=1e-10, budget_s=10)
    res2 = integrate(sys, (0.0, 1e5), [1.0, 0.0, 0.0], **report.recommended_options)
"""

from autokinetics._ext._core import (
    LinearOdeSystem,
    OdeSystemHandle,
    PythonOdeSystem,
    RobertsonSystem,
    VanDerPolSystem,
)
from autokinetics.numerics import (
    consistency,
    convergence,
    interpolation,
    linalg,
    quadrature,
    stability,
)
from autokinetics.numerics.autotune import AutoTuneReport, auto_optimize
from autokinetics.numerics.integrator import IntegrationResult, as_system, integrate
from autokinetics.numerics.methods import LINEAR_SOLVERS, METHODS, NONLINEAR_SOLVERS, get_method
from autokinetics.numerics.tableaux import ButcherTableau

__all__ = [
    "integrate",
    "IntegrationResult",
    "as_system",
    "auto_optimize",
    "AutoTuneReport",
    "METHODS",
    "LINEAR_SOLVERS",
    "NONLINEAR_SOLVERS",
    "get_method",
    "ButcherTableau",
    "OdeSystemHandle",
    "PythonOdeSystem",
    "LinearOdeSystem",
    "RobertsonSystem",
    "VanDerPolSystem",
    "linalg",
    "stability",
    "consistency",
    "convergence",
    "interpolation",
    "quadrature",
]
