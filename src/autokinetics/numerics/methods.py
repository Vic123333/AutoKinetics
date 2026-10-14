"""Registries of integration methods, linear solvers and nonlinear solvers.

Each entry carries German labels/descriptions for the UI, the method family,
theoretical order, stability properties and the lecture reference
("ODE" = Numerik gewöhnlicher Differentialgleichungen (Lederer),
"Numerik I" = Vorlesungsskript Numerik I).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = [
    "MethodInfo",
    "LinearSolverInfo",
    "NonlinearSolverInfo",
    "METHODS",
    "LINEAR_SOLVERS",
    "NONLINEAR_SOLVERS",
    "get_method",
]

FAMILY_EXPLICIT_RK = "explicit_rk"
FAMILY_IMPLICIT_RK = "implicit_rk"
FAMILY_ROSENBROCK = "rosenbrock"
FAMILY_MULTISTEP = "multistep"
FAMILY_SCIPY = "scipy"

FAMILY_LABELS_DE = {
    FAMILY_EXPLICIT_RK: "Explizites Runge-Kutta-Verfahren",
    FAMILY_IMPLICIT_RK: "Implizites Runge-Kutta-Verfahren",
    FAMILY_ROSENBROCK: "Linear-implizites Rosenbrock-Verfahren",
    FAMILY_MULTISTEP: "Lineares Mehrschrittverfahren",
    FAMILY_SCIPY: "SciPy-Referenzverfahren",
}


@dataclass(frozen=True)
class MethodInfo:
    id: str
    label_de: str
    description_de: str
    family: str
    order: int
    stages: int | None
    implicit: bool
    adaptive: bool  # supports adaptive step-size control
    error_estimator: str  # "embedded" | "step_doubling" | "predictor" | "internal"
    embedded_order: int | None
    a_stable: bool
    l_stable: bool
    stability_de: str
    lecture_ref: str
    backend: str  # "cpp" | "scipy"
    supports_fixed_step: bool
    uses_nonlinear_solver: bool  # implicit stage equations solved iteratively
    uses_linear_solver: bool
    preserves_linear_invariants: bool
    tableau: str | None = None  # name of the built-in Butcher tableau

    @property
    def family_label_de(self) -> str:
        return FAMILY_LABELS_DE[self.family]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["family_label_de"] = self.family_label_de
        return d


def _rk(
    id,
    label,
    desc,
    family,
    order,
    stages,
    implicit,
    estimator,
    emb,
    a_stable,
    l_stable,
    stab,
    ref,
    tableau,
):
    newton = implicit
    return MethodInfo(
        id=id,
        label_de=label,
        description_de=desc,
        family=family,
        order=order,
        stages=stages,
        implicit=implicit,
        adaptive=True,
        error_estimator=estimator,
        embedded_order=emb,
        a_stable=a_stable,
        l_stable=l_stable,
        stability_de=stab,
        lecture_ref=ref,
        backend="cpp",
        supports_fixed_step=True,
        uses_nonlinear_solver=newton,
        uses_linear_solver=newton,
        preserves_linear_invariants=True,
        tableau=tableau,
    )


_EXPL_STAB = "Beschränktes Stabilitätsgebiet (Polynom als Stabilitätsfunktion), nicht A-stabil"

METHODS: dict[str, MethodInfo] = {
    m.id: m
    for m in [
        _rk(
            "euler",
            "Explizites Euler-Verfahren",
            "Einfachstes Einschrittverfahren: ein Schritt entlang der Steigung am Intervallanfang. "
            "Ordnung 1, Stabilitätsgebiet ist die Kreisscheibe |1+z| ≤ 1.",
            FAMILY_EXPLICIT_RK,
            1,
            1,
            False,
            "step_doubling",
            None,
            False,
            False,
            _EXPL_STAB + "; S = {|1+z| ≤ 1}",
            "ODE Kap. 2.1 (Bsp. 2.1), Korollar 2.9",
            "euler",
        ),
        _rk(
            "heun",
            "Heun-Verfahren (verbessertes Euler)",
            "Zweistufiges explizites RK-Verfahren der Ordnung 2 (Trapezregel mit Euler-Prädiktor); "
            "eingebettetes Euler-Verfahren liefert den Fehlerschätzer (Heun-Euler 2(1)).",
            FAMILY_EXPLICIT_RK,
            2,
            2,
            False,
            "embedded",
            1,
            False,
            False,
            _EXPL_STAB + "; g(z) = 1 + z + z²/2",
            "ODE Kap. 2.1 / Kap. 3, Kap. 4.3 (eingebettete Verfahren)",
            "heun",
        ),
        _rk(
            "rk4",
            "Klassisches Runge-Kutta-Verfahren (RK4)",
            "Vierstufiges explizites Verfahren der Ordnung 4 (Simpson-Regel als Quadratur-Analogon). "
            "Adaptiv über Schrittweitenverdopplung (Richardson-Fehlerschätzung).",
            FAMILY_EXPLICIT_RK,
            4,
            4,
            False,
            "step_doubling",
            None,
            False,
            False,
            _EXPL_STAB + "; reelles Stabilitätsintervall ≈ [-2.79, 0]",
            "ODE Kap. 3 (Satz 3.4, Bem. 3.5)",
            "rk4",
        ),
        _rk(
            "dopri5",
            "Dormand-Prince 5(4)",
            "Eingebettetes explizites RK-Verfahren der Ordnung 5 mit Fehlerschätzer der Ordnung 4, "
            "FSAL-Eigenschaft und PI-Schrittweitensteuerung (entspricht scipy RK45). "
            "Standardverfahren für nicht-steife Probleme.",
            FAMILY_EXPLICIT_RK,
            5,
            7,
            False,
            "embedded",
            4,
            False,
            False,
            _EXPL_STAB + "; reelles Stabilitätsintervall ≈ [-3.31, 0]",
            "ODE Kap. 4.3 (eingebettete RK, Fehlberg-Trick)",
            "dopri5",
        ),
        _rk(
            "implicit_euler",
            "Implizites Euler-Verfahren",
            "Implizites Einschrittverfahren der Ordnung 1 (= Radau IIA mit s=1 = BDF1). "
            "L-stabil, sehr robust für steife Probleme, aber nur Ordnung 1.",
            FAMILY_IMPLICIT_RK,
            1,
            1,
            True,
            "step_doubling",
            None,
            True,
            True,
            "A- und L-stabil; g(z) = 1/(1-z)",
            "ODE Kap. 2.1 (Lemma 2.3), Kap. 5.6 (Def. 5.18/5.19)",
            "implicit_euler",
        ),
        _rk(
            "implicit_midpoint",
            "Implizite Mittelpunktregel (Gauß, s=1)",
            "Einstufiges Gauß-Kollokationsverfahren der Ordnung 2; erhält quadratische Invarianten, "
            "A-stabil aber nicht L-stabil (g(∞) = -1).",
            FAMILY_IMPLICIT_RK,
            2,
            1,
            True,
            "step_doubling",
            None,
            True,
            False,
            "A-stabil, nicht L-stabil; g(z) = (1+z/2)/(1-z/2)",
            "ODE Kap. 5.4 (Bsp. 5.10), Kap. 6",
            "implicit_midpoint",
        ),
        _rk(
            "trapezoidal",
            "Implizite Trapezregel (Lobatto IIIA, s=2)",
            "Zweistufiges Lobatto-IIIA-Kollokationsverfahren der Ordnung 2. A-stabil, aber nicht L-stabil: "
            "steife Komponenten werden oszillierend statt dämpfend dargestellt.",
            FAMILY_IMPLICIT_RK,
            2,
            2,
            True,
            "step_doubling",
            None,
            True,
            False,
            "A-stabil, nicht L-stabil; g(z) = (1+z/2)/(1-z/2)",
            "ODE Kap. 5.4 (Bem. 5.12), Kap. 5.6",
            "trapezoidal",
        ),
        _rk(
            "gauss4",
            "Gauß-Legendre-Verfahren (s=2)",
            "Zweistufiges Gauß-Kollokationsverfahren mit maximaler Ordnung 2s = 4; B-stabil, "
            "erhält quadratische Invarianten, aber nicht L-stabil.",
            FAMILY_IMPLICIT_RK,
            4,
            2,
            True,
            "step_doubling",
            None,
            True,
            False,
            "A- und B-stabil, nicht L-stabil (|g(∞)| = 1)",
            "ODE Kap. 5.4 (Bsp. 5.10), Lemma 5.21",
            "gauss4",
        ),
        _rk(
            "radau3",
            "Radau IIA (s=2)",
            "Zweistufiges Radau-IIA-Kollokationsverfahren der Ordnung 2s-1 = 3; L-stabil und steif genau.",
            FAMILY_IMPLICIT_RK,
            3,
            2,
            True,
            "step_doubling",
            None,
            True,
            True,
            "A- und L-stabil",
            "ODE Kap. 5.4 (Bem. 5.12)",
            "radau3",
        ),
        _rk(
            "radau5",
            "Radau IIA (s=3, Radau5)",
            "Dreistufiges Radau-IIA-Kollokationsverfahren der Ordnung 5; L-stabil, steif genau. "
            "Robustes Standardverfahren für steife Kinetik (vgl. RADAU5 von Hairer/Wanner).",
            FAMILY_IMPLICIT_RK,
            5,
            3,
            True,
            "step_doubling",
            None,
            True,
            True,
            "A- und L-stabil",
            "ODE Kap. 5.4 (Bem. 5.12)",
            "radau5",
        ),
        _rk(
            "sdirk2",
            "SDIRK2 (L-stabil, Alexander)",
            "Zweistufiges einfach diagonal-implizites RK-Verfahren der Ordnung 2 mit γ = 1-√2/2; "
            "L-stabil und steif genau. Stufen nacheinander lösbar (DIRK-Struktur).",
            FAMILY_IMPLICIT_RK,
            2,
            2,
            True,
            "step_doubling",
            None,
            True,
            True,
            "A- und L-stabil",
            "ODE Kap. 5.5 (DIRK/SDIRK)",
            "sdirk2",
        ),
        _rk(
            "sdirk3",
            "SDIRK3 (Crouzeix, A-stabil)",
            "Zweistufiges SDIRK-Verfahren der Ordnung 3 mit γ = (3+√3)/6 (A-stabile Wahl); "
            "nicht L-stabil. Vgl. das SDIRK-Beispiel im Skript (dort γ = 1/2-√3/6, nicht A-stabil).",
            FAMILY_IMPLICIT_RK,
            3,
            2,
            True,
            "step_doubling",
            None,
            True,
            False,
            "A-stabil, nicht L-stabil (|g(∞)| ≈ 0.73)",
            "ODE Kap. 5.5 (SDIRK-Beispiel)",
            "sdirk3",
        ),
        MethodInfo(
            id="ros2",
            label_de="Rosenbrock ROS2",
            description_de="Linear-implizites Rosenbrock-Verfahren der Ordnung 2 (Verwer et al. 1999, γ = 1+1/√2): "
            "statt Newton-Iterationen nur zwei lineare Gleichungssysteme pro Schritt mit derselben Matrix "
            "I − γhJ. L-stabil; eingebettete Lösung der Ordnung 1 als Fehlerschätzer.",
            family=FAMILY_ROSENBROCK,
            order=2,
            stages=2,
            implicit=True,
            adaptive=True,
            error_estimator="embedded",
            embedded_order=1,
            a_stable=True,
            l_stable=True,
            stability_de="A- und L-stabil",
            lecture_ref="ODE Kap. 5.5 (Linearisierung der Newton-Iteration)",
            backend="cpp",
            supports_fixed_step=True,
            uses_nonlinear_solver=False,
            uses_linear_solver=True,
            preserves_linear_invariants=True,
        ),
        MethodInfo(
            id="bdf2",
            label_de="BDF2 (variable Schrittweite)",
            description_de="Zweischritt-BDF-Verfahren (Rückwärtsdifferenzenformel) der Ordnung 2 mit variabler "
            "Schrittweite, Start mit implizitem Euler. Fehlerschätzung über die Differenz zu einem "
            "expliziten Extrapolations-Prädiktor (Milne-Trick). A-stabil, steife Komponenten werden gedämpft.",
            family=FAMILY_MULTISTEP,
            order=2,
            stages=None,
            implicit=True,
            adaptive=True,
            error_estimator="predictor",
            embedded_order=None,
            a_stable=True,
            l_stable=True,
            stability_de="A-stabil; Wurzeln von ρ(ζ)−zσ(ζ) → 0 für z → −∞ (steif dämpfend)",
            lecture_ref="ODE Kap. 7.3 (Bsp. 7.8), Kap. 7.5/7.6",
            backend="cpp",
            supports_fixed_step=True,
            uses_nonlinear_solver=True,
            uses_linear_solver=True,
            preserves_linear_invariants=True,
        ),
        MethodInfo(
            id="scipy:RK45",
            label_de="SciPy RK45 (Dormand-Prince)",
            description_de="Referenzimplementierung aus scipy.integrate.solve_ivp (explizit, Ordnung 5(4)).",
            family=FAMILY_SCIPY,
            order=5,
            stages=7,
            implicit=False,
            adaptive=True,
            error_estimator="internal",
            embedded_order=4,
            a_stable=False,
            l_stable=False,
            stability_de=_EXPL_STAB,
            lecture_ref="ODE Kap. 4.3",
            backend="scipy",
            supports_fixed_step=False,
            uses_nonlinear_solver=False,
            uses_linear_solver=False,
            preserves_linear_invariants=True,
        ),
        MethodInfo(
            id="scipy:BDF",
            label_de="SciPy BDF (variable Ordnung 1–5)",
            description_de="Referenzimplementierung aus scipy.integrate.solve_ivp: BDF/NDF variabler Ordnung "
            "(1–5) in Quasi-Konstant-Schrittweitenform.",
            family=FAMILY_SCIPY,
            order=5,
            stages=None,
            implicit=True,
            adaptive=True,
            error_estimator="internal",
            embedded_order=None,
            a_stable=False,
            l_stable=False,
            stability_de="A(α)-stabil (Ordnungen > 2 nicht A-stabil, Dahlquist-Schranke)",
            lecture_ref="ODE Kap. 7.6",
            backend="scipy",
            supports_fixed_step=False,
            uses_nonlinear_solver=True,
            uses_linear_solver=True,
            preserves_linear_invariants=True,
        ),
        MethodInfo(
            id="scipy:Radau",
            label_de="SciPy Radau (Radau IIA, Ordnung 5)",
            description_de="Referenzimplementierung aus scipy.integrate.solve_ivp (Radau IIA s=3 mit "
            "eingebettetem Fehlerschätzer nach Hairer/Wanner).",
            family=FAMILY_SCIPY,
            order=5,
            stages=3,
            implicit=True,
            adaptive=True,
            error_estimator="internal",
            embedded_order=3,
            a_stable=True,
            l_stable=True,
            stability_de="A- und L-stabil",
            lecture_ref="ODE Kap. 5.4",
            backend="scipy",
            supports_fixed_step=False,
            uses_nonlinear_solver=True,
            uses_linear_solver=True,
            preserves_linear_invariants=True,
        ),
        MethodInfo(
            id="scipy:LSODA",
            label_de="SciPy LSODA (automatische Steifheitserkennung)",
            description_de="Referenzimplementierung (ODEPACK LSODA): wechselt automatisch zwischen Adams "
            "(nicht-steif) und BDF (steif).",
            family=FAMILY_SCIPY,
            order=12,
            stages=None,
            implicit=True,
            adaptive=True,
            error_estimator="internal",
            embedded_order=None,
            a_stable=False,
            l_stable=False,
            stability_de="abhängig vom aktiven Verfahren (Adams bzw. BDF)",
            lecture_ref="ODE Kap. 7",
            backend="scipy",
            supports_fixed_step=False,
            uses_nonlinear_solver=True,
            uses_linear_solver=True,
            preserves_linear_invariants=True,
        ),
    ]
}


@dataclass(frozen=True)
class LinearSolverInfo:
    id: str
    label_de: str
    description_de: str
    kind: str  # "direct" | "iterative"
    requirements_de: str
    lecture_ref: str

    def to_dict(self) -> dict:
        return asdict(self)


LINEAR_SOLVERS: dict[str, LinearSolverInfo] = {
    s.id: s
    for s in [
        LinearSolverInfo(
            "lu_pivot",
            "LU-Zerlegung mit Spaltenpivotsuche",
            "Gauß-Elimination mit Zeilenvertauschung nach dem betragsgrößten Pivot; "
            "rückwärtsstabil in der Praxis, Standardwahl.",
            "direct",
            "Matrix regulär",
            "Numerik I, Kap. 4.1.4",
        ),
        LinearSolverInfo(
            "lu_nopivot",
            "LU-Zerlegung ohne Pivotsuche",
            "Gauß-Elimination ohne Zeilenvertauschung; bricht bei (fast) verschwindenden Pivots ab "
            "(wird erkannt und gemeldet).",
            "direct",
            "alle Hauptabschnittsdeterminanten ≠ 0 (z.B. strikt diagonaldominant)",
            "Numerik I, Kap. 4.1.3",
        ),
        LinearSolverInfo(
            "qr_householder",
            "QR-Zerlegung (Householder)",
            "Orthogonale Zerlegung durch Householder-Spiegelungen; teurer als LU (≈ doppelter "
            "Aufwand), aber ohne Fehlerverstärkung durch Pivotwahl.",
            "direct",
            "Matrix regulär",
            "Numerik I, Kap. 4.3",
        ),
        LinearSolverInfo(
            "gauss_seidel",
            "Gauß-Seidel-Verfahren",
            "Einzelschrittverfahren; konvergiert genau dann, wenn der Spektralradius der "
            "Iterationsmatrix (D+L)⁻¹U kleiner als 1 ist (z.B. bei strikter Diagonaldominanz).",
            "iterative",
            "ρ((D+L)⁻¹U) < 1",
            "Numerik I, Kap. 6.1.3",
        ),
        LinearSolverInfo(
            "jacobi",
            "Jacobi-Verfahren",
            "Gesamtschrittverfahren; konvergiert genau dann, wenn ρ(D⁻¹(L+U)) < 1.",
            "iterative",
            "ρ(D⁻¹(L+U)) < 1",
            "Numerik I, Kap. 6.1.2",
        ),
        LinearSolverInfo(
            "cgnr",
            "CGNR (CG auf Normalgleichungen)",
            "Verfahren der konjugierten Gradienten angewandt auf AᵀA x = Aᵀb; konvergiert für jede "
            "reguläre Matrix, aber mit κ(A)² – bei schlecht konditionierten Newton-Matrizen langsam.",
            "iterative",
            "Matrix regulär (Konvergenzrate hängt von κ(A)² ab)",
            "Numerik I, Kap. 6.2",
        ),
    ]
}


@dataclass(frozen=True)
class NonlinearSolverInfo:
    id: str
    label_de: str
    description_de: str
    convergence_de: str
    lecture_ref: str

    def to_dict(self) -> dict:
        return asdict(self)


NONLINEAR_SOLVERS: dict[str, NonlinearSolverInfo] = {
    s.id: s
    for s in [
        NonlinearSolverInfo(
            "full_newton",
            "Newton-Verfahren (voll)",
            "In jeder Iteration neue Jacobi-Matrizen an allen Stufenpunkten und neue "
            "Faktorisierung der (s·n)-dimensionalen Newton-Matrix.",
            "lokal quadratisch",
            "Numerik I, Kap. 5.2 (Alg. 5.10); ODE Kap. 5.5",
        ),
        NonlinearSolverInfo(
            "simplified_newton",
            "Vereinfachtes Newton-Verfahren",
            "Jacobi-Matrix und LU-Zerlegung werden pro Schritt eingefroren (frozen Newton); "
            "nur Vorwärts-/Rückwärtseinsetzen pro Iteration.",
            "linear mit Kontraktionsfaktor θ",
            "Numerik I, Bem. 5.11; ODE Kap. 5.5",
        ),
        NonlinearSolverInfo(
            "fixed_point",
            "Fixpunktiteration (Banach)",
            "Z ← h(A⊗I)F(Z) ohne lineare Gleichungssysteme; konvergiert nur, wenn die "
            "Abbildung kontrahiert (h·‖A‖·‖J‖ < 1) – für steife Probleme ungeeignet.",
            "linear, falls q = h·ρ(A)·ρ(J) < 1",
            "Numerik I, Kap. 5.1 (Banachscher Fixpunktsatz)",
        ),
    ]
}


def get_method(method_id: str) -> MethodInfo:
    try:
        return METHODS[method_id]
    except KeyError:
        raise ValueError(f"unknown method {method_id!r}; available: {sorted(METHODS)}") from None
