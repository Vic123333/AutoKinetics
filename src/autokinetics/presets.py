"""Conversion of user-facing reactor/simulation settings (with units) into SI objects.

Used by the example library, the REST API and the CLI so all entry points share
one interpretation of units and defaults.
"""

from __future__ import annotations

from typing import Any

from .errors import SimulationError
from .mechanism import Mechanism
from .reactor import ReactorSpec
from .simulation import SimulationOptions
from .units import concentration_to_SI, parse_quantity, pressure_to_Pa, time_to_s


def _temperature(v: Any) -> float:
    val, unit = parse_quantity(v, "K")
    if unit == "K":
        return val
    if unit in ("C", "°C", "degC"):
        return val + 273.15
    raise SimulationError(f"Unbekannte Temperatureinheit '{unit}'")


def _program(points: Any) -> list[tuple[float, float]] | None:
    if not points:
        return None
    out = []
    for p in points:
        if isinstance(p, dict):
            out.append((time_to_s(*parse_quantity(p["t"], "s")), _temperature(p["T"])))
        else:
            out.append((float(p[0]), _temperature(p[1])))
    out.sort()
    return out


def reactor_spec_from_dict(d: dict, mech: Mechanism) -> ReactorSpec:
    """Build a :class:`ReactorSpec` from a dict with optional units.

    Keys: kind, constraint, energy, T, P, basis ('concentration'|'mole-fraction'),
    composition, concentration_unit (default mol/L), T_program [(t, T)...],
    UA_V [W/(m³K)], T_env, T_env_program, rho_cp [J/(m³K)], tau [s], feed, T_feed,
    fixed_species.
    """
    d = dict(d or {})
    gas = mech.phase == "ideal-gas"
    basis = d.get("basis", "mole-fraction" if gas else "concentration")
    cunit = d.get("concentration_unit", "mol/L")
    comp = {str(k): float(v) for k, v in (d.get("composition") or {}).items()}
    if basis == "concentration":
        comp = {k: concentration_to_SI(v, cunit) for k, v in comp.items()}
    feed = {str(k): concentration_to_SI(float(v), cunit) for k, v in (d.get("feed") or {}).items()}
    P = d.get("P", "1 atm")
    P_val, P_unit = parse_quantity(P, "Pa")
    spec = ReactorSpec(
        kind=d.get("kind", "batch"),
        constraint=d.get("constraint", "volume"),
        energy=d.get("energy", "isothermal"),
        T0=_temperature(d.get("T", 298.15)),
        P0=pressure_to_Pa(P_val, P_unit),
        initial=comp,
        basis=basis,
        T_program=_program(d.get("T_program")),
        UA_V=float(d.get("UA_V", 0.0)),
        T_env=_temperature(d["T_env"]) if d.get("T_env") is not None else None,
        T_env_program=_program(d.get("T_env_program")),
        rho_cp=float(d.get("rho_cp", 4.18e6)),
        tau=time_to_s(*parse_quantity(d["tau"], "s")) if d.get("tau") is not None else None,
        feed=feed,
        T_feed=_temperature(d["T_feed"]) if d.get("T_feed") is not None else None,
        fixed_species=list(d.get("fixed_species") or []),
    )
    spec.validate(mech)
    return spec


def options_from_dict(d: dict) -> SimulationOptions:
    d = dict(d or {})
    if "t_end" not in d:
        raise SimulationError("Simulationsdauer t_end fehlt")
    t_end = time_to_s(*parse_quantity(d["t_end"], "s"))
    opts = SimulationOptions(
        t_end=t_end,
        method=str(d.get("method", SimulationOptions.method)),
        rtol=float(d.get("rtol", 1e-6)),
        atol_rel=float(d.get("atol_rel", 1e-12)),
        max_points=int(d.get("max_points", 1500)),
        log_time=bool(d.get("log_time", False)),
        linear_solver=str(d.get("linear_solver", "lu_pivot")),
        nonlinear_solver=str(d.get("nonlinear_solver", "simplified_newton")),
        fixed_step=float(d["fixed_step"]) if d.get("fixed_step") else None,
        max_wall_time=float(d.get("max_wall_time", 120.0)),
    )
    opts.validate()
    return opts
