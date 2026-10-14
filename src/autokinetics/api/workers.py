"""Functions executed in worker processes. Inputs/outputs are JSON-compatible."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import yaml

from ..errors import AutoKineticsError, MechanismError
from ..io.kin_legacy import load_kin
from ..io.yaml_format import mechanism_from_dict, safe_load_yaml
from ..mechanism import Mechanism
from ..presets import options_from_dict, reactor_spec_from_dict

UNIT_FACTOR = {"mol/L": 1e-3, "mol/m3": 1.0, "mmol/L": 1.0}


def build_mechanism(payload: dict) -> Mechanism:
    if payload.get("document") is not None:
        return mechanism_from_dict(payload["document"], payload.get("phase"))
    if payload.get("yaml") is not None:
        try:
            doc = safe_load_yaml(payload["yaml"])
        except yaml.YAMLError as e:
            raise MechanismError(f"YAML-Syntaxfehler: {e}") from None
        return mechanism_from_dict(doc, payload.get("phase"))
    if payload.get("kin") is not None:
        return load_kin(payload["kin"])
    raise MechanismError("Kein Mechanismus übergeben")


def _limits(mech: Mechanism, limits: dict) -> None:
    if mech.n_species > limits["max_species"]:
        raise AutoKineticsError(f"Zu viele Spezies ({mech.n_species} > {limits['max_species']})")
    if mech.n_reactions > limits["max_reactions"]:
        raise AutoKineticsError(f"Zu viele Reaktionen ({mech.n_reactions} > {limits['max_reactions']})")


def _sim_options(sim: dict, limits: dict):
    opts = options_from_dict(sim)
    opts.max_wall_time = min(opts.max_wall_time, limits["max_wall_time"])
    opts.max_points = min(opts.max_points, 20000)
    return opts


def _wrap(fn):
    """Return errors as data so they survive the process boundary with their message.
    ``functools.wraps`` keeps the qualified name so the job stays picklable."""
    @functools.wraps(fn)
    def inner(*args):
        try:
            return {"ok": True, "result": fn(*args)}
        except AutoKineticsError as e:
            return {"ok": False, "error": str(e), "kind": type(e).__name__}
        except (ValueError, ArithmeticError) as e:
            return {"ok": False, "error": f"Ungültige Eingabe: {e}", "kind": "ValueError"}
    return inner


@_wrap
def job_simulate(req: dict, limits: dict) -> dict:
    from ..simulation import heat_release_rate, simulate

    mech = build_mechanism(req["mechanism"])
    _limits(mech, limits)
    spec = reactor_spec_from_dict(req["reactor"], mech)
    opts = _sim_options(req["simulation"], limits)
    res = simulate(mech, spec, opts)
    out = res.to_dict(req.get("concentration_unit", "mol/L"))
    if req.get("include_heat_release") and mech.phase == "ideal-gas" and mech.has_thermo:
        out["heat_release_rate"] = heat_release_rate(res).tolist()
    return out


@_wrap
def job_equilibrium(req: dict, limits: dict) -> dict:
    from ..equilibrium import equilibrate

    mech = build_mechanism(req["mechanism"])
    _limits(mech, limits)
    spec = reactor_spec_from_dict(req["reactor"], mech)
    c0 = spec.initial_concentrations(mech)
    mode = req.get("mode", "TP")
    kw: dict[str, Any] = {}
    if mech.phase == "ideal-gas" and mode in ("TP", "HP"):
        kw["P"] = spec.P0
    else:
        kw["V"] = 1.0
        if mech.phase == "ideal-gas":
            kw["P"] = spec.P0
    eq = equilibrate(mech, c0, spec.T0, mode=mode, basis=req.get("basis", "auto"), **kw)
    out = eq.to_dict(req.get("concentration_unit", "mol/L"))
    out["initial_concentrations"] = (c0 * UNIT_FACTOR[req.get("concentration_unit", "mol/L")]).tolist()
    return out


@_wrap
def job_sensitivity(req: dict, limits: dict) -> dict:
    from ..analysis import sensitivity

    mech = build_mechanism(req["mechanism"])
    _limits(mech, limits)
    spec = reactor_spec_from_dict(req["reactor"], mech)
    opts = _sim_options(req["simulation"], limits)
    return sensitivity(mech, spec, opts, req["target"], req.get("reactions"), req.get("perturbation", 0.05),
                       req.get("max_reactions", 25), budget_s=limits["max_wall_time"])


@_wrap
def job_rop(req: dict, limits: dict) -> dict:
    from ..analysis import rates_of_production
    from ..simulation import simulate

    mech = build_mechanism(req["mechanism"])
    _limits(mech, limits)
    spec = reactor_spec_from_dict(req["reactor"], mech)
    opts = _sim_options(req["simulation"], limits)
    opts.max_points = min(opts.max_points, 600)
    return rates_of_production(simulate(mech, spec, opts), req["species"], req.get("top", 10))


@_wrap
def job_fit(req: dict, limits: dict) -> dict:
    from ..analysis import estimate_parameters

    mech = build_mechanism(req["mechanism"])
    _limits(mech, limits)
    spec = reactor_spec_from_dict(req["reactor"], mech)
    opts = _sim_options(req["simulation"], limits)
    f = 1.0 / UNIT_FACTOR[req.get("concentration_unit", "mol/L")]
    data = [{"species": d["species"], "t": d["t"], "c": [v * f for v in d["c"]],
             "sigma": [v * f for v in d["sigma"]] if d.get("sigma") else None} for d in req["data"]]
    out = estimate_parameters(mech, spec, opts, data, req["parameters"], budget_s=limits["max_wall_time"] * 2)
    inv = UNIT_FACTOR[req.get("concentration_unit", "mol/L")]
    for s in out["fit"]:
        s["measured"] = [v * inv for v in s["measured"]]
        s["model"] = [v * inv for v in s["model"]]
    return out


@_wrap
def job_numerics(req: dict, limits: dict) -> dict:
    try:
        from ..numerics import auto_optimize
    except ImportError as e:
        raise AutoKineticsError(f"Numerik-Modul nicht verfügbar: {e}") from None
    from ..reactor import build_reactor

    mech = build_mechanism(req["mechanism"])
    _limits(mech, limits)
    spec = reactor_spec_from_dict(req["reactor"], mech)
    opts = _sim_options(req["simulation"], limits)
    built = build_reactor(mech, spec, opts.rtol, opts.atol_rel)
    # The analysis targets a *global* accuracy; the tuner chooses method and local tolerance.
    target = float(req.get("target_rtol", 1e-4))
    atol = built.atol * (target / opts.rtol) if opts.rtol > 0 else built.atol
    if built.energy:
        atol[-1] = max(1e-6, target * 1e-2)
    kwargs: dict[str, Any] = dict(rtol=target, atol=atol, budget_s=min(req.get("budget_s", 20.0),
                                                                     limits["max_wall_time"]))
    if req.get("candidates"):
        kwargs["candidates"] = req["candidates"]
    if spec.kind == "batch" and not spec.fixed_species and mech.elements and all(s.composition for s in mech.species):
        E = mech.element_matrix()
        ns = mech.n_species
        kwargs["invariants"] = lambda y, E=E, ns=ns: E @ np.asarray(y)[..., :ns].T
    report = auto_optimize(built.system, (0.0, opts.t_end), built.y0, **kwargs)
    return report.to_dict()
