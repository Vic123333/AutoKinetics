"""Flatten a :class:`Mechanism` into plain arrays (CSR layouts, SI units).

The same arrays feed the C++ kernel (via Cython) and the numpy reference
kernel in :mod:`autokinetics.kinetics_ref`, which is used for differential
testing and as a fallback when the extension is not built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .constants import GAS_CONSTANT as R
from .mechanism import Mechanism

RXN_ELEMENTARY, RXN_THREE_BODY, RXN_FALLOFF, RXN_PLOG = 0, 1, 2, 3
REV_NONE, REV_THERMO, REV_ARRHENIUS, REV_EQCONST = 0, 1, 2, 3
FALLOFF_LINDEMANN, FALLOFF_TROE, FALLOFF_SRI = 0, 1, 2
PHASE_GAS, PHASE_LIQUID = 0, 1


def _csr(rows: list[list[tuple[int, float]]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ptr = np.zeros(len(rows) + 1, dtype=np.int32)
    idx: list[int] = []
    val: list[float] = []
    for j, row in enumerate(rows):
        for i, v in row:
            idx.append(i)
            val.append(v)
        ptr[j + 1] = len(idx)
    return ptr, np.asarray(idx, dtype=np.int32), np.asarray(val, dtype=float)


@dataclass
class CompiledMechanism:
    n_species: int
    n_reactions: int
    phase: int
    # forward orders (reactants), reverse orders (products), net stoichiometry
    fwd_ptr: np.ndarray
    fwd_idx: np.ndarray
    fwd_ord: np.ndarray
    rev_ptr: np.ndarray
    rev_idx: np.ndarray
    rev_ord: np.ndarray
    nu_ptr: np.ndarray
    nu_idx: np.ndarray
    nu_val: np.ndarray
    dnu: np.ndarray  # sum(nu_products) - sum(nu_reactants)
    rtype: np.ndarray
    A: np.ndarray
    b: np.ndarray
    Ea_R: np.ndarray
    # third bodies (three-body + falloff rows)
    tb_rxn: np.ndarray
    tb_eff: np.ndarray  # (n_tb, n_species)
    # fall-off
    fo_rxn: np.ndarray
    fo_kind: np.ndarray
    fo_A: np.ndarray
    fo_b: np.ndarray
    fo_Ea_R: np.ndarray
    fo_params: np.ndarray  # (n_fo, 5): Troe A,T3,T1,T2,hasT2 | SRI a,b,c,d,e
    # PLOG
    plog_rxn: np.ndarray
    plog_ptr: np.ndarray
    plog_lnP: np.ndarray
    plog_A: np.ndarray
    plog_b: np.ndarray
    plog_Ea_R: np.ndarray
    # reverse
    rev_mode: np.ndarray
    rev_A: np.ndarray
    rev_b: np.ndarray
    rev_Ea_R: np.ndarray
    K_ref: np.ndarray
    K_Tref: np.ndarray
    K_dH_R: np.ndarray
    # thermo (NASA9 piecewise per species)
    has_thermo: np.ndarray  # (n_species,) uint8
    th_ptr: np.ndarray  # (n_species+1,)
    th_Tlo: np.ndarray
    th_Thi: np.ndarray
    th_coef: np.ndarray  # (n_ranges_total, 9)
    # optional per-reaction enthalpy (J/mol), NaN = compute from thermo
    dH_rxn: np.ndarray
    mw: np.ndarray  # kg/mol, NaN if unknown
    species_names: list[str] = field(default_factory=list)


def compile_mechanism(mech: Mechanism) -> CompiledMechanism:
    mech.check()
    ns, nr = mech.n_species, mech.n_reactions
    fwd_rows, rev_rows, nu_rows = [], [], []
    rtype = np.zeros(nr, np.int32)
    A = np.zeros(nr)
    b = np.zeros(nr)
    Ea_R = np.zeros(nr)
    dnu = np.zeros(nr)
    tb_rxn, tb_eff = [], []
    fo_rxn, fo_kind, fo_A, fo_b, fo_Ea_R, fo_params = [], [], [], [], [], []
    plog_rxn, plog_rows = [], []
    rev_mode = np.zeros(nr, np.int32)
    rev_A, rev_b, rev_Ea_R = np.zeros(nr), np.zeros(nr), np.zeros(nr)
    K_ref, K_Tref, K_dH_R = np.ones(nr), np.full(nr, 298.15), np.zeros(nr)
    dH_rxn = np.full(nr, np.nan)
    type_code = {"elementary": RXN_ELEMENTARY, "three-body": RXN_THREE_BODY, "falloff": RXN_FALLOFF,
                 "pressure-dependent-Arrhenius": RXN_PLOG}

    for j, r in enumerate(mech.reactions):
        fwd_rows.append([(mech.species_index(n), o) for n, o in r.forward_orders().items() if o != 0])
        rev_rows.append([(mech.species_index(n), o) for n, o in r.products.items()])
        net: dict[int, float] = {}
        for n, v in r.reactants.items():
            net[mech.species_index(n)] = net.get(mech.species_index(n), 0.0) - v
        for n, v in r.products.items():
            net[mech.species_index(n)] = net.get(mech.species_index(n), 0.0) + v
        nu_rows.append([(i, v) for i, v in net.items() if v != 0])
        dnu[j] = sum(r.products.values()) - sum(r.reactants.values())
        rtype[j] = type_code[r.type]
        A[j], b[j], Ea_R[j] = r.rate.A, r.rate.b, r.rate.Ea / R
        if r.type in ("three-body", "falloff"):
            eff = np.full(ns, r.default_efficiency)
            for n, e in r.efficiencies.items():
                eff[mech.species_index(n)] = e
            tb_rxn.append(j)
            tb_eff.append(eff)
        if r.type == "falloff":
            assert r.low_rate is not None and r.falloff is not None
            fo_rxn.append(j)
            fo_A.append(r.low_rate.A)
            fo_b.append(r.low_rate.b)
            fo_Ea_R.append(r.low_rate.Ea / R)
            p = r.falloff.params
            if r.falloff.kind == "Troe":
                fo_kind.append(FALLOFF_TROE)
                fo_params.append([p["A"], p["T3"], p["T1"], p.get("T2", 0.0), 1.0 if "T2" in p else 0.0])
            elif r.falloff.kind == "SRI":
                fo_kind.append(FALLOFF_SRI)
                fo_params.append([p["a"], p["b"], p["c"], p["d"], p["e"]])
            else:
                fo_kind.append(FALLOFF_LINDEMANN)
                fo_params.append([0.0] * 5)
        if r.type == "pressure-dependent-Arrhenius":
            # merge entries at identical pressure (they are summed, as in Cantera)
            plog_rxn.append(j)
            plog_rows.append([(math.log(P), k.A, k.b, k.Ea / R) for P, k in r.plog_rates])
        if r.reversible:
            if r.reverse_rate is not None:
                rev_mode[j] = REV_ARRHENIUS
                rev_A[j], rev_b[j], rev_Ea_R[j] = r.reverse_rate.A, r.reverse_rate.b, r.reverse_rate.Ea / R
            elif r.equilibrium_constant is not None:
                rev_mode[j] = REV_EQCONST
                ec = r.equilibrium_constant
                K_ref[j], K_Tref[j], K_dH_R[j] = ec.K_ref, ec.T_ref, ec.dH / R
            else:
                rev_mode[j] = REV_THERMO
        if r.reaction_enthalpy is not None:
            dH_rxn[j] = r.reaction_enthalpy

    fwd_ptr, fwd_idx, fwd_ord = _csr(fwd_rows)
    rev_ptr, rev_idx, rev_ord = _csr(rev_rows)
    nu_ptr, nu_idx, nu_val = _csr(nu_rows)

    plog_ptr = np.zeros(len(plog_rows) + 1, np.int32)
    plog_lnP, plog_A, plog_b, plog_Ea_R = [], [], [], []
    for k, rows in enumerate(plog_rows):
        for lnP, a, bb, e in rows:
            plog_lnP.append(lnP)
            plog_A.append(a)
            plog_b.append(bb)
            plog_Ea_R.append(e)
        plog_ptr[k + 1] = len(plog_lnP)

    has_thermo = np.zeros(ns, np.uint8)
    th_ptr = np.zeros(ns + 1, np.int32)
    th_Tlo, th_Thi, th_coef = [], [], []
    for i, s in enumerate(mech.species):
        if s.thermo is not None:
            has_thermo[i] = 1
            # Gas-phase entropies are referred to p° = 1 bar; liquids ignore the pressure reference.
            th = s.thermo.at_standard_pressure() if mech.phase == "ideal-gas" else s.thermo
            tr = th.t_ranges
            for k, c in enumerate(th.coeffs):
                th_Tlo.append(tr[k])
                th_Thi.append(tr[k + 1])
                th_coef.append(c)
        th_ptr[i + 1] = len(th_Tlo)

    def arr(x, dtype=float, shape=None):
        a = np.asarray(x, dtype=dtype)
        if shape is not None and a.size == 0:
            a = a.reshape(shape)
        return np.ascontiguousarray(a)

    return CompiledMechanism(
        n_species=ns, n_reactions=nr, phase=PHASE_GAS if mech.phase == "ideal-gas" else PHASE_LIQUID,
        fwd_ptr=fwd_ptr, fwd_idx=fwd_idx, fwd_ord=fwd_ord, rev_ptr=rev_ptr, rev_idx=rev_idx,
        rev_ord=rev_ord, nu_ptr=nu_ptr, nu_idx=nu_idx, nu_val=nu_val, dnu=dnu, rtype=rtype,
        A=A, b=b, Ea_R=Ea_R,
        tb_rxn=arr(tb_rxn, np.int32), tb_eff=arr(tb_eff, shape=(0, ns)),
        fo_rxn=arr(fo_rxn, np.int32), fo_kind=arr(fo_kind, np.int32), fo_A=arr(fo_A), fo_b=arr(fo_b),
        fo_Ea_R=arr(fo_Ea_R), fo_params=arr(fo_params, shape=(0, 5)),
        plog_rxn=arr(plog_rxn, np.int32), plog_ptr=plog_ptr, plog_lnP=arr(plog_lnP), plog_A=arr(plog_A),
        plog_b=arr(plog_b), plog_Ea_R=arr(plog_Ea_R),
        rev_mode=rev_mode, rev_A=rev_A, rev_b=rev_b, rev_Ea_R=rev_Ea_R, K_ref=K_ref, K_Tref=K_Tref,
        K_dH_R=K_dH_R, has_thermo=has_thermo, th_ptr=th_ptr, th_Tlo=arr(th_Tlo), th_Thi=arr(th_Thi),
        th_coef=arr(th_coef, shape=(0, 9)), dH_rxn=dH_rxn, mw=mech.molecular_weights(),
        species_names=mech.species_names,
    )
