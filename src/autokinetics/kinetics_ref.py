"""Reference (pure numpy) kinetics kernel.

Mirrors the C++ kernel in ``cpp/src/kinetics`` one-to-one. It is used for
differential testing of the compiled code and as a fallback when the extension
module is unavailable. Not optimised for speed.
"""

from __future__ import annotations

import math

import numpy as np

from .compiled import (
    FALLOFF_SRI,
    FALLOFF_TROE,
    PHASE_GAS,
    REV_ARRHENIUS,
    REV_EQCONST,
    REV_THERMO,
    RXN_THREE_BODY,
    CompiledMechanism,
)
from .constants import GAS_CONSTANT as R
from .constants import STANDARD_CONCENTRATION_LIQUID, STANDARD_PRESSURE

SMALL = 1e-300


def species_thermo(cm: CompiledMechanism, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Molar cp [J/mol/K], h [J/mol], s° [J/mol/K]; NaN for species without thermo."""
    ns = cm.n_species
    cp, h, s = np.full(ns, np.nan), np.full(ns, np.nan), np.full(ns, np.nan)
    lnT = math.log(T)
    for i in range(ns):
        lo, hi = cm.th_ptr[i], cm.th_ptr[i + 1]
        if hi == lo:
            continue
        k = hi - 1
        for kk in range(lo, hi):
            if T <= cm.th_Thi[kk]:
                k = kk
                break
        a = cm.th_coef[k]
        cp[i] = R * (a[0] / T**2 + a[1] / T + a[2] + T * (a[3] + T * (a[4] + T * (a[5] + T * a[6]))))
        h[i] = R * T * (-a[0] / T**2 + a[1] * lnT / T + a[2]
                        + T * (a[3] / 2 + T * (a[4] / 3 + T * (a[5] / 4 + T * a[6] / 5))) + a[7] / T)
        s[i] = R * (-a[0] / (2 * T**2) - a[1] / T + a[2] * lnT
                    + T * (a[3] + T * (a[4] / 2 + T * (a[5] / 3 + T * a[6] / 4))) + a[8])
    return cp, h, s


def equilibrium_constants(cm: CompiledMechanism, T: float) -> np.ndarray:
    """Kc in SI concentration units for every reaction (NaN if not determinable)."""
    _, h, s = species_thermo(cm, T)
    g_RT = (h - T * s) / (R * T)
    c0 = STANDARD_PRESSURE / (R * T) if cm.phase == PHASE_GAS else STANDARD_CONCENTRATION_LIQUID
    Kc = np.full(cm.n_reactions, np.nan)
    for j in range(cm.n_reactions):
        mode = cm.rev_mode[j]
        if mode == REV_EQCONST:
            Kc[j] = cm.K_ref[j] * math.exp(-cm.K_dH_R[j] * (1.0 / T - 1.0 / cm.K_Tref[j]))
            continue
        sl = slice(cm.nu_ptr[j], cm.nu_ptr[j + 1])
        dg = float(np.dot(cm.nu_val[sl], g_RT[cm.nu_idx[sl]]))
        if not math.isnan(dg):
            Kc[j] = math.exp(min(-dg, 700.0)) * c0 ** cm.dnu[j]
    return Kc


def _conc_product(ptr, idx, order, c, j) -> float:
    p = 1.0
    for k in range(ptr[j], ptr[j + 1]):
        ci, o = c[idx[k]], order[k]
        if o == 1.0:
            p *= ci
        elif float(o).is_integer():
            p *= ci ** int(o)
        else:
            p *= max(ci, 0.0) ** o
    return p


def rate_constants(cm: CompiledMechanism, T: float, P: float, c: np.ndarray):
    """Return (kf, kr, M) where kf already contains fall-off corrections."""
    nr = cm.n_reactions
    kf = cm.A * T**cm.b * np.exp(-cm.Ea_R / T)
    M = np.ones(nr)
    for k, j in enumerate(cm.tb_rxn):
        if cm.rtype[j] == RXN_THREE_BODY:
            M[j] = float(np.dot(cm.tb_eff[k], c))
    for k, j in enumerate(cm.plog_rxn):
        lo, hi = cm.plog_ptr[k], cm.plog_ptr[k + 1]
        lnPs = cm.plog_lnP[lo:hi]
        rates = cm.plog_A[lo:hi] * T ** cm.plog_b[lo:hi] * np.exp(-cm.plog_Ea_R[lo:hi] / T)
        # sum duplicates at identical pressure
        uniq = np.unique(lnPs)
        ksum = np.array([rates[lnPs == u].sum() for u in uniq])
        lnP = math.log(P)
        if lnP <= uniq[0]:
            kf[j] = ksum[0]
        elif lnP >= uniq[-1]:
            kf[j] = ksum[-1]
        else:
            m = int(np.searchsorted(uniq, lnP)) - 1
            l1, l2 = math.log(max(ksum[m], SMALL)), math.log(max(ksum[m + 1], SMALL))
            kf[j] = math.exp(l1 + (l2 - l1) * (lnP - uniq[m]) / (uniq[m + 1] - uniq[m]))
    tb_pos = {int(j): k for k, j in enumerate(cm.tb_rxn)}
    for k, j in enumerate(cm.fo_rxn):
        k0 = cm.fo_A[k] * T ** cm.fo_b[k] * math.exp(-cm.fo_Ea_R[k] / T)
        kinf = kf[j]
        conc_m = float(np.dot(cm.tb_eff[tb_pos[int(j)]], c))
        pr = conc_m * k0 / kinf if kinf > 0 else 0.0
        F = 1.0
        kind = cm.fo_kind[k]
        p = cm.fo_params[k]
        if kind == FALLOFF_TROE:
            A, T3, T1, T2, hasT2 = p
            fcent = (1 - A) * (math.exp(-T / T3) if abs(T3) > 1e-300 else 0.0) \
                + A * (math.exp(-T / T1) if abs(T1) > 1e-300 else 0.0)
            if hasT2:
                fcent += math.exp(-T2 / T)
            log_fcent = math.log10(max(fcent, SMALL))
            lpr = math.log10(max(pr, SMALL))
            cc = -0.4 - 0.67 * log_fcent
            nn = 0.75 - 1.27 * log_fcent
            f1 = (lpr + cc) / (nn - 0.14 * (lpr + cc))
            F = 10.0 ** (log_fcent / (1.0 + f1 * f1))
        elif kind == FALLOFF_SRI:
            a, bb, cc_, d, e = p
            lpr = math.log10(max(pr, SMALL))
            X = 1.0 / (1.0 + lpr * lpr)
            F = d * (a * math.exp(-bb / T) + math.exp(-T / cc_)) ** X * T**e
        kf[j] = kinf * pr / (1.0 + pr) * F
        M[j] = 1.0  # [M] is part of k for fall-off reactions
    kr = np.zeros(nr)
    Kc = None
    for j in range(nr):
        mode = cm.rev_mode[j]
        if mode == REV_ARRHENIUS:
            kr[j] = cm.rev_A[j] * T ** cm.rev_b[j] * math.exp(-cm.rev_Ea_R[j] / T)
        elif mode in (REV_THERMO, REV_EQCONST):
            if Kc is None:
                Kc = equilibrium_constants(cm, T)
            kr[j] = kf[j] / Kc[j]
    return kf, kr, M


def rates_of_progress(cm: CompiledMechanism, T: float, P: float, c: np.ndarray):
    kf, kr, M = rate_constants(cm, T, P, c)
    qf = np.empty(cm.n_reactions)
    qr = np.empty(cm.n_reactions)
    for j in range(cm.n_reactions):
        qf[j] = M[j] * kf[j] * _conc_product(cm.fwd_ptr, cm.fwd_idx, cm.fwd_ord, c, j)
        qr[j] = M[j] * kr[j] * _conc_product(cm.rev_ptr, cm.rev_idx, cm.rev_ord, c, j) if kr[j] else 0.0
    return qf, qr, qf - qr


def production_rates(cm: CompiledMechanism, T: float, P: float, c: np.ndarray) -> np.ndarray:
    _, _, q = rates_of_progress(cm, T, P, c)
    wdot = np.zeros(cm.n_species)
    for j in range(cm.n_reactions):
        for k in range(cm.nu_ptr[j], cm.nu_ptr[j + 1]):
            wdot[cm.nu_idx[k]] += cm.nu_val[k] * q[j]
    return wdot
