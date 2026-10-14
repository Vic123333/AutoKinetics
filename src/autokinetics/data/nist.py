"""NIST Chemistry WebBook (https://webbook.nist.gov) gas-phase thermochemistry.

The WebBook has no JSON API; the SI HTML pages (``Mask=1``) are parsed. Data are
public domain (U.S. Government). Shomate parameters are converted exactly into
the internal NASA-9 representation (reference pressure 1 bar).
"""

from __future__ import annotations

import html
import re
from urllib.parse import quote

from ..errors import ExternalDataError
from ..thermo import Nasa9Thermo, from_shomate
from .http import fetch

BASE = "https://webbook.nist.gov/cgi/cbook.cgi"


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _num(text: str) -> float | None:
    m = re.match(r"\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)", text.replace("−", "-").replace("&#8722;", "-"))
    return float(m.group(1)) if m else None


def _page_url(cas: str | None = None, name: str | None = None, formula: str | None = None) -> str:
    if cas:
        return f"{BASE}?ID=C{cas.replace('-', '')}&Units=SI&Mask=1"
    if name:
        return f"{BASE}?Name={quote(name)}&Units=SI&Mask=1"
    if formula:
        return f"{BASE}?Formula={quote(formula)}&NoIon=on&Units=SI&Mask=1"
    raise ValueError("cas, name or formula required")


def parse_thermo_page(page: str) -> dict:
    """Extract name, formula, CAS, ΔfH°gas, S°gas and Shomate tables from a WebBook page."""
    out: dict = {"name": None, "formula": None, "cas": None, "dHf_gas": None, "S_gas": None,
                 "cp_gas_298": None, "cp_points": [], "shomate": None, "references": []}
    m = re.search(r"<title>([^<]*)</title>", page)
    out["name"] = html.unescape(m.group(1)).strip() if m else None
    m = re.search(r'"molecularFormula"\s*:\s*"([^"]+)"', page)
    out["formula"] = m.group(1) if m else None
    m = re.search(r"CAS Registry Number:</strong>\s*([\d-]+)", page)
    out["cas"] = m.group(1) if m else None

    # ΔfH°gas rows: prefer the first (review) value
    for row in re.finditer(r"<tr[^>]*>\s*<td[^>]*>&#916;<sub>f</sub>H&deg;<sub>gas</sub></td>(.*?)</tr>", page, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row.group(1), re.S)
        if len(cells) >= 2 and "kJ/mol" in _clean(cells[1]):
            v = _num(_clean(cells[0]))
            if v is not None:
                out["dHf_gas"] = v * 1e3
                out["references"].append(_clean(cells[3]) if len(cells) > 3 else "")
                break
    s_candidates = []
    for row in re.finditer(r"<tr[^>]*>\s*<td[^>]*>S&deg;<sub>gas(,1 bar)?</sub></td>(.*?)</tr>", page, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row.group(2), re.S)
        if len(cells) >= 2 and "J/mol*K" in _clean(cells[1]):
            v = _num(_clean(cells[0]))
            if v is not None:
                s_candidates.append((0 if row.group(1) else 1, v))  # prefer the explicit 1-bar value
    if s_candidates:
        out["S_gas"] = sorted(s_candidates, key=lambda x: x[0])[0][1]
    # constant-pressure heat capacity tables: collect all (T, cp) points
    points: dict[float, float] = {}
    for cp_tab in re.finditer(r'aria-label="Constant pressure heat capacity of gas"[^>]*>(.*?)</table>', page, re.S):
        for row in re.finditer(r"<tr[^>]*>(.*?)</tr>", cp_tab.group(1), re.S):
            cells = [_clean(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row.group(1), re.S)]
            if len(cells) >= 2:
                cp, T = _num(cells[0]), _num(cells[1])
                if cp is not None and T is not None and 50.0 < T < 6000.0 and cp > 0:
                    points.setdefault(round(T, 2), cp)
    out["cp_points"] = sorted(points.items())
    if points:
        T_near = min(points, key=lambda T: abs(T - 298.15))
        if abs(T_near - 298.15) < 30:
            out["cp_gas_298"] = points[T_near]
    tab = re.search(r'<table[^>]*aria-label="Gas Phase Heat Capacity \(Shomate Equation\)"[^>]*>(.*?)</table>', page, re.S)
    if tab:
        rows = {}
        for row in re.finditer(r"<tr>\s*<th[^>]*>(.*?)</th>(.*?)</tr>", tab.group(1), re.S):
            key = _clean(row.group(1))
            rows[key] = [_clean(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", row.group(2), re.S)]
        ranges = rows.get("Temperature (K)", [])
        coeffs = []
        t_bounds = []
        for k, rng in enumerate(ranges):
            m = re.match(r"([\d.]+)\s*to\s*([\d.]+)", rng)
            if not m:
                continue
            vals = [_num(rows.get(letter, [""] * len(ranges))[k] if k < len(rows.get(letter, [])) else "")
                    for letter in "ABCDEFGH"]
            if any(v is None for v in vals[:7]):
                continue
            t_bounds.append((float(m.group(1)), float(m.group(2))))
            coeffs.append(vals)
        if coeffs:
            out["shomate"] = {"ranges": t_bounds, "coeffs": coeffs,
                              "reference": rows.get("Reference", [""])[0] if rows.get("Reference") else ""}
    return out


def thermo_from_page(info: dict) -> Nasa9Thermo | None:
    sh = info.get("shomate")
    if not sh:
        return None
    ranges = sh["ranges"]
    # Use contiguous ranges only; NIST occasionally lists overlapping/discontinuous ranges.
    t_ranges = [ranges[0][0]]
    data = []
    for (lo, hi), c in zip(ranges, sh["coeffs"], strict=True):
        if abs(lo - t_ranges[-1]) > 1.0:
            break
        t_ranges.append(hi)
        data.append(c[:7])
    src = f"NIST WebBook (Shomate), {sh.get('reference', '')}".strip(", ")
    return from_shomate(t_ranges, data, source=src)


def lookup(cas: str | None = None, name: str | None = None, formula: str | None = None) -> dict | None:
    """Fetch and parse the WebBook gas-phase thermochemistry page. None if not found."""
    page = fetch(_page_url(cas, name, formula), source="NIST WebBook", not_found_ok=True)
    if page is None or "Name Not Found" in page or "No Matching Species Found" in page:
        return None
    if "<title>Search Results</title>" in page:
        m = re.search(r'href="/cgi/cbook\.cgi\?ID=(C?\d+)[^"]*"', page)
        if not m:
            return None
        page = fetch(f"{BASE}?ID={m.group(1)}&Units=SI&Mask=1", source="NIST WebBook", not_found_ok=True)
        if page is None:
            return None
    info = parse_thermo_page(page)
    if info["dHf_gas"] is None and info["shomate"] is None:
        return info | {"thermo": None}
    th = thermo_from_page(info)
    if th is not None and info["dHf_gas"] is not None:
        # consistency check: the Shomate fit must reproduce ΔfH°(298.15 K)
        h298 = th.h(298.15)
        if abs(h298 - info["dHf_gas"]) > 2000.0:
            raise ExternalDataError(f"NIST-Daten inkonsistent für {info['name']}: "
                                    f"h(298 K)={h298 / 1e3:.2f} vs ΔfH°={info['dHf_gas'] / 1e3:.2f} kJ/mol")
    info["thermo"] = th
    return info


def thermo_from_cp_points(dHf: float, S: float, points: list[tuple[float, float]],
                          source: str) -> Nasa9Thermo | None:
    """NASA-7 polynomial from tabulated cp(T): least-squares fit (Householder QR) of
    cp/R = a1 + a2 T + a3 T² + a4 T³, with a6, a7 fixed by ΔfH°(298.15) and S°(298.15)."""
    import numpy as np

    from ..constants import GAS_CONSTANT as R
    from ..thermo import from_constant_cp, from_nasa7

    if not points:
        return None
    T = np.array([p[0] for p in points])
    cp = np.array([p[1] for p in points]) / R
    span = T.max() - T.min()
    if len(points) < 3 or span < 100.0:
        near = int(np.argmin(np.abs(T - 298.15)))
        if abs(T[near] - 298.15) > 50:
            return None
        return from_constant_cp(dHf, S, cp[near] * R, source=source + "; konstantes cp")
    deg = min(3, len(points) - 1)
    x = (T - 1000.0) / 1000.0  # scaled for conditioning
    V = np.vander(x, deg + 1, increasing=True)
    q, r = np.linalg.qr(V)
    c_scaled = np.linalg.solve(r, q.T @ cp)
    # back to powers of T: sum_k c_k ((T-1000)/1000)^k
    poly = np.polynomial.Polynomial(c_scaled)
    shifted = poly(np.polynomial.Polynomial([-1.0, 1e-3]))
    a = list(shifted.coef) + [0.0] * (5 - len(shifted.coef))
    a1, a2, a3, a4, a5 = a[:5]
    T0 = 298.15
    h_rt_noconst = a1 + a2 * T0 / 2 + a3 * T0**2 / 3 + a4 * T0**3 / 4 + a5 * T0**4 / 5
    a6 = (dHf / (R * T0) - h_rt_noconst) * T0
    s_noconst = a1 * np.log(T0) + a2 * T0 + a3 * T0**2 / 2 + a4 * T0**3 / 3 + a5 * T0**4 / 4
    a7 = S / R - s_noconst
    t_lo = float(max(200.0, min(T.min(), 298.15) - 50.0))
    t_hi = float(T.max() + 100.0)
    return from_nasa7([t_lo, t_hi], [[a1, a2, a3, a4, a5, a6, a7]],
                      source=source + f"; cp-Polynom (Grad {deg}) aus {len(points)} Messpunkten "
                      f"{T.min():.0f}–{T.max():.0f} K", reference_pressure=1e5)
