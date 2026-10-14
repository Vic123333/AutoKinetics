"""Chemical formulas, element compositions and molar masses."""

from __future__ import annotations

import re

from .errors import MechanismError

# Standard atomic weights (IUPAC, conventional values) in g/mol.
ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.008, "D": 2.014101778, "T": 3.01604928, "He": 4.002602, "Li": 6.94,
    "Be": 9.0121831, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163,
    "Ne": 20.1797, "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044,
    "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.630, "As": 74.921595, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95,
    "Tc": 97.0, "Ru": 101.07, "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682,
    "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196, "Ba": 137.327, "La": 138.90547,
    "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36,
    "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535, "Dy": 162.500, "Ho": 164.93033,
    "Er": 167.259, "Tm": 168.93422, "Yb": 173.045, "Lu": 174.9668, "Hf": 178.49,
    "Ta": 180.94788, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217,
    "Pt": 195.084, "Au": 196.966569, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98040, "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0,
    "Ac": 227.0, "Th": 232.0377, "Pa": 231.03588, "U": 238.02891,
    "E": 5.48579909065e-4,  # electron (only via explicit composition / charge)
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?|(\()|(\))(\d+(?:\.\d+)?)?")
_CHARGE = re.compile(r"(?:\^?(\d*)([+-])|([+-]+))$")


def _parse_group(formula: str) -> dict[str, float]:
    stack: list[dict[str, float]] = [{}]
    pos = 0
    while pos < len(formula):
        m = _TOKEN.match(formula, pos)
        if not m or m.end() == pos:
            raise MechanismError(f"Ungültige Summenformel '{formula}' (Position {pos + 1})")
        element, count, open_p, close_p, group_count = m.groups()
        if element:
            if element not in ATOMIC_WEIGHTS or element == "E":
                raise MechanismError(f"Unbekanntes Element '{element}' in '{formula}'")
            stack[-1][element] = stack[-1].get(element, 0.0) + float(count or 1)
        elif open_p:
            stack.append({})
        elif close_p:
            if len(stack) == 1:
                raise MechanismError(f"Klammerfehler in '{formula}'")
            inner = stack.pop()
            mult = float(group_count or 1)
            for el, n in inner.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + n * mult
        pos = m.end()
    if len(stack) != 1:
        raise MechanismError(f"Klammerfehler in '{formula}'")
    return stack[0]


def parse_formula(formula: str) -> tuple[dict[str, float], int]:
    """Parse a molecular formula into (composition, charge).

    Supports parentheses ``Ca(OH)2``, hydrates ``CuSO4·5H2O`` / ``CuSO4*5H2O``,
    and charges ``Na+``, ``SO4-2``, ``SO4^2-``, ``Fe+++``.
    """
    text = formula.strip().replace(" ", "")
    if not text:
        raise MechanismError("Leere Summenformel")
    charge = 0
    m = _CHARGE.search(text)
    if m and m.start() > 0:
        if m.group(3):
            sign = 1 if m.group(3)[0] == "+" else -1
            charge = sign * len(m.group(3))
        else:
            digits, sign_char = m.group(1), m.group(2)
            charge = (1 if sign_char == "+" else -1) * int(digits or 1)
        text = text[: m.start()]
    elif "-" in text or "+" in text[1:]:
        # "SO4-2" style (sign before digits)
        m2 = re.search(r"([+-])(\d+)$", text)
        if m2:
            charge = (1 if m2.group(1) == "+" else -1) * int(m2.group(2))
            text = text[: m2.start()]
    composition: dict[str, float] = {}
    for part in re.split(r"[·*.](?=\d*[A-Z(])", text):
        mult_match = re.match(r"^(\d+)(?=[A-Z(])", part)
        mult = 1.0
        if mult_match:
            mult = float(mult_match.group(1))
            part = part[mult_match.end():]
        for el, n in _parse_group(part).items():
            composition[el] = composition.get(el, 0.0) + n * mult
    return composition, charge


def try_parse_formula(formula: str) -> tuple[dict[str, float], int] | None:
    try:
        return parse_formula(formula)
    except MechanismError:
        return None


def molar_mass(composition: dict[str, float], charge: float = 0.0) -> float:
    """Molar mass in kg/mol (electrons removed/added for ions)."""
    total = 0.0
    for el, n in composition.items():
        if el not in ATOMIC_WEIGHTS:
            raise MechanismError(f"Unbekanntes Element '{el}'")
        total += ATOMIC_WEIGHTS[el] * n
    total -= charge * ATOMIC_WEIGHTS["E"]
    return total * 1.0e-3


def hill_formula(composition: dict[str, float]) -> str:
    def fmt(el: str) -> str:
        n = composition[el]
        if n == 1:
            return el
        return f"{el}{int(n) if float(n).is_integer() else n}"

    els = [e for e in composition if composition[e] != 0]
    if "C" in els:
        order = ["C"] + (["H"] if "H" in els else []) + sorted(e for e in els if e not in ("C", "H"))
    else:
        order = sorted(els)
    return "".join(fmt(e) for e in order)
