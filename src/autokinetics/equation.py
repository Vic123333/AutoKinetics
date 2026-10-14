"""Parser for reaction equations (Cantera syntax plus common shorthands).

Examples::

    2 H2 + O2 => 2 H2O            irreversible
    N2O4 <=> 2 NO2                reversible  ('=', '<->', '⇌' also accepted)
    2 O + M <=> O2 + M            three-body
    H + O2 (+M) <=> HO2 (+M)      fall-off, generic collider
    H + O2 (+AR) <=> HO2 (+AR)    fall-off, specific collider
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .errors import MechanismError

_ARROWS = [
    ("<=>", True), ("<->", True), ("⇌", True), ("⇄", True), ("=>", False), ("->", False),
    ("→", False), ("⟶", False), ("=", True),
]
_FALLOFF = re.compile(r"\(\s*\+\s*([^()\s]+(?:\([^()]*\))?)\s*\)")
_COEFF = re.compile(r"^(\d+(?:\.\d*)?|\.\d+)(?:\s+|\*)?(.+)$")


@dataclass
class ParsedEquation:
    reactants: dict[str, float]
    products: dict[str, float]
    reversible: bool
    third_body: str | None = None  # "M" for '+ M', None otherwise
    falloff_collider: str | None = None  # "M" or species name for '(+X)'
    warnings: list[str] = field(default_factory=list)

    def canonical(self) -> str:
        return format_equation(self.reactants, self.products, self.reversible,
                               self.third_body, self.falloff_collider)


def _fmt_side(side: dict[str, float]) -> str:
    parts = []
    for name, nu in side.items():
        if nu == 1:
            parts.append(name)
        else:
            parts.append(f"{int(nu) if float(nu).is_integer() else nu:g} {name}")
    return " + ".join(parts)


def format_equation(reactants: dict[str, float], products: dict[str, float], reversible: bool,
                    third_body: str | None = None, falloff_collider: str | None = None) -> str:
    lhs, rhs = _fmt_side(reactants), _fmt_side(products)
    if third_body:
        lhs, rhs = f"{lhs} + {third_body}", f"{rhs} + {third_body}"
    if falloff_collider:
        lhs, rhs = f"{lhs} (+{falloff_collider})", f"{rhs} (+{falloff_collider})"
    return f"{lhs} {'<=>' if reversible else '=>'} {rhs}"


def _split_species(side: str, known: set[str] | None) -> list[str]:
    side = side.strip()
    if not side:
        return []
    chunks = [c.strip() for c in re.split(r"\s+\+\s+|\s+\+$|^\+\s+", side) if c.strip()]
    result: list[str] = []
    for chunk in chunks:
        if known is not None and "+" in chunk and _strip_coeff(chunk)[1] not in known:
            parts = [p.strip() for p in chunk.split("+") if p.strip()]
            if all(_strip_coeff(p)[1] in known or _strip_coeff(p)[1] == "M" for p in parts):
                result.extend(parts)
                continue
        result.append(chunk)
    return result


def _strip_coeff(token: str, known: set[str] | None = None) -> tuple[float, str]:
    token = token.strip()
    if known is not None and token in known:
        return 1.0, token
    m = _COEFF.match(token)
    if m:
        name = m.group(2).strip()
        # "2O" is ambiguous with species names starting with a digit; prefer known names.
        if known is None or name in known or name == "M" or token not in known:
            return float(m.group(1)), name
    return 1.0, token


def parse_equation(text: str, known_species: set[str] | list[str] | None = None) -> ParsedEquation:
    known = set(known_species) if known_species is not None else None
    eq = " ".join(text.strip().split())
    if not eq:
        raise MechanismError("Leere Reaktionsgleichung")

    falloff = _FALLOFF.findall(eq)
    collider = None
    if falloff:
        if len(falloff) != 2 or falloff[0] != falloff[1]:
            raise MechanismError(f"'{text}': '(+X)' muss auf beiden Seiten identisch stehen")
        collider = falloff[0]
        eq = _FALLOFF.sub(" ", eq)
        eq = " ".join(eq.split())

    for arrow, arrow_reversible in _ARROWS:
        if arrow in eq:
            lhs, _, rhs = eq.partition(arrow)
            if arrow == "=" and ("<" in lhs[-1:] or ">" in rhs[:1]):
                continue
            reversible = arrow_reversible
            break
    else:
        raise MechanismError(f"'{text}': kein Reaktionspfeil gefunden (=>, <=>, ->, =)")

    warnings: list[str] = []
    sides: list[dict[str, float]] = []
    third_body_flags = []
    for side_text in (lhs, rhs):
        side: dict[str, float] = {}
        has_m = False
        for token in _split_species(side_text, known):
            nu, name = _strip_coeff(token, known)
            if nu <= 0:
                raise MechanismError(f"'{text}': stöchiometrischer Koeffizient muss > 0 sein")
            if name == "M":
                if has_m:
                    raise MechanismError(f"'{text}': M darf nur einmal pro Seite stehen")
                has_m = True
                continue
            if known is not None and name not in known:
                raise MechanismError(f"'{text}': unbekannte Spezies '{name}'")
            side[name] = side.get(name, 0.0) + nu
        sides.append(side)
        third_body_flags.append(has_m)

    if third_body_flags[0] != third_body_flags[1]:
        raise MechanismError(f"'{text}': '+ M' muss auf beiden Seiten stehen")
    if third_body_flags[0] and collider:
        raise MechanismError(f"'{text}': '+ M' und '(+M)' gleichzeitig ist nicht erlaubt")
    if not sides[0]:
        raise MechanismError(f"'{text}': keine Edukte")
    if not sides[1]:
        warnings.append("Reaktion ohne Produkte (Senke)")
    if known is not None and collider and collider != "M" and collider not in known:
        raise MechanismError(f"'{text}': unbekannter Stoßpartner '{collider}'")
    return ParsedEquation(sides[0], sides[1], reversible, "M" if third_body_flags[0] else None,
                          collider, warnings)
