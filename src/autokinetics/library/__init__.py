"""Example mechanisms with reactor presets and literature references."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from ..errors import MechanismError
from ..io.yaml_format import mechanism_from_dict, safe_load_yaml
from ..mechanism import Mechanism


@dataclass
class Example:
    id: str
    title: str
    category: str
    summary: str
    parameter_quality: str
    sources: list[str]
    n_species: int
    n_reactions: int
    phase: str
    preset: dict  # {"reactor": {...}, "simulation": {...}, "plot": {...}, "expected": [...]}

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _files() -> list[str]:
    root = resources.files(__name__)
    return sorted(p.name for p in root.iterdir() if p.name.endswith(".yaml"))


@lru_cache(maxsize=64)
def _load_text(fname: str) -> str:
    return resources.files(__name__).joinpath(fname).read_text(encoding="utf-8")


def load_example(example_id: str) -> tuple[Mechanism, Example, str]:
    """Return (mechanism, metadata, original YAML text)."""
    fname = f"{example_id}.yaml"
    if fname not in _files():
        raise MechanismError(f"Beispiel '{example_id}' nicht gefunden")
    text = _load_text(fname)
    doc = safe_load_yaml(text)
    mech = mechanism_from_dict(doc, name=example_id)
    meta = doc.get("autokinetics") or {}
    ex = Example(
        id=example_id, title=meta.get("title", example_id), category=meta.get("category", "Sonstiges"),
        summary=meta.get("summary", doc.get("description", "")), parameter_quality=meta.get("parameter_quality", "literature"),
        sources=list(doc.get("references", [])) + list(meta.get("sources", [])), n_species=mech.n_species,
        n_reactions=mech.n_reactions, phase=mech.phase,
        preset={k: meta.get(k) for k in ("reactor", "simulation", "plot", "expected") if meta.get(k) is not None},
    )
    return mech, ex, text


def list_examples() -> list[Example]:
    return [load_example(f[:-5])[1] for f in _files()]
