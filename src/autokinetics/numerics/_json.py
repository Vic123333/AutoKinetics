"""Conversion of analysis results into strict-JSON-compatible Python objects."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np


def jsonable(obj: Any) -> Any:
    """Recursively converts numpy types, dataclasses and tuples to JSON-compatible
    types. Non-finite floats become None (strict JSON has no NaN/Infinity)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (complex, np.complexfloating)):
        return {"re": jsonable(obj.real), "im": jsonable(obj.imag)}
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"re": jsonable(obj.real), "im": jsonable(obj.imag)}
        return [jsonable(v) for v in obj.tolist()] if obj.ndim else jsonable(obj.item())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    if dataclasses.is_dataclass(obj):
        return jsonable(dataclasses.asdict(obj))
    raise TypeError(f"cannot convert {type(obj).__name__} to JSON")
