"""Shared helpers for the test-suite (importable because tests/ is on pythonpath)."""

import os
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def cantera_data(name: str) -> str:
    ct = pytest.importorskip("cantera")
    return os.path.join(os.path.dirname(ct.__file__), "data", name)


def rel_err(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
