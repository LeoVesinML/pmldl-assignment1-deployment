"""Shared pytest fixtures and import paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for extra_path in (PROJECT_ROOT / "code", PROJECT_ROOT / "code" / "models", PROJECT_ROOT / "code" / "datasets"):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))


@pytest.fixture()
def raw_like_frame() -> pd.DataFrame:
    """A small frame with the same schema as the combined raw dataset."""
    rng = np.random.default_rng(0)
    size = 200
    frame = pd.DataFrame(
        {
            "fixed_acidity": rng.normal(7.2, 1.0, size),
            "volatile_acidity": rng.normal(0.34, 0.1, size).clip(0.05),
            "citric_acid": rng.normal(0.32, 0.1, size).clip(0),
            "residual_sugar": rng.normal(5.4, 2.0, size).clip(0.6),
            "chlorides": rng.normal(0.056, 0.01, size).clip(0.01),
            "free_sulfur_dioxide": rng.normal(30, 8, size).clip(1),
            "total_sulfur_dioxide": rng.normal(115, 25, size).clip(6),
            "density": rng.normal(0.9947, 0.002, size),
            "ph": rng.normal(3.2, 0.12, size),
            "sulphates": rng.normal(0.53, 0.1, size).clip(0.2),
            "alcohol": rng.normal(10.5, 1.1, size).clip(8),
            "quality": rng.integers(3, 9, size),
            "wine_type": rng.choice(["red", "white"], size),
        }
    )
    return frame
