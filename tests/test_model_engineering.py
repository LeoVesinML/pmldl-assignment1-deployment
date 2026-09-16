"""Stage 2 tests: feature engineering and the packaged pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import (
    ENGINEERED_FEATURES,
    RAW_FEATURES,
    RAW_NUMERIC_FEATURES,
    add_engineered_features,
)
from train import build_pipeline


def _sample(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {name: rng.normal(5, 1, n).clip(0.1) for name in RAW_NUMERIC_FEATURES}
    )
    frame["density"] = rng.normal(0.995, 0.002, n)
    frame["wine_type"] = rng.choice(["red", "white"], n)
    return frame[RAW_FEATURES]


def test_engineered_features_are_added_and_input_is_not_mutated() -> None:
    frame = _sample()
    before = frame.copy()
    out = add_engineered_features(frame)
    assert list(out.columns) == RAW_FEATURES + ENGINEERED_FEATURES
    pd.testing.assert_frame_equal(frame, before)


def test_engineered_features_survive_zero_division() -> None:
    frame = _sample(3)
    frame.loc[:, "total_sulfur_dioxide"] = 0.0
    frame.loc[:, "volatile_acidity"] = 0.0
    out = add_engineered_features(frame)
    assert np.isfinite(out[ENGINEERED_FEATURES].to_numpy()).all()


def test_bound_sulfur_dioxide_is_never_negative() -> None:
    frame = _sample(10)
    frame["free_sulfur_dioxide"] = 200.0
    frame["total_sulfur_dioxide"] = 50.0
    out = add_engineered_features(frame)
    assert (out["bound_sulfur_dioxide"] >= 0).all()


def test_wine_type_is_normalised() -> None:
    frame = _sample(4)
    frame["wine_type"] = [" RED ", "White", "red", "WHITE"]
    out = add_engineered_features(frame)
    assert set(out["wine_type"]) == {"red", "white"}


@pytest.mark.parametrize(
    "model_name", ["logistic_regression", "random_forest", "hist_gradient_boosting"]
)
def test_pipeline_trains_and_predicts_from_raw_input(model_name: str) -> None:
    frame = _sample(120)
    target = (frame["alcohol"] > frame["alcohol"].median()).astype(int)
    pipeline = build_pipeline(model_name, random_state=0)
    pipeline.fit(frame, target)

    proba = pipeline.predict_proba(frame.head(5))[:, 1]
    predictions = pipeline.predict(frame.head(5))
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(np.unique(predictions)) <= {0, 1}


def test_pipeline_handles_missing_values_at_inference_time() -> None:
    frame = _sample(80)
    target = (frame["alcohol"] > frame["alcohol"].median()).astype(int)
    pipeline = build_pipeline("logistic_regression", random_state=0).fit(frame, target)

    degraded = frame.head(1).copy()
    degraded.loc[:, "chlorides"] = np.nan
    assert 0.0 <= float(pipeline.predict_proba(degraded)[0, 1]) <= 1.0
