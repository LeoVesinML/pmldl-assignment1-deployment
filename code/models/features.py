"""Feature engineering for the wine-quality model.

This module is intentionally dependency-light (pandas / numpy / scikit-learn
only) and is **shipped both with the training code and with the API image**.
The trained artifact is an end-to-end ``sklearn`` pipeline whose first step is a
``FunctionTransformer`` bound to :func:`add_engineered_features`, therefore the
module must be importable under the very same name (``features``) wherever the
model is unpickled.  Keeping the feature logic inside the artifact guarantees
that training and serving can never drift apart.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#: Raw numeric columns of the UCI Wine Quality dataset (after normalisation).
RAW_NUMERIC_FEATURES: List[str] = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]

#: Raw categorical columns.
RAW_CATEGORICAL_FEATURES: List[str] = ["wine_type"]

#: Full raw input schema expected by the served pipeline.
RAW_FEATURES: List[str] = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES

#: Domain-driven features derived from the raw ones.
ENGINEERED_FEATURES: List[str] = [
    "total_acidity",
    "acidity_ratio",
    "bound_sulfur_dioxide",
    "free_sulfur_ratio",
    "sugar_alcohol_ratio",
    "sulphates_chlorides_ratio",
    "alcohol_density_index",
]

TARGET = "is_good_quality"

_EPS = 1e-6


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *data* enriched with domain-specific ratio features.

    The function is deliberately defensive: it accepts a ``DataFrame`` (or
    anything convertible to one), never mutates its input and never raises on
    division by zero, so it is safe to call on a single API request as well as
    on the full training set.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=RAW_FEATURES)
    out = data.copy()

    for column in RAW_NUMERIC_FEATURES:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if "wine_type" not in out.columns:
        out["wine_type"] = "red"
    out["wine_type"] = out["wine_type"].astype(str).str.strip().str.lower()

    # Total titratable acidity of the wine.
    out["total_acidity"] = (
        out["fixed_acidity"] + out["volatile_acidity"] + out["citric_acid"]
    )
    # Balance between pleasant (fixed) and vinegary (volatile) acidity.
    out["acidity_ratio"] = out["fixed_acidity"] / (out["volatile_acidity"] + _EPS)
    # SO2 that is already bound and no longer protects the wine.
    out["bound_sulfur_dioxide"] = (
        out["total_sulfur_dioxide"] - out["free_sulfur_dioxide"]
    ).clip(lower=0.0)
    # Share of the SO2 that is still active.
    out["free_sulfur_ratio"] = out["free_sulfur_dioxide"] / (
        out["total_sulfur_dioxide"] + _EPS
    )
    # Sweetness relative to the alcoholic strength.
    out["sugar_alcohol_ratio"] = out["residual_sugar"] / (out["alcohol"] + _EPS)
    # Mineral balance.
    out["sulphates_chlorides_ratio"] = out["sulphates"] / (out["chlorides"] + _EPS)
    # Body proxy: strong but light wines score differently from heavy ones.
    out["alcohol_density_index"] = out["alcohol"] / (out["density"] + _EPS)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out[RAW_FEATURES + ENGINEERED_FEATURES]


def build_preprocessor() -> ColumnTransformer:
    """Median imputation + scaling for numeric columns, one-hot for the wine type."""
    numeric_columns = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, RAW_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
