"""Stage 1 tests: cleaning and splitting behave as documented."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prepare_data import TARGET, clean_data, remove_outliers, split_data


def test_clean_data_removes_duplicates(raw_like_frame: pd.DataFrame) -> None:
    duplicated = pd.concat([raw_like_frame, raw_like_frame.head(10)], ignore_index=True)
    cleaned, report = clean_data(duplicated)
    assert report["duplicates_removed"] >= 10
    assert not cleaned.duplicated().any()


def test_clean_data_imputes_missing_values(raw_like_frame: pd.DataFrame) -> None:
    frame = raw_like_frame.copy()
    frame.loc[0:4, "alcohol"] = np.nan
    cleaned, report = clean_data(frame)
    assert report["missing_values_imputed"].get("alcohol", 0) == 5
    assert not cleaned.isna().any().any()


def test_clean_data_creates_binary_target(raw_like_frame: pd.DataFrame) -> None:
    cleaned, report = clean_data(raw_like_frame)
    assert TARGET in cleaned.columns
    assert "quality" not in cleaned.columns
    assert set(cleaned[TARGET].unique()) <= {0, 1}
    assert report["rows_after_cleaning"] == len(cleaned)


def test_remove_outliers_drops_extreme_rows(raw_like_frame: pd.DataFrame) -> None:
    frame = raw_like_frame.copy()
    frame.loc[0, "residual_sugar"] = 1000.0  # obvious outlier
    cleaned, per_column = remove_outliers(frame)
    assert len(cleaned) < len(frame)
    assert "residual_sugar" in per_column


def test_split_is_stratified_and_complete(raw_like_frame: pd.DataFrame) -> None:
    cleaned, _ = clean_data(raw_like_frame)
    train, test = split_data(cleaned)
    assert len(train) + len(test) == len(cleaned)
    assert abs(len(test) / len(cleaned) - 0.2) < 0.02
    train_ratio = train[TARGET].mean()
    test_ratio = test[TARGET].mean()
    assert abs(train_ratio - test_ratio) < 0.1
