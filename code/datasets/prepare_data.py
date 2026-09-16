"""Stage 1b - data engineering: load, clean and split the raw data.

Input  artifacts : ``data/raw/winequality-red.csv``, ``data/raw/winequality-white.csv``
Output artifacts : ``data/processed/train.csv``, ``data/processed/test.csv``
                   (plus ``data_report.json`` documenting what the stage did)

Operations
----------
1. **Load**      - read both semicolon-separated CSV files and tag the wine type.
2. **Clean**     - normalise column names, drop exact duplicates, impute missing
                   values (median per wine type), remove IQR outliers.
3. **Split**     - stratified train/test split, written to ``data/processed``.

Usage
-----
    python code/datasets/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_logger, load_params, resolve  # noqa: E402

LOGGER = get_logger("stage1.prepare")

TARGET = "is_good_quality"
QUALITY_COLUMN = "quality"
TYPE_COLUMN = "wine_type"


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_raw_data() -> pd.DataFrame:
    """Read the red and white CSV files and concatenate them."""
    params = load_params()["data"]
    raw_dir = resolve(params["raw_dir"])

    frames = []
    for wine_type, key in (("red", "red_file"), ("white", "white_file")):
        path = raw_dir / params[key]
        if not path.exists():
            raise FileNotFoundError(
                f"Raw file {path} is missing - run code/datasets/download_data.py first."
            )
        frame = pd.read_csv(path, sep=";")
        frame[TYPE_COLUMN] = wine_type
        LOGGER.info("Loaded %-5s wines: %d rows x %d columns", wine_type, *frame.shape)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
    LOGGER.info("Combined dataset: %d rows x %d columns", *data.shape)
    return data


# --------------------------------------------------------------------------- #
# 2. Clean
# --------------------------------------------------------------------------- #
def impute_missing_values(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Impute missing numeric values with the median of the same wine type."""
    numeric_columns = data.select_dtypes(include="number").columns
    missing_before = data[numeric_columns].isna().sum()
    missing_map = {col: int(n) for col, n in missing_before.items() if n > 0}

    if missing_map:
        LOGGER.info("Imputing missing values (median per wine type): %s", missing_map)
        data[numeric_columns] = data.groupby(TYPE_COLUMN)[numeric_columns].transform(
            lambda col: col.fillna(col.median())
        )
        # Fallback for columns that are fully missing inside a group.
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
    else:
        LOGGER.info("No missing values found in the numeric columns.")

    data = data.dropna(subset=[QUALITY_COLUMN, TYPE_COLUMN])
    return data, missing_map


def _iqr_mask(data: pd.DataFrame, columns, multiplier: float) -> Tuple[pd.Series, Dict[str, int]]:
    """Boolean mask of the rows outside the IQR fence of any of *columns*."""
    mask = pd.Series(False, index=data.index)
    per_column: Dict[str, int] = {}
    for column in columns:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        column_mask = (data[column] < lower) | (data[column] > upper)
        if column_mask.any():
            per_column[column] = int(column_mask.sum())
        mask |= column_mask
    return mask, per_column


def remove_outliers(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop rows that are IQR outliers in any physicochemical feature.

    The quality column is excluded (it is the label source) and the removal is
    capped by ``data.max_outlier_fraction``: if the fence would delete too much
    of the dataset (e.g. after a bad data drop), it is widened so that only the
    truly extreme rows are removed and the pipeline keeps running.
    """
    params = load_params()["data"]
    multiplier = float(params["iqr_multiplier"])
    max_fraction = float(params["max_outlier_fraction"])

    feature_columns = [
        c for c in data.select_dtypes(include="number").columns if c != QUALITY_COLUMN
    ]
    mask, per_column = _iqr_mask(data, feature_columns, multiplier)

    fraction = int(mask.sum()) / len(data) if len(data) else 0.0
    if fraction > max_fraction:
        LOGGER.warning(
            "Outlier detection flagged %.1f%% of the rows (cap is %.1f%%) - widening the fence "
            "to %.1f x IQR and keeping the rest of the data.",
            fraction * 100,
            max_fraction * 100,
            2 * multiplier,
        )
        mask, per_column = _iqr_mask(data, feature_columns, 2 * multiplier)

    n_outliers = int(mask.sum())
    LOGGER.info(
        "Removing %d outlier rows (%.2f%%); per-column counts: %s",
        n_outliers,
        100 * n_outliers / max(len(data), 1),
        per_column or "none",
    )
    return data.loc[~mask].reset_index(drop=True), per_column


def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Run the full cleaning routine and return the data plus a report."""
    params = load_params()["data"]
    report: Dict[str, object] = {"rows_loaded": int(len(data))}

    duplicates = int(data.duplicated().sum())
    if duplicates:
        data = data.drop_duplicates().reset_index(drop=True)
    LOGGER.info("Removed %d duplicated rows.", duplicates)
    report["duplicates_removed"] = duplicates

    data, missing_map = impute_missing_values(data)
    report["missing_values_imputed"] = missing_map

    rows_before = len(data)
    data, per_column = remove_outliers(data)
    report["outliers_removed"] = rows_before - len(data)
    report["outliers_per_column"] = per_column

    # Binary target: is this wine rated as "good" by the experts?
    threshold = int(params["good_quality_threshold"])
    data[TARGET] = (data[QUALITY_COLUMN] >= threshold).astype(int)
    data = data.drop(columns=[QUALITY_COLUMN])
    report["good_quality_threshold"] = threshold
    report["rows_after_cleaning"] = int(len(data))
    report["class_balance"] = {
        str(k): int(v) for k, v in data[TARGET].value_counts().sort_index().items()
    }
    LOGGER.info("Clean dataset: %d rows; class balance: %s", len(data), report["class_balance"])
    return data, report


# --------------------------------------------------------------------------- #
# 3. Split
# --------------------------------------------------------------------------- #
def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = load_params()["data"]
    train, test = train_test_split(
        data,
        test_size=float(params["test_size"]),
        random_state=int(params["random_state"]),
        stratify=data[TARGET],
    )
    LOGGER.info("Split into %d train rows and %d test rows.", len(train), len(test))
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> int:
    params = load_params()["data"]
    processed_dir = resolve(params["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    data = load_raw_data()
    data, report = clean_data(data)
    train, test = split_data(data)

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    report.update(
        {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "features": [c for c in train.columns if c != TARGET],
            "target": TARGET,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    report_path = processed_dir / "data_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    LOGGER.info("Wrote %s, %s and %s", train_path, test_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
