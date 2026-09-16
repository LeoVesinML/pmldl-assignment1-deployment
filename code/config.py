"""Shared configuration helpers for every stage of the pipeline.

The module resolves the project root, loads ``params.yaml`` and exposes a
consistently configured logger, so that the data / model / deployment scripts
stay free of boilerplate.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

#: Repository root (``code/config.py`` -> ``code`` -> repository root).
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

#: ``code/models`` is added to ``sys.path`` so that ``features`` is importable
#: under the *same* module name both during training and inside the API image
#: (this keeps the pickled sklearn pipeline loadable everywhere).
MODELS_CODE_DIR = PROJECT_ROOT / "code" / "models"
if str(MODELS_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_CODE_DIR))

PARAMS_FILE = PROJECT_ROOT / "params.yaml"


@lru_cache(maxsize=1)
def load_params() -> Dict[str, Any]:
    """Load ``params.yaml`` once and cache it."""
    with PARAMS_FILE.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    if not isinstance(params, dict):  # pragma: no cover - defensive
        raise ValueError(f"{PARAMS_FILE} must contain a YAML mapping")
    return params


def resolve(path: str | Path) -> Path:
    """Resolve *path* relative to the project root (absolute paths pass through)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints to stdout (captured by Airflow task logs)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger
