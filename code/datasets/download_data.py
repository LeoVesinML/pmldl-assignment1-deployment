"""Stage 1a - data acquisition.

Downloads the UCI *Wine Quality* dataset (red + white) into ``data/raw``.
The download is idempotent: if the raw files are already present and
non-empty, the network call is skipped (``--force`` overrides this).

Usage
-----
    python code/datasets/download_data.py [--force]
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_logger, load_params, resolve  # noqa: E402

LOGGER = get_logger("stage1.download")
EXPECTED_FILES = ("red_file", "white_file")


def _raw_paths() -> dict:
    params = load_params()["data"]
    raw_dir = resolve(params["raw_dir"])
    return {key: raw_dir / params[key] for key in EXPECTED_FILES}


def already_downloaded() -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in _raw_paths().values())


def download(force: bool = False) -> None:
    params = load_params()["data"]
    raw_dir = resolve(params["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if already_downloaded() and not force:
        for name, path in _raw_paths().items():
            LOGGER.info("Raw file already present: %s (%d bytes)", path, path.stat().st_size)
        LOGGER.info("Skipping download (use --force to re-download).")
        return

    url = params["source_url"]
    LOGGER.info("Downloading raw data from %s", url)
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed HTTPS URL
        payload = response.read()
    LOGGER.info("Downloaded %.1f KiB", len(payload) / 1024)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = {Path(name).name: name for name in archive.namelist()}
        for key, target in _raw_paths().items():
            member = members.get(target.name)
            if member is None:
                raise FileNotFoundError(
                    f"{target.name} not found in the archive (contains: {sorted(members)})"
                )
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            LOGGER.info("Saved %s (%d bytes)", target, target.stat().st_size)

    LOGGER.info("Raw data is ready in %s", raw_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the raw Wine Quality dataset.")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()
    try:
        download(force=args.force)
    except Exception as exc:  # pragma: no cover - surfaced in Airflow logs
        if already_downloaded():
            LOGGER.warning("Download failed (%s) but raw files exist - continuing.", exc)
            return 0
        LOGGER.error("Download failed and no raw data is available: %s", exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
