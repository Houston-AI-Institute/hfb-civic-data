"""Locating fetched sources through ``manifest.csv``."""

import csv
from pathlib import Path


def raw_path(raw: Path, key: str) -> Path:
    """Path of a fetched source, looked up in the manifest (path is relative to raw)."""
    with open(Path(raw) / "manifest.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["key"] == key:
                return Path(raw) / row["path"]
    raise KeyError(f"{key!r} not in {Path(raw) / 'manifest.csv'}")
