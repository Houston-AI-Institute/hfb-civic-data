"""One-command build: fetch every source, then assemble the table, QA report and map.

Usage: python -m hfb.build RAW_DIR [--seed SEED_ROOT] [--out DIR] [--qa DIR]

With --seed, sources are copied from the seed root with no network access. Without
it, every source is downloaded. assemble writes the CSV, XLSX, QA report, PNG map and
GeoJSON. Exits non-zero if a fetch fails or any QA check fails.
"""

import argparse
import logging
import sys
from pathlib import Path

from . import assemble, fetch


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hfb.build", description=__doc__)
    ap.add_argument("raw", type=Path, help="raw data directory (written by fetch)")
    ap.add_argument("--seed", type=Path, default=None, help="seed root to copy from (no network)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--qa", type=Path, default=Path("qa"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    fetch.fetch_all(a.raw, a.seed)
    if not assemble.main(a.raw, out_dir=a.out, qa_dir=a.qa):
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
