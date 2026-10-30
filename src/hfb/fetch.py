"""Fetch every source in config.SOURCES into raw_dir and write manifest.csv.

Usage: python -m hfb.fetch RAW_DIR [--seed SEED_ROOT]

With a seed root, every key is copied (or, for zips, extracted) from the seed with
no network access (manifest ``source`` = ``seed``); a missing seed file raises
FileNotFoundError. Without a seed root every key is downloaded (``download``). Downloaded zips are cached at ``raw_dir/_zips/<key>.zip``.
Manifest ``path`` is relative to raw_dir.
"""

import argparse
import csv
import hashlib
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .config import Source

COLUMNS = ["key", "url", "path", "source", "sha256", "bytes", "fetched_utc"]
TIMEOUT_S = 120
ATTEMPTS = 3


def _download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hfb-fetch/0.1"})
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp, open(part, "wb") as f:
                shutil.copyfileobj(resp, f)
            part.replace(out)
            return
        except Exception:
            if attempt == ATTEMPTS:
                raise
            time.sleep(2 * attempt)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_all(
    raw_dir: Path,
    seed_root: Path | None,
    sources: dict[str, Source] | None = None,
    seed: dict[str, str] | None = None,
) -> Path:
    """Fetch all sources; return the path of raw_dir/manifest.csv."""
    raw_dir = Path(raw_dir)
    sources = config.SOURCES if sources is None else sources
    seed = config.SEED if seed is None else seed
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, src in sources.items():
        if seed_root is not None:
            # With a seed, never touch the network: a missing seed file is an error.
            seed_file = Path(seed_root) / seed[key] if key in seed else None
            if seed_file is None or not seed_file.is_file():
                expected = seed_file if seed_file is not None else f"<no SEED entry under {seed_root}>"
                raise FileNotFoundError(f"seed file for {key!r} not found: {expected}")
            origin, payload = "seed", seed_file
        else:
            origin = "download"
            payload = raw_dir / "_zips" / f"{key}.zip" if src.unzip else raw_dir / src.dest
            _download(src.url, payload)
        dest = raw_dir / src.dest
        if src.unzip:
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(payload) as z:
                z.extractall(dest)
        elif origin == "seed":
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(payload, dest)
        rows.append({
            "key": key,
            "url": src.url,
            "path": src.dest,
            "source": origin,
            "sha256": _sha256(payload),
            "bytes": payload.stat().st_size,
            "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        print(f"{key}: {origin}")
    manifest = raw_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("--seed", type=Path, default=None, help="seed root to copy from (no network)")
    args = ap.parse_args()
    print(f"wrote {fetch_all(args.raw_dir, args.seed)}")


if __name__ == "__main__":
    main()
