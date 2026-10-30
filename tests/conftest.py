import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def raw():
    """Raw data directory (HFB_RAW, default data/raw); skipped without a manifest."""
    path = Path(os.environ.get("HFB_RAW", ROOT / "data" / "raw"))
    if not (path / "manifest.csv").exists():
        pytest.skip(f"no manifest.csv in {path}; run `python -m hfb.fetch` first")
    return path


@pytest.fixture(scope="session")
def table(raw):
    """The built ZCTA table, cached per session."""
    from hfb import pipeline  # Task 5; imported lazily so collection never fails

    return pipeline.build_table(raw)


@pytest.fixture(scope="session")
def weights(blocks):
    """VTD -> ZCTA weights from the real block table, built once per session."""
    from hfb import crosswalk

    return crosswalk.vtd_weights(blocks)


@pytest.fixture(scope="session")
def blocks(raw):
    """2020 block table for the region, built once per session."""
    from hfb import crosswalk

    return crosswalk.build_blocks(raw, write=False)
