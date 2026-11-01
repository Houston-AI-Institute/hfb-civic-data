import csv
import hashlib
import zipfile

import pytest

from hfb import copy as copy_mod
from hfb import fetch
from hfb.config import ACS_TABLES, COUNTIES, SEED, SOURCES, Source

COLUMNS = ["key", "url", "path", "source", "sha256", "bytes", "fetched_utc"]


def read_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_manifest_written(tmp_path):
    payload = tmp_path / "payload.csv"
    payload.write_text("a,b\n1,2\n")
    sources = {"demo": Source(payload.as_uri(), "demo.csv", False)}
    raw = tmp_path / "raw"

    manifest = fetch.fetch_all(raw, None, sources=sources, seed={})

    assert manifest == raw / "manifest.csv"
    rows = read_manifest(manifest)
    assert list(rows[0]) == COLUMNS
    assert len(rows) == 1
    row = rows[0]
    assert row["key"] == "demo"
    assert row["source"] == "download"
    assert len(row["sha256"]) == 64
    assert row["sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert int(row["bytes"]) > 0
    assert (raw / row["path"]).read_text() == "a,b\n1,2\n"


def test_seed_used(tmp_path):
    seed_root = tmp_path / "seed"
    (seed_root / "x").mkdir(parents=True)
    (seed_root / "x" / "f.csv").write_text("seeded\n")
    zpath = seed_root / "x" / "z.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("inner.txt", "zipped")
    url = "https://example.invalid/never-fetched"  # would fail if touched
    sources = {
        "plain": Source(url + ".csv", "plain.csv", False),
        "zipped": Source(url + ".zip", "zipped", True),
    }
    seed = {"plain": "x/f.csv", "zipped": "x/z.zip"}
    raw = tmp_path / "raw"

    manifest = fetch.fetch_all(raw, seed_root, sources=sources, seed=seed)

    rows = {r["key"]: r for r in read_manifest(manifest)}
    assert rows["plain"]["source"] == "seed"
    assert rows["plain"]["url"] == url + ".csv"
    assert (raw / rows["plain"]["path"]).read_text() == "seeded\n"
    assert rows["zipped"]["source"] == "seed"
    assert rows["zipped"]["url"] == url + ".zip"
    assert (raw / rows["zipped"]["path"]).is_dir()
    assert (raw / rows["zipped"]["path"] / "inner.txt").read_text() == "zipped"
    assert rows["zipped"]["sha256"] == hashlib.sha256(zpath.read_bytes()).hexdigest()
    assert (seed_root / "x" / "f.csv").read_text() == "seeded\n"  # seed untouched


def test_source_config_complete():
    assert len(SOURCES) == 36
    assert set(SEED) == set(SOURCES)
    assert sum(k.startswith("blk_") for k in SOURCES) == 10
    assert sum(k.startswith("faces_") for k in SOURCES) == 10
    assert sum(k.startswith("acs_") for k in SOURCES) == 6


def test_copy_constants():
    assert copy_mod.MISUSE == (
        "Not for decisions about food distribution, eligibility, pantry siting or any individual."
    )
    assert copy_mod.USE_LIMIT == (
        "This table is for choosing where to listen first. "
        "It is not for planning voter registration or turnout drives."
    )
    assert copy_mod.REQUIRED_LINES == (
        "Missing data is not missing civic life.",
        "A low score is not low interest.",
    )
    assert copy_mod.RANK_DIRECTION == (
        "P1_rank 100 = highest measured participation among ranked ZCTAs."
    )
    assert copy_mod.P1_FOOTNOTE == (
        "P1 = average of 2022 and 2024 general-election turnout divided by citizen "
        "voting-age population, less adults in correctional facilities. "
        "A reference measure, not a score of a community."
    )
    assert copy_mod.MAP_FOOTER == (
        "Not for decisions about food distribution, eligibility, pantry siting or any individual."
        " For the food bank's hand-off only; use county views in recorded talks."
    )
    assert copy_mod.CREDITS == (
        "Texas Legislative Council, Capitol Data Portal "
        "(CC BY, http://www.opendefinition.org/licenses/cc-by). "
        "Allocated to ZCTA by HAI; not a TLC product.",
        "U.S. Census Bureau: ACS 2020-2024 5-year, 2020 PL 94-171, "
        "TIGER/Line and cartographic boundaries.",
        "IRS Exempt Organizations Business Master File, Texas extract.",
        "Not a TLC, Census Bureau or IRS product.",
    )


def test_seed_missing_raises(tmp_path):
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    sources = {"gone": Source("https://example.invalid/gone.csv", "gone.csv", False)}
    raw = tmp_path / "raw"

    with pytest.raises(FileNotFoundError, match=r"gone.*missing/gone\.csv"):
        fetch.fetch_all(raw, seed_root, sources=sources, seed={"gone": "missing/gone.csv"})
    # unmapped key with a seed_root also never falls back to the network
    with pytest.raises(FileNotFoundError, match="gone"):
        fetch.fetch_all(raw, seed_root, sources=sources, seed={})


def test_source_urls_match_brief():
    """Pin every URL against the brief's table (typed independently of config.py)."""
    expected = {
        "tlc_vd24": "https://ted.capitol.texas.gov/api/Offices/510/1428/vtd",
        "tlc_vd22": "https://ted.capitol.texas.gov/api/Offices/497/1428/vtd",
        "tlc_vtds24": "https://data.capitol.texas.gov/dataset/4d8298d0-d176-4c19-b174-42837027b73e/resource/906f47e4-4e39-4156-b1bd-4969be0b2780/download/vtds_24pg.zip",
        "tlc_vtdpop24": "https://data.capitol.texas.gov/dataset/4d8298d0-d176-4c19-b174-42837027b73e/resource/bf9b54a8-090c-41d0-8f00-e263fc1789c5/download/vtds_24pg_pop.zip",
        "tlc_blocks_pop": "https://data.capitol.texas.gov/dataset/2b59f5ce-5fa4-4040-a550-caffbe8986c4/resource/33ed77c5-951b-4a88-9de7-9c90c4ba50db/download/blocks_pop.zip",
        "pl": "https://www2.census.gov/programs-surveys/decennial/2020/data/01-Redistricting_File--PL_94-171/Texas/tx2020.pl.zip",
        "zcta_county": "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_county20_natl.txt",
        "zcta_tract": "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_tract20_natl.txt",
        "irs_eo_tx": "https://www.irs.gov/pub/irs-soi/eo_tx.csv",
        "cb_zcta": "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip",
    }
    assert COUNTIES == ["201", "157", "039", "015", "473", "471", "071", "455", "407", "291"]
    for c in COUNTIES:
        expected[f"blk_{c}"] = (
            f"https://www2.census.gov/geo/tiger/TIGER2020PL/STATE/48_TEXAS/48{c}/"
            f"tl_2020_48{c}_tabblock20.zip"
        )
        expected[f"faces_{c}"] = (
            f"https://www2.census.gov/geo/tiger/TIGER2020/FACES/tl_2020_48{c}_faces.zip"
        )
    assert ACS_TABLES == ["b05003", "b07003", "b25003", "b15003", "c16002", "b28002"]
    for t in ACS_TABLES:
        expected[f"acs_{t}"] = (
            "https://www2.census.gov/programs-surveys/acs/summary_file/2024/"
            f"table-based-SF/data/5YRData/acsdt5y2024-{t}.dat"
        )
    assert len(expected) == 36
    assert {k: v.url for k, v in SOURCES.items()} == expected
