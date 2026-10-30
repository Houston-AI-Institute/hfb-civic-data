import csv
import hashlib
import zipfile

from hfb import copy as copy_mod
from hfb import fetch
from hfb.config import SEED, SOURCES, Source

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
        "This table does not choose where to run voter registration or GOTV; "
        "that is the food bank's decision with its counsel. "
        "Until then it supports a listening shortlist only."
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
