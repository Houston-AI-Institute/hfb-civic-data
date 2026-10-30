"""Source registry and seed map.

Raw layout under ``raw_dir`` (stable; later tasks resolve paths through the
``manifest.csv`` ``key`` -> ``path`` lookup, with ``path`` relative to ``raw_dir``):

* single-file sources are saved at ``Source.dest`` (e.g. ``tlc_vd24.csv``);
* zip sources are extracted into the folder ``Source.dest`` (= ``raw_dir/<key>/``).
"""

from typing import NamedTuple

# Harris, Fort Bend, Brazoria, Austin, Waller, Walker, Chambers, Trinity, San Jacinto, Liberty
COUNTIES: list[str] = [
    "201", "157", "039", "015", "473", "471", "071", "455", "407", "291",
]

# Short county names, keyed like COUNTIES; checked against the 2020 PL geography file in the tests.
COUNTY_NAMES: dict[str, str] = {
    "201": "Harris", "157": "Fort Bend", "039": "Brazoria", "015": "Austin", "473": "Waller",
    "471": "Walker", "071": "Chambers", "455": "Trinity", "407": "San Jacinto", "291": "Liberty",
}

ACS_TABLES = ["b05003", "b07003", "b25003", "b15003", "c16002", "b28002"]

_TLC_API = "https://ted.capitol.texas.gov/api/Offices/{office}/1428/vtd"
_TLC_DATA = "https://data.capitol.texas.gov/dataset"
_TIGER = "https://www2.census.gov/geo/tiger"
_REL = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520"
_ACS = "https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/data/5YRData"


class Source(NamedTuple):
    url: str
    dest: str  # path relative to raw_dir: a file, or a folder when unzip is True
    unzip: bool


SOURCES: dict[str, Source] = {
    # The TLC API is GET only (HEAD returns 405); the response is saved as-is.
    "tlc_vd24": Source(_TLC_API.format(office=510), "tlc_vd24.csv", False),
    "tlc_vd22": Source(_TLC_API.format(office=497), "tlc_vd22.csv", False),
    "tlc_vtds24": Source(
        f"{_TLC_DATA}/4d8298d0-d176-4c19-b174-42837027b73e/resource/"
        "906f47e4-4e39-4156-b1bd-4969be0b2780/download/vtds_24pg.zip",
        "tlc_vtds24", True),
    "tlc_vtdpop24": Source(
        f"{_TLC_DATA}/4d8298d0-d176-4c19-b174-42837027b73e/resource/"
        "bf9b54a8-090c-41d0-8f00-e263fc1789c5/download/vtds_24pg_pop.zip",
        "tlc_vtdpop24", True),
    "tlc_blocks_pop": Source(  # statewide
        f"{_TLC_DATA}/2b59f5ce-5fa4-4040-a550-caffbe8986c4/resource/"
        "33ed77c5-951b-4a88-9de7-9c90c4ba50db/download/blocks_pop.zip",
        "tlc_blocks_pop", True),
    "pl": Source(
        "https://www2.census.gov/programs-surveys/decennial/2020/data/"
        "01-Redistricting_File--PL_94-171/Texas/tx2020.pl.zip",
        "pl", True),
    "zcta_county": Source(f"{_REL}/tab20_zcta520_county20_natl.txt", "zcta_county.txt", False),
    "zcta_tract": Source(f"{_REL}/tab20_zcta520_tract20_natl.txt", "zcta_tract.txt", False),
    "irs_eo_tx": Source("https://www.irs.gov/pub/irs-soi/eo_tx.csv", "irs_eo_tx.csv", False),
    "cb_zcta": Source(f"{_TIGER}/GENZ2020/shp/cb_2020_us_zcta520_500k.zip", "cb_zcta", True),
}
for _c in COUNTIES:
    SOURCES[f"blk_{_c}"] = Source(
        f"{_TIGER}/TIGER2020PL/STATE/48_TEXAS/48{_c}/tl_2020_48{_c}_tabblock20.zip",
        f"blk_{_c}", True)
    SOURCES[f"faces_{_c}"] = Source(
        f"{_TIGER}/TIGER2020/FACES/tl_2020_48{_c}_faces.zip", f"faces_{_c}", True)
for _t in ACS_TABLES:
    SOURCES[f"acs_{_t}"] = Source(f"{_ACS}/acsdt5y2024-{_t}.dat", f"acs_{_t}.dat", False)

# key -> path relative to seed_root (data/raw/seed); one file per source key.
SEED: dict[str, str] = {
    "tlc_vd24": "geography/dl_tlc_vd24/vd24.csv",
    "tlc_vd22": "geography/dl_tlc_vd22/vd22.csv",
    "tlc_vtds24": "geography/dl_tlc_vtds/vtds_24pg.zip",
    "tlc_vtdpop24": "geography/dl_tlc_vtdpop/vtds_24pg_pop.zip",
    "tlc_blocks_pop": "geography/dl_tlc_blkpop/blocks_pop.zip",
    "pl": "geography/dl_pl/pl.zip",
    "zcta_county": "geography/dl_zcta_county/rel.txt",
    "zcta_tract": "geography/dl_zcta_tract/rel.txt",
    "irs_eo_tx": "gaps/irs/eo_tx.csv",
    "cb_zcta": "cb_zcta/cb_2020_us_zcta520_500k.zip",
}
for _c in COUNTIES:
    SEED[f"blk_{_c}"] = f"geography/dl_blk_{_c}/tabblock20.zip"
    SEED[f"faces_{_c}"] = f"geography/dl_faces_{_c}/faces.zip"
for _t in ACS_TABLES:
    SEED[f"acs_{_t}"] = f"gaps/acs_{_t}/{_t}.dat"

# ZCTAs whose corr_adult may be immigration detention rather than prison; flagged
# CAUTION FACILITY_CHECK until the facility type is checked (77032 = Houston IAH area).
FACILITY_CHECK: set[str] = {"77032"}
