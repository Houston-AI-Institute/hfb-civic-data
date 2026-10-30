"""The ZCTA frame: which ZCTAs touch the region, and how much of each is inside it.

``land_share`` is the fraction of a ZCTA's land in the 10 region counties. ``inreg_share20``
is the fraction of a ZCTA's 2020 population living inside them, estimated by spreading each
out-of-region tract's population over the ZCTAs it overlaps by land area.
"""

import logging
from pathlib import Path

import pandas as pd

from .config import COUNTIES
from .paths import raw_path

log = logging.getLogger(__name__)

REGION_GEOIDS = {"48" + c for c in COUNTIES}


def read_rel(path) -> pd.DataFrame:
    """Census 2020 ZCTA relationship file (pipe-delimited, BOM) as strings.

    Rows with a blank ZCTA are county or tract land outside any ZCTA and are dropped.
    """
    r = pd.read_csv(Path(path), sep="|", dtype=str, encoding="utf-8-sig", keep_default_na=False)
    return r[r.GEOID_ZCTA5_20 != ""].reset_index(drop=True)


def read_blocks_pop(raw: Path) -> pd.DataFrame:
    """Statewide TLC Blocks_Pop: FIPS (3-digit county), TRT (6-digit tract), total."""
    bp = pd.read_csv(
        raw_path(raw, "tlc_blocks_pop") / "Blocks_Pop.txt",
        usecols=["FIPS", "TRT", "total"], dtype={"FIPS": str, "TRT": str})
    return bp


def region_zctas(rel_county: pd.DataFrame) -> pd.DataFrame:
    """ZCTA -> land_share (region land / ZCTA land, > 0) and county_main.

    ``county_main`` is the 5-digit GEOID of the region county holding most of the ZCTA.
    """
    r = rel_county[["GEOID_ZCTA5_20", "AREALAND_ZCTA5_20", "GEOID_COUNTY_20", "AREALAND_PART"]]
    r = r.assign(
        AREALAND_ZCTA5_20=pd.to_numeric(r.AREALAND_ZCTA5_20),
        AREALAND_PART=pd.to_numeric(r.AREALAND_PART))
    zland = r.groupby("GEOID_ZCTA5_20").AREALAND_ZCTA5_20.first()
    ins = r[r.GEOID_COUNTY_20.isin(REGION_GEOIDS)]
    part = ins.groupby("GEOID_ZCTA5_20").AREALAND_PART.sum()
    zland = zland.where(zland > 0)  # zero ZCTA land -> NaN, not inf
    out = pd.DataFrame({"land_share": part / zland.reindex(part.index)})
    out = out[out.land_share > 0]
    ins = ins[ins.AREALAND_PART > 0]
    main = ins.loc[ins.groupby("GEOID_ZCTA5_20").AREALAND_PART.idxmax()]
    out["county_main"] = main.set_index("GEOID_ZCTA5_20").GEOID_COUNTY_20.reindex(out.index)
    out.index.name = "ZCTA"
    return out.sort_index()


def out_of_region_2020(rel_tract: pd.DataFrame, blocks_pop: pd.DataFrame, zctas=None):
    """(Series ZCTA -> 2020 people in out-of-region Texas tract pieces, stats).

    Each piece counts ``tract_total * AREALAND_PART / AREALAND_TRACT_20``. Non-Texas
    tracts are ignored; a tract with zero land, or one missing from ``blocks_pop``,
    contributes 0. ``stats`` counts the distinct non-Texas tracts ignored and the
    out-of-region Texas tracts missing from ``blocks_pop``. ``zctas`` limits the pieces to
    those ZCTAs (the relationship file is national); the stats then cover only them.
    """
    if zctas is not None:
        rel_tract = rel_tract[rel_tract.GEOID_ZCTA5_20.isin(set(zctas))]
    tp = blocks_pop.assign(
        tract="48" + blocks_pop.FIPS.astype(str).str.zfill(3)
        + blocks_pop.TRT.astype(str).str.zfill(6)).groupby("tract").total.sum()
    t = rel_tract[["GEOID_ZCTA5_20", "GEOID_TRACT_20", "AREALAND_TRACT_20", "AREALAND_PART"]]
    non_tx = ~t.GEOID_TRACT_20.str.startswith("48")
    out_tx = t[~non_tx & ~t.GEOID_TRACT_20.str[:5].isin(REGION_GEOIDS)].copy()
    land = pd.to_numeric(out_tx.AREALAND_TRACT_20).astype(float)
    part = pd.to_numeric(out_tx.AREALAND_PART).astype(float)
    total = out_tx.GEOID_TRACT_20.map(tp).astype(float).fillna(0)
    out_tx["people"] = (total * part / land.where(land > 0)).fillna(0)
    stats = {
        "non_texas_tracts": int(t.GEOID_TRACT_20[non_tx].nunique()),
        "missing_tracts": int((~pd.Index(out_tx.GEOID_TRACT_20.unique()).isin(tp.index)).sum()),
    }
    if stats["non_texas_tracts"] or stats["missing_tracts"]:
        log.warning("out_of_region_2020: %s", stats)
    return out_tx.groupby("GEOID_ZCTA5_20").people.sum(), stats


def inreg_share20(rel_tract: pd.DataFrame, blocks_pop: pd.DataFrame, zpop: pd.Series) -> pd.Series:
    """total_inregion / (total_inregion + out_of_region_2020), indexed like ``zpop``.

    NaN where the ZCTA has zero 2020 population both in and out of the region (77341).
    """
    out, _ = out_of_region_2020(rel_tract, blocks_pop, zpop.index)
    outside = out.reindex(zpop.index).fillna(0.0)
    return zpop / (zpop + outside)
