"""Prison-adjusted VTD -> ZCTA crosswalk.

Blocks are the bridge: each 2020 block's internal point is placed in a 2024 VTD
(spatial ``within`` join), its ZCTA comes from TIGER2020 FACES, population from TLC
``Blocks_Pop`` and group-quarters counts from PL 94-171 segment 3. VTD votes are then
allocated to ZCTAs by each block group's share of adjusted voting-age population
(``vap_adj = clip(vap - corr_adult, 0)``), falling back to land area when a VTD has no
adjusted population (e.g. a prison-only VTD).

Block-level data is an interim product (``data/interim/blocks.parquet``); it never goes
to ``out/``.
"""

import csv
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

from .config import COUNTIES

log = logging.getLogger(__name__)

NONE = "NONE"
BLOCK_COLUMNS = [
    "GEOID20", "VTDKEY", "ZCTA", "total", "vap", "gq", "corr_adult", "college", "ALAND20",
]


def _path(raw: Path, key: str) -> Path:
    """Path of a fetched source, looked up in the manifest (path is relative to raw)."""
    with open(Path(raw) / "manifest.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["key"] == key:
                return Path(raw) / row["path"]
    raise KeyError(f"{key!r} not in {Path(raw) / 'manifest.csv'}")


def zcta_from_faces(joined: pd.Series) -> pd.Series:
    """Blank or missing ZCTA -> 'NONE'; a block spanning several ZCTAs is an error."""
    s = joined.fillna("").astype(str)
    multi = s.str.contains("|", regex=False)
    if multi.any():
        raise ValueError(
            f"{int(multi.sum())} blocks fall in more than one ZCTA in FACES "
            f"(e.g. {s[multi].iloc[0]!r}); the block-to-ZCTA assignment assumes one")
    return s.where(s != "", NONE)


def add_vap_adj(blocks: pd.DataFrame) -> pd.DataFrame:
    """vap_adj = voting-age population less adults in correctional facilities."""
    out = blocks.copy()
    out["vap_adj"] = (out["vap"] - out["corr_adult"]).clip(lower=0)
    return out


def _read_pop(raw: Path) -> pd.DataFrame:
    pop = pd.read_csv(
        _path(raw, "tlc_blocks_pop") / "Blocks_Pop.txt",
        dtype={"FIPS": str, "CTBKEY": str}, usecols=["FIPS", "CTBKEY", "total", "vap"])
    pop = pop[pop.FIPS.isin(COUNTIES)].copy()
    pop["GEOID20"] = "48" + pop.CTBKEY
    return pop[["GEOID20", "total", "vap"]]


def _read_gq(raw: Path) -> pd.DataFrame:
    """PL segment 3 by block: gq (P0050001), corr_adult (P0050003), college (P0050008)."""
    pl = _path(raw, "pl")
    geo = pd.read_csv(
        pl / "txgeo2020.pl", sep="|", header=None, usecols=[2, 7, 9, 14], dtype=str,
        encoding="latin-1", quoting=csv.QUOTE_NONE)
    geo.columns = ["sumlev", "logrecno", "geoid", "county"]
    geo = geo[(geo.sumlev == "750") & geo.county.isin(COUNTIES)]
    seg = pd.read_csv(
        pl / "tx000032020.pl", sep="|", header=None, usecols=[4, 5, 7, 12],
        dtype={4: str}, encoding="latin-1", quoting=csv.QUOTE_NONE)
    seg.columns = ["logrecno", "gq", "corr_adult", "college"]
    g = geo.merge(seg, on="logrecno", how="inner")
    g = g.rename(columns={"geoid": "GEOID20"})[["GEOID20", "gq", "corr_adult", "college"]]
    return g.astype({"gq": int, "corr_adult": int, "college": int})


def build_blocks(raw: Path, write: bool = True) -> pd.DataFrame:
    """Block table for the region; written to ``<raw>/../interim/blocks.parquet``."""
    raw = Path(raw)
    pop = _read_pop(raw)
    gq = _read_gq(raw)
    vtds = gpd.read_file(_path(raw, "tlc_vtds24") / "VTDs_24PG.shp", engine="pyogrio")
    parts = []
    for c in COUNTIES:
        blk = pyogrio.read_dataframe(
            _path(raw, f"blk_{c}") / f"tl_2020_48{c}_tabblock20.shp", read_geometry=False,
            columns=["GEOID20", "ALAND20", "INTPTLON20", "INTPTLAT20"])
        fc = pyogrio.read_dataframe(
            _path(raw, f"faces_{c}") / f"tl_2020_48{c}_faces.shp", read_geometry=False,
            columns=["STATEFP20", "COUNTYFP20", "TRACTCE20", "BLOCKCE20", "ZCTA5CE20"])
        fc["GEOID20"] = fc.STATEFP20 + fc.COUNTYFP20 + fc.TRACTCE20 + fc.BLOCKCE20
        z = fc[fc.ZCTA5CE20.fillna("") != ""]
        fz = z.groupby("GEOID20").ZCTA5CE20.agg(lambda s: "|".join(sorted(set(s))))
        pts = gpd.GeoDataFrame(
            blk[["GEOID20", "ALAND20"]],
            geometry=gpd.points_from_xy(blk.INTPTLON20.astype(float), blk.INTPTLAT20.astype(float)),
            crs="EPSG:4269").to_crs(vtds.crs)
        v = vtds[vtds.CNTY == int(c)][["VTDKEY", "geometry"]]
        j = gpd.sjoin(pts, v, how="left", predicate="within").drop(columns="index_right")
        j = j[~j.GEOID20.duplicated()]  # a point on a shared edge can match two VTDs
        j["ZCTA"] = zcta_from_faces(j.GEOID20.map(fz))
        parts.append(pd.DataFrame(j.drop(columns="geometry")))
    b = pd.concat(parts, ignore_index=True)
    b = b.merge(pop, on="GEOID20", how="left").merge(gq, on="GEOID20", how="left")
    for col in ["total", "vap", "gq", "corr_adult", "college"]:
        b[col] = b[col].fillna(0).astype(int)
    missing = b.VTDKEY.isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} blocks (population {int(b.loc[missing, 'total'].sum())}) "
            "fall in no 2024 VTD; the crosswalk assumes none")
    b["VTDKEY"] = b.VTDKEY.astype(int)
    b = b[BLOCK_COLUMNS]
    if write:
        interim = raw.parent / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        b.to_parquet(interim / "blocks.parquet", index=False)
    return b


def vtd_weights(blocks: pd.DataFrame, col: str = "vap_adj") -> pd.DataFrame:
    """Share of each VTD's weight (``col``) in each ZCTA, with a land-area fallback.

    A VTD with no weight falls back to land area; one with neither gets share 0 (and is
    logged). ``NONE`` is kept as a ZCTA so shares sum to one; callers decide what to do
    with it.
    """
    b = add_vap_adj(blocks) if col == "vap_adj" and "vap_adj" not in blocks else blocks
    g = b.groupby(["VTDKEY", "ZCTA"])[[col, "ALAND20"]].sum().reset_index()
    tot = g.groupby("VTDKEY")[col].transform("sum")
    land = g.groupby("VTDKEY").ALAND20.transform("sum")
    share = np.zeros(len(g))
    use_w, use_l = (tot > 0).to_numpy(), ((tot <= 0) & (land > 0)).to_numpy()
    share[use_w] = (g[col] / tot.where(tot > 0, 1))[use_w]
    share[use_l] = (g.ALAND20 / land.where(land > 0, 1))[use_l]
    g["share"] = share
    dead = sorted(set(g.VTDKEY[~(use_w | use_l)]))
    if dead:
        log.warning("%d VTDs have no %s and no land; share set to 0: %s", len(dead), col, dead[:20])
    g["VTDKEY"] = g.VTDKEY.astype(int)
    return g[["VTDKEY", "ZCTA", "share"]]


def zcta_pop(blocks: pd.DataFrame) -> pd.DataFrame:
    """ZCTA -> total, vap, gq, corr_adult, college, vap_adj (NONE excluded)."""
    b = add_vap_adj(blocks)
    b = b[b.ZCTA != NONE]
    return b.groupby("ZCTA")[["total", "vap", "gq", "corr_adult", "college", "vap_adj"]].sum()
