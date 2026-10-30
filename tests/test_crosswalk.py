import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hfb import crosswalk
from hfb.config import COUNTIES

GOLDEN = Path(__file__).parent / "golden"
TAG = "_".join(COUNTIES)


def blocks_df(rows):
    """Hand-built block table: (VTDKEY, ZCTA, vap, corr_adult, ALAND20)."""
    df = pd.DataFrame(rows, columns=["VTDKEY", "ZCTA", "vap", "corr_adult", "ALAND20"])
    df["VTDKEY"] = df.VTDKEY.astype(int)
    df["total"] = df.vap
    df["gq"] = df.corr_adult
    df["college"] = 0
    df["GEOID20"] = [f"48000{i:010d}" for i in range(len(df))]
    return df


# ---------- unit tests on hand-built frames ----------


def test_shares_sum_to_one():
    b = blocks_df([
        (1, "77001", 100, 0, 10), (1, "77002", 300, 0, 10), (1, "NONE", 100, 0, 5),
        (2, "77002", 50, 0, 7),
        (3, "77003", 0, 0, 0),  # no weight and no land: excluded from the check below
    ])
    w = crosswalk.vtd_weights(b)
    sums = w.groupby("VTDKEY").share.sum()
    assert sums[1] == pytest.approx(1.0)
    assert sums[2] == pytest.approx(1.0)
    # Only VTDs with weight > 0 or land > 0 are required to sum to one.
    live = b.groupby("VTDKEY")[["vap", "ALAND20"]].sum().gt(0).any(axis=1)
    assert np.allclose(sums[live[live].index], 1.0)


def test_prison_vtd_falls_back_to_land():
    # VTD 7 is all prison: vap_adj is 0 everywhere, so shares follow land area.
    b = blocks_df([
        (7, "77010", 500, 500, 30), (7, "77011", 200, 200, 10),
        (8, "77010", 100, 0, 1),
    ])
    w = crosswalk.vtd_weights(b)
    s = w[w.VTDKEY == 7].set_index("ZCTA").share
    assert s["77010"] == pytest.approx(0.75)
    assert s["77011"] == pytest.approx(0.25)
    # Without the correctional adjustment the same VTD would use population.
    raw = crosswalk.vtd_weights(b, col="vap")
    assert raw[raw.VTDKEY == 7].set_index("ZCTA").share["77010"] == pytest.approx(500 / 700)


def test_zero_weight_and_zero_land_vtd_emits_zero_and_logs(caplog):
    # R1: the zero-share check that raises lives in Task 3's allocate.
    b = blocks_df([(9, "77020", 0, 0, 0), (9, "NONE", 0, 0, 0), (10, "77020", 10, 0, 1)])
    with caplog.at_level(logging.WARNING, logger="hfb.crosswalk"):
        w = crosswalk.vtd_weights(b)
    assert (w[w.VTDKEY == 9].share == 0).all()
    assert w[w.VTDKEY == 10].share.sum() == pytest.approx(1.0)
    assert "9" in caplog.text


def test_vtdkey_is_int():
    b = blocks_df([(1, "77001", 5, 0, 1)])
    b["VTDKEY"] = b.VTDKEY.astype(float)
    assert pd.api.types.is_integer_dtype(crosswalk.vtd_weights(b).VTDKEY)


def test_blank_zcta_becomes_NONE():
    assert crosswalk.zcta_from_faces(pd.Series(["", None, "77001"])).tolist() == [
        "NONE", "NONE", "77001"]


def test_multi_zcta_block_raises():
    with pytest.raises(ValueError, match="more than one ZCTA"):
        crosswalk.zcta_from_faces(pd.Series(["77001|77002"]))


def test_block_missing_from_join_source_raises():
    blocks = pd.DataFrame({"GEOID20": ["a", "b"]})
    other = pd.DataFrame({"GEOID20": ["a"], "total": [1]})
    with pytest.raises(ValueError, match=r"1 blocks have no row in Blocks_Pop.*'b'"):
        crosswalk._join_complete(blocks, other, "Blocks_Pop")
    ok = crosswalk._join_complete(blocks.iloc[:1], other, "Blocks_Pop")
    assert ok.total.tolist() == [1]


def test_zcta_pop_excludes_none():
    b = blocks_df([(1, "77001", 100, 10, 1), (1, "77001", 50, 0, 1), (1, "NONE", 9, 0, 1)])
    z = crosswalk.zcta_pop(b)
    assert z.index.tolist() == ["77001"]
    assert z.loc["77001", "vap"] == 150
    assert z.loc["77001", "vap_adj"] == 140
    assert z.loc["77001", "corr_adult"] == 10


def test_vap_adj_clips_at_zero():
    b = blocks_df([(1, "77001", 5, 9, 1)])
    assert crosswalk.add_vap_adj(b).vap_adj.iloc[0] == 0


# ---------- real-data tests ----------


@pytest.fixture(scope="module")
def built(raw):
    import time

    t0 = time.time()
    blocks = crosswalk.build_blocks(raw)
    return blocks, time.time() - t0


@pytest.mark.real
def test_blocks_match_golden(built, raw):
    blocks, secs = built
    g = pd.read_csv(GOLDEN / f"blocks_{TAG}.csv", dtype={"GEOID20": str, "ZCTA": str})
    gq = pd.read_csv(GOLDEN / f"block_gq_{TAG}.csv", dtype={"GEOID20": str})
    g = g.merge(gq, on="GEOID20", how="left")
    g["ZCTA"] = g.ZCTA.fillna("").replace("", "NONE")
    g["VTDKEY"] = g.VTDKEY.astype(int)
    assert len(blocks) == len(g) == 79250
    assert (raw.parent / "interim" / "blocks.parquet").exists()
    shared = ["GEOID20", "VTDKEY", "ZCTA", "total", "vap", "corr_adult", "college", "ALAND20"]
    a = blocks.sort_values("GEOID20").reset_index(drop=True)[shared]
    b = g.sort_values("GEOID20").reset_index(drop=True)[shared]
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert blocks.VTDKEY.dtype.kind == "i"
    assert not blocks.ZCTA.str.contains(r"\|").any()
    # gq (PL index 5) is new, so golden cannot check it: bound it by its own components
    # (so a wrong split index fails) and pin the region total (measured 93,854).
    assert (blocks.gq >= blocks.corr_adult + blocks.college).all()
    assert blocks.gq.sum() == 93854
    print(f"\nbuild_blocks seconds: {secs:.1f}")


@pytest.fixture(scope="module")
def region_vtdkeys(raw):
    import geopandas as gpd

    v = gpd.read_file(raw / "tlc_vtds24" / "VTDs_24PG.shp", engine="pyogrio", ignore_geometry=True)
    return set(v[v.CNTY.isin([int(c) for c in COUNTIES])].VTDKEY.astype(int))


@pytest.mark.real
def test_weights_match_golden_z(built, raw, region_vtdkeys):
    blocks, _ = built
    w = crosswalk.vtd_weights(blocks)
    assert len(region_vtdkeys) == 1561
    assert set(w.VTDKEY) == region_vtdkeys
    vd24 = pd.read_csv(raw / "tlc_vd24.csv", usecols=["VTDKEY"])
    assert region_vtdkeys <= set(vd24.VTDKEY.astype(int))

    # Rebuild c5.py's in-memory adjusted weights from the golden CSVs.
    B = pd.read_csv(GOLDEN / f"blocks_{TAG}.csv", dtype={"GEOID20": str, "ZCTA": str}).merge(
        pd.read_csv(GOLDEN / f"block_gq_{TAG}.csv", dtype={"GEOID20": str}),
        on="GEOID20", how="left").fillna({"corr_adult": 0})
    B["ZCTA"] = B.ZCTA.fillna("")
    B["vap_adj"] = (B.vap - B.corr_adult).clip(lower=0)
    g = B.groupby(["VTDKEY", "ZCTA"])[["vap_adj", "ALAND20"]].sum().reset_index()
    t = g.groupby("VTDKEY").vap_adj.transform("sum")
    ta = g.groupby("VTDKEY").ALAND20.transform("sum")
    g["s"] = np.where(t > 0, g.vap_adj / t, g.ALAND20 / ta)
    g["VTDKEY"] = g.VTDKEY.astype(int)
    g["ZCTA"] = g.ZCTA.replace("", "NONE")
    m = w.merge(g[["VTDKEY", "ZCTA", "s"]], on=["VTDKEY", "ZCTA"], how="outer", indicator=True)
    assert (m._merge == "both").all()
    assert np.abs(m.share - m.s).max() < 1e-9
    live = blocks.groupby("VTDKEY")[["vap", "ALAND20"]].sum().gt(0).any(axis=1)
    sums = w.groupby("VTDKEY").share.sum()
    assert np.allclose(sums[live[live].index], 1.0)
    print(f"\nNONE weight rows: {(w.ZCTA == 'NONE').sum()}; VTDs: {w.VTDKEY.nunique()}")


@pytest.mark.real
def test_vtd_3807_uses_land(built):
    blocks, _ = built
    w = crosswalk.vtd_weights(blocks)
    v = blocks[blocks.VTDKEY == 3807]
    assert len(v) > 0
    assert (v.vap - v.corr_adult).clip(lower=0).sum() == 0  # all adults are correctional
    land = v.groupby("ZCTA").ALAND20.sum()
    expected = land / land.sum()
    got = w[w.VTDKEY == 3807].set_index("ZCTA").share
    assert set(got.index) == set(expected.index)
    assert np.allclose(got.sort_index(), expected.sort_index(), atol=1e-12)
    print(f"\nVTD 3807 shares: {got.round(6).to_dict()}")
