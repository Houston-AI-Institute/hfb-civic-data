import logging
import time

import numpy as np
import pandas as pd
import pytest

from hfb import pipeline

COLS = ["ZCTA", "county_main", "land_share", "inreg_share20", "acs_pop", "reg22", "to22",
        "reg24", "to24", "to_reg24", "cvap", "cvap_moe", "corr_adult", "cvap_adj", "P1",
        "P1_rank", "rank_p05", "rank_p95", "not_ranked", "not_ranked_reason", "caution",
        "split_exposure", "gq_share", "college_share"]
EXTRA = ["gq_share", "college_share", "split_exposure", "corr_share", "vap", "reg24"]


# ---------- assemble on synthetic pieces ----------

def pieces(zs=("77001", "77002", "77003")):
    ix = pd.Index(zs, name="ZCTA")
    region = pd.DataFrame({"land_share": 0.9, "county_main": "48201"}, index=ix)
    acs = pd.DataFrame({"acs_pop": 5000.0, "cvap": 3000.0, "cvap_moe": 200.0,
                        "acs_vap": 3900.0}, index=ix)
    zpop = pd.DataFrame({"total": 5000, "vap": 4000, "gq": 10, "corr_adult": 0,
                         "college": 0, "vap_adj": 4000}, index=ix)
    a22 = pd.DataFrame({"reg22": 2000.0, "to22": 900.0 + 100 * np.arange(len(zs))}, index=ix)
    a22.loc["NONE"] = [5.0, 5.0]
    a24 = pd.DataFrame({"reg24": 2200.0, "to24": 1000.0 + 100 * np.arange(len(zs))}, index=ix)
    a24.loc["NONE"] = [5.0, 5.0]
    split = pd.Series(0.1, index=ix, name="split_exposure")
    inreg = pd.Series(0.99, index=ix)
    return dict(region=region, acs=acs, zpop=zpop, alloc22=a22, alloc24=a24,
                split=split, inreg=inreg)


def test_assemble_columns_and_rows():
    t = pipeline.assemble(**pieces())
    assert list(t.index) == ["77001", "77002", "77003"]
    assert set(COLS) <= set(t.columns) and set(EXTRA) <= set(t.columns)
    assert t.ZCTA.tolist() == ["77001", "77002", "77003"]
    assert "NONE" not in t.index
    assert t.not_ranked.dtype == bool
    assert t.P1_rank.tolist() == [pytest.approx(100 / 3), pytest.approx(200 / 3), 100.0]
    assert t.P1.iloc[0] == pytest.approx((900 + 1000) / (2 * 3000))


def test_assemble_raises_when_acs_missing_a_region_zcta():
    p = pieces()
    p["acs"] = p["acs"].drop("77002")
    with pytest.raises(ValueError, match="77002"):
        pipeline.assemble(**p)


def test_assemble_raises_when_zpop_missing_a_region_zcta():
    p = pieces()
    p["zpop"] = p["zpop"].drop("77003")
    with pytest.raises(ValueError, match="77003"):
        pipeline.assemble(**p)


def test_assemble_fills_zero_alloc_for_zcta_absent_from_allocation(caplog):
    p = pieces()
    p["alloc24"] = p["alloc24"].drop("77001")
    with caplog.at_level(logging.WARNING):
        t = pipeline.assemble(**p)
    assert t.loc["77001", "reg24"] == 0 and t.loc["77001", "to24"] == 0
    assert np.isnan(t.loc["77001", "to_reg24"])
    assert "77001" in caplog.text
    assert "LOW_N" in t.loc["77001", "caution"]


def test_assemble_missing_split_is_zero():
    p = pieces()
    p["split"] = p["split"].drop("77001")
    assert pipeline.assemble(**p).loc["77001", "split_exposure"] == 0.0


def test_assemble_zero_population_zcta_is_not_ranked_with_nan_inreg():
    p = pieces(("77001", "77002", "77341"))
    p["zpop"].loc["77341"] = [0, 0, 0, 0, 0, 0]
    p["acs"].loc["77341"] = [0.0, np.nan, np.nan, np.nan]
    p["inreg"].loc["77341"] = np.nan
    t = pipeline.assemble(**p)
    r = t.loc["77341"]
    assert np.isnan(r.inreg_share20) and np.isnan(r.P1)
    assert r.not_ranked and r.not_ranked_reason == "NO_CVAP;POP"
    assert "EDGE" not in r.caution
    assert np.isnan(r.P1_rank) and np.isnan(r.rank_p05) and np.isnan(r.rank_p95)


def test_assemble_not_ranked_row_keeps_nan_rank_but_ranked_rows_rank_among_themselves():
    p = pieces(("77001", "77002", "77003", "77004"))
    p["acs"].loc["77004", "acs_pop"] = 100.0  # POP
    p["alloc22"].loc["77004", "to22"] = 1e6
    p["alloc24"].loc["77004", "to24"] = 1e6
    t = pipeline.assemble(**p)
    assert t.loc["77004", "not_ranked_reason"] == "POP"
    assert t.loc["77004", "P1"] > 1  # kept as measured, not capped or blanked
    assert t.P1_rank.dropna().max() == 100.0 and np.isnan(t.loc["77004", "P1_rank"])


def test_assemble_is_deterministic():
    a = pipeline.assemble(**pieces())
    b = pipeline.assemble(**pieces())
    pd.testing.assert_frame_equal(a, b)


SIDE = ["same_house_1yr", "same_house_1yr_moe", "owner_occupied", "owner_occupied_moe",
        "hs_diploma_25plus", "hs_diploma_25plus_moe", "not_limited_english_hh",
        "not_limited_english_hh_moe", "broadband_hh", "broadband_hh_moe",
        "civic_orgs_n", "civic_orgs_band", "hoa_like_n"]


def side_frame(zs=("77001", "77002", "77003")):
    ix = pd.Index(zs, name="ZCTA")
    d = {c: 0.5 for c in SIDE}
    d.update(civic_orgs_n=3, civic_orgs_band="3-5", hoa_like_n=1)
    return pd.DataFrame(d, index=ix)[SIDE]


def test_side_columns_are_appended_and_change_nothing_else():
    base = pipeline.assemble(**pieces())
    with_side = pipeline.assemble(**pieces(), side=side_frame())
    assert list(with_side.columns[:len(base.columns)]) == list(base.columns)
    assert list(with_side.columns[len(base.columns):]) == SIDE
    pd.testing.assert_frame_equal(with_side[base.columns], base)


def test_side_columns_never_feed_flags_p1_or_rank():
    base = pipeline.assemble(**pieces())
    side = side_frame()
    side[["owner_occupied", "same_house_1yr"]] = np.nan  # blank context must not unrank a row
    side["civic_orgs_n"] = 0
    t = pipeline.assemble(**pieces(), side=side)
    for c in ("P1", "P1_rank", "rank_p05", "rank_p95", "not_ranked", "not_ranked_reason",
              "caution"):
        pd.testing.assert_series_equal(t[c], base[c])


def test_side_is_aligned_to_region_index():
    side = side_frame(("77003", "77001", "77002")).drop("77002")  # shuffled, one missing
    t = pipeline.assemble(**pieces(), side=side)
    assert t.loc["77003", "civic_orgs_n"] == 3 and np.isnan(t.loc["77002", "owner_occupied"])


# ---------- real data ----------

def codes(t, col, code):
    return {z for z, v in t[col].items() if code in v.split(";")}


@pytest.fixture(scope="module")
def timed(raw):
    t0 = time.perf_counter()
    t = pipeline.build_table(raw)
    return t, time.perf_counter() - t0


@pytest.mark.real
def test_real_table_shape_and_columns(table):
    assert len(table) == 240 and table.ZCTA.is_unique
    assert set(COLS) <= set(table.columns) and set(EXTRA) <= set(table.columns)
    assert list(table.ZCTA) == sorted(table.ZCTA)
    assert not (table.ZCTA == "NONE").any()


@pytest.mark.real
def test_real_77320_is_correctional_with_expected_p1(table):
    r = table.loc["77320"]
    assert r.P1 == pytest.approx(0.470, abs=0.005)
    assert "CORRECTIONAL" in r.not_ranked_reason.split(";")


@pytest.mark.real
def test_real_not_ranked_correctional_set(table):
    assert codes(table, "not_ranked_reason", "CORRECTIONAL") == {"77002", "77320", "77535", "77583"}


@pytest.mark.real
def test_real_77422_correctional_caution_not_ranked_out(table):
    r = table.loc["77422"]
    assert "CORRECTIONAL" in r.caution.split(";")
    assert not r.not_ranked
    assert r.corr_share == pytest.approx(0.0997, abs=0.0005)


@pytest.mark.real
def test_real_partial_set_and_blank_p1(table):
    partial = codes(table, "not_ranked_reason", "PARTIAL")
    in_land = set(table.index[table.land_share >= 0.50])
    assert len(partial) == 30
    assert partial & in_land == {"77328", "77339", "77358", "78950"}
    assert table.loc[sorted(partial), "P1"].isna().all()
    # every other PARTIAL row is a LAND row (land_share < 0.50)
    assert partial - in_land == codes(table, "not_ranked_reason", "LAND")
    assert partial - in_land == {z for z in partial if table.loc[z, "land_share"] < 0.50}


@pytest.mark.real
def test_real_77665_ranked_with_edge(table):
    r = table.loc["77665"]
    assert not r.not_ranked and "EDGE" in r.caution.split(";")


@pytest.mark.real
def test_real_gq_includes_77204_and_77446(table):
    assert {"77204", "77446"} <= codes(table, "not_ranked_reason", "GQ")


@pytest.mark.real
def test_real_77441_growth_caution(table):
    assert "GROWTH" in table.loc["77441", "caution"].split(";")


@pytest.mark.real
def test_real_ranked_count(table, caplog):
    n = int((~table.not_ranked).sum())
    print(f"ranked count: {n}")
    assert 170 <= n <= 180


@pytest.mark.real
def test_real_77341_not_ranked_nan_inreg(table):
    r = table.loc["77341"]
    assert np.isnan(r.inreg_share20) and r.not_ranked
    assert {"POP", "NO_CVAP"} <= set(r.not_ranked_reason.split(";"))


@pytest.mark.real
def test_real_ranks_only_on_ranked_rows(table):
    ranked = ~table.not_ranked
    assert table.loc[ranked, ["P1", "P1_rank", "rank_p05", "rank_p95"]].notna().all().all()
    assert table.loc[~ranked, ["P1_rank", "rank_p05", "rank_p95"]].isna().all().all()
    assert table.loc[ranked, "P1_rank"].max() == 100.0
    assert (table.loc[ranked, "rank_p05"] <= table.loc[ranked, "rank_p95"]).all()
    assert (table.not_ranked == (table.not_ranked_reason != "")).all()


@pytest.mark.real
def test_real_build_is_fast_enough(timed):
    print(f"build_table seconds: {timed[1]:.1f}")
    assert timed[1] < 300


@pytest.mark.real
def test_real_table_has_side_columns_after_college_share(table):
    cols = list(table.columns)
    assert cols[-len(SIDE):] == SIDE
    assert table.civic_orgs_n.notna().all() and table.hoa_like_n.notna().all()
    assert table.civic_orgs_band.isin(["0", "1-2", "3-5", "6+"]).all()
