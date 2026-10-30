"""Task 7: ACS context proportions and their margins of error."""

import logging

import numpy as np
import pandas as pd
import pytest

from hfb import context

CTX = ["same_house_1yr", "owner_occupied", "hs_diploma_25plus", "not_limited_english_hh",
       "broadband_hh"]


def acs_frame(rows: dict) -> pd.DataFrame:
    """ACS columns for all five tables; rows = {zcta: {column: value}}, others 0."""
    cols = ["B07003_E001", "B07003_M001", "B07003_E004", "B07003_M004",
            "B25003_E001", "B25003_M001", "B25003_E002", "B25003_M002",
            "B15003_E001", "B15003_M001",
            *[f"B15003_{k}{i:03d}" for i in range(17, 26) for k in "EM"],
            "C16002_E001", "C16002_M001",
            *[f"C16002_{k}{i:03d}" for i in (4, 7, 10, 13) for k in "EM"],
            "B28002_E001", "B28002_M001", "B28002_E004", "B28002_M004"]
    df = pd.DataFrame(0.0, index=pd.Index(list(rows), name="ZCTA"), columns=cols)
    for z, vals in rows.items():
        for c, v in vals.items():
            df.loc[z, c] = v
    return df


def good_rows():
    return {
        "77001": {"B07003_E001": 1000, "B07003_M001": 20, "B07003_E004": 300, "B07003_M004": 40,
                  "B25003_E001": 400, "B25003_M001": 30, "B25003_E002": 100, "B25003_M002": 10,
                  "B15003_E001": 2000, "B15003_M001": 50,
                  "B15003_E017": 500, "B15003_M017": 30, "B15003_E018": 100, "B15003_M018": 40,
                  "B15003_E025": 100, "B15003_M025": 0,
                  "C16002_E001": 800, "C16002_M001": 25,
                  "C16002_E004": 40, "C16002_M004": 12, "C16002_E013": 20, "C16002_M013": 5,
                  "B28002_E001": 800, "B28002_M001": 25,
                  "B28002_E004": 600, "B28002_M004": 35},
        "77002": {"B07003_E001": 50, "B07003_E004": 50, "B25003_E001": 10, "B25003_E002": 0,
                  "B15003_E001": 30, "B15003_E017": 30, "C16002_E001": 20,
                  "B28002_E001": 20, "B28002_E004": 20},
    }


def test_proportions_between_0_and_1():
    c = context.context(acs_frame(good_rows()))
    est = c[CTX]
    assert ((est >= 0) & (est <= 1)).all().all()
    assert (c[[f"{k}_moe" for k in CTX]].dropna() >= 0).all().all()
    assert list(c.columns) == [x for k in CTX for x in (k, f"{k}_moe")]


def test_proportion_moe_formula():
    c = context.context(acs_frame(good_rows()))
    # p = 300/1000; sqrt(40^2 - 0.3^2 * 20^2)/1000
    assert c.loc["77001", "same_house_1yr"] == pytest.approx(0.3)
    assert c.loc["77001", "same_house_1yr_moe"] == pytest.approx(np.sqrt(1600 - 0.09 * 400) / 1000)
    assert c.loc["77001", "owner_occupied"] == pytest.approx(0.25)
    assert c.loc["77001", "owner_occupied_moe"] == pytest.approx(
        np.sqrt(100 - 0.0625 * 900) / 400)


def test_moe_falls_back_to_ratio_form_when_root_negative():
    rows = good_rows()
    rows["77001"].update({"B07003_M004": 5, "B07003_M001": 100})  # 25 - 0.09*10000 < 0
    c = context.context(acs_frame(rows))
    assert c.loc["77001", "same_house_1yr_moe"] == pytest.approx(
        np.sqrt(25 + 0.09 * 10000) / 1000)


def test_sum_terms_use_root_sum_square():
    c = context.context(acs_frame(good_rows()))
    p = (500 + 100 + 100) / 2000
    m_num = np.sqrt(30**2 + 40**2 + 0**2)  # nonzero terms only; the other terms are 0
    assert c.loc["77001", "hs_diploma_25plus"] == pytest.approx(p)
    assert c.loc["77001", "hs_diploma_25plus_moe"] == pytest.approx(
        np.sqrt(m_num**2 - p**2 * 50**2) / 2000)


def test_not_limited_english_is_complement_with_same_moe():
    c = context.context(acs_frame(good_rows()))
    p = (40 + 20) / 800
    m_num = np.sqrt(12**2 + 5**2)
    moe = np.sqrt(m_num**2 - p**2 * 25**2) / 800
    assert c.loc["77001", "not_limited_english_hh"] == pytest.approx(1 - p)
    assert c.loc["77001", "not_limited_english_hh_moe"] == pytest.approx(moe)


def test_zero_or_nan_denominator_is_blank():
    rows = good_rows()
    rows["77003"] = {}  # every denominator is 0
    a = acs_frame(rows)
    a.loc["77002", "B07003_E001"] = np.nan
    a.loc["77002", "C16002_E004"] = np.nan  # NaN numerator term
    c = context.context(a)
    assert c.loc["77003"].isna().all()
    assert np.isnan(c.loc["77002", "same_house_1yr"])
    assert np.isnan(c.loc["77002", "same_house_1yr_moe"])
    assert np.isnan(c.loc["77002", "not_limited_english_hh"])
    assert c.loc["77002", "broadband_hh"] == 1.0  # other columns unaffected


def test_estimate_survives_a_missing_moe():
    a = acs_frame(good_rows())
    a.loc["77001", "B07003_M004"] = np.nan
    c = context.context(a)
    assert c.loc["77001", "same_house_1yr"] == pytest.approx(0.3)
    assert np.isnan(c.loc["77001", "same_house_1yr_moe"])


def write_dat(path, table, rows):
    cols = [f"{table}_{k}{i:03d}" for i in range(1, 26) for k in "EM"]
    with open(path, "w") as f:
        f.write("|".join(["GEO_ID"] + cols) + "\n")
        for z, vals in rows.items():
            f.write("|".join([f"860Z200US{z}"] + [str(vals.get(c, 0)) for c in cols]) + "\n")


def test_read_context_missing_zcta_is_blank_and_logged(tmp_path, caplog):
    files = []
    for t in ("b07003", "b25003", "b15003", "c16002", "b28002"):
        rows = {"77001": {f"{t.upper()}_E001": 100, f"{t.upper()}_M001": 10,
                          f"{t.upper()}_E004": 50, f"{t.upper()}_M004": 9,
                          f"{t.upper()}_E002": 25, f"{t.upper()}_M002": 5}}
        if t != "b28002":
            rows["77002"] = dict(rows["77001"])
        write_dat(tmp_path / f"acs_{t}.dat", t.upper(), rows)
        files.append(f"acs_{t},u,acs_{t}.dat,seed,x,1,t")
    (tmp_path / "manifest.csv").write_text(
        "key,url,path,source,sha256,bytes,fetched_utc\n" + "\n".join(files) + "\n")
    with caplog.at_level(logging.WARNING, logger="hfb.context"):
        c = context.read_context(tmp_path, pd.Index(["77001", "77002", "77009"], name="ZCTA"))
    assert c.index.tolist() == ["77001", "77002", "77009"]
    assert c.loc["77001", "broadband_hh"] == 0.5
    assert np.isnan(c.loc["77002", "broadband_hh"]) and c.loc["77002", "owner_occupied"] == 0.25
    assert c.loc["77009"].isna().all()
    assert "acs_b28002" in caplog.text and "2" in caplog.text


@pytest.mark.real
def test_real_context_bounds(table):
    for k in CTX:
        s = table[k].dropna()
        assert len(s) > 200, k
        assert s.between(0, 1).all(), k
        assert (table[f"{k}_moe"].dropna() >= 0).all(), k
