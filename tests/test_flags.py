import numpy as np
import pandas as pd
import pytest

from hfb import config, flags

BASE = dict(
    land_share=0.9, acs_pop=5000, cvap=3000.0, corr_adult=0, gq=10, total=5000, vap=4000,
    college=0, inreg_share20=0.99, reg24=2000.0, split_exposure=0.0)


def row(zcta="77001", **kw) -> pd.DataFrame:
    d = {**BASE, **kw}
    df = pd.DataFrame(d, index=[zcta])
    df["cvap_adj"] = df.cvap - df.corr_adult
    return df


def flag(**kw):
    out = flags.apply_flags(row(**kw))
    return out.not_ranked_reason.iloc[0], out.caution.iloc[0], bool(out.not_ranked.iloc[0])


def test_clean_row_has_no_flags():
    assert flag() == ("", "", False)


def test_facility_check_constant():
    assert config.FACILITY_CHECK == {"77032"}


# (kwarg, value that trips the flag, value just on the safe side, expected code)
NOT_RANKED_CASES = [
    ("land_share", 0.4999, 0.50, "LAND"),
    ("acs_pop", 1499, 1500, "POP"),
    ("gq", 2501, 2500, "GQ"),  # gq / total > 0.50, total 5000
    ("corr_adult", 400, 399, "CORRECTIONAL"),  # corr_adult / vap >= 0.10, vap 4000
    ("inreg_share20", 0.8999, 0.90, "PARTIAL"),
]


@pytest.mark.parametrize("col,bad,ok,code", NOT_RANKED_CASES)
def test_not_ranked_threshold(col, bad, ok, code):
    reason, _, nr = flag(**{col: bad})
    assert code in reason.split(";") and nr
    reason, _, nr = flag(**{col: ok})
    assert code not in reason.split(";")


def test_no_cvap_when_missing():
    reason, _, nr = flag(cvap=np.nan)
    assert reason == "NO_CVAP" and nr


def test_no_cvap_when_adj_zero_or_negative():
    # cvap_adj = cvap - corr_adult; 0 and negative both trip, 1 does not
    assert "NO_CVAP" in flag(cvap=300.0, corr_adult=300)[0].split(";")
    assert "NO_CVAP" in flag(cvap=300.0, corr_adult=301)[0].split(";")
    assert "NO_CVAP" not in flag(cvap=300.0, corr_adult=299)[0].split(";")


# (kwarg, value that trips the caution, value just on the safe side, code)
CAUTION_CASES = [
    ("reg24", 499.0, 500.0, "LOW_N"),
    ("cvap", 999.0, 1000.0, "LOW_N"),
    ("reg24", 4401.0, 4400.0, "GROWTH"),  # reg24 / vap > 1.1, vap 4000
    ("split_exposure", 0.5001, 0.50, "SPLIT"),
    ("college", 500, 499, "COLLEGE"),  # college / total >= 0.10, total 5000
    ("corr_adult", 200, 199, "CORRECTIONAL"),  # lower edge, 0.05 <= corr / vap
    ("corr_adult", 399, 400, "CORRECTIONAL"),  # upper edge, corr / vap < 0.10
    ("inreg_share20", 0.90, 0.8999, "EDGE"),  # lower edge, 0.90 <= inreg
    ("inreg_share20", 0.9799, 0.98, "EDGE"),  # upper edge, inreg < 0.98
]


@pytest.mark.parametrize("col,trips,safe,code", CAUTION_CASES)
def test_caution_threshold(col, trips, safe, code):
    assert code in flag(**{col: trips})[1].split(";")
    assert code not in flag(**{col: safe})[1].split(";")


def test_low_n_is_either_condition():
    assert flag(reg24=100.0)[1] == "LOW_N"
    assert flag(cvap=900.0)[1] == "LOW_N"
    assert flag(reg24=100.0, cvap=900.0)[1] == "LOW_N"


def test_facility_check_only_listed_zcta():
    assert flag(zcta="77032")[1] == "FACILITY_CHECK"
    assert flag(zcta="77033")[1] == ""


def test_partial_and_edge_are_exclusive_at_boundary():
    r, c, _ = flag(inreg_share20=0.89)
    assert "PARTIAL" in r and "EDGE" not in c
    r, c, _ = flag(inreg_share20=0.90)
    assert "PARTIAL" not in r and "EDGE" in c


def test_correctional_tiers_do_not_overlap():
    r, c, _ = flag(corr_adult=400)
    assert "CORRECTIONAL" in r and "CORRECTIONAL" not in c
    r, c, _ = flag(corr_adult=200)
    assert "CORRECTIONAL" not in r and "CORRECTIONAL" in c


def test_reasons_and_cautions_sorted_and_joined():
    r, c, nr = flag(land_share=0.1, acs_pop=100, gq=4000, reg24=100.0, college=900,
                    zcta="77032")
    assert r == "GQ;LAND;POP"
    assert c == "COLLEGE;FACILITY_CHECK;LOW_N"
    assert nr is True


def test_not_ranked_follows_reason():
    out = flags.apply_flags(pd.concat([row("77001"), row("77002", land_share=0.1)]))
    assert out.not_ranked.tolist() == [False, True]
    assert out.not_ranked_reason.tolist() == ["", "LAND"]
    assert out.not_ranked.dtype == bool
    assert list(out.columns) == ["not_ranked", "not_ranked_reason", "caution"]


def test_nan_flag_input_does_not_trigger_that_flag():
    # 77341-like: zero-population ZCTA, so inreg_share20, gq/total and corr/vap are NaN
    r, c, nr = flag(zcta="77341", acs_pop=0, total=0, gq=0, vap=0, cvap=np.nan,
                    corr_adult=0, inreg_share20=np.nan, college=0, reg24=0.0,
                    split_exposure=np.nan)
    codes = r.split(";")
    assert "PARTIAL" not in codes and "GQ" not in codes and "CORRECTIONAL" not in codes
    assert "EDGE" not in c and "COLLEGE" not in c and "SPLIT" not in c and "GROWTH" not in c
    assert codes == ["NO_CVAP", "POP"] and nr


def test_nan_inreg_alone_neither_partial_nor_edge():
    r, c, nr = flag(inreg_share20=np.nan)
    assert r == "" and c == "" and not nr


def test_nan_land_share_does_not_trigger_land():
    assert "LAND" not in flag(land_share=np.nan)[0].split(";")


def test_does_not_mutate_input():
    df = row()
    before = df.copy()
    flags.apply_flags(df)
    pd.testing.assert_frame_equal(df, before)


def test_apply_flags_computes_cvap_adj_when_absent():
    df = row(cvap=300.0, corr_adult=300).drop(columns="cvap_adj")
    assert "NO_CVAP" in flags.apply_flags(df).not_ranked_reason.iloc[0]
