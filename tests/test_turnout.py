from pathlib import Path

import pandas as pd
import pytest

from hfb import turnout
from hfb.config import COUNTIES

GOLDEN = Path(__file__).parent / "golden"
TARGETS = {
    24: (3_740_459, 2_239_980),
    22: (3_535_245, 1_564_982),
}


def vd_df(rows):
    """Hand-built TLC frame: (VTDKEY, CNTYVTD, Voter_Registration, Turnout)."""
    return pd.DataFrame(rows, columns=["VTDKEY", "CNTYVTD", "Voter_Registration", "Turnout"])


def w_df(rows):
    return pd.DataFrame(rows, columns=["VTDKEY", "ZCTA", "share"])


# ---------- unit tests on hand-built frames ----------


def test_totals_preserved():
    weights = w_df([
        (1, "77001", 0.25), (1, "77002", 0.50), (1, "NONE", 0.25),
        (2, "77002", 1.0),
    ])
    vd = vd_df([(1, "0010001", 1000, 600), (2, "0010002", 200, 50),
                (99, "0010099", 5000, 4000)])  # VTD 99 not in the region: ignored
    a = turnout.allocate(weights, vd, 24)
    assert list(a.columns) == ["reg24", "to24"]
    assert a.loc["77001"].tolist() == pytest.approx([250, 150])
    assert a.loc["77002"].tolist() == pytest.approx([700, 350])
    assert a.loc["NONE"].tolist() == pytest.approx([250, 150])
    assert a.reg24.sum() == pytest.approx(1200)
    assert a.to24.sum() == pytest.approx(650)
    assert turnout.none_bucket(a) == {"reg24": pytest.approx(250), "to24": pytest.approx(150)}


def test_column_names_follow_yy():
    weights = w_df([(1, "77001", 1.0)])
    a = turnout.allocate(weights, vd_df([(1, "0010001", 10, 5)]), 22)
    assert list(a.columns) == ["reg22", "to22"]
    assert "NONE" in a.index and a.loc["NONE"].tolist() == [0, 0]


def test_missing_vtdkey_raises():
    weights = w_df([(1, "77001", 1.0), (2, "77001", 1.0)])
    vd = vd_df([(1, "0010001", 10, 5)])
    with pytest.raises(ValueError, match="2"):
        turnout.allocate(weights, vd, 24)


def test_zero_share_vtd_with_registrations_raises():
    weights = w_df([(1, "77001", 1.0), (7, "77001", 0.0), (7, "NONE", 0.0)])
    vd = vd_df([(1, "0010001", 10, 5), (7, "0010007", 40, 20)])
    with pytest.raises(ValueError, match="7"):
        turnout.allocate(weights, vd, 24)


def test_zero_share_vtd_without_registrations_is_fine():
    weights = w_df([(1, "77001", 1.0), (7, "77001", 0.0)])
    vd = vd_df([(1, "0010001", 10, 5), (7, "0010007", 0, 0)])
    a = turnout.allocate(weights, vd, 24)
    assert a.reg24.sum() == pytest.approx(10)


def test_no_surname_columns(tmp_path):
    p = tmp_path / "vd.csv"
    p.write_text(
        "CNTYVTD,VTDKEY,Voter_Registration,Turnout,"
        "Spanish_Surname_Voter_Registration,Spanish_Surname_Turnout\n"
        '"0010001",1,1870,1118,208,0.07\n'
        '"2010002",2,2850,1873,120,0.03\n')
    vd = turnout.load_vd(p)
    assert not [c for c in vd.columns if "spanish" in c.lower()]
    assert {"VTDKEY", "Voter_Registration", "Turnout"} <= set(vd.columns)
    assert vd.CNTYVTD.tolist() == ["0010001", "2010002"]  # leading zeros kept
    assert vd.VTDKEY.dtype.kind == "i"


def test_duplicate_vtdkey_in_vd_raises():
    weights = w_df([(1, "77001", 1.0)])
    vd = vd_df([(1, "0010001", 10, 5), (1, "0010001", 10, 5)])
    with pytest.raises(ValueError, match="duplicated.*1"):
        turnout.allocate(weights, vd, 24)


@pytest.mark.parametrize("col", ["Voter_Registration", "Turnout"])
def test_nan_input_raises(col):
    weights = w_df([(1, "77001", 1.0), (8, "77001", 1.0)])
    vd = vd_df([(1, "0010001", 10, 5), (8, "0010008", 40, 20)])
    vd.loc[vd.VTDKEY == 8, col] = float("nan")
    with pytest.raises(ValueError, match=r"NaN.*8"):
        turnout.allocate(weights, vd, 24)


def test_turnout_without_registration_and_zero_share_raises():
    weights = w_df([(1, "77001", 1.0), (9, "77001", 0.0)])
    vd = vd_df([(1, "0010001", 10, 5), (9, "0010009", 0, 12)])
    with pytest.raises(ValueError, match="9"):
        turnout.allocate(weights, vd, 24)


def test_shares_not_summing_to_one_raise():
    weights = w_df([(1, "77001", 1.0), (6, "77001", 0.5), (6, "77002", 0.3)])
    vd = vd_df([(1, "0010001", 10, 5), (6, "0010006", 100, 60)])
    with pytest.raises(ValueError, match=r"6.*0\.8"):
        turnout.allocate(weights, vd, 24)


def test_split_exposure_hand_built():
    # ZCTA A: pieces 0.05 (reg 5) ; 0.10 (reg 10) ; 0.90 (reg 90 of VTD 3) ; 1.0 (VTD 4, reg 100)
    # exposed share in [0.10, 0.90] inclusive: VTD2 piece + VTD3 piece = 10 + 90 = 100
    # total = 5 + 10 + 90 + 100 = 205 -> 100/205
    weights = w_df([
        (1, "A", 0.05), (1, "B", 0.95),
        (2, "A", 0.10), (2, "B", 0.90),
        (3, "A", 0.90), (3, "B", 0.10),
        (4, "A", 1.0),
        (5, "C", 1.0), (5, "NONE", 0.0),  # C: zero registration -> 0.0
        (6, "NONE", 0.5), (6, "D", 0.5),
    ])
    vd = vd_df([
        (1, "x", 100, 0), (2, "x", 100, 0), (3, "x", 100, 0), (4, "x", 100, 0),
        (5, "x", 0, 0), (6, "x", 100, 0),
    ])
    s = turnout.split_exposure(weights, vd)
    assert "NONE" not in s.index
    assert s["A"] == pytest.approx(100 / 205)
    assert s["B"] == pytest.approx((10 * 0 + 90 + 10 + 0) / (95 + 90 + 10))  # VTD1 0.95 out
    assert s["C"] == 0.0
    assert s["D"] == pytest.approx(1.0)  # single piece at share 0.5, all exposed


# ---------- real data ----------


@pytest.fixture(scope="module")
def vds(raw):
    return {yy: turnout.load_vd(raw / f"tlc_vd{yy}.csv") for yy in (22, 24)}


@pytest.fixture(scope="module")
def allocs(weights, vds):
    return {yy: turnout.allocate(weights, vds[yy], yy) for yy in (22, 24)}


@pytest.mark.real
@pytest.mark.parametrize("yy", [22, 24])
def test_real_totals_match_targets(allocs, yy):
    reg, to = TARGETS[yy]
    a = allocs[yy]
    assert a[f"reg{yy}"].sum() == pytest.approx(reg, abs=0.5)
    assert a[f"to{yy}"].sum() == pytest.approx(to, abs=0.5)


@pytest.mark.real
@pytest.mark.parametrize("yy", [22, 24])
def test_real_none_registration_small(allocs, yy):
    nb = turnout.none_bucket(allocs[yy])
    assert 0 <= nb[f"reg{yy}"] <= 50, nb


@pytest.mark.real
@pytest.mark.parametrize("yy", [22, 24])
def test_county_totals_match_tlc(weights, vds, yy):
    vd = vds[yy]
    pieces = turnout.pieces(weights, vd, yy)
    county = vd.set_index("VTDKEY").CNTYVTD.str[:3]
    pieces["county"] = pieces.VTDKEY.map(county)
    got = pieces.groupby("county")[[f"reg{yy}", f"to{yy}"]].sum()
    tlc = vd[vd.CNTYVTD.str[:3].isin(COUNTIES)].groupby(vd.CNTYVTD.str[:3])[
        ["Voter_Registration", "Turnout"]].sum()
    assert sorted(got.index) == sorted(COUNTIES) == sorted(tlc.index)
    diff = max(
        (got[f"reg{yy}"] - tlc.Voter_Registration.reindex(got.index)).abs().max(),
        (got[f"to{yy}"] - tlc.Turnout.reindex(got.index)).abs().max())
    assert diff <= 0.5, diff


@pytest.mark.real
def test_real_split_exposure_valid(weights, vds):
    s = turnout.split_exposure(weights, vds[24])
    assert "NONE" not in s.index
    assert s.between(0, 1).all()


@pytest.mark.real
@pytest.mark.parametrize("yy", [22, 24])
def test_allocation_matches_golden_z(allocs, yy):
    z = pd.read_csv(GOLDEN / "z.csv", dtype={"ZCTA": str}, usecols=["ZCTA", f"reg{yy}", f"to{yy}"])
    z = z.set_index("ZCTA")
    a = allocs[yy]
    a = a[a.index != "NONE"]
    assert set(a.index) == set(z.index)
    z = z.reindex(a.index)
    assert (a[[f"reg{yy}", f"to{yy}"]] - z).abs().to_numpy().max() < 1e-6
