import numpy as np
import pandas as pd
import pytest

from hfb import crosswalk, frame
from hfb.config import COUNTIES


def rel_county(rows):
    """(ZCTA, ZCTA land, county GEOID, part land)."""
    return pd.DataFrame(
        rows, columns=["GEOID_ZCTA5_20", "AREALAND_ZCTA5_20", "GEOID_COUNTY_20", "AREALAND_PART"])


def rel_tract(rows):
    """(ZCTA, tract GEOID, tract land, part land)."""
    return pd.DataFrame(
        rows, columns=["GEOID_ZCTA5_20", "GEOID_TRACT_20", "AREALAND_TRACT_20", "AREALAND_PART"])


def test_region_zctas_land_share_and_county_main():
    rc = rel_county([
        ("77001", 1000, "48201", 300), ("77001", 1000, "48157", 500),
        ("77001", 1000, "48099", 200),                            # out of region
        ("77002", 400, "48201", 400),                              # whole
        ("77003", 800, "48099", 800),                              # no region land: dropped
        ("77004", 600, "48473", 0), ("77004", 600, "48099", 600),  # zero region land: dropped
    ])
    z = frame.region_zctas(rc)
    assert z.index.tolist() == ["77001", "77002"]
    assert z.loc["77001", "land_share"] == pytest.approx(0.8)
    assert z.loc["77001", "county_main"] == "48157"
    assert z.loc["77002", "land_share"] == pytest.approx(1.0)
    assert z.loc["77002", "county_main"] == "48201"


def test_region_zctas_county_main_ignores_larger_out_of_region_county():
    rc = rel_county([("77001", 1000, "48099", 600), ("77001", 1000, "48201", 100)])
    assert frame.region_zctas(rc).loc["77001", "county_main"] == "48201"


def test_inreg_share20_fixture():
    # Z1 = 750 people in region; one in-region tract and one out-of-region Texas tract
    # (total 1000, 1/4 of its land in Z1) -> 250 outside -> 750 / 1000.
    rt = rel_tract([
        ("Z1", "48201000100", 5000, 5000),
        ("Z1", "48099000200", 4000, 1000),
        ("Z2", "48201000100", 5000, 100),
    ])
    # two blocks in tract 48099000200 sum to 1000
    bp = pd.DataFrame({"FIPS": ["099", "099", "201"], "TRT": ["200", "200", "100"],
                       "total": [600, 400, 5000]})
    zpop = pd.Series({"Z1": 750.0, "Z2": 50.0})
    s = frame.inreg_share20(rt, bp, zpop)
    assert s["Z1"] == pytest.approx(0.75)
    assert s["Z2"] == pytest.approx(1.0)


def test_inreg_share20_edge_cases():
    rt = rel_tract([
        ("Z1", "48099000200", 4000, 1000),   # contributes 250
        ("Z1", "48099000300", 0, 500),       # zero tract land: contributes 0, no ZeroDivision
        ("Z1", "48099000400", 100, 100),     # tract absent from Blocks_Pop: contributes 0
        ("Z1", "01003010100", 100, 100),     # non-Texas: ignored
    ])
    bp = pd.DataFrame({"FIPS": ["99", "099", "099"], "TRT": ["200", "300", "999999"],
                       "total": [1000, 70, 5]})  # unpadded FIPS/TRT must be zero-filled
    out, stats = frame.out_of_region_2020(rt, bp)
    assert out["Z1"] == pytest.approx(250.0)
    assert stats == {"non_texas_tracts": 1, "missing_tracts": 1}
    assert frame.inreg_share20(rt, bp, pd.Series({"Z1": 750.0}))["Z1"] == pytest.approx(0.75)


def test_out_of_region_limited_to_requested_zctas():
    rt = rel_tract([("Z1", "48099000200", 4000, 1000), ("Z9", "01003010100", 100, 100),
                    ("Z9", "48099000200", 4000, 4000)])
    bp = pd.DataFrame({"FIPS": ["099"], "TRT": ["200"], "total": [1000]})
    out, stats = frame.out_of_region_2020(rt, bp, ["Z1"])
    assert out.to_dict() == {"Z1": 250.0}
    assert stats == {"non_texas_tracts": 0, "missing_tracts": 0}


def test_inreg_share20_in_region_tracts_do_not_count_as_outside():
    rt = rel_tract([("Z1", f"48{c}000100", 100, 100) for c in COUNTIES])
    bp = pd.DataFrame({"FIPS": COUNTIES, "TRT": ["100"] * len(COUNTIES), "total": [999] * len(COUNTIES)})
    assert frame.inreg_share20(rt, bp, pd.Series({"Z1": 500.0}))["Z1"] == pytest.approx(1.0)


def test_read_rel_skips_blank_zcta_and_bom(tmp_path):
    p = tmp_path / "rel.txt"
    p.write_text(
        "﻿OID_ZCTA5_20|GEOID_ZCTA5_20|AREALAND_ZCTA5_20|GEOID_COUNTY_20|AREALAND_PART\n"
        "||||5\n1|77001|10|48201|4\n", encoding="utf-8")
    r = frame.read_rel(p)
    assert r.GEOID_ZCTA5_20.tolist() == ["77001"]
    assert r.GEOID_COUNTY_20.tolist() == ["48201"]


# ---------- real-data tests ----------


@pytest.fixture(scope="module")
def region(raw):
    return frame.region_zctas(frame.read_rel(crosswalk._path(raw, "zcta_county")))


@pytest.fixture(scope="module")
def inreg(raw, blocks, region):
    rt = frame.read_rel(crosswalk._path(raw, "zcta_tract"))
    bp = frame.read_blocks_pop(raw)
    return frame.inreg_share20(rt, bp, crosswalk.zcta_pop(blocks).total)


@pytest.mark.real
def test_region_has_240_zctas(region):
    assert len(region) == 240


@pytest.mark.real
def test_all_region_zctas_in_b05003(raw, region):
    from hfb import acs

    a = acs.read_acs(crosswalk._path(raw, "acs_b05003"), region.index)
    assert set(region.index) <= set(a.index)


@pytest.mark.real
@pytest.mark.parametrize("zcta,expected", [
    ("77328", 0.537), ("77339", 0.788), ("77358", 0.885), ("78950", 0.633), ("77665", 0.942)])
def test_inreg_share20_known(inreg, zcta, expected):
    assert inreg[zcta] == pytest.approx(expected, abs=0.005)


@pytest.mark.real
def test_whole_zctas_share_near_one(region, inreg):
    whole = region.index[region.land_share >= 0.999]
    assert len(whole) > 0
    assert inreg.reindex(whole).min() >= 0.99
