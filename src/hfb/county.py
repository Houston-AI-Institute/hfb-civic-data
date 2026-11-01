"""County roll-up: 10 rows, one per county in ``config.COUNTIES``, in that order.

Where the ZCTA table cannot describe a county (in Walker, Liberty and San Jacinto the ranked
ZCTAs hold only about half to two thirds of the county's registered voters), this table gives a
county view. P1 here uses the exact Texas Legislative Council county totals (straight sums of
the VTD files, no allocation) over the ACS county estimate of citizen voting-age population,
less adults in correctional facilities from the 2020 PL block file. No rank, no rank interval
and no flags: with 10 counties a rank would read as a league table.

``county_table(raw, t)`` reads ``raw`` (nothing is fetched); ``build`` is the same computation
on already-loaded inputs. Integers are nullable ``Int64`` and ratios are rounded to 4 places,
so the frame is what the CSV and the xlsx sheet hold.
"""

from pathlib import Path

import pandas as pd

from . import acs, context, crosswalk, flags, turnout
from .config import COUNTIES, COUNTY_NAMES
from .paths import raw_path

COLUMNS = [
    "county_fips", "county_label", "reg22", "to22", "reg24", "to24", "to_reg24", "cvap",
    "cvap_moe", "corr_adult", "cvap_adj", "P1", "P1_low", "P1_high", "zctas_total",
    "zctas_ranked", "reg24_share_ranked", *context.CONTEXT_COLS, "civic_orgs_main",
]
INT_COLS = ["reg22", "to22", "reg24", "to24", "cvap", "cvap_moe", "corr_adult", "cvap_adj",
            "zctas_total", "zctas_ranked", "civic_orgs_main"]
DP4_COLS = ["to_reg24", "P1", "P1_low", "P1_high", "reg24_share_ranked", *context.CONTEXT_COLS]
TEXT_COLS = ["county_fips", "county_label"]
YEARS = (22, 24)
SHEET = "county"  # xlsx sheet name, after ``data``

COLUMN_DEFS = {
    "county_fips": "5-digit FIPS code (text) of the county: Texas 48 followed by the 3-digit "
                   "county code.",
    "county_label": "County name, for reading the table.",
    "reg22": "Registered voters, 2022 general election: the exact county total from the Texas "
             "Legislative Council voting-district file, summed with no allocation.",
    "to22": "Ballots cast, 2022 general election: the exact county total from the Texas "
            "Legislative Council voting-district file, summed with no allocation.",
    "reg24": "Registered voters, 2024 general election: the exact county total from the Texas "
             "Legislative Council voting-district file, summed with no allocation.",
    "to24": "Ballots cast, 2024 general election: the exact county total from the Texas "
            "Legislative Council voting-district file, summed with no allocation.",
    "to_reg24": "to24 divided by reg24: 2024 general-election turnout as a share of registered "
                "voters (0 to 1).",
    "cvap": "Citizen voting-age population of the county (native-born and naturalized citizens "
            "aged 18 and over), ACS 2020-2024 5-year estimate for the whole county, computed "
            "as in the ZCTA table. A rounded estimate with sampling error.",
    "cvap_moe": "90 percent margin of error on cvap (root sum of squares of the component "
                "margins). Rounded.",
    "corr_adult": "People living in adult correctional facilities in the county, 2020 Census "
                  "group quarters count, summed over the county's blocks. Subtracted from "
                  "cvap in P1.",
    "cvap_adj": "cvap less corr_adult; the denominator of P1.",
    "P1": "Average of 2022 and 2024 general-election turnout divided by cvap_adj, that is "
          "(to22 + to24) / (2 * cvap_adj). A reference measure, not a score of a community. "
          "Not ranked and not flagged at county level.",
    "P1_low": "P1 recomputed with cvap raised by cvap_moe, so the denominator is cvap + "
              "cvap_moe - corr_adult. With P1_high it gives a 90 percent range from ACS "
              "sampling error only.",
    "P1_high": "P1 recomputed with cvap lowered by cvap_moe, so the denominator is cvap - "
               "cvap_moe - corr_adult. With P1_low it gives a 90 percent range from ACS "
               "sampling error only.",
    "zctas_total": "Number of rows in the ZCTA table whose county_main is this county.",
    "zctas_ranked": "Number of those ZCTAs that are ranked (not NOT_RANKED).",
    "reg24_share_ranked": "Share of the county's reg24 in the ZCTA table (summed over ZCTAs "
                          "whose county_main is this county) that sits in ranked ZCTAs, 0 to 1. "
                          "Low values mean the ZCTA ranking describes little of the county. "
                          "Blank where the county has no ZCTA assigned.",
    "same_house_1yr": "Share of people aged 1 and over in the county who lived in the same "
                      "house one year earlier (ACS 2020-2024 5-year, table B07003), 0 to 1. "
                      "Context only; not used in P1. A sampling estimate.",
    "same_house_1yr_moe": "90 percent margin of error on same_house_1yr, as a share (0 to 1).",
    "owner_occupied": "Share of occupied housing units in the county that are owner occupied "
                      "(ACS 2020-2024 5-year, table B25003), 0 to 1. Context only; not used in "
                      "P1. A sampling estimate.",
    "owner_occupied_moe": "90 percent margin of error on owner_occupied, as a share (0 to 1).",
    "hs_diploma_25plus": "Share of people aged 25 and over in the county whose highest "
                         "attainment is a regular high school diploma or above (ACS 2020-2024 "
                         "5-year, table B15003), 0 to 1. Same definition as the ZCTA table. "
                         "Context only; not used in P1. A sampling estimate.",
    "hs_diploma_25plus_moe": "90 percent margin of error on hs_diploma_25plus, as a share "
                             "(0 to 1).",
    "not_limited_english_hh": "Share of households in the county that are not limited English "
                              "speaking (ACS 2020-2024 5-year, table C16002), 0 to 1. Context "
                              "only; not used in P1. A sampling estimate.",
    "not_limited_english_hh_moe": "90 percent margin of error on not_limited_english_hh, as a "
                                  "share (0 to 1).",
    "broadband_hh": "Share of households in the county with a broadband internet subscription "
                    "of any type (ACS 2020-2024 5-year, table B28002), 0 to 1. Context only; "
                    "not used in P1. A sampling estimate.",
    "broadband_hh_moe": "90 percent margin of error on broadband_hh, as a share (0 to 1).",
    "civic_orgs_main": "Sum of civic_orgs_n over the ZCTAs whose county_main is this county. "
                       "ZCTAs with less than half their land in the region are left out. "
                       "The IRS file counts by mailing ZIP, and each ZCTA is assigned to the "
                       "region county holding most of its in-region land, so this is not an "
                       "exact count of organizations located in the county. Zero means none "
                       "registered; "
                       "registration is not a measure of civic life.",
}

COUNTY_NOTE = (
    "P1 at county level uses the exact Texas Legislative Council county totals for turnout and "
    "the ACS county estimate for citizen voting-age population, not the ZCTA allocation. The "
    "county table has 10 rows, in the order of the region's counties, with no rank, no rank "
    "interval and no flags. Read it with skiprows=1 like the ZCTA CSV.")


def tlc_sums(vd: pd.DataFrame) -> pd.DataFrame:
    """Straight sums of a TLC VTD file by county (``CNTYVTD[:3]``), in ``COUNTIES`` order."""
    cn = vd.CNTYVTD.str[:3]
    tot = vd[cn.isin(COUNTIES)].groupby(cn)[["Voter_Registration", "Turnout"]].sum()
    return tot.reindex(COUNTIES)


def corr_adult_by_county(gq: pd.DataFrame) -> pd.Series:
    """Adult correctional group quarters by county (3-digit code) from the PL block table.

    Grouped by the block's own county (``GEOID20[2:5]``), never by the county of the ZCTA the
    block falls in. A county with blocks but no correctional people is 0; a region county with
    no blocks at all raises ``ValueError`` (a truncated PL read must not become 0).
    """
    got = gq.groupby(gq.GEOID20.str[2:5]).corr_adult.sum().reindex(COUNTIES)
    absent = sorted(got.index[got.isna()])
    if absent:
        raise ValueError(f"counties with no blocks in the PL group quarters table: {absent}")
    return got


def _p1(to22, to24, den: pd.Series) -> pd.Series:
    return (to22 + to24) / (2 * den.where(den > 0))


def build(vd_sums: dict, b05003: pd.DataFrame, ctx: pd.DataFrame, corr: pd.Series,
          t: pd.DataFrame) -> pd.DataFrame:
    """The county table from loaded inputs.

    ``vd_sums`` maps 22 and 24 to ``tlc_sums`` frames (index = 3-digit county); ``b05003`` and
    ``ctx`` are indexed by 5-digit county FIPS (raw B05003 columns, and ``context.context``
    output); ``corr`` is indexed by 3-digit county; ``t`` is the finished ZCTA table.
    """
    fips = ["48" + c for c in COUNTIES]
    c = pd.DataFrame({"county_fips": fips,
                      "county_label": [COUNTY_NAMES[x] for x in COUNTIES]})
    for yy in YEARS:
        s = vd_sums[yy].reindex(COUNTIES)
        c[f"reg{yy}"] = s.Voter_Registration.to_numpy()
        c[f"to{yy}"] = s.Turnout.to_numpy()
    c["to_reg24"] = c.to24 / c.reg24.where(c.reg24 > 0)

    v = acs.cvap(b05003.reindex(fips))
    c["cvap"] = v.cvap.round().to_numpy()
    c["cvap_moe"] = v.cvap_moe.round().to_numpy()
    corr = corr.reindex(COUNTIES)
    if corr.isna().any():
        raise ValueError(f"counties with no corr_adult value: {sorted(corr.index[corr.isna()])}")
    c["corr_adult"] = corr.to_numpy()
    c["cvap_adj"] = c.cvap - c.corr_adult
    c["P1"] = _p1(c.to22, c.to24, c.cvap_adj)
    c["P1_low"] = _p1(c.to22, c.to24, c.cvap_adj + c.cvap_moe)
    c["P1_high"] = _p1(c.to22, c.to24, c.cvap_adj - c.cvap_moe)

    by = t.groupby("county_main")
    ranked = ~t.not_ranked.astype(bool)
    total = by.size().reindex(fips).fillna(0)
    c["zctas_total"] = total.to_numpy()
    c["zctas_ranked"] = ranked.groupby(t.county_main).sum().reindex(fips).fillna(0).to_numpy()
    reg_all = t.reg24.groupby(t.county_main).sum().reindex(fips)
    reg_rk = t.reg24.where(ranked, 0.0).groupby(t.county_main).sum().reindex(fips)
    c["reg24_share_ranked"] = (reg_rk / reg_all.where(reg_all > 0)).to_numpy()

    x = ctx.reindex(fips)
    for col in context.CONTEXT_COLS:
        c[col] = x[col].to_numpy()
    mostly_in = t.land_share >= flags.LAND_MIN  # skip ZCTAs mostly outside the region
    orgs = t.civic_orgs_n[mostly_in].groupby(t.county_main[mostly_in]).sum()
    orgs = orgs.reindex(fips).fillna(0)
    c["civic_orgs_main"] = orgs.to_numpy()

    c = c[COLUMNS].copy()
    for col in INT_COLS:
        c[col] = c[col].round().astype("Int64")
    c[DP4_COLS] = c[DP4_COLS].astype(float).round(4)
    return c.reset_index(drop=True)


def county_table(raw, t: pd.DataFrame) -> pd.DataFrame:
    """The county roll-up from ``raw`` (``data/raw``) and the finished ZCTA table ``t``."""
    raw = Path(raw)
    fips = ["48" + c for c in COUNTIES]
    vd_sums = {yy: tlc_sums(turnout.load_vd(raw_path(raw, f"tlc_vd{yy}"))) for yy in YEARS}
    for yy, s in vd_sums.items():
        if s.isna().any().any():
            raise ValueError(f"counties missing from the 20{yy} TLC file: "
                             f"{sorted(s.index[s.isna().any(axis=1)])}")
    b05003 = acs.read_acs_county(raw_path(raw, "acs_b05003"), fips)
    missing = sorted(set(fips) - set(b05003.index))
    if missing:
        raise ValueError(f"counties missing from ACS B05003: {missing}")
    ctx = context.read_context(raw, pd.Index(fips), reader=acs.read_acs_county)
    corr = corr_adult_by_county(crosswalk._read_gq(raw))
    return build(vd_sums, b05003, ctx, corr, t)
