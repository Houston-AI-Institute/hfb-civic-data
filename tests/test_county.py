"""Task 10: the 10-row county roll-up (``county_civic.csv``, the ``county`` sheet, a QA check).

Synthetic tests run in the default pass; the real-data ones are marked ``real`` like the rest
of the suite. Nothing here writes to the repo's ``out/`` or ``qa/``.
"""

import csv
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openpyxl
import pandas as pd
import pytest
from test_acs import COLS as ACS_COLS
from test_acs import write_dat
from test_assemble import synth_ctx, synth_table

from hfb import acs, assemble, context, county, safeguards
from hfb.config import COUNTIES
from hfb.copy import MISUSE
from hfb.paths import raw_path

ROOT = Path(__file__).resolve().parent.parent
FIPS = ["48" + c for c in COUNTIES]
CTX_BASES = list(context.SPECS)
EXPECTED_COLUMNS = [
    "county_fips", "county_label", "reg22", "to22", "reg24", "to24", "to_reg24", "cvap",
    "cvap_moe", "corr_adult", "cvap_adj", "P1", "P1_low", "P1_high", "zctas_total",
    "zctas_ranked", "reg24_share_ranked", *context.CONTEXT_COLS, "civic_orgs_main",
]


# ---------- synthetic inputs ----------

def synth_inputs(t=None):
    """County inputs whose turnout sums match ``synth_table`` plus its NONE bucket (5.0)."""
    t = synth_table() if t is None else t
    vd = {}
    for yy in (22, 24):
        reg, to = round(t[f"reg{yy}"].sum() + 5.0), round(t[f"to{yy}"].sum() + 5.0)
        base_r, base_t = reg // 10, to // 10
        r = pd.Series(base_r, index=COUNTIES)
        o = pd.Series(base_t, index=COUNTIES)
        r["201"] += reg - base_r * 10
        o["201"] += to - base_t * 10
        vd[yy] = pd.DataFrame({"Voter_Registration": r, "Turnout": o})
    b = pd.DataFrame(0.0, index=pd.Index(FIPS, name="ZCTA"), columns=ACS_COLS)
    for c, v in (("E009", 40000), ("E011", 30000), ("E020", 45000), ("E022", 35000),
                 ("E008", 160000), ("E019", 170000), ("E001", 400000)):
        b[f"B05003_{c}"] = float(v)
    for c in ("M009", "M011", "M020", "M022"):
        b[f"B05003_{c}"] = 3000.0
    ctxdf = pd.DataFrame({c: 0.4123456 for c in context.CONTEXT_COLS}, index=FIPS)
    corr = pd.Series((np.arange(10) + 1) * 10, index=COUNTIES)
    return vd, b, ctxdf, corr, t


@pytest.fixture(scope="module")
def synth_county():
    vd, b, ctxdf, corr, t = synth_inputs()
    return county.build(vd, b, ctxdf, corr, t), t


@pytest.fixture(scope="module")
def synth_built(tmp_path_factory, synth_county):
    c, t = synth_county
    ctx = synth_ctx(t)
    ctx.county = c
    base = tmp_path_factory.mktemp("synth_county")
    out, qa = base / "out", base / "qa"
    ok = assemble.emit(t, ctx, out, qa)
    return SimpleNamespace(out=out, qa=qa, ok=ok, county=c, table=t, ctx=ctx, kind="synthetic")


@pytest.fixture(scope="session")
def county_real(raw, table):
    return county.county_table(raw, table)


@pytest.fixture(params=["synthetic", pytest.param("real", marks=pytest.mark.real)])
def cb(request):
    """(county table, outputs) on synthetic inputs and on the real build."""
    if request.param == "synthetic":
        s = request.getfixturevalue("synth_built")
        return SimpleNamespace(county=s.county, out=s.out, qa=s.qa, kind="synthetic")
    out, qa, _ = request.getfixturevalue("real_outputs")
    return SimpleNamespace(county=request.getfixturevalue("county_real"), out=out, qa=qa,
                           kind="real")


def read_county_csv(cb, **kw):
    return pd.read_csv(cb.out / "county_civic.csv", skiprows=1, dtype={"county_fips": str}, **kw)


# ---------- the table ----------

def test_county_rows_10_in_config_order(cb):
    c = cb.county
    assert len(c) == 10
    assert c.county_fips.tolist() == FIPS
    assert c.county_fips.map(type).eq(str).all()
    assert c.county_label.tolist() == [county.COUNTY_NAMES[x] for x in COUNTIES]


def test_county_columns_in_order(cb):
    assert list(cb.county.columns) == EXPECTED_COLUMNS == county.COLUMNS
    assert not any(c in cb.county for c in ("P1_rank", "rank_p05", "rank_p95", "not_ranked",
                                           "caution"))  # no rank, no flags


def test_county_integer_and_rounded_columns(cb):
    c = cb.county
    for col in county.INT_COLS:
        assert c[col].dtype == "Int64", col
    for col in county.DP4_COLS:
        assert (c[col].dropna().round(4) == c[col].dropna()).all(), col


@pytest.mark.real
def test_county_turnout_matches_tlc_totals(raw, county_real):
    for yy in (22, 24):
        vd = pd.read_csv(raw_path(raw, f"tlc_vd{yy}"), dtype={"CNTYVTD": str})
        cn = vd.CNTYVTD.str[:3]
        tot = vd[cn.isin(COUNTIES)].groupby(cn).agg(
            reg=("Voter_Registration", "sum"), to=("Turnout", "sum")).reindex(COUNTIES)
        assert county_real[f"reg{yy}"].astype(int).tolist() == tot.reg.astype(int).tolist()
        assert county_real[f"to{yy}"].astype(int).tolist() == tot.to.astype(int).tolist()
    assert (county_real.reg24 > 0).all() and (county_real.to24 > 0).all()


def test_county_turnout_is_a_straight_sum_synthetic(synth_county):
    c, t = synth_county
    vd = synth_inputs(t)[0]
    for yy in (22, 24):
        assert c[f"reg{yy}"].astype(int).tolist() == vd[yy].Voter_Registration.tolist()
        assert c[f"to{yy}"].astype(int).tolist() == vd[yy].Turnout.tolist()
    assert (c.to_reg24 == (c.to24 / c.reg24).round(4)).all()


def test_county_p1_between_low_and_high(cb):
    c = cb.county
    assert c.P1.notna().all() and c.P1_low.notna().all() and c.P1_high.notna().all()
    assert ((c.P1_low <= c.P1) & (c.P1 <= c.P1_high)).all()
    assert (c.P1_low < c.P1_high).all()


def test_county_p1_formula_synthetic(synth_county):
    c, t = synth_county
    r = c.set_index("county_fips").loc["48201"]
    adj = r.cvap - r.corr_adult
    assert r.P1 == pytest.approx((r.to22 + r.to24) / (2 * adj), abs=5e-5)
    assert r.P1_low == pytest.approx((r.to22 + r.to24) / (2 * (adj + r.cvap_moe)), abs=5e-5)
    assert r.P1_high == pytest.approx((r.to22 + r.to24) / (2 * (adj - r.cvap_moe)), abs=5e-5)
    assert r.cvap == 150000 and r.cvap_moe == 6000 and r.cvap_adj == r.cvap - r.corr_adult


def test_county_zcta_counts_and_orgs_from_table_synthetic(synth_county):
    c, t = synth_county
    c = c.set_index("county_fips")
    assert c.loc["48201", "zctas_total"] == 5 and c.loc["48039", "zctas_total"] == 1
    ranked = ~t.not_ranked.astype(bool)
    assert c.loc["48201", "zctas_ranked"] == int((ranked & (t.county_main == "48201")).sum())
    assert c.loc["48201", "civic_orgs_main"] == int(
        t.civic_orgs_n[t.county_main == "48201"].sum())
    reg = t.reg24[t.county_main == "48201"]
    assert c.loc["48201", "reg24_share_ranked"] == pytest.approx(
        reg[ranked[reg.index]].sum() / reg.sum(), abs=5e-5)
    assert c.loc["48015", "zctas_total"] == 0 and c.loc["48015", "civic_orgs_main"] == 0
    assert pd.isna(c.loc["48015", "reg24_share_ranked"])


@pytest.mark.real
def test_county_real_sanity(county_real, table, raw):
    c = county_real.set_index("county_fips")
    assert c.zctas_total.sum() == 240 and c.zctas_ranked.sum() == int((~table.not_ranked).sum())
    assert c.civic_orgs_main.sum() == int(table.civic_orgs_n.sum())
    # the rural counties the food bank asked about hold only part of their voters in ranked ZCTAs
    for f in ("48471", "48291", "48407"):
        assert 0.4 < c.loc[f, "reg24_share_ranked"] < 0.7, f
    assert c.loc["48201", "reg24_share_ranked"] > 0.9
    assert (c.corr_adult >= 0).all() and (c.cvap_adj < c.cvap).any()
    # every correctional adult is in some block of some region county, and the ZCTA column is
    # the same blocks grouped by ZCTA (NONE excluded), so the totals must agree exactly
    assert int(c.corr_adult.sum()) == int(table.corr_adult.sum())
    assert 2_500_000 < c.loc["48201", "cvap"] < 4_838_303  # below Harris total population
    # names agree with the 2020 PL geography file
    geo = pd.read_csv(raw_path(raw, "pl") / "txgeo2020.pl", sep="|", header=None, dtype=str,
                      encoding="latin-1", quoting=csv.QUOTE_NONE, usecols=[2, 9, 86])
    geo = geo[geo[2] == "050"].set_index(9)[86]
    assert {f: geo[f] for f in FIPS} == dict(zip(c.index, c.county_label))


# ---------- corr_adult by county ----------

def _gq(rows):
    """PL-style block frame: (county code, tract, block, corr_adult[, ZCTA]) -> GEOID20 frame."""
    return pd.DataFrame({
        "GEOID20": [f"48{c}{tr}{b}" for c, tr, b, *_ in rows],
        "corr_adult": [r[3] for r in rows],
        "ZCTA": [r[4] if len(r) > 4 else "" for r in rows]})


def _all_counties(extra=None):
    rows = [(c, "000100", "1000", 0) for c in COUNTIES]  # a block with no correctional people
    return rows + (extra or [])


def test_corr_adult_by_county_groups_by_the_blocks_own_county():
    # 77001 is a Harris (201) ZCTA by land, but block 157/000200/2000 lies in Fort Bend: its
    # 40 people belong to Fort Bend. Grouping by the ZCTA's county would credit Harris.
    gq = _gq(_all_counties([
        ("201", "000100", "2000", 100, "77001"),
        ("201", "000300", "1000", 7, "77001"),
        ("157", "000200", "2000", 40, "77001"),
        ("157", "000200", "3000", 5, "77002"),
        ("471", "000100", "2000", 250, ""),  # outside every ZCTA, still Walker's
    ]))
    got = county.corr_adult_by_county(gq)
    assert got.index.tolist() == COUNTIES
    assert got["201"] == 107 and got["157"] == 45 and got["471"] == 250
    assert got.drop(["201", "157", "471"]).eq(0).all()  # blocks but no one: 0, not missing
    assert got.sum() == gq.corr_adult.sum()


def test_corr_adult_by_county_key_is_the_county_digits_not_the_state_prefix():
    gq = _gq(_all_counties([("201", "000100", "2000", 9)]))
    assert county.corr_adult_by_county(gq).loc["201"] == 9
    assert all(len(k) == 3 for k in county.corr_adult_by_county(gq).index)


def test_county_with_no_blocks_raises_not_zero():
    gq = _gq([r for r in _all_counties([("201", "000100", "2000", 9)]) if r[0] != "455"])
    with pytest.raises(ValueError, match="455"):
        county.corr_adult_by_county(gq)


def test_build_raises_when_a_county_has_no_corr_adult():
    vd, b, ctxdf, corr, t = synth_inputs()
    with pytest.raises(ValueError, match="407"):
        county.build(vd, b, ctxdf, corr.drop("407"), t)
    county.build(vd, b, ctxdf, corr * 0, t)  # all zeros is legitimate


# ---------- ACS county rows ----------

def test_read_acs_county_reads_county_rows_only(tmp_path):
    p = tmp_path / "b05003.dat"
    write_dat(p, {
        "0100000US": {"B05003_E001": 999},
        "0400000US48": {"B05003_E001": 888},
        "0500000US48201": {"B05003_E001": 1000, "B05003_M009": -555555555},
        "0500000US48039": {"B05003_E001": 500, "B05003_E009": 7},
        "0500000US06037": {"B05003_E001": 777},  # another state
        "860Z200US77001": {"B05003_E001": 1},
    })
    a = acs.read_acs_county(p, ["48201", "48039", "48999"])
    assert a.index.tolist() == ["48201", "48039"] and a.index.name == "county_fips"
    assert a.loc["48201", "B05003_E001"] == 1000 and np.isnan(a.loc["48201", "B05003_M009"])
    z = acs.read_acs(p, ["77001", "48201"])
    assert z.index.tolist() == ["77001"]  # ZCTA reader behaviour is unchanged


# ---------- outputs ----------

def test_county_csv_row1_is_misuse(cb):
    path = cb.out / "county_civic.csv"
    with open(path, newline="") as f:
        assert next(csv.reader(f)) == [MISUSE]
    assert path.read_text().splitlines()[0] == '"' + MISUSE + '"'


def test_county_csv_reads_with_skiprows(cb):
    df = read_county_csv(cb)
    assert len(df) == 10
    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.county_fips.tolist() == FIPS
    assert df.county_fips.str.len().eq(5).all()


def test_county_csv_matches_table(cb):
    df = read_county_csv(cb)
    for col in ("reg24", "to24", "cvap", "zctas_ranked", "civic_orgs_main"):
        assert df[col].tolist() == cb.county[col].astype(int).tolist(), col
    assert df.P1.round(4).tolist() == pytest.approx(cb.county.P1.astype(float).tolist())


def test_county_sheet_present_with_misuse(cb):
    wb = openpyxl.load_workbook(cb.out / "zcta_civic.xlsx")
    assert wb.sheetnames == ["data", "county", "read_me"]
    ws = wb["county"]
    assert ws["A1"].value == MISUSE
    assert [c.value for c in ws[2]] == EXPECTED_COLUMNS
    assert ws["A3"].data_type == "s" and ws["A3"].value == "48201"
    assert ws["A12"].value == "48291" and ws["A13"].value is None  # 10 rows, then nothing
    assert ws["A1"].font.bold and ws.freeze_panes == "A3"


def test_county_no_banned_words_or_forbidden_columns(cb):
    assert safeguards.check_forbidden_columns(read_county_csv(cb).columns) == []
    wb = openpyxl.load_workbook(cb.out / "zcta_civic.xlsx")
    assert safeguards.check_forbidden_columns(c.value for c in wb["county"][2]) == []
    texts = {"county csv": (cb.out / "county_civic.csv").read_text()}
    texts.update(safeguards._xlsx_texts(wb))
    texts["qa"] = (cb.qa / "qa_report.md").read_text()
    assert safeguards.check_banned_words(texts) == []
    low = texts["county csv"].lower()
    for w in safeguards.BANNED_WORDS:
        if w not in safeguards.WHOLE_WORD:
            assert w.lower() not in low
    res = safeguards.run_all(cb.out)
    assert res["csv"] == [] and res["csv_county"] == [] and res["xlsx"] == []


def test_read_me_defines_every_county_column(cb):
    wb = openpyxl.load_workbook(cb.out / "zcta_civic.xlsx")
    rows = [[c.value for c in row] for row in wb["read_me"].iter_rows()]
    start = next(i for i, r in enumerate(rows) if r[0] == "County table")
    block = rows[start:]
    assert "exact" in block[0][1] and "county" in block[0][1].lower()
    defined = {r[0]: r[1] for r in block[1:] if r[0] and r[1]}
    assert set(EXPECTED_COLUMNS) <= set(defined)
    assert all(len(defined[c]) > 10 for c in EXPECTED_COLUMNS)
    assert "mailing ZIP" in defined["civic_orgs_main"] and "most of its land" in defined[
        "civic_orgs_main"]
    assert all("—" not in str(v) and "–" not in str(v) for r in rows for v in r if v)
    assert re.search(r"no rank", " ".join(str(v) for r in block for v in r if v), re.I)


def test_county_column_defs_cover_columns():
    assert set(county.COLUMN_DEFS) == set(county.COLUMNS)


def test_county_files_in_allowlist():
    assert "county_civic.csv" in safeguards.ALLOWED_OUT


# ---------- QA check ----------

def test_qa_county_line_passes(cb):
    text = (cb.qa / "qa_report.md").read_text()
    m = re.search(r"^(PASS|FAIL) county_sums: county reg/to sums equal the ZCTA table sums "
                  r"plus the NONE bucket", text, re.M)
    assert m and m.group(1) == "PASS", text[:2000]


def test_qa_county_check_can_fail(synth_county):
    c, t = synth_county
    ctx = synth_ctx(t)
    ctx.county = c
    name, ok, detail = assemble.county_check(t, ctx)
    assert name == "county_sums" and ok is True
    for col in ("reg22", "to24"):
        bad = c.copy()
        bad.loc[0, col] = bad.loc[0, col] + 2  # two votes off in one county
        ctx.county = bad
        _, ok, detail = assemble.county_check(t, ctx)
        assert ok is False and col in detail
    off_by_one = c.copy()
    off_by_one.loc[3, "reg24"] = off_by_one.loc[3, "reg24"] - 1  # 0.6 off
    ctx.county = off_by_one
    assert assemble.county_check(t, ctx)[1] is True  # within 1 vote
    ctx.county = None
    assert assemble.county_check(t, ctx)[1] is False  # no county table is a failure


def test_failing_county_check_fails_overall_but_writes_files(tmp_path, synth_county):
    c, t = synth_county
    ctx = synth_ctx(t)
    ctx.county = c.assign(to22=c.to22 + 5)
    ok = assemble.emit(t, ctx, tmp_path / "out", tmp_path / "qa")
    text = (tmp_path / "qa" / "qa_report.md").read_text()
    assert ok is False and re.search(r"^FAIL county_sums: ", text, re.M)
    assert (tmp_path / "out" / "county_civic.csv").exists()


def test_safeguards_catch_a_bad_county_csv(tmp_path):
    (tmp_path / "county_civic.csv").write_text('"wrong"\nowner_street,HFB\n1,2\n')
    fails = safeguards.run_all(tmp_path)["csv_county"]
    joined = " ".join(fails)
    assert "row 1" in joined and "forbidden column" in joined and "banned word" in joined
