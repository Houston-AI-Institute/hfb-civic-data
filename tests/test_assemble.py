"""Task 6: CSV/XLSX hand-off files and the internal QA report.

The eight named checks run twice: on a small synthetic table (default run) and on the full
real build (``-m real``). Nothing here writes to the repo's ``out/`` or ``qa/``.
"""

import csv
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openpyxl
import pandas as pd
import pytest

from hfb import assemble, pipeline, safeguards
from hfb.copy import CREDITS, MAP_FOOTER, MISUSE, P1_FOOTNOTE, RANK_DIRECTION, REQUIRED_LINES, USE_LIMIT

ROOT = Path(__file__).resolve().parent.parent
COLS = ["ZCTA", "county_main", "land_share", "inreg_share20", "acs_pop", "reg22", "to22",
        "reg24", "to24", "to_reg24", "cvap", "cvap_moe", "corr_adult", "cvap_adj", "P1",
        "P1_rank", "rank_p05", "rank_p95", "not_ranked", "not_ranked_reason", "caution",
        "split_exposure", "gq_share", "college_share"]
INT_COLS = ["acs_pop", "reg22", "to22", "reg24", "to24", "cvap", "cvap_moe", "corr_adult",
            "cvap_adj"]
DP4 = ["land_share", "inreg_share20", "to_reg24", "P1", "split_exposure", "gq_share",
       "college_share"]
DP1 = ["P1_rank", "rank_p05", "rank_p95"]


# ---------- synthetic table ----------

def synth_table() -> pd.DataFrame:
    """Six ZCTAs. 77004 LAND, 77003 CORRECTIONAL (P1 > 1, measured), 77032 ranked with
    corr_adult > 0, 77002 CORRECTIONAL caution. 00501 tests a leading zero."""
    zs = ["00501", "77001", "77002", "77003", "77032", "77004"]
    ix = pd.Index(zs, name="ZCTA")
    n = len(zs)
    region = pd.DataFrame({"land_share": 0.9, "county_main": "48201"}, index=ix)
    region.loc["77004", "land_share"] = 0.3
    region.loc["00501", "county_main"] = "48039"
    acs = pd.DataFrame({"acs_pop": 5000.0, "cvap": 3000.0, "cvap_moe": 200.0,
                        "acs_vap": 3900.0}, index=ix)
    corr = pd.Series([0, 0, 200, 500, 100, 0], index=ix, dtype=float)
    zpop = pd.DataFrame({"total": 5000, "vap": 4000, "gq": 10, "corr_adult": corr,
                         "college": 0, "vap_adj": 4000}, index=ix)
    a22 = pd.DataFrame({"reg22": 2000.4, "to22": 900.0 + 130 * np.arange(n)}, index=ix)
    a24 = pd.DataFrame({"reg24": 2200.6, "to24": 1000.0 + 130 * np.arange(n)}, index=ix)
    a22.loc["77003", "to22"] = 3600.0
    a24.loc["77003", "to24"] = 3700.0
    a22.loc["NONE"] = [5.0, 5.0]
    a24.loc["NONE"] = [5.0, 5.0]
    split = pd.Series(0.123456, index=ix)
    inreg = pd.Series(0.99, index=ix)
    return pipeline.assemble(region, acs, zpop, a22, a24, split, inreg)


def synth_ctx(table) -> assemble.QAContext:
    tlc = [dict(yy=yy, reg=1000.0 * yy, to=500.0 * yy, tlc_reg=1000.0 * yy, tlc_to=500.0 * yy,
                max_diff=0.0, none_reg=5.0, none_to=5.0) for yy in (22, 24)]
    return assemble.QAContext(commit="abc1234", seconds=1.5, tlc=tlc, sensitivity=None,
                              sensitivity_note="skipped in unit test")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    t = synth_table()
    base = tmp_path_factory.mktemp("synth")
    out, qa = base / "out", base / "qa"
    ok = assemble.emit(t, synth_ctx(t), out, qa)
    return SimpleNamespace(out=out, qa=qa, ok=ok, table=t, n=len(t))


@pytest.fixture(params=["synthetic", pytest.param("real", marks=pytest.mark.real)])
def built(request):
    if request.param == "synthetic":
        s = request.getfixturevalue("synth")
        return SimpleNamespace(out=s.out, qa=s.qa, n=s.n, kind="synthetic")
    out, qa, _ = request.getfixturevalue("real_outputs")
    return SimpleNamespace(out=out, qa=qa, n=240, kind="real")


def read_csv(built, strings=False, **kw):
    """The CSV as the brief's readers load it; ``strings`` keeps every field as raw text."""
    dtype = str if strings else {"ZCTA": str, "county_main": str}
    return pd.read_csv(built.out / "zcta_civic.csv", skiprows=1, dtype=dtype, **kw)


# ---------- the eight named tests ----------

def test_csv_row1_is_misuse_statement(built):
    path = built.out / "zcta_civic.csv"
    with open(path, newline="") as f:
        first = next(csv.reader(f))
    assert first == [MISUSE]
    assert path.read_text().splitlines()[0] == '"' + MISUSE + '"'  # fully quoted


def test_csv_reads_with_skiprows(built):
    df = read_csv(built)
    assert len(df) == built.n
    assert list(df.columns) == COLS
    assert df.ZCTA.str.len().eq(5).all()
    assert df.county_main.str.len().eq(5).all()


def test_out_dir_allowlist(built):
    names = {p.name for p in built.out.iterdir()}
    assert names <= {"zcta_civic.csv", "zcta_civic.xlsx", "zcta_p1_map.png", "zcta_civic.geojson"}
    assert {"zcta_civic.csv", "zcta_civic.xlsx"} <= names
    repo_out = ROOT / "out"
    if repo_out.exists():
        assert {p.name for p in repo_out.iterdir()} <= safeguards.ALLOWED_OUT


def test_no_forbidden_columns(built):
    bad = [c for c in read_csv(built).columns
           if any(f.lower() in c.lower() for f in ["NAME", "ICO", "STREET", "Spanish", "income",
                                                   "poverty", "GEOID20", "CTBKEY"])]
    assert bad == []
    wb = openpyxl.load_workbook(built.out / "zcta_civic.xlsx", read_only=True)
    header = [c for c in next(wb["data"].iter_rows(min_row=2, max_row=2, values_only=True))]
    assert header == COLS


def _all_text(built):
    parts = {"csv": (built.out / "zcta_civic.csv").read_text(),
             "qa": (built.qa / "qa_report.md").read_text()}
    wb = openpyxl.load_workbook(built.out / "zcta_civic.xlsx")
    for ws in wb:
        parts["xlsx:" + ws.title] = "\n".join(
            str(c.value) for row in ws.iter_rows() for c in row if c.value is not None)
    return parts


def test_no_banned_words(built):
    for where, text in _all_text(built).items():
        low = text.lower()
        for w in ["disengaged", "civic desert", "underperforming", "houston food bank"]:
            assert w not in low, (where, w)
        assert not re.search(r"\bhfb\b", low), where


def test_required_lines_present(built):
    wb = openpyxl.load_workbook(built.out / "zcta_civic.xlsx")
    cells = [str(c.value) for row in wb["read_me"].iter_rows() for c in row if c.value]
    for line in REQUIRED_LINES:
        assert line in cells
    for s in (MISUSE, USE_LIMIT, RANK_DIRECTION, P1_FOOTNOTE, *CREDITS, "Data starts on row 2"):
        assert s in cells, s


def test_not_ranked_rows_have_blank_rank(built):
    df = read_csv(built, strings=True, keep_default_na=False)
    nr = df[df.not_ranked == "TRUE"]
    assert len(nr) > 0
    assert (nr[DP1] == "").all().all()
    ranked = df[df.not_ranked == "FALSE"]
    assert (ranked[DP1] != "").all().all()


def test_not_ranked_sorted_last_and_grey(built):
    df = read_csv(built, strings=True)
    nr = (df.not_ranked == "TRUE").to_numpy()
    assert not (nr[:-1] & ~nr[1:]).any(), "a ranked row follows a NOT_RANKED row"
    for part in (df[~nr], df[nr]):
        assert part.ZCTA.is_monotonic_increasing
    ws = openpyxl.load_workbook(built.out / "zcta_civic.xlsx")["data"]
    for i, is_nr in enumerate(nr):
        for c in ws[3 + i]:
            if is_nr:
                assert c.fill.fill_type == "solid" and c.fill.fgColor.rgb.endswith("BDBDBD")
            else:
                assert c.fill.fill_type is None


# ---------- rulings 2-4, 7: format ----------

def test_integer_and_rounded_columns(built):
    df = read_csv(built, strings=True, keep_default_na=False)
    for c in INT_COLS:
        assert not df[c].str.contains(r"\.").any(), c
        assert df[c].str.fullmatch(r"-?\d*").all(), c
    for cols, dp in ((DP4, 4), (DP1, 1)):
        for c in cols:
            frac = df[c].str.extract(r"\.(\d+)$")[0].dropna()
            assert (frac.str.len() <= dp).all(), c
    assert set(df.not_ranked) <= {"TRUE", "FALSE"}
    assert df.not_ranked_reason.map(lambda s: " " not in s).all()


def test_xlsx_text_cells_and_layout(built):
    wb = openpyxl.load_workbook(built.out / "zcta_civic.xlsx")
    assert wb.sheetnames == ["data", "read_me"]
    ws = wb["data"]
    assert ws["A1"].value == MISUSE
    assert [c.value for c in ws[2]] == COLS
    for r in range(3, 3 + built.n):
        for col in ("A", "B"):
            c = ws[f"{col}{r}"]
            assert c.data_type == "s" and isinstance(c.value, str) and len(c.value) == 5
    assert ws[f"A{3 + built.n}"].value is None
    # blanks stay blank, not 'nan' or 0
    p1rank = COLS.index("P1_rank") + 1
    vals = [ws.cell(r, p1rank).value for r in range(3, 3 + built.n)]
    assert any(v is None for v in vals) and all(v is None or isinstance(v, (int, float)) for v in vals)
    nrcol = COLS.index("not_ranked") + 1
    assert {ws.cell(r, nrcol).value for r in range(3, 3 + built.n)} <= {True, False}


def test_read_me_defines_every_column_and_notes_estimates(built):
    wb = openpyxl.load_workbook(built.out / "zcta_civic.xlsx")
    rows = [[c.value for c in row] for row in wb["read_me"].iter_rows()]
    defined = {r[0]: r[1] for r in rows if len(r) > 1 and r[1] and r[0] in COLS}
    assert set(defined) == set(COLS)
    assert all(len(d) > 10 for d in defined.values())
    text = "\n".join(str(v) for r in rows for v in r if v)
    assert "estimate" in text.lower()
    assert MAP_FOOTER not in text  # the map footer belongs to the map, not the table


def test_column_defs_cover_columns():
    assert set(assemble.COLUMN_DEFS) == set(assemble.COLUMNS) == set(COLS)
    assert list(assemble.COLUMNS) == COLS


def test_format_table_sorts_and_types():
    t = synth_table()
    f = assemble.format_table(t)
    assert list(f.columns) == COLS
    assert f.not_ranked.tolist() == sorted(f.not_ranked.tolist())
    assert f.reg22.dtype == "Int64" and f.reg22.iloc[0] == 2000
    assert f.loc[f.ZCTA == "77004", "P1"].isna().all()
    assert (f.P1_rank.dropna().round(1) == f.P1_rank.dropna()).all()


def test_nan_count_stays_blank_not_zero():
    t = synth_table()
    t.loc["77001", "cvap"] = np.nan
    t.loc["77001", "cvap_adj"] = np.nan
    f = assemble.format_table(t)
    assert pd.isna(f.set_index("ZCTA").loc["77001", "cvap_adj"])


# ---------- QA report ----------

def test_qa_report_header_and_order(built):
    text = (built.qa / "qa_report.md").read_text()
    lines = text.splitlines()
    assert lines[0] == "internal - not for teams or slides"
    assert re.search(r"^Build commit: \S+$", text, re.M)
    assert re.search(r"^Build duration: [\d.]+ s$", text, re.M)
    assert not re.search(r"20\d\d-\d\d-\d\d", text), "no wall-clock date"
    i_block = text.index("OVERALL:")
    i_eye = text.index("Eyeball")
    i_app = text.index("Appendix")
    assert i_block < i_eye < i_app


def test_qa_block_lines(built):
    text = (built.qa / "qa_report.md").read_text()
    lines = [ln for ln in text.splitlines() if re.match(r"^(PASS|FAIL) \w+: ", ln)]
    names = [re.match(r"^(?:PASS|FAIL) (\w+):", ln).group(1) for ln in lines]
    assert names == ["tlc_totals", "row_count", "none_registration", "ranked_count",
                     "p1_gt_1_cvap_adj_le_0", "safeguards"]
    assert re.search(r"^OVERALL: (PASS|FAIL)$", text, re.M)


def test_eyeball_items_present(built):
    text = (built.qa / "qa_report.md").read_text()
    for h in ("PARTIAL", "CORRECTIONAL", "Highest 10", "Lowest 10", "NONE bucket",
              "corr_adult > 0", "Counts by flag"):
        assert h in text, h
    assert "| ZCTA |" in text  # markdown tables


def test_corr_adult_table_has_77032_first_and_both_p1s():
    t = synth_table()
    sec = assemble.corr_adult_section(t)
    rows = [ln for ln in sec.splitlines() if re.match(r"^\| \d{5} ", ln)]
    assert rows[0].startswith("| 77032 ")
    assert [r.split("|")[1].strip() for r in rows] == ["77032", "77002"]
    # 77032: with subtraction (to22+to24)/(2*(3000-100)); without /(2*3000)
    r = t.loc["77032"]
    with_ = (r.to22 + r.to24) / (2 * (r.cvap - r.corr_adult))
    without = (r.to22 + r.to24) / (2 * r.cvap)
    assert f"{with_:.4f}" in rows[0] and f"{without:.4f}" in rows[0]


def test_highest_lowest_cover_ranked_rows_only():
    t = synth_table()
    sec = assemble.extremes_section(t)
    assert "77003" not in sec and "77004" not in sec  # NOT_RANKED never appear
    assert "77001" in sec


def test_p1_over_1_listed_with_reason_and_still_passes():
    t = synth_table()
    checks, lists = assemble.qa_checks(t, synth_ctx(t))
    c = {name: (ok, detail) for name, ok, detail in checks}
    assert c["p1_gt_1_cvap_adj_le_0"][0] is True  # 77003 is NOT_RANKED
    flat = "\n".join(lists)
    assert "77003" in flat and "CORRECTIONAL" in flat


def test_p1_over_1_ranked_row_fails_and_is_listed():
    t = synth_table()
    t.loc["77001", "P1"] = 1.2  # a ranked row above 1
    checks, lists = assemble.qa_checks(t, synth_ctx(t))
    ok, detail = {n: (o, d) for n, o, d in checks}["p1_gt_1_cvap_adj_le_0"]
    assert ok is False and "77001" in detail


def test_cvap_adj_nonpositive_ranked_row_fails():
    t = synth_table()
    t.loc["77002", "cvap_adj"] = 0.0
    checks, _ = assemble.qa_checks(t, synth_ctx(t))
    assert {n: o for n, o, _ in checks}["p1_gt_1_cvap_adj_le_0"] is False


def test_failing_check_still_writes_outputs_and_reports_fail(synth):
    # synthetic table has 6 rows, so row_count and ranked_count fail by construction
    assert synth.ok is False
    text = (synth.qa / "qa_report.md").read_text()
    assert re.search(r"^FAIL row_count: ", text, re.M)
    assert re.search(r"^FAIL ranked_count: ", text, re.M)
    assert re.search(r"^PASS none_registration: ", text, re.M)
    assert text.count("\nOVERALL: FAIL") == 1
    assert (synth.out / "zcta_civic.csv").exists() and (synth.out / "zcta_civic.xlsx").exists()


def test_tlc_and_none_checks_follow_context():
    t = synth_table()
    ctx = synth_ctx(t)
    ctx.tlc[1]["max_diff"] = 3.0
    ctx.tlc[0]["none_reg"] = 51.0
    checks, _ = assemble.qa_checks(t, ctx)
    c = {n: o for n, o, _ in checks}
    assert c["tlc_totals"] is False and c["none_registration"] is False


def test_qa_failing_safeguard_is_fail(tmp_path):
    t = synth_table()
    out, qa = tmp_path / "out", tmp_path / "qa"
    out.mkdir()
    (out / "blocks.parquet").write_text("x")  # not allowed in out/
    ok = assemble.emit(t, synth_ctx(t), out, qa)
    text = (qa / "qa_report.md").read_text()
    assert ok is False
    assert re.search(r"^FAIL safeguards: ", text, re.M) and "blocks.parquet" in text


def test_banned_word_in_report_text_is_caught(tmp_path):
    t = synth_table()
    ctx = synth_ctx(t)
    ctx.commit = "HFB"
    ok = assemble.emit(t, ctx, tmp_path / "out", tmp_path / "qa")
    text = (tmp_path / "qa" / "qa_report.md").read_text()
    assert ok is False and re.search(r"^FAIL safeguards: ", text, re.M)


def test_sensitivity_section_skipped_note_and_stats():
    assert "skipped in unit test" in assemble.appendix_section(None, "skipped in unit test")
    ix = pd.Index([f"7700{i}" for i in range(6)], name="ZCTA")
    s = pd.DataFrame({"rank_main": [10., 30, 50, 70, 90, 100],
                      "rank_total": [10., 50, 30, 70, 100, 90],
                      "P1": 0.5, "P1_total": 0.5}, index=ix)
    text = assemble.appendix_section(s, "")
    assert "Spearman" in text and "0.886" in text  # 1 - 6*4/(6*35)
    assert re.search(r"Median absolute rank shift: [\d.]+", text)
    assert re.search(r"Maximum absolute rank shift: 20\.0", text)


def test_cli_exit_code_follows_overall(monkeypatch, tmp_path):
    monkeypatch.setattr(assemble, "main", lambda raw, out_dir, qa_dir: False)
    with pytest.raises(SystemExit) as e:
        assemble.cli(["x", "--out", str(tmp_path), "--qa", str(tmp_path)])
    assert e.value.code == 1
    monkeypatch.setattr(assemble, "main", lambda raw, out_dir, qa_dir: True)
    assert assemble.cli(["x"]) in (0, None)


# ---------- real build ----------

@pytest.mark.real
def test_real_overall_pass_and_counts(real_outputs):
    out, qa, ok = real_outputs
    text = (qa / "qa_report.md").read_text()
    assert ok is True, text.split("Eyeball")[0]
    assert re.search(r"^OVERALL: PASS$", text, re.M)
    assert not re.search(r"^FAIL ", text, re.M)
    df = pd.read_csv(out / "zcta_civic.csv", skiprows=1, dtype={"ZCTA": str, "county_main": str})
    assert len(df) == 240 and 170 <= (df.not_ranked == False).sum() <= 180  # noqa: E712


@pytest.mark.real
def test_real_sensitivity_appendix_present(real_outputs):
    _, qa, _ = real_outputs
    text = (qa / "qa_report.md").read_text()
    assert "Spearman" in text and "Maximum absolute rank shift" in text
