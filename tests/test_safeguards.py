"""Safeguard checks: plain functions returning lists of failures (empty = pass).

The same functions drive the QA PASS/FAIL block, so these tests are what "safeguards green"
means in the QA report.
"""

import csv

import openpyxl
import pytest
from openpyxl.styles import PatternFill
from test_assemble import synth_table

from hfb import assemble
from hfb import safeguards as sg
from hfb.copy import MISUSE, REQUIRED_LINES


def test_constants_are_the_briefed_lists():
    assert set(sg.FORBIDDEN_COLUMNS) == {
        "NAME", "ICO", "STREET", "Spanish", "income", "poverty", "GEOID20", "CTBKEY"}
    assert set(sg.BANNED_WORDS) == {
        "disengaged", "civic desert", "underperforming", "Houston Food Bank", "HFB"}
    # pinned to the brief / global-constraints values, in order
    assert sg.FORBIDDEN_COLUMNS == (
        "NAME", "ICO", "STREET", "Spanish", "income", "poverty", "GEOID20", "CTBKEY")
    assert sg.BANNED_WORDS == (
        "disengaged", "civic desert", "underperforming", "Houston Food Bank", "HFB")
    assert set(sg.ALLOWED_OUT) == {
        "zcta_civic.csv", "zcta_civic.xlsx", "zcta_p1_map.png", "zcta_civic.geojson",
        "county_civic.csv"}


@pytest.mark.parametrize("cols", [["ZCTA", "P1"], []])
def test_forbidden_columns_clean(cols):
    assert sg.check_forbidden_columns(cols) == []


@pytest.mark.parametrize("bad", [
    "NAME", "name", "owner_street", "Spanish_Surname_x", "SPANISH", "Income_med",
    "poverty_rate", "GEOID20", "ctbkey", "ICO", "ico_code"])
def test_forbidden_columns_caught_case_insensitive_substring(bad):
    fails = sg.check_forbidden_columns(["ZCTA", bad])
    assert len(fails) == 1 and bad in fails[0]


@pytest.mark.parametrize("text", [
    "a disengaged tract", "DISENGAGED", "Civic Desert",
    "Underperforming", "Houston Food Bank", "houston food bank", "HFB", "hfb", "Hfb.",
    "see hfb.turnout", "(HFB)"])
def test_banned_words_caught(text):
    fails = sg.check_banned_words({"doc": text})
    assert len(fails) == 1 and "doc" in fails[0]


@pytest.mark.parametrize("text", [
    "a Houston-area food bank", "chfbx", "hfb_x", "engaged", "desert", "food bank",
    "P1_rank 100 = highest"])
def test_banned_words_not_overmatched(text):
    assert sg.check_banned_words({"doc": text}) == []


def test_banned_word_failure_does_not_echo_the_word():
    (msg,) = sg.check_banned_words({"qa_report": "x HFB y"})
    assert "hfb" not in msg.lower()


def test_out_dir_allowlist(tmp_path):
    assert sg.check_out_dir(tmp_path) == []
    (tmp_path / "zcta_civic.csv").write_text("x")
    (tmp_path / "zcta_p1_map.png").write_text("x")
    (tmp_path / "county_civic.csv").write_text("x")
    assert sg.check_out_dir(tmp_path) == []
    (tmp_path / "blocks.parquet").write_text("x")
    (tmp_path / "zcta_county.csv").write_text("x")
    fails = sg.check_out_dir(tmp_path)
    assert len(fails) == 1 and "blocks.parquet" in fails[0] and "zcta_county.csv" in fails[0]


def test_out_dir_subdirectory_is_a_failure(tmp_path):
    (tmp_path / "sub").mkdir()
    assert sg.check_out_dir(tmp_path)


def _write_csv(path, first, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f, quoting=csv.QUOTE_ALL).writerow(first)
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_csv_row1_check(tmp_path):
    ok = tmp_path / "ok.csv"
    _write_csv(ok, [MISUSE], ["ZCTA", "not_ranked"], [["77001", "FALSE"]])
    assert sg.check_csv_row1(ok) == []
    bad = tmp_path / "bad.csv"
    _write_csv(bad, [MISUSE + "!"], ["ZCTA"], [])
    assert sg.check_csv_row1(bad)
    two = tmp_path / "two.csv"
    _write_csv(two, [MISUSE, "x"], ["ZCTA"], [])
    assert sg.check_csv_row1(two)
    unq = tmp_path / "unq.csv"
    unq.write_text(MISUSE.replace(",", ";") + "\nZCTA\n")
    assert sg.check_csv_row1(unq)  # not quoted


def test_blank_rank_check(tmp_path):
    cols = ["ZCTA", "P1_rank", "rank_p05", "rank_p95", "not_ranked"]
    p = tmp_path / "a.csv"
    _write_csv(p, [MISUSE], cols, [["77001", "50.0", "40.0", "60.0", "FALSE"],
                                    ["77002", "", "", "", "TRUE"]])
    assert sg.check_not_ranked_blank(p) == []
    _write_csv(p, [MISUSE], cols, [["77002", "5.0", "", "", "TRUE"]])
    assert sg.check_not_ranked_blank(p)


def test_csv_sorted_check(tmp_path):
    cols = ["ZCTA", "not_ranked"]
    p = tmp_path / "a.csv"
    _write_csv(p, [MISUSE], cols, [["77001", "FALSE"], ["77003", "FALSE"], ["77002", "TRUE"]])
    assert sg.check_csv_sorted(p) == []
    _write_csv(p, [MISUSE], cols, [["77002", "TRUE"], ["77001", "FALSE"]])
    assert sg.check_csv_sorted(p)
    _write_csv(p, [MISUSE], cols, [["77003", "FALSE"], ["77001", "FALSE"]])
    assert sg.check_csv_sorted(p)


def test_run_all_reports_missing_files(tmp_path):
    res = sg.run_all(tmp_path)
    assert set(res) >= {"out_dir", "csv", "xlsx"}
    assert any(res.values())  # missing csv/xlsx are failures, not silent passes


# ---------- negative tests on written files ----------

@pytest.fixture
def good(tmp_path):
    """A clean synthetic build in tmp; every mutation below starts from this."""
    assemble.write_outputs(synth_table(), tmp_path)
    return tmp_path


def _wb(out):
    return openpyxl.load_workbook(out / "zcta_civic.xlsx")


def _row_of(ws, nr: bool):
    """First data row of the sheet whose not_ranked cell is ``nr``."""
    col = [c.value for c in ws[2]].index("not_ranked")
    return next(r for r in range(3, ws.max_row + 1) if ws.cell(r, col + 1).value is nr)


def test_good_build_is_clean(good):
    assert all(v == [] for v in sg.run_all(good).values())
    wb = _wb(good)
    assert sg.check_read_me(wb) == [] and sg.check_xlsx_data(wb) == []


def test_xlsx_a1_not_misuse(good):
    wb = _wb(good)
    wb["data"]["A1"] = "something else"
    assert any("A1" in f for f in sg.check_xlsx_data(wb))


def test_xlsx_numeric_zcta(good):
    wb = _wb(good)
    ws = wb["data"]
    ws["A3"] = 77001  # a number, not text
    assert any("text" in f for f in sg.check_xlsx_data(wb))


def test_xlsx_numeric_county_main(good):
    wb = _wb(good)
    wb["data"]["B3"] = 48201
    assert any("text" in f for f in sg.check_xlsx_data(wb))


def test_xlsx_not_ranked_row_without_grey(good):
    wb = _wb(good)
    ws = wb["data"]
    ws.cell(_row_of(ws, True), 5).fill = PatternFill(fill_type=None)
    assert any("grey" in f for f in sg.check_xlsx_data(wb))


def test_xlsx_ranked_row_with_grey(good):
    wb = _wb(good)
    ws = wb["data"]
    ws.cell(_row_of(ws, False), 5).fill = PatternFill("solid", start_color="BDBDBD",
                                                      end_color="BDBDBD")
    assert any("grey" in f for f in sg.check_xlsx_data(wb))


def test_xlsx_wrong_grey(good):
    wb = _wb(good)
    ws = wb["data"]
    r = _row_of(ws, True)
    for c in ws[r]:
        c.fill = PatternFill("solid", start_color="CCCCCC", end_color="CCCCCC")
    assert any("grey" in f for f in sg.check_xlsx_data(wb))


def test_xlsx_not_ranked_row_not_last(good):
    wb = _wb(good)
    ws = wb["data"]
    col = [c.value for c in ws[2]].index("not_ranked") + 1
    ws.cell(_row_of(ws, False), col).value = True  # a NOT_RANKED flag before ranked rows
    assert any("last" in f or "grey" in f for f in sg.check_xlsx_data(wb))


@pytest.mark.parametrize("needed", [*REQUIRED_LINES, MISUSE, sg.DATA_STARTS])
def test_read_me_missing_required_string(good, needed):
    wb = _wb(good)
    for row in wb["read_me"].iter_rows():
        for c in row:
            if c.value == needed:
                c.value = None
    fails = sg.check_read_me(wb)
    assert len(fails) == 1 and needed[:30] in fails[0]


def test_not_ranked_row_with_p1_fails(tmp_path):
    cols = ["ZCTA", "P1", "P1_rank", "rank_p05", "rank_p95", "not_ranked",
            "P1_unranked_reference"]
    p = tmp_path / "a.csv"
    ok = [["77001", "0.5", "50.0", "40.0", "60.0", "FALSE", ""],
          ["77002", "", "", "", "", "TRUE", "0.9"]]
    _write_csv(p, [MISUSE], cols, ok)
    assert sg.check_not_ranked_blank(p) == []
    _write_csv(p, [MISUSE], cols, [ok[0], ["77002", "0.9", "", "", "", "TRUE", ""]])
    fails = sg.check_not_ranked_blank(p)
    assert len(fails) == 1 and "P1" in fails[0]
    _write_csv(p, [MISUSE], cols, [["77001", "0.5", "50.0", "40.0", "60.0", "FALSE", "0.5"]])
    fails = sg.check_not_ranked_blank(p)
    assert len(fails) == 1 and "P1_unranked_reference" in fails[0]


def test_blank_rank_on_ranked_row(tmp_path):
    cols = ["ZCTA", "P1_rank", "rank_p05", "rank_p95", "not_ranked"]
    p = tmp_path / "a.csv"
    _write_csv(p, [MISUSE], cols, [["77001", "", "40.0", "60.0", "FALSE"]])
    fails = sg.check_not_ranked_blank(p)
    assert len(fails) == 1 and "blank" in fails[0]


def test_run_all_catches_banned_word_in_csv(good):
    p = good / "zcta_civic.csv"
    p.write_text(p.read_text().replace("FACILITY_CHECK", "disengaged"))
    res = sg.run_all(good)
    assert any("banned word" in f for f in res["csv"]) and res["xlsx"] == []


def test_run_all_catches_banned_word_in_xlsx_either_sheet(good):
    for sheet, hit in (("data", "FACILITY_CHECK"), ("read_me", "Credits")):
        wb = _wb(good)
        for row in wb[sheet].iter_rows():
            for c in row:
                if c.value == hit:
                    c.value = "HFB"  # whole word, any case is banned
        wb.save(good / "zcta_civic.xlsx")
        res = sg.run_all(good)
        assert any("banned word" in f and sheet in f for f in res["xlsx"]), sheet
        assemble.write_outputs(synth_table(), good)  # restore


def test_run_all_catches_forbidden_column_in_csv(good):
    p = good / "zcta_civic.csv"
    p.write_text(p.read_text().replace("college_share", "college_income"))
    assert any("forbidden column" in f for f in sg.run_all(good)["csv"])


def test_run_all_catches_forbidden_column_in_xlsx(good):
    wb = _wb(good)
    wb["data"]["X2"] = "Spanish_Surname_Pct"  # header cell of the last column
    wb.save(good / "zcta_civic.xlsx")
    assert any("forbidden column" in f for f in sg.run_all(good)["xlsx"])


def test_run_all_catches_extra_file_in_out(good):
    (good / "voters.csv").write_text("x")
    assert any("voters.csv" in f for f in sg.run_all(good)["out_dir"])
