"""Safeguard checks: plain functions returning lists of failures (empty = pass).

The same functions drive the QA PASS/FAIL block, so these tests are what "safeguards green"
means in the QA report.
"""

import csv

import pytest

from hfb import safeguards as sg
from hfb.copy import MISUSE


def test_constants_are_the_briefed_lists():
    assert set(sg.FORBIDDEN_COLUMNS) == {
        "NAME", "ICO", "STREET", "Spanish", "income", "poverty", "GEOID20", "CTBKEY"}
    assert set(sg.BANNED_WORDS) == {
        "disengaged", "civic desert", "underperforming", "Houston Food Bank", "HFB"}
    assert set(sg.ALLOWED_OUT) == {
        "zcta_civic.csv", "zcta_civic.xlsx", "zcta_p1_map.png", "zcta_civic.geojson"}


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
