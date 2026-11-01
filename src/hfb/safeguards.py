"""Output safeguards: plain checks that each return a list of failure messages (empty = pass).

Used by the tests and by the QA PASS/FAIL block, so "safeguards green" means these functions
returned no failures on that build. A failure message never repeats a banned word.
"""

import csv
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import openpyxl
import pandas as pd

from .copy import (CONDITIONS_NOTE, CREDITS, MISUSE, P1_FOOTNOTE, RANK_DIRECTION,
                   REQUIRED_LINES, USE_LIMIT)

BANNED_WORDS = ("disengaged", "civic desert", "underperforming", "Houston Food Bank", "HFB")
WHOLE_WORD = {"HFB"}  # matched as a whole word only; the rest as substrings
FORBIDDEN_COLUMNS = ("NAME", "ICO", "STREET", "Spanish", "income", "poverty", "GEOID20", "CTBKEY")
ALLOWED_OUT = frozenset({"zcta_civic.csv", "zcta_civic.xlsx", "zcta_p1_map.png",
                         "zcta_civic.geojson", "county_civic.csv"})
DATA_STARTS = "Data starts on row 2"
GREY = "BDBDBD"
RANK_COLS = ("P1_rank", "rank_p05", "rank_p95")
TEXT_COLS = ("ZCTA", "county_main")
CSV_NAME, XLSX_NAME = "zcta_civic.csv", "zcta_civic.xlsx"
GEOJSON_NAME = "zcta_civic.geojson"
COUNTY_CSV_NAME, COUNTY_SHEET = "county_civic.csv", "county"  # sheet name = county.SHEET

_BANNED_RE = [
    re.compile(r"\b" + re.escape(w) + r"\b" if w in WHOLE_WORD else re.escape(w), re.I)
    for w in BANNED_WORDS]


def check_forbidden_columns(columns: Iterable[str]) -> list[str]:
    """Column names containing a forbidden fragment (case-insensitive substring)."""
    fails = []
    for c in columns:
        hit = [f for f in FORBIDDEN_COLUMNS if f.lower() in str(c).lower()]
        if hit:
            fails.append(f"forbidden column {c!r} (matches {hit})")
    return fails


def check_banned_words(texts: Mapping[str, str]) -> list[str]:
    """One failure per (document, line) holding a banned word; the word itself is not echoed."""
    fails = []
    for where, text in texts.items():
        for n, line in enumerate(str(text).splitlines(), 1):
            if any(rx.search(line) for rx in _BANNED_RE):
                fails.append(f"banned word in {where}, line {n}")
    return fails


def check_out_dir(out_dir) -> list[str]:
    """Everything in ``out_dir`` must be an allowlisted file name."""
    extra = sorted(p.name for p in Path(out_dir).iterdir() if p.name not in ALLOWED_OUT)
    return [f"not allowed in out/: {extra}"] if extra else []


def check_csv_row1(path) -> list[str]:
    """Row 1 is exactly one fully quoted field holding MISUSE."""
    with open(path, newline="") as f:
        raw = f.readline().rstrip("\r\n")
    with open(path, newline="") as f:
        first = next(csv.reader(f), [])
    fails = []
    if first != [MISUSE]:
        fails.append(f"csv row 1 is not the single MISUSE field (got {len(first)} field(s))")
    if raw != '"' + MISUSE + '"':
        fails.append("csv row 1 is not fully quoted")
    return fails


def check_county_csv(path) -> list[str]:
    """county_civic.csv: row 1 is MISUSE, no forbidden column, no banned word."""
    return (check_csv_row1(path) + check_forbidden_columns(_read(path).columns)
            + check_banned_words({COUNTY_CSV_NAME: Path(path).read_text()}))


def check_county_sheet(wb) -> list[str]:
    """xlsx ``county`` sheet: MISUSE in A1, header on row 2, no forbidden column, text FIPS."""
    if COUNTY_SHEET not in wb.sheetnames:
        return [f"xlsx has no {COUNTY_SHEET} sheet"]
    ws = wb[COUNTY_SHEET]
    fails = []
    if ws["A1"].value != MISUSE:
        fails.append(f"xlsx {COUNTY_SHEET} A1 is not MISUSE")
    fails += check_forbidden_columns(c.value for c in ws[2] if c.value)
    bad = [r[0].value for r in ws.iter_rows(min_row=3) if r[0].value is not None
           and r[0].data_type != "s"]
    if bad:
        fails.append(f"xlsx {COUNTY_SHEET} county_fips not stored as text: {bad[:10]}")
    return fails


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1, dtype=str, keep_default_na=False)


def check_not_ranked_blank(path) -> list[str]:
    """NOT_RANKED rows have blank P1_rank, rank_p05 and rank_p95 (ranked rows do not). Where
    the columns exist, NOT_RANKED rows have a blank P1 and ranked rows a blank
    P1_unranked_reference."""
    df = _read(path)
    nr = df.not_ranked == "TRUE"
    bad = df.ZCTA[nr & (df[list(RANK_COLS)] != "").any(axis=1)].tolist()
    miss = df.ZCTA[~nr & (df[list(RANK_COLS)] == "").any(axis=1)].tolist()
    fails = []
    if bad:
        fails.append(f"NOT_RANKED rows with a rank value: {bad[:10]}")
    if miss:
        fails.append(f"ranked rows with a blank rank value: {miss[:10]}")
    if "P1" in df and (nr & (df.P1 != "")).any():
        fails.append(f"NOT_RANKED rows with a P1 value: {df.ZCTA[nr & (df.P1 != '')].tolist()[:10]}")
    ref = "P1_unranked_reference"
    if ref in df and (~nr & (df[ref] != "")).any():
        fails.append(f"ranked rows with a {ref} value: {df.ZCTA[~nr & (df[ref] != '')].tolist()[:10]}")
    return fails


def check_csv_sorted(path) -> list[str]:
    """Rows sorted by not_ranked (False first), then ZCTA."""
    df = _read(path)
    key = list(zip(df.not_ranked == "TRUE", df.ZCTA))
    return [] if key == sorted(key) else ["csv rows are not sorted by not_ranked, then ZCTA"]


def check_read_me(wb) -> list[str]:
    """The read_me sheet holds every required string and the data-start line."""
    if "read_me" not in wb.sheetnames:
        return ["xlsx has no read_me sheet"]
    cells = {str(c.value) for row in wb["read_me"].iter_rows() for c in row if c.value}
    need = (MISUSE, USE_LIMIT, *REQUIRED_LINES, RANK_DIRECTION, P1_FOOTNOTE, CONDITIONS_NOTE,
            *CREDITS, DATA_STARTS)
    return [f"read_me lacks: {s[:50]!r}" for s in need if s not in cells]


def check_xlsx_data(wb) -> list[str]:
    """data sheet: MISUSE in A1, header row 2, text ZCTA/county_main, NOT_RANKED rows grey and last."""
    if "data" not in wb.sheetnames:
        return ["xlsx has no data sheet"]
    ws = wb["data"]
    fails = []
    if ws["A1"].value != MISUSE:
        fails.append("xlsx data A1 is not MISUSE")
    header = [c.value for c in ws[2]]
    if "not_ranked" not in header or "ZCTA" not in header:
        return fails + ["xlsx data row 2 is not the header"]
    ix = {h: i for i, h in enumerate(header)}
    flags, bad_text, bad_fill = [], [], []
    for row in ws.iter_rows(min_row=3):
        if row[0].value is None:
            continue
        nr = row[ix["not_ranked"]].value is True
        flags.append(nr)
        z = row[ix["ZCTA"]].value
        if any(row[ix[c]].data_type != "s" for c in TEXT_COLS):
            bad_text.append(z)
        grey = [c.fill.fill_type == "solid" and str(c.fill.fgColor.rgb).endswith(GREY)
                for c in row]
        if (nr and not all(grey)) or (not nr and any(c.fill.fill_type for c in row)):
            bad_fill.append(z)
    if bad_text:
        fails.append(f"xlsx ZCTA/county_main not stored as text: {bad_text[:10]}")
    if bad_fill:
        fails.append(f"xlsx grey fill does not match NOT_RANKED: {bad_fill[:10]}")
    if flags != sorted(flags):
        fails.append("xlsx NOT_RANKED rows are not last")
    return fails


def _xlsx_texts(wb) -> dict[str, str]:
    return {f"xlsx {ws.title}": "\n".join(
        str(c.value) for row in ws.iter_rows() for c in row if c.value is not None)
        for ws in wb}


def _check_geojson(path) -> list[str]:
    """Banned words anywhere in the text, forbidden fragments in any property name."""
    text = Path(path).read_text(encoding="utf-8")
    keys = {k for f in json.loads(text)["features"] for k in f["properties"]}
    return check_banned_words({GEOJSON_NAME: text}) + check_forbidden_columns(sorted(keys))


def _guard(fn, *a) -> list[str]:
    try:
        return fn(*a)
    except Exception as e:  # an unreadable output is a failure, not a crash
        return [f"{fn.__name__} could not run: {type(e).__name__}: {e}"]


def run_all(out_dir) -> dict[str, list[str]]:
    """Every output safeguard on ``out_dir``: ``{group: failures}`` for out_dir, csv, xlsx and,
    when present, csv_county (county_civic.csv, plus the xlsx county sheet under xlsx)."""
    out = Path(out_dir)
    res = {"out_dir": _guard(check_out_dir, out) if out.is_dir() else [f"{out} is not a directory"]}
    csv_p, xlsx_p = out / CSV_NAME, out / XLSX_NAME
    if not csv_p.exists():
        res["csv"] = [f"{CSV_NAME} is missing"]
    else:
        f = _guard(check_csv_row1, csv_p) + _guard(check_not_ranked_blank, csv_p)
        f += _guard(check_csv_sorted, csv_p)
        f += _guard(lambda p: check_forbidden_columns(_read(p).columns), csv_p)
        f += _guard(lambda p: check_banned_words({CSV_NAME: p.read_text()}), csv_p)
        res["csv"] = f
    if not xlsx_p.exists():
        res["xlsx"] = [f"{XLSX_NAME} is missing"]
    else:
        try:
            wb = openpyxl.load_workbook(xlsx_p)
        except Exception as e:
            res["xlsx"] = [f"{XLSX_NAME} could not be opened: {type(e).__name__}"]
        else:
            f = _guard(check_read_me, wb) + _guard(check_xlsx_data, wb)
            f += _guard(check_banned_words, _xlsx_texts(wb))
            if "data" in wb.sheetnames:
                f += check_forbidden_columns(c.value for c in wb["data"][2] if c.value)
            res["xlsx"] = f
    cty_p = out / COUNTY_CSV_NAME
    if cty_p.exists():  # written by hfb.county via assemble; the CSV and the sheet go together
        res["csv_county"] = _guard(check_county_csv, cty_p)
        if "xlsx" in res and xlsx_p.exists():
            try:
                wb = openpyxl.load_workbook(xlsx_p)
            except Exception:
                pass  # already reported under "xlsx"
            else:
                res["xlsx"] = res["xlsx"] + _guard(check_county_sheet, wb)
    gj_p = out / GEOJSON_NAME
    if gj_p.exists():  # written by hfb.map; the same word and column rules apply
        res["geojson"] = _guard(_check_geojson, gj_p)
    return res
