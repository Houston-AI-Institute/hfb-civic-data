"""Write the hand-off files (CSV, XLSX, map PNG, GeoJSON) and the internal QA report.

``python -m hfb.assemble data/raw`` builds the 240-row table, writes ``out/zcta_civic.csv``,
``out/zcta_civic.xlsx``, ``out/zcta_p1_map.png`` and ``out/zcta_civic.geojson``, runs the output safeguards, and writes ``qa/qa_report.md``. Outputs
are written even when a QA check fails; the command then exits non-zero.
"""

import argparse
import csv
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

from . import context, crosswalk, flags, pipeline, safeguards, score, turnout
from .config import COUNTIES
from .copy import CREDITS, MISUSE, P1_FOOTNOTE, RANK_DIRECTION, REQUIRED_LINES, USE_LIMIT
from .paths import raw_path

log = logging.getLogger(__name__)

COLUMNS = [
    "ZCTA", "county_main", "land_share", "inreg_share20", "acs_pop", "reg22", "to22",
    "reg24", "to24", "to_reg24", "cvap", "cvap_moe", "corr_adult", "cvap_adj", "P1",
    "P1_rank", "rank_p05", "rank_p95", "not_ranked", "not_ranked_reason", "caution",
    "split_exposure", "gq_share", "college_share",
    *context.CONTEXT_COLS, "civic_orgs_n", "civic_orgs_band", "hoa_like_n",
]
INT_COLS = ["acs_pop", "reg22", "to22", "reg24", "to24", "cvap", "cvap_moe", "corr_adult",
            "cvap_adj", "civic_orgs_n", "hoa_like_n"]
DP4_COLS = ["land_share", "inreg_share20", "to_reg24", "P1", "split_exposure", "gq_share",
            "college_share", *context.CONTEXT_COLS]
DP1_COLS = ["P1_rank", "rank_p05", "rank_p95"]
TEXT_COLS = ["ZCTA", "county_main"]

QA_FIRST_LINE = "internal - not for teams or slides"
ROWS_EXPECTED = 240
NONE_REG_MAX = 50
RANKED_RANGE = (170, 180)
TLC_TOL = 0.5
FIRST_ZCTA = "77032"  # listed first where corr_adult > 0
REASON_CODES = ["LAND", "POP", "NO_CVAP", "GQ", "CORRECTIONAL", "PARTIAL"]
CAUTION_CODES = ["LOW_N", "GROWTH", "SPLIT", "COLLEGE", "CORRECTIONAL", "EDGE", "FACILITY_CHECK"]

COLUMN_DEFS = {
    "ZCTA": "ZIP Code Tabulation Area (2020 Census geography), 5 characters, stored as text. "
            "One row per ZCTA that touches the 10-county region.",
    "county_main": "5-digit FIPS code (text) of the region county that holds most of the "
                   "ZCTA's land.",
    "land_share": "Share of the ZCTA's land area inside the 10-county region (0 to 1).",
    "inreg_share20": "Estimated share of the ZCTA's 2020 population living inside the "
                     "10-county region (0 to 1). Blank where the ZCTA had no 2020 population.",
    "acs_pop": "Total population, ACS 2020-2024 5-year estimate. A rounded estimate.",
    "reg22": "Registered voters, 2022 general election, allocated from voting districts to "
             "the ZCTA. A rounded estimate.",
    "to22": "Ballots cast, 2022 general election, allocated from voting districts to the "
            "ZCTA. A rounded estimate.",
    "reg24": "Registered voters, 2024 general election, allocated from voting districts to "
             "the ZCTA. A rounded estimate.",
    "to24": "Ballots cast, 2024 general election, allocated from voting districts to the "
            "ZCTA. A rounded estimate.",
    "to_reg24": "to24 divided by reg24: 2024 general-election turnout as a share of "
                "registered voters.",
    "cvap": "Citizen voting-age population (native-born and naturalized citizens aged 18 and "
            "over), ACS 2020-2024 5-year estimate. A rounded estimate with sampling error.",
    "cvap_moe": "90 percent margin of error on cvap (root sum of squares of the component "
                "margins). Rounded.",
    "corr_adult": "People living in adult correctional facilities, 2020 Census group "
                  "quarters count. Subtracted from cvap in P1.",
    "cvap_adj": "cvap less corr_adult; the denominator of P1. Zero or below means P1 cannot "
                "be measured (flag NO_CVAP).",
    "P1": "Average of 2022 and 2024 general-election turnout divided by cvap_adj, that is "
          "(to22 + to24) / (2 * cvap_adj). A reference measure, not a score of a community. "
          "Above 1 is kept as measured. Blank where not_ranked_reason has LAND, PARTIAL or "
          "NO_CVAP.",
    "P1_rank": "Percentile rank of P1 among ranked ZCTAs, 0 to 100; 100 = highest. "
               "Blank for NOT_RANKED rows.",
    "rank_p05": "5th percentile of P1_rank over 1,000 simulated draws of the ACS sampling "
                "error in cvap. Blank for NOT_RANKED rows.",
    "rank_p95": "95th percentile of P1_rank over the same 1,000 draws. Blank for NOT_RANKED "
                "rows.",
    "not_ranked": "TRUE when the ZCTA is left out of the ranking (see not_ranked_reason). "
                  "These rows are shaded grey in the xlsx and sort last.",
    "not_ranked_reason": (
        "Semicolon-separated codes, blank when ranked: "
        f"LAND (land_share below {flags.LAND_MIN}); POP (acs_pop below {flags.POP_MIN}); "
        "NO_CVAP (cvap missing or cvap_adj zero or below); "
        f"GQ (group quarters above {flags.GQ_MAX:.0%} of total population); "
        f"CORRECTIONAL (corr_adult at least {flags.CORR_RANK_OUT:.0%} of voting-age "
        f"population); PARTIAL (inreg_share20 below {flags.PARTIAL_MIN})."),
    "caution": (
        "Semicolon-separated codes, blank when none: "
        f"LOW_N (reg24 below {flags.LOW_N_REG} or cvap below {flags.LOW_N_CVAP}); "
        f"GROWTH (reg24 above {flags.GROWTH_MAX} times voting-age population); "
        f"SPLIT (split_exposure above {flags.SPLIT_MAX}); "
        f"COLLEGE (college housing at least {flags.COLLEGE_MIN:.0%} of total population); "
        f"CORRECTIONAL (corr_adult {flags.CORR_CAUTION_MIN:.0%} to {flags.CORR_RANK_OUT:.0%} "
        "of voting-age population); "
        f"EDGE (inreg_share20 from {flags.PARTIAL_MIN} to {flags.EDGE_MAX}); "
        "FACILITY_CHECK (corr_adult may be a detention facility; type not yet checked)."),
    "split_exposure": "Share of the ZCTA's allocated 2024 registration that comes from voting "
                      "districts only partly inside it (district weight in the ZCTA between "
                      "0.10 and 0.90).",
    "gq_share": "Group quarters population divided by total 2020 population.",
    "college_share": "College and university housing population divided by total 2020 "
                     "population.",
    "same_house_1yr": (
        "Share of people aged 1 and over who lived in the same house one year earlier "
        "(ACS 2020-2024 5-year, table B07003), 0 to 1. Context only; not used in P1 or the "
        "rank. A sampling estimate. Blank where the table has no row for the ZCTA or its "
        "universe is zero."),
    "same_house_1yr_moe": "90 percent margin of error on same_house_1yr, as a share (0 to 1).",
    "owner_occupied": (
        "Share of occupied housing units that are owner occupied (ACS 2020-2024 5-year, "
        "table B25003), 0 to 1. Context only; not used in P1 or the rank. A sampling "
        "estimate. Blank where the table has no row for the ZCTA or its universe is zero."),
    "owner_occupied_moe": "90 percent margin of error on owner_occupied, as a share (0 to 1).",
    "hs_diploma_25plus": (
        "Share of people aged 25 and over whose highest attainment is a regular high school "
        "diploma or above: regular diploma through doctorate, so GED and some-college "
        "categories are included (ACS 2020-2024 5-year, table B15003), 0 to 1. Context only; "
        "not used in P1 or the rank. A sampling estimate. Blank where the table has no row "
        "for the ZCTA or its universe is zero."),
    "hs_diploma_25plus_moe": "90 percent margin of error on hs_diploma_25plus, as a share "
                             "(0 to 1).",
    "not_limited_english_hh": (
        "Share of households that are not limited English speaking: one minus the share that "
        "are, across all language groups (ACS 2020-2024 5-year, table C16002), 0 to 1. "
        "Context only; not used in P1 or the rank. A sampling estimate. Blank where the table "
        "has no row for the ZCTA or its universe is zero."),
    "not_limited_english_hh_moe": (
        "90 percent margin of error on not_limited_english_hh, as a share (0 to 1); the same "
        "as the margin of the limited English speaking share."),
    "broadband_hh": (
        "Share of households with a broadband internet subscription of any type (ACS "
        "2020-2024 5-year, table B28002), 0 to 1. Context only; not used in P1 or the rank. A "
        "sampling estimate. Blank where the table has no row for the ZCTA or its universe is "
        "zero."),
    "broadband_hh_moe": "90 percent margin of error on broadband_hh, as a share (0 to 1).",
    "civic_orgs_n": (
        "Count of active tax-exempt organizations registered in the ZCTA with a civic-purpose "
        "activity code (NTEE S20, S21, S22, S80, S81, S82 or W24, any subsection; R40 and R6* "
        "left out), IRS Business Master File, Texas extract. An exact count of registered "
        "organizations, not an estimate. Zero means none registered; registration is not a "
        "measure of civic life. Not used in P1 or the rank."),
    "civic_orgs_band": (
        "Text band of civic_orgs_n: 0, 1 to 2, 3 to 5 or 6 or more."),
    "hoa_like_n": (
        "Count of active registered organizations in the ZCTA that look like homeowner or "
        "property-owner associations: NTEE code starting L5, or a registered name containing "
        "homeowner, home owner, property owner, owners assoc, POA, HOA, townhome or condo "
        "(R40 and R6* left out). An exact count of registered organizations, not an estimate. "
        "An organization can be in both this count and civic_orgs_n. Names are matched during "
        "the build and are not stored in this file. Not used in P1 or the rank."),
}


# ---------- formatting and writers ----------

def format_table(t: pd.DataFrame) -> pd.DataFrame:
    """Briefed columns and order, integers as nullable Int64, rounded, sorted not_ranked then ZCTA."""
    f = t[COLUMNS].copy()
    for c in INT_COLS:
        f[c] = f[c].round().astype("Int64")
    f[DP4_COLS] = f[DP4_COLS].round(4)
    f[DP1_COLS] = f[DP1_COLS].round(1)
    f["not_ranked"] = f["not_ranked"].astype(bool)
    return f.sort_values(["not_ranked", "ZCTA"], kind="stable").reset_index(drop=True)


def write_csv(t: pd.DataFrame, path) -> None:
    """Row 1 = MISUSE as one fully quoted field, row 2 = header, then the rows."""
    f = format_table(t)
    f["not_ranked"] = f.not_ranked.map({True: "TRUE", False: "FALSE"})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh, quoting=csv.QUOTE_ALL, lineterminator="\n").writerow([MISUSE])
        f.to_csv(fh, index=False, lineterminator="\n")


def _cell(v):
    if v is None or v is pd.NA or (isinstance(v, float) and v != v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return v


def _read_me_rows() -> list[tuple]:
    rows = [("Use statement", MISUSE), ("Use limit", USE_LIMIT)]
    rows += [("Required line", s) for s in REQUIRED_LINES]
    rows += [("Rank direction", RANK_DIRECTION), ("P1", P1_FOOTNOTE)]
    rows += [("Credits", s) for s in CREDITS]
    rows += [("Layout", "Data starts on row 2"),
             ("", "Row 1 holds the use statement; row 2 is the column header and the table "
                  "starts there. Read the CSV with skiprows=1."),
             ("Counts", "Counts are rounded estimates, not exact figures, except "
                        "civic_orgs_n and hoa_like_n, which are exact counts of registered "
                        "organizations. ZCTA-level counts are not suppressed below 50, so read "
                        "small counts with care."),
             ("", ""), ("Column", "Definition")]
    return rows + [(c, COLUMN_DEFS[c]) for c in COLUMNS]


def write_xlsx(t: pd.DataFrame, path) -> None:
    """Sheet ``data`` (MISUSE in row 1, header row 2, NOT_RANKED rows filled grey) and ``read_me``."""
    f = format_table(t)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "data"
    ws.append([MISUSE])
    ws.append(COLUMNS)
    grey = PatternFill("solid", start_color="BDBDBD", end_color="BDBDBD")
    for rec in f.itertuples(index=False):
        ws.append([_cell(v) for v in rec])
        row = ws[ws.max_row]
        for c in row:
            if c.column_letter in ("A", "B"):
                c.number_format = "@"
        if rec[COLUMNS.index("not_ranked")]:
            for c in row:
                c.fill = grey
    ws["A1"].font = Font(bold=True)
    for c in ws[2]:
        c.font = Font(bold=True)
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:{ws.cell(2, len(COLUMNS)).column_letter}{ws.max_row}"
    for i, name in enumerate(COLUMNS, 1):
        ws.column_dimensions[ws.cell(2, i).column_letter].width = max(11, min(len(name) + 3, 24))

    rm = wb.create_sheet("read_me")
    for a, b in _read_me_rows():
        rm.append([a or None, b or None])
    rm.column_dimensions["A"].width = 18
    rm.column_dimensions["B"].width = 120
    for row in rm.iter_rows():
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)


def write_outputs(t: pd.DataFrame, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(t, out / safeguards.CSV_NAME)
    write_xlsx(t, out / safeguards.XLSX_NAME)


# ---------- QA ----------

@dataclass
class QAContext:
    """Facts the QA report needs beyond the table. ``tlc`` has one dict per year (yy, reg, to,
    tlc_reg, tlc_to, max_diff, none_reg, none_to, optional table_diff)."""
    commit: str = "unknown"
    seconds: float = 0.0
    tlc: list = field(default_factory=list)
    sensitivity: pd.DataFrame | None = None
    sensitivity_note: str = ""
    side: dict = field(default_factory=dict)  # IRS counts and ACS blank counts (pipeline.build)


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent, timeout=10, check=True)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _fmt(col: str, v) -> str:
    if v is None or v is pd.NA or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return v
    if col in INT_COLS:
        return f"{v:.0f}"
    if col in DP1_COLS:
        return f"{v:.1f}"
    return f"{v:.4f}"


def md_table(df: pd.DataFrame, cols: list[str], headers: dict | None = None) -> str:
    """Markdown table of ``cols``; ``headers`` renames columns for display."""
    h = [(headers or {}).get(c, c) for c in cols]
    lines = ["| " + " | ".join(h) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(c, r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _ranked(t):
    return t[~t.not_ranked.astype(bool)]


def _has(series: pd.Series, code: str) -> pd.Series:
    return series.map(lambda s: code in s.split(";"))


def qa_checks(t: pd.DataFrame, ctx: QAContext):
    """The table and TLC checks: ``([(name, ok, detail)], [markdown blocks])``."""
    checks = []
    total = {d["yy"]: d for d in ctx.tlc}
    diffs = [max(d["max_diff"], d.get("table_diff", 0.0)) for d in ctx.tlc]
    worst = max(diffs, default=float("inf"))
    parts = [f"{yy} reg {total[yy]['tlc_reg']:,.0f} / turnout {total[yy]['tlc_to']:,.0f}"
             for yy in (24, 22) if yy in total]
    tail = f"max county abs diff {max((d['max_diff'] for d in ctx.tlc), default=float('nan')):.3f}"
    if any("table_diff" in d for d in ctx.tlc):
        tail += f"; table rows + NONE vs TLC {max(d.get('table_diff', 0.0) for d in ctx.tlc):.3f}"
    checks.append(("tlc_totals", bool(ctx.tlc) and worst <= TLC_TOL,
                   "; ".join(parts) + f"; {tail} (limit {TLC_TOL})"))

    checks.append(("row_count", len(t) == ROWS_EXPECTED, f"{len(t)} rows (expected {ROWS_EXPECTED})"))

    nones = {d["yy"]: d["none_reg"] for d in ctx.tlc}
    checks.append(("none_registration",
                   bool(nones) and all(v <= NONE_REG_MAX for v in nones.values()),
                   "NONE reg " + " / ".join(f"20{yy} {nones[yy]:.1f}" for yy in sorted(nones))
                   + f" (limit {NONE_REG_MAX})"))

    n_ranked = int((~t.not_ranked.astype(bool)).sum())
    lo, hi = RANKED_RANGE
    checks.append(("ranked_count", lo <= n_ranked <= hi,
                   f"{n_ranked} ranked, {len(t) - n_ranked} NOT_RANKED (range {lo}-{hi})"))

    over1 = t[t.P1 > 1]
    nonpos = t[t.cvap_adj <= 0]
    bad1 = sorted(over1.ZCTA[~over1.not_ranked.astype(bool)])
    badc = sorted(nonpos.ZCTA[~nonpos.not_ranked.astype(bool)])
    detail = (f"P1 > 1: {len(over1)} row(s), "
              + ("all NOT_RANKED" if not bad1 else f"RANKED: {bad1}")
              + f"; cvap_adj <= 0: {len(nonpos)} row(s), "
              + ("all NOT_RANKED" if not badc else f"RANKED: {badc}"))
    checks.append(("p1_gt_1_cvap_adj_le_0", not bad1 and not badc, detail))

    cols = ["ZCTA", "county_main", "P1", "cvap_adj", "not_ranked", "not_ranked_reason"]
    lists = []
    for title, rows in (("Rows with P1 > 1", over1), ("Rows with cvap_adj <= 0", nonpos)):
        body = md_table(rows.sort_values("ZCTA"), cols) if len(rows) else "None."
        lists.append(f"{title} ({len(rows)}):\n\n{body}")
    return checks, lists


def safeguards_check(result: dict[str, list[str]]):
    fails = [f for v in result.values() for f in v]
    if fails:
        return ("safeguards", False, f"{len(fails)} failure(s): " + "; ".join(fails))
    return ("safeguards", True,
            "output allowlist, CSV row 1, blank ranks, sort order, XLSX read_me, text cells, "
            "grey fills, forbidden columns and banned words: 0 failures")


def extremes_section(t: pd.DataFrame) -> str:
    r = _ranked(t).sort_values(["P1", "ZCTA"], ascending=[False, True])
    cols = ["ZCTA", "county_main", "P1", "P1_rank", "rank_p05", "rank_p95", "caution"]
    return (f"Highest 10 measured P1 (ranked ZCTAs only):\n\n{md_table(r.head(10), cols)}\n\n"
            f"Lowest 10 measured P1 (ranked ZCTAs only):\n\n"
            f"{md_table(r.tail(10).iloc[::-1], cols)}")


def partial_correctional_section(t: pd.DataFrame) -> str:
    part = t[_has(t.not_ranked_reason, "PARTIAL")].sort_values("ZCTA")
    corr = t[_has(t.not_ranked_reason, "CORRECTIONAL")].sort_values("ZCTA")
    caut = t[_has(t.caution, "CORRECTIONAL") & ~t.not_ranked.astype(bool)].sort_values(
        "corr_share", ascending=False)
    return "\n\n".join([
        f"PARTIAL (NOT_RANKED, {len(part)}):\n\n" + (md_table(
            part, ["ZCTA", "county_main", "acs_pop", "inreg_share20", "not_ranked_reason"])
            if len(part) else "None."),
        f"CORRECTIONAL (NOT_RANKED, {len(corr)}):\n\n" + (md_table(
            corr, ["ZCTA", "county_main", "corr_adult", "corr_share", "P1", "not_ranked_reason"])
            if len(corr) else "None."),
        f"CORRECTIONAL caution (ranked, {len(caut)}):\n\n" + (md_table(
            caut, ["ZCTA", "county_main", "corr_adult", "corr_share", "P1", "P1_rank"])
            if len(caut) else "None."),
    ])


def none_section(ctx: QAContext) -> str:
    if not ctx.tlc:
        return "No TLC data in this context."
    rows = pd.DataFrame([{
        "year": f"20{d['yy']}", "none_reg": d["none_reg"], "none_to": d["none_to"],
        "reg_share": d["none_reg"] / d["tlc_reg"] if d["tlc_reg"] else float("nan"),
        "to_share": d["none_to"] / d["tlc_to"] if d["tlc_to"] else float("nan"),
    } for d in sorted(ctx.tlc, key=lambda d: d["yy"])])
    lines = ["| year | registration in NONE | turnout in NONE | share of region registration "
             "| share of region turnout |", "|---|---|---|---|---|"]
    for _, r in rows.iterrows():
        lines.append(f"| {r.year} | {r.none_reg:.1f} | {r.none_to:.1f} | {r.reg_share:.6f} "
                     f"| {r.to_share:.6f} |")
    return ("NONE is the part of a voting district that lies outside every ZCTA "
            "(registered voters and ballots that cannot be placed in a ZCTA).\n\n" + "\n".join(lines))


def corr_adult_section(t: pd.DataFrame) -> str:
    r = _ranked(t)
    r = r[r.corr_adult > 0].copy()
    r["P1_without"] = (r.to22 + r.to24) / (2 * r.cvap.where(r.cvap > 0))
    r["_first"] = r.ZCTA != FIRST_ZCTA
    r = r.sort_values(["_first", "corr_share", "ZCTA"], ascending=[True, False, True])
    cols = ["ZCTA", "county_main", "corr_adult", "corr_share", "P1", "P1_without", "caution"]
    heads = {"P1": "P1 (less corr_adult)", "P1_without": "P1 (without subtraction)"}
    out = md_table(r, cols, heads) if len(r) else "None."
    if FIRST_ZCTA not in set(r.ZCTA):
        row = t[t.ZCTA == FIRST_ZCTA]
        if len(row):
            x = row.iloc[0]
            out += (f"\n\n{FIRST_ZCTA} is not in this list: not_ranked={bool(x.not_ranked)}, "
                    f"reason {x.not_ranked_reason or 'none'}, corr_adult {x.corr_adult:.0f}.")
        else:
            out += f"\n\n{FIRST_ZCTA} is not a row of this table."
    return out


def flag_counts_section(t: pd.DataFrame) -> str:
    n_rank = int((~t.not_ranked.astype(bool)).sum())
    seen_r = {c for s in t.not_ranked_reason for c in s.split(";") if c}
    seen_c = {c for s in t.caution for c in s.split(";") if c}
    rl = ["| NOT_RANKED reason | rows |", "|---|---|"]
    for c in REASON_CODES + sorted(seen_r - set(REASON_CODES)):
        rl.append(f"| {c} | {int(_has(t.not_ranked_reason, c).sum())} |")
    rl.append(f"| any reason (NOT_RANKED rows) | {int(t.not_ranked.astype(bool).sum())} |")
    rl.append(f"| ranked | {n_rank} |")
    cl = ["| CAUTION code | all rows | ranked rows |", "|---|---|---|"]
    rk = ~t.not_ranked.astype(bool)
    for c in CAUTION_CODES + sorted(seen_c - set(CAUTION_CODES)):
        m = _has(t.caution, c)
        cl.append(f"| {c} | {int(m.sum())} | {int((m & rk).sum())} |")
    return "\n".join(rl) + "\n\n" + "\n".join(cl)


def appendix_section(s: pd.DataFrame | None, note: str) -> str:
    if s is None:
        return note or "Not run."
    s = s.reset_index() if "ZCTA" not in s else s.reset_index(drop=True)
    a, b = s.rank_main, s.rank_total
    shift = (b - a).abs()
    top = s.assign(shift=shift).sort_values(["shift", "ZCTA"], ascending=[False, True]).head(10)
    lines = [
        "Re-allocation of 2022 and 2024 turnout using total 2020 population as the weight "
        "instead of voting-age population less correctional adults. Flags, the ranked set, "
        "registration and CVAP stay as in the main build; P1 and its rank are recomputed over "
        "the same ranked ZCTAs.", "",
        f"- Ranked ZCTAs compared: {len(s)}",
        f"- Spearman correlation of P1_rank, main vs total weights: {a.rank().corr(b.rank()):.3f}",
        f"- Median absolute rank shift: {shift.median():.1f}",
        f"- Maximum absolute rank shift: {shift.max():.1f}", "",
        "Ten largest rank shifts:", "",
        "| ZCTA | P1_rank main | P1_rank total weights | abs shift | P1 main | P1 total weights |",
        "|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        lines.append(f"| {r.ZCTA} | {r.rank_main:.1f} | {r.rank_total:.1f} | {r['shift']:.1f} "
                     f"| {r.P1:.4f} | {r.P1_total:.4f} |")
    return "\n".join(lines)


CONTEXT_BASES = list(context.SPECS)


def side_appendix_section(t: pd.DataFrame, side: dict) -> str:
    """Aggregate numbers for the IRS and ACS side columns; no ZCTA-level IRS detail."""
    lines = ["The side columns are context only; they feed no flag, P1 or rank.", ""]
    if side:
        blank = ", ".join(f"{k} {v}" for k, v in side["context_blank"].items())
        lines += [
            f"IRS Business Master File, Texas extract (read as {side['encoding']}):", "",
            f"- Rows with STATUS 01: {side['irs_total_active']:,}",
            f"- civic_orgs_n, region total: {side['civic_region']:,}",
            f"- hoa_like_n, region total: {side['hoa_like_region']:,}",
            f"- Overlap (rows counted in both): {side['overlap']:,}",
            "- Lost IRS rows (civic rows in a region ZIP3 whose ZIP is in no ZCTA): "
            f"{side['lost_irs_rows']:,}",
            f"- Out of region (other civic rows outside the {len(t)} ZCTAs): "
            f"{side['out_of_region']:,}", "",
            f"ACS context, ZCTAs with a blank estimate (no table row or zero universe): {blank}",
            ""]
    else:
        lines += ["IRS and ACS coverage counts: not run in this context.", ""]
    r = _ranked(t)
    lines += ["### correlation with conditions", "",
              "Spearman correlation of P1_rank with each context column, over ranked ZCTAs that "
              "have a value. Descriptive only; it says nothing about cause.", "",
              "| context column | ZCTAs | Spearman |", "|---|---|---|"]
    for c in CONTEXT_BASES:
        if c not in r:
            continue
        pair = r[["P1_rank", c]].dropna()
        rho = pair.P1_rank.rank().corr(pair[c].rank()) if len(pair) > 2 else float("nan")
        lines.append(f"| {c} | {len(pair)} | {rho:.3f} |" if rho == rho else f"| {c} | {len(pair)} | n/a |")
    return "\n".join(lines)


def render_report(t, ctx, checks, lists, commit=None) -> str:
    """The QA report text; ``checks`` is the full PASS/FAIL list including safeguards."""
    block = [f"{'PASS' if ok else 'FAIL'} {name}: {detail}" for name, ok, detail in checks]
    overall = "PASS" if all(ok for _, ok, _ in checks) else "FAIL"
    return "\n".join([
        QA_FIRST_LINE, "",
        "# ZCTA civic table: QA report", "",
        f"Build commit: {commit or ctx.commit}",
        f"Build duration: {ctx.seconds:.1f} s", "",
        "## 1. Automated checks", "", *block, f"OVERALL: {overall}", "",
        *[x + "\n" for x in lists],
        "## 2. Eyeball list", "",
        "### 2.1 PARTIAL and CORRECTIONAL lists", "", partial_correctional_section(t), "",
        "### 2.2 Highest and lowest 10 measured P1 (internal QA, with rank intervals)", "",
        extremes_section(t), "",
        "### 2.3 The NONE bucket", "", none_section(ctx), "",
        "### 2.4 Ranked ZCTAs with corr_adult > 0", "",
        "P1 with the subtraction is the published measure; the other column shows what P1 "
        "would be without it, (to22 + to24) / (2 * cvap).", "", corr_adult_section(t), "",
        "### 2.5 Counts by flag reason", "", flag_counts_section(t), "",
        "## 3. Appendix: weight sensitivity", "",
        appendix_section(ctx.sensitivity, ctx.sensitivity_note), "",
        "## 4. Appendix: ACS context and IRS civic counts", "",
        side_appendix_section(t, ctx.side), ""])


def emit(t: pd.DataFrame, ctx: QAContext, out_dir, qa_dir, raw=None) -> bool:
    """Write the outputs, run safeguards, write the QA report. True when every check passes.

    The map PNG and GeoJSON are written after the CSV and XLSX when ``raw`` is given (they need
    its shapefiles); without it only the CSV and XLSX are written.
    """
    write_outputs(t, out_dir)
    if raw is not None:
        from . import map as hmap  # lazy: map imports this module

        hmap.write_map(t, raw, out_dir)
    sg = safeguards.run_all(out_dir)
    checks, lists = qa_checks(t, ctx)
    full = checks + [safeguards_check(sg)]
    text = render_report(t, ctx, full, lists)
    leaks = safeguards.check_banned_words({"qa_report": text})
    if leaks:  # the report itself is covered by the banned-word rule
        sg = {**sg, "qa_report": leaks}
        full = checks + [safeguards_check(sg)]
        text = render_report(t, ctx, full, lists)
    qa = Path(qa_dir)
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "qa_report.md").write_text(text, encoding="utf-8")
    return all(ok for _, ok, _ in full)


# ---------- build ----------

def _tlc_year(yy, weights, vd, alloc, t) -> dict:
    p = turnout.pieces(weights, vd, yy)
    v = vd[vd.VTDKEY.isin(set(weights.VTDKEY))]
    p["county"] = p.VTDKEY.map(v.set_index("VTDKEY").CNTYVTD.str[:3])
    reg, to = f"reg{yy}", f"to{yy}"
    got = p.groupby("county")[[reg, to]].sum().reindex(COUNTIES).fillna(0.0)
    cnty = vd.CNTYVTD.str[:3]
    tlc = vd[cnty.isin(COUNTIES)].groupby(cnty)[["Voter_Registration", "Turnout"]].sum()
    tlc = tlc.reindex(COUNTIES).fillna(0.0)
    max_diff = max((got[reg] - tlc.Voter_Registration).abs().max(),
                   (got[to] - tlc.Turnout).abs().max())
    none = turnout.none_bucket(alloc)
    tlc_reg, tlc_to = float(tlc.Voter_Registration.sum()), float(tlc.Turnout.sum())
    table_diff = max(abs(t[reg].sum() + none[reg] - tlc_reg), abs(t[to].sum() + none[to] - tlc_to))
    return dict(yy=yy, reg=float(got[reg].sum()), to=float(got[to].sum()), tlc_reg=tlc_reg,
                tlc_to=tlc_to, max_diff=float(max_diff), none_reg=float(none[reg]),
                none_to=float(none[to]), table_diff=float(table_diff))


def sensitivity(t, blocks, vds) -> pd.DataFrame:
    """Ranked ZCTAs: P1 and P1_rank as built vs re-allocated with total-population weights."""
    wt = crosswalk.vtd_weights(blocks, col="total")
    t2 = t.copy()
    for yy, vd in vds.items():
        a = turnout.allocate(wt, vd, yy)
        t2[f"to{yy}"] = a[f"to{yy}"].reindex(t.index).fillna(0.0)
    t2["P1"] = score.p1(t2)
    t2["P1_rank"] = score.rank_pct(t2)
    r = ~t.not_ranked.astype(bool)
    return pd.DataFrame({"ZCTA": t.ZCTA[r], "rank_main": t.P1_rank[r], "rank_total": t2.P1_rank[r],
                         "P1": t.P1[r], "P1_total": t2.P1[r]})


def build_all(raw) -> tuple[pd.DataFrame, QAContext]:
    """The table plus everything the QA report needs (reads ``raw`` once more for TLC checks)."""
    raw = Path(raw)
    t0 = time.perf_counter()
    t, side = pipeline.build(raw)
    blocks = crosswalk.build_blocks(raw, write=False)
    weights = crosswalk.vtd_weights(blocks)
    vds = {yy: turnout.load_vd(raw_path(raw, f"tlc_vd{yy}")) for yy in (22, 24)}
    tlc = [_tlc_year(yy, weights, vd, turnout.allocate(weights, vd, yy), t) for yy, vd in vds.items()]
    s = sensitivity(t, blocks, vds)
    ctx = QAContext(commit=git_commit(), tlc=tlc, sensitivity=s, side=side)
    ctx.seconds = time.perf_counter() - t0
    return t, ctx


def main(raw, out_dir=Path("out"), qa_dir=Path("qa")) -> bool:
    """Build and write everything; True when every QA check passes."""
    t0 = time.perf_counter()
    t, ctx = build_all(raw)
    ctx.seconds = time.perf_counter() - t0
    ok = emit(t, ctx, out_dir, qa_dir, raw=raw)
    lines = (Path(qa_dir) / "qa_report.md").read_text(encoding="utf-8").splitlines()
    for ln in lines:
        if ln.startswith(("PASS ", "FAIL ", "OVERALL:")):
            print(ln)
    print(f"QA report: {Path(qa_dir) / 'qa_report.md'}")
    return ok


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hfb.assemble", description=__doc__)
    ap.add_argument("raw", type=Path, help="raw data directory (holds manifest.csv)")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--qa", type=Path, default=Path("qa"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if not main(a.raw, out_dir=a.out, qa_dir=a.qa):
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
