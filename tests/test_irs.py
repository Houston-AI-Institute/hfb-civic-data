"""Task 7: IRS civic-organization counts. Counts only; names never leave this module."""

import io
import re

import numpy as np
import openpyxl
import pandas as pd
import pytest

from hfb import irs

HEAD = "EIN,NAME,ICO,STREET,CITY,STATE,ZIP,GROUP,SUBSECTION,STATUS,NTEE_CD,SORT_NAME\n"


def write_irs(path, rows, encoding="utf-8"):
    """rows = (name, zip, status, ntee[, subsection]); street/ICO hold sentinel text."""
    lines = [HEAD]
    for i, r in enumerate(rows):
        name, z, status, ntee = r[:4]
        sub = r[4] if len(r) > 4 else "03"
        lines.append(f"{i:09d},{name},C/O SECRETICO,1 SECRETSTREET,X,TX,{z},0,{sub},{status},"
                     f"{ntee},\n")
    path.write_bytes("".join(lines).encode(encoding))
    return path


def counts(tmp_path, rows, zctas=("77001", "77002", "77003"), all_extra=(), **kw):
    p = write_irs(tmp_path / "irs.csv", rows, **kw)
    zcta_all = set(zctas) | set(all_extra) | {"75001"}
    return irs.civic_counts(p, list(zctas), zcta_all)


def test_band_edges():
    got = {n: irs.band(n) for n in (0, 1, 2, 3, 5, 6, 40)}
    assert got == {0: "0", 1: "1-2", 2: "1-2", 3: "3-5", 5: "3-5", 6: "6+", 40: "6+"}


def test_prefix_includes_subcodes(tmp_path):
    rows = [("A ORG", "77001-1111", "01", c) for c in
            ("S20", "S21", "S22", "S80", "S81", "S82", "W24", "S20Z", "S2100", "W240", "S82X")]
    rows += [("B ORG", "77001", "01", c) for c in ("S23", "S19", "S2", "W2", "W25", "", "X20")]
    df, qa = counts(tmp_path, rows)
    assert df.loc["77001", "civic_orgs_n"] == 11
    assert qa["civic_region"] == 11


def test_any_subsection_counts_and_status_filter(tmp_path):
    rows = [("A", "77001", "01", "S20", "04"), ("B", "77001", "01", "S20", "03"),
            ("C", "77001", "01", "S20", "07"),
            ("D", "77001", "02", "S20"), ("E", "77001", "12", "S20"), ("F", "77001", "25", "S20")]
    df, qa = counts(tmp_path, rows)
    assert df.loc["77001", "civic_orgs_n"] == 3
    assert qa["irs_total_active"] == 3  # STATUS 01 rows of the extract


def test_zip5_uses_first_five_characters(tmp_path):
    rows = [("A", "770021234", "01", "S20"), ("B", "77002-1234", "01", "S20"),
            ("C", "77002", "01", "S20"), ("D", "7700", "01", "S20")]
    df, qa = counts(tmp_path, rows)
    assert df.loc["77002", "civic_orgs_n"] == 3


def test_hoa_like_name_ntee_overlap_and_exclusions(tmp_path):
    rows = [
        ("OAK HOMEOWNERS ASSN", "77001", "01", ""),        # name
        ("pine home owners club", "77001", "01", ""),      # name, case-insensitive, two words
        ("RIVER PROPERTY OWNER GROUP", "77001", "01", ""),
        ("LAKE OWNERS ASSOCIATION", "77001", "01", ""),
        ("MILL POA INC", "77001", "01", ""),               # whole word
        ("ELM HOA", "77001", "01", ""),
        ("ROW TOWNHOME COUNCIL", "77001", "01", ""),
        ("SKY CONDOMINIUM TRUST", "77001", "01", ""),
        ("SPOAT SHOA", "77001", "01", ""),                 # POA/HOA inside a word: no
        ("PLAIN ORG", "77001", "01", "L50"),               # NTEE L5 prefix
        ("PLAIN ORG TWO", "77001", "01", "L5"),
        ("PLAIN ORG THREE", "77001", "01", "L40"),         # not L5
        ("BOTH ORG HOA", "77001", "01", "S20"),            # counts in both
        ("CIVIC ONLY", "77001", "01", "S81"),
        ("X HOA", "77001", "01", "R40"),                   # R40 excluded from both
        ("X HOMEOWNERS", "77001", "01", "R61"),            # R6* excluded from both
        ("Y HOA", "77001", "02", "L50"),                   # inactive
    ]
    df, qa = counts(tmp_path, rows)
    assert df.loc["77001", "hoa_like_n"] == 8 + 2 + 1
    assert df.loc["77001", "civic_orgs_n"] == 2
    assert qa["hoa_like_region"] == 11 and qa["civic_region"] == 2 and qa["overlap"] == 1


def test_r_codes_excluded_from_civic_too(tmp_path):
    # R40 / R6* never match the civic prefixes, but the exclusion applies to both counts
    df, qa = counts(tmp_path, [("A", "77001", "01", "R40"), ("B", "77001", "01", "R6X")])
    assert df.loc["77001", ["civic_orgs_n", "hoa_like_n"]].tolist() == [0, 0]


def test_empty_region_zcta_is_zero_not_blank(tmp_path):
    df, _ = counts(tmp_path, [("A", "77001", "01", "S20")])
    assert df.loc["77003"].tolist() == [0, "0", 0]
    assert df.index.tolist() == ["77001", "77002", "77003"]
    assert list(df.columns) == ["civic_orgs_n", "civic_orgs_band", "hoa_like_n"]
    assert df.civic_orgs_n.dtype.kind == "i" and df.hoa_like_n.dtype.kind == "i"
    assert df.loc["77001", "civic_orgs_band"] == "1-2"


def test_lost_rows_fixture(tmp_path):
    rows = [
        ("A", "77001", "01", "S20"),         # in region
        ("B", "77099", "01", "S20"),         # PO-box style ZIP, not a ZCTA, region ZIP3: lost
        ("C", "77099-0001", "01", "S21"),    # lost
        ("D", "99999", "01", "S20"),         # non-region ZIP3: out_of_region
        ("E", "75001", "01", "S20"),         # a ZCTA but outside the region: out_of_region
        ("F", "77099", "02", "S20"),         # inactive: not counted anywhere
        ("G", "77099", "01", "X20"),         # not civic: not counted anywhere
    ]
    df, qa = counts(tmp_path, rows)
    assert qa["lost_irs_rows"] == 2
    assert qa["out_of_region"] == 2
    assert qa["civic_region"] == 1 and df.civic_orgs_n.sum() == 1
    assert qa["irs_total_active"] == 6


def test_encoding_utf8_then_latin1(tmp_path):
    rows = [("CAFÉ CIVIC", "77001", "01", "S20")]
    _, qa = counts(tmp_path, rows)
    assert qa["encoding"] == "utf-8"
    _, qa = counts(tmp_path, rows, encoding="latin-1")
    assert qa["encoding"] == "latin-1"


def test_output_has_no_names_or_addresses(tmp_path):
    df, qa = counts(tmp_path, [("UNIQUEORGNAME HOA", "77001", "01", "S20")])
    blob = df.to_csv() + repr(qa)
    for s in ("UNIQUEORGNAME", "SECRETICO", "SECRETSTREET"):
        assert s not in blob
    assert all(isinstance(v, (int, str)) for v in qa.values())


# ---------- real data ----------

@pytest.fixture(scope="module")
def real_counts(raw):
    from hfb import frame
    from hfb.paths import raw_path

    zall = set(frame.read_rel(raw_path(raw, "zcta_county")).GEOID_ZCTA5_20)
    region = frame.region_zctas(frame.read_rel(raw_path(raw, "zcta_county"))).index
    return irs.civic_counts(raw_path(raw, "irs_eo_tx"), region, zall)


@pytest.mark.real
def test_real_counts_in_expected_ranges(real_counts, capsys):
    df, qa = real_counts
    print(f"REAL IRS: {qa}")
    with capsys.disabled():
        print(f"\nREAL IRS counts: {qa}")
    assert len(df) == 240
    assert 530 <= df.civic_orgs_n.sum() <= 570, qa
    assert 407 * 0.85 <= qa["hoa_like_region"] <= 407 * 1.15, qa
    assert 39 * 0.85 <= qa["overlap"] <= 39 * 1.15, qa
    assert df.hoa_like_n.sum() == qa["hoa_like_region"]


def name_hits(names, text):
    """Names found in ``text`` (case-insensitive).

    Hybrid rule: a name of 6 or more characters (after strip) matches as a substring; a
    shorter name matches only as a whole word (``\\b`` around the escaped name). A plain
    substring rule gave false positives: OWL inside "allowlist" and DAT inside "data". Whole-word
    matching had 0 hits across all 27,948 region names.
    """
    low = text.lower()
    hits = []
    for n in names:
        n = n.strip().lower()
        if not n:
            continue
        if len(n) >= 6:
            found = n in low
        else:
            found = re.search(r"\b" + re.escape(n) + r"\b", low) is not None
        if found:
            hits.append(n)
    return sorted(hits)


def test_name_hits_short_names_are_whole_word_only():
    assert name_hits(["owl", "dat"], "output allowlist; Data starts on row 2") == []
    assert name_hits(["owl", "dat"], "the OWL sat; a DAT file") == ["dat", "owl"]
    # 6 or more characters still match inside longer text
    assert name_hits(["maple"], "maplewood") == []
    assert name_hits(["maplew"], "maplewood") == ["maplew"]


@pytest.mark.real
def test_irs_names_never_in_output(raw, real_outputs):
    from hfb import frame
    from hfb.paths import raw_path

    out, qa_dir, _ = real_outputs
    region = set(frame.region_zctas(frame.read_rel(raw_path(raw, "zcta_county"))).index)
    b = pd.read_csv(raw_path(raw, "irs_eo_tx"), dtype=str, keep_default_na=False,
                    usecols=["NAME", "ZIP", "STATUS"])
    b = b[(b.STATUS == "01") & b.ZIP.str[:5].isin(region)]
    names = {n.strip().lower() for n in b.NAME if n.strip()}
    assert len(names) > 1000
    texts = {"csv": (out / "zcta_civic.csv").read_text(),
             "qa": (qa_dir / "qa_report.md").read_text()}
    wb = openpyxl.load_workbook(out / "zcta_civic.xlsx")
    texts["xlsx"] = "\n".join(str(c.value) for ws in wb for row in ws.iter_rows()
                              for c in row if c.value is not None)
    hits = {where: name_hits(names, text) for where, text in texts.items()}
    assert hits == {"csv": [], "qa": [], "xlsx": []}
