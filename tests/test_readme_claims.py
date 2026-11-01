"""README numbers that describe the built table must match it (real data)."""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


def readme_caution_counts():
    """{code: (ranked, all)} from the README "Caution flags" table."""
    text = README.read_text()
    block = text.split("## Caution flags", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| ([A-Z_]+) \|.*\| (\d+) \| (\d+) \|$", block, re.M)
    return {code: (int(r), int(a)) for code, r, a in rows}


def test_readme_caution_table_parses():
    counts = readme_caution_counts()
    assert set(counts) == {"SPLIT", "LOW_N", "GROWTH", "CORRECTIONAL", "COLLEGE", "EDGE",
                           "FACILITY_CHECK"}


@pytest.mark.real
def test_readme_caution_counts_match_table(table):
    codes = table.caution.fillna("").str.split(";")
    ranked = ~table.not_ranked.astype(bool)
    for code, (n_ranked, n_all) in readme_caution_counts().items():
        has = codes.apply(lambda c, code=code: code in c)
        assert (int((has & ranked).sum()), int(has.sum())) == (n_ranked, n_all), code


@pytest.mark.real
def test_readme_p1_above_one_claim(table):
    """README: P1 is above 1 for 5 rows, all not ranked."""
    assert "P1 is above 1 for 5 rows. All 5 are not ranked" in README.read_text()
    high = table[table.P1 > 1]
    assert len(high) == 5 and high.not_ranked.astype(bool).all()
