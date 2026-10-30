"""Check that the ACS variables used by hfb.context mean what the formulas assume.

One read-only GET of the Census 2024 ACS 5-year variables.json, saved to
data/interim/acs2024_variables.json (never data/raw). If the file is already saved it is
read instead of fetched again. Prints each variable's label; exits non-zero on a mismatch.

    uv run python scripts/check_acs_labels.py
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

URL = "https://api.census.gov/data/2024/acs/acs5/variables.json"
SAVED = Path(__file__).resolve().parent.parent / "data" / "interim" / "acs2024_variables.json"

# variable -> regex the label must match (case-insensitive)
EXPECT = {
    "B07003_001E": r"^Estimate!!Total:?$",
    "B07003_004E": r"Same house 1 year ago",
    "B25003_001E": r"^Estimate!!Total:?$",
    "B25003_002E": r"Owner occupied",
    "B15003_001E": r"^Estimate!!Total:?$",
    "B15003_017E": r"Regular high school diploma",
    "B15003_018E": r"GED or alternative credential",
    "B15003_019E": r"Some college, less than 1 year",
    "B15003_020E": r"Some college, 1 or more years, no degree",
    "B15003_021E": r"Associate'?s degree",
    "B15003_022E": r"Bachelor'?s degree",
    "B15003_023E": r"Master'?s degree",
    "B15003_024E": r"Professional school degree",
    "B15003_025E": r"Doctorate degree",
    "C16002_001E": r"^Estimate!!Total:?$",
    "C16002_004E": r"Spanish:?!!Limited English speaking household",
    "C16002_007E": r"Other Indo-European languages:?!!Limited English speaking household",
    "C16002_010E": r"Asian and Pacific Island languages:?!!Limited English speaking household",
    "C16002_013E": r"Other languages:?!!Limited English speaking household",
    "B28002_001E": r"^Estimate!!Total:?$",
    "B28002_004E": r"Broadband of any type",
}


def load() -> dict:
    if not SAVED.exists():
        SAVED.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as r:  # the single read-only GET
            SAVED.write_bytes(r.read())
        print(f"fetched {URL} -> {SAVED}")
    return json.loads(SAVED.read_text(encoding="utf-8"))["variables"]


def check(variables: dict) -> list[str]:
    bad = []
    for var, pat in EXPECT.items():
        label = variables.get(var, {}).get("label")
        ok = label is not None and re.search(pat, label, re.I) is not None
        print(f"{'OK  ' if ok else 'FAIL'} {var}: {label}")
        if not ok:
            bad.append(var)
    return bad


if __name__ == "__main__":
    bad = check(load())
    if bad:
        print(f"MISMATCH: {bad}")
        sys.exit(1)
    print(f"all {len(EXPECT)} labels match")
