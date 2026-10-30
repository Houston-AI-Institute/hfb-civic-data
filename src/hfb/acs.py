"""ACS 2020-2024 5-year B05003 (sex by age by nativity and citizenship) by ZCTA.

The table-based summary file is pipe-delimited, one row per geography. ZCTA rows have a
``GEO_ID`` starting ``860Z200US``; the ZCTA is its last five characters. Negative values
(-555555555, -666666666, ...) are Census annotation sentinels, not data, and become NaN.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ZCTA_PREFIX = "860Z200US"


def read_acs(path, zctas) -> pd.DataFrame:
    """B05003 E and M columns (float) for the requested ZCTAs, indexed by ZCTA.

    A requested ZCTA with no row is simply absent from the result; callers that need
    full coverage check for it.
    """
    want = set(zctas)
    rows, keys = [], []
    with open(Path(path), encoding="utf-8-sig") as f:
        header = f.readline().rstrip("\n").split("|")
        for line in f:
            if not line.startswith(ZCTA_PREFIX):
                continue
            z = line[:line.index("|")][-5:]
            if z in want:
                rows.append(line.rstrip("\n").split("|")[1:])
                keys.append(z)
    out = pd.DataFrame(rows, index=pd.Index(keys, name="ZCTA"), columns=header[1:])
    out = out.replace("", np.nan).astype(float)
    return out.mask(out < 0)


def cvap(b05003: pd.DataFrame) -> pd.DataFrame:
    """acs_pop, cvap (citizen adults), cvap_moe (root sum of squares) and acs_vap.

    CVAP is native-born plus naturalized adults, male and female (E009, E011, E020,
    E022); acs_vap is all adults, male and female (E008 + E019). NaN in any input makes
    the dependent output NaN.
    """
    b = b05003
    cv = b.B05003_E009 + b.B05003_E011 + b.B05003_E020 + b.B05003_E022
    moe = np.sqrt(b.B05003_M009**2 + b.B05003_M011**2 + b.B05003_M020**2 + b.B05003_M022**2)
    return pd.DataFrame({
        "acs_pop": b.B05003_E001, "cvap": cv, "cvap_moe": moe,
        "acs_vap": b.B05003_E008 + b.B05003_E019})
