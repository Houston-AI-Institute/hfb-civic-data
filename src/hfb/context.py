"""ACS 2020-2024 context proportions with margins of error, by ZCTA.

Five side columns for reading a ZCTA's conditions next to its civic measures. They never feed
flags, P1 or the rank. Each is a proportion of a table's E001 universe, with a 90 percent MOE.

    proportion MOE  = sqrt(M_num^2 - p^2 * M_den^2) / den
    fallback        = sqrt(M_num^2 + p^2 * M_den^2) / den   (when the term under the root < 0)

For a sum of numerator terms, M_num is the root sum of squares of the terms' MOEs. A ZCTA whose
E001 is 0 or missing gets a blank estimate and MOE, and so does one absent from a table.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import acs
from .paths import raw_path

log = logging.getLogger(__name__)

# name -> (table, numerator variable numbers, complement?)
SPECS = {
    "same_house_1yr": ("B07003", [4], False),
    "owner_occupied": ("B25003", [2], False),
    "hs_diploma_25plus": ("B15003", list(range(17, 26)), False),
    "not_limited_english_hh": ("C16002", [4, 7, 10, 13], True),
    "broadband_hh": ("B28002", [4], False),
}
CONTEXT_COLS = [c for k in SPECS for c in (k, f"{k}_moe")]
TABLES = sorted({t.lower() for t, _, _ in SPECS.values()})


def proportion(num, m_num, den, m_den) -> tuple[pd.Series, pd.Series]:
    """Proportion ``num / den`` and its MOE; both NaN where ``den`` is 0 or missing."""
    d = den.where(den > 0)
    p = num / d
    root = m_num**2 - p**2 * m_den**2
    moe = np.sqrt(root.where(root >= 0, m_num**2 + p**2 * m_den**2)) / d
    return p, moe


def context(a: pd.DataFrame) -> pd.DataFrame:
    """The five context proportions and ``_moe`` columns for the rows of ``a``.

    ``a`` holds the E and M columns of B07003, B25003, B15003, C16002 and B28002 (as read by
    ``acs.read_acs``). A NaN in any input of a result makes that result NaN.
    """
    out = {}
    for name, (table, nums, complement) in SPECS.items():
        e = lambda i, k="E": a[f"{table}_{k}{i:03d}"]
        num = sum((e(i) for i in nums[1:]), e(nums[0]))  # NaN propagates
        m_num = np.sqrt(sum((e(i, "M") ** 2 for i in nums[1:]), e(nums[0], "M") ** 2))
        p, moe = proportion(num, m_num, e(1), e(1, "M"))
        out[name] = 1 - p if complement else p  # the complement has the same MOE
        out[f"{name}_moe"] = moe
    return pd.DataFrame(out, index=a.index)[CONTEXT_COLS]


def read_context(raw: Path, zctas: pd.Index, reader=acs.read_acs) -> pd.DataFrame:
    """``context`` for ``zctas``, read from the raw ACS files; absent ZCTAs are blank (logged).

    ``reader`` is ``acs.read_acs`` for ZCTAs; pass ``acs.read_acs_county`` with county FIPS to
    read county rows instead (same formulas).
    """
    zctas = pd.Index(zctas)
    parts = []
    for t in TABLES:
        key = f"acs_{t}"
        part = reader(raw_path(raw, key), zctas).reindex(zctas)
        absent = int(part.isna().all(axis=1).sum())
        if absent:
            log.warning("%d of %d region geographies have no %s row; their context values are "
                        "blank", absent, len(zctas), key)
        parts.append(part)
    return context(pd.concat(parts, axis=1))
