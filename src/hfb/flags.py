"""NOT_RANKED reasons and CAUTION codes for the ZCTA table.

Every rule is a named threshold below. A NaN input never trips its own flag (a NaN
comparison is False), with one exception: ``NO_CVAP`` fires when ``cvap`` (hence
``cvap_adj``) is missing. Reasons and cautions are ``;``-joined, sorted, and ``""`` when none.
"""

import pandas as pd

from .config import FACILITY_CHECK

# NOT_RANKED
LAND_MIN = 0.50            # LAND: land_share < LAND_MIN
POP_MIN = 1500             # POP: acs_pop < POP_MIN
GQ_MAX = 0.50              # GQ: gq / total > GQ_MAX
CORR_RANK_OUT = 0.10       # CORRECTIONAL: corr_adult / vap >= CORR_RANK_OUT
PARTIAL_MIN = 0.90         # PARTIAL: inreg_share20 < PARTIAL_MIN
# CAUTION
LOW_N_REG = 500            # LOW_N: reg24 < LOW_N_REG ...
LOW_N_CVAP = 1000          # ... or cvap < LOW_N_CVAP
GROWTH_MAX = 1.1           # GROWTH: reg24 / vap > GROWTH_MAX
SPLIT_MAX = 0.50           # SPLIT: split_exposure > SPLIT_MAX
COLLEGE_MIN = 0.10         # COLLEGE: college / total >= COLLEGE_MIN
CORR_CAUTION_MIN = 0.05    # CORRECTIONAL caution: CORR_CAUTION_MIN <= corr_adult / vap < CORR_RANK_OUT
EDGE_MAX = 0.98            # EDGE: PARTIAL_MIN <= inreg_share20 < EDGE_MAX


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, NaN where den is 0 or missing."""
    return num / den.where(den > 0)


def _join(masks: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """Per row, the sorted ``;``-joined names whose mask is True."""
    names = sorted(masks)
    rows = zip(*[masks[k].fillna(False).to_numpy(bool) for k in names])
    return pd.Series([";".join(k for k, hit in zip(names, r) if hit) for r in rows],
                     index=index, dtype=object)


def apply_flags(df: pd.DataFrame) -> pd.DataFrame:
    """``not_ranked`` (bool), ``not_ranked_reason`` and ``caution`` for each row of ``df``.

    ``df`` is indexed by ZCTA and has land_share, acs_pop, cvap, corr_adult, gq, total, vap,
    college, inreg_share20, reg24 and split_exposure; ``cvap_adj`` is computed if absent.
    """
    cvap_adj = df["cvap_adj"] if "cvap_adj" in df else df.cvap - df.corr_adult
    gq = _ratio(df.gq, df.total)
    college = _ratio(df.college, df.total)
    corr = _ratio(df.corr_adult, df.vap)
    growth = _ratio(df.reg24, df.vap)
    inreg = df.inreg_share20

    reasons = {
        "LAND": df.land_share < LAND_MIN,
        "POP": df.acs_pop < POP_MIN,
        "NO_CVAP": cvap_adj.isna() | (cvap_adj <= 0),
        "GQ": gq > GQ_MAX,
        "CORRECTIONAL": corr >= CORR_RANK_OUT,
        "PARTIAL": inreg < PARTIAL_MIN,
    }
    cautions = {
        "LOW_N": (df.reg24 < LOW_N_REG) | (df.cvap < LOW_N_CVAP),
        "GROWTH": growth > GROWTH_MAX,
        "SPLIT": df.split_exposure > SPLIT_MAX,
        "COLLEGE": college >= COLLEGE_MIN,
        "CORRECTIONAL": (corr >= CORR_CAUTION_MIN) & (corr < CORR_RANK_OUT),
        "EDGE": (inreg >= PARTIAL_MIN) & (inreg < EDGE_MAX),
        "FACILITY_CHECK": pd.Series(df.index.astype(str).isin(FACILITY_CHECK), index=df.index),
    }
    reason = _join(reasons, df.index)
    return pd.DataFrame({
        "not_ranked": reason != "",
        "not_ranked_reason": reason,
        "caution": _join(cautions, df.index),
    }, index=df.index)
