"""P1 (the reference measure), its percentile rank and the rank's sampling interval.

P1 = (to22 + to24) / (2 * cvap_adj), with cvap_adj = cvap - corr_adult. It is a reference
measure, not a score. P1 above 1 is kept as measured. Ranks run over ranked rows only
(``not_ranked`` False); 100 is the highest P1.
"""

import numpy as np
import pandas as pd

BLANK_REASONS = {"LAND", "PARTIAL", "NO_CVAP"}
MOE_Z = 1.645  # ACS margins of error are 90%; standard error = MOE / 1.645


def derived(df: pd.DataFrame) -> pd.DataFrame:
    """to_reg24, gq_share, college_share, corr_share; NaN where the denominator is 0."""
    def ratio(n, d):
        return df[n] / df[d].where(df[d] > 0)

    return pd.DataFrame({
        "to_reg24": ratio("to24", "reg24"),
        "gq_share": ratio("gq", "total"),
        "college_share": ratio("college", "total"),
        "corr_share": ratio("corr_adult", "vap"),
    }, index=df.index)


def _adj(df: pd.DataFrame) -> pd.Series:
    return df["cvap_adj"] if "cvap_adj" in df else df.cvap - df.corr_adult


def p1(df: pd.DataFrame) -> pd.Series:
    """P1 per row; NaN where cvap_adj <= 0 (or missing) or the reason has LAND/PARTIAL/NO_CVAP."""
    adj = _adj(df)
    blank = df.not_ranked_reason.map(lambda s: bool(BLANK_REASONS & set(s.split(";"))))
    out = (df.to22 + df.to24) / (2 * adj.where(adj > 0))
    out = out.mask(blank)
    out.name = "P1"
    return out


def rank_pct(df: pd.DataFrame) -> pd.Series:
    """Percentile rank (0-100, 100 = highest P1) among ``not_ranked == False`` rows only."""
    ranked = ~df.not_ranked.astype(bool)
    out = pd.Series(np.nan, index=df.index, name="P1_rank")
    out[ranked] = df.P1[ranked].rank(pct=True) * 100
    return out


def rank_interval(df: pd.DataFrame, n: int = 1000, seed: int = 7) -> pd.DataFrame:
    """5th and 95th percentile of the rank over ``n`` draws of CVAP sampling error.

    Each draw: cvap_d = max(Normal(cvap, cvap_moe / 1.645), 1), cvap_adj_d = max(cvap_d -
    corr_adult, 1), re-ranked among ranked rows. NaN for not-ranked rows. Deterministic
    for a fixed ``seed``. Raises ValueError if a ranked row has a missing ``cvap_moe``.
    """
    ranked = ~df.not_ranked.astype(bool)
    r = df[ranked]
    out = pd.DataFrame(np.nan, index=df.index, columns=["rank_p05", "rank_p95"])
    if r.cvap_moe.isna().any():
        bad = sorted(r.index[r.cvap_moe.isna()])
        raise ValueError(f"{len(bad)} ranked rows have a missing cvap_moe: {bad[:10]}")
    if r.empty:
        return out
    rng = np.random.default_rng(seed)
    cvap, se = r.cvap.to_numpy(float), r.cvap_moe.to_numpy(float) / MOE_Z
    corr = r.corr_adult.to_numpy(float)
    num = ((r.to22 + r.to24) / 2).to_numpy(float)
    ranks = np.empty((n, len(r)))
    for i in range(n):
        cvap_d = np.maximum(rng.normal(cvap, se), 1)
        adj_d = np.maximum(cvap_d - corr, 1)
        ranks[i] = pd.Series(num / adj_d).rank(pct=True).to_numpy() * 100
    out.loc[ranked, "rank_p05"] = np.percentile(ranks, 5, axis=0)
    out.loc[ranked, "rank_p95"] = np.percentile(ranks, 95, axis=0)
    return out
