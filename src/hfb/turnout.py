"""Allocate TLC general-election registration and turnout from VTDs to ZCTAs.

TLC publishes statewide VTD totals (``tlc_vd22.csv``, ``tlc_vd24.csv``). Each region VTD's
totals are split across ZCTAs by the VTD -> ZCTA weights from ``crosswalk.vtd_weights``.
Allocations stay float (no rounding here); ``NONE`` collects the share of a VTD that lies
outside every ZCTA. Files are joined on ``VTDKEY`` (int) only, never ``CNTYVTD``.
"""

from pathlib import Path

import pandas as pd

from .crosswalk import NONE

SPLIT_LOW, SPLIT_HIGH = 0.10, 0.90


def load_vd(path) -> pd.DataFrame:
    """TLC VTD file -> VTDKEY, CNTYVTD, Voter_Registration, Turnout.

    CNTYVTD is kept as a string (county = ``CNTYVTD[:3]``) for county checks only; it is
    never a join key. Every ``Spanish_Surname_*`` column is dropped here.
    """
    df = pd.read_csv(Path(path), dtype={"CNTYVTD": str})
    df = df[[c for c in df.columns if "spanish" not in c.lower()]]
    return df[["VTDKEY", "CNTYVTD", "Voter_Registration", "Turnout"]].astype(
        {"VTDKEY": int})


def _check(weights: pd.DataFrame, vd: pd.DataFrame) -> pd.DataFrame:
    """Region VTD rows of ``vd`` (those in ``weights``); raise on a missing or unallocatable VTD."""
    region = set(weights.VTDKEY)
    dup = vd.VTDKEY[vd.VTDKEY.isin(region) & vd.VTDKEY.duplicated()]
    if len(dup):
        raise ValueError(f"VTDKEY duplicated in vd: {sorted(set(dup))[:10]}")
    missing = sorted(region - set(vd.VTDKEY))
    if missing:
        raise ValueError(f"{len(missing)} region VTDKEYs missing from vd: {missing[:10]}")
    v = vd[vd.VTDKEY.isin(region)]
    share_sum = weights.groupby("VTDKEY").share.sum()
    bad = v[(v.Voter_Registration > 0) & (v.VTDKEY.map(share_sum) <= 0)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} VTDs have registrations but no positive share: "
            f"{sorted(bad.VTDKEY)[:10]}")
    return v


def pieces(weights: pd.DataFrame, vd: pd.DataFrame, yy: int) -> pd.DataFrame:
    """VTD x ZCTA pieces: VTDKEY, ZCTA, share, reg{yy}, to{yy} (share x VTD total)."""
    v = _check(weights, vd)
    p = weights.merge(v[["VTDKEY", "Voter_Registration", "Turnout"]], on="VTDKEY", how="left")
    p[f"reg{yy}"] = p.share * p.Voter_Registration
    p[f"to{yy}"] = p.share * p.Turnout
    return p[["VTDKEY", "ZCTA", "share", f"reg{yy}", f"to{yy}"]]


def allocate(weights: pd.DataFrame, vd: pd.DataFrame, yy: int) -> pd.DataFrame:
    """ZCTA -> reg{yy}, to{yy}, including a ``NONE`` row (zero if no block falls outside).

    Raises ``ValueError`` naming the key if a weights VTD is absent from ``vd`` or a VTD
    with registrations has no positive share. Rows of ``vd`` outside the weights are ignored.
    """
    cols = [f"reg{yy}", f"to{yy}"]
    out = pieces(weights, vd, yy).groupby("ZCTA")[cols].sum()
    if NONE not in out.index:
        out.loc[NONE] = 0.0
    return out


def none_bucket(alloc: pd.DataFrame) -> dict:
    """The ``NONE`` row as ``{column: value}``."""
    return alloc.loc[NONE].to_dict()


def split_exposure(weights: pd.DataFrame, vd: pd.DataFrame) -> pd.Series:
    """Per ZCTA, share of allocated registration from VTD pieces with share in [0.10, 0.90].

    Uses the ``Voter_Registration`` of the ``vd`` given (2024 in practice). ``NONE`` is
    excluded; a ZCTA with no allocated registration gets 0.0.
    """
    p = pieces(weights, vd, 0)
    p = p[p.ZCTA != NONE]
    exposed = p.share.between(SPLIT_LOW, SPLIT_HIGH, inclusive="both")
    tot = p.groupby("ZCTA").reg0.sum()
    part = p.reg0.where(exposed, 0.0).groupby(p.ZCTA).sum()
    s = (part / tot.where(tot > 0)).fillna(0.0)
    s.name = "split_exposure"
    return s
