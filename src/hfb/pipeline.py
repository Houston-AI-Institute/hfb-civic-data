"""Build the ZCTA table: one row per region ZCTA, inputs from Tasks 2-4, then P1 and flags.

Order of operations: inputs, derived columns, flags, P1 (blanked by the flags), ranks, then the
side columns (ACS context, IRS civic counts), which never feed any of the earlier steps.
``NONE`` (turnout outside every ZCTA) is not a row. Count columns stay float here.
"""

import logging
from pathlib import Path

import pandas as pd

from . import acs, context, crosswalk, flags, frame, irs, score, turnout
from .paths import raw_path

log = logging.getLogger(__name__)

def _require(index: pd.Index, have: pd.Index, what: str) -> None:
    missing = sorted(set(index) - set(have))
    if missing:
        raise ValueError(f"{len(missing)} region ZCTAs missing from {what}: {missing[:20]}")


def assemble(region, acs, zpop, alloc22, alloc24, split, inreg, side=None) -> pd.DataFrame:
    """Combine the per-ZCTA inputs into the table, rows = ``region.index``.

    Raises ValueError if ``acs`` or ``zpop`` lacks a region ZCTA. A region ZCTA absent from
    an allocation gets registration and turnout 0 (logged); a missing split_exposure is 0.
    ``side`` (ACS context and IRS civic columns) is joined last, aligned to the region, and
    changes no other column.
    """
    z = region.index
    _require(z, acs.index, "ACS B05003")
    _require(z, zpop.index, "zcta_pop (no 2020 blocks)")
    t = region[["county_main", "land_share"]].copy()
    t["inreg_share20"] = inreg.reindex(z)
    t["acs_pop"] = acs.acs_pop.reindex(z)
    for alloc, cols in ((alloc22, ["reg22", "to22"]), (alloc24, ["reg24", "to24"])):
        absent = sorted(set(z) - set(alloc.index))
        if absent:
            log.warning("%d region ZCTAs absent from the %s allocation, set to 0: %s",
                        len(absent), cols[0], absent)
        for c in cols:
            t[c] = alloc[c].reindex(z).fillna(0.0)
    t[["cvap", "cvap_moe"]] = acs[["cvap", "cvap_moe"]].reindex(z)
    t[["total", "vap", "gq", "corr_adult", "college"]] = zpop[
        ["total", "vap", "gq", "corr_adult", "college"]].reindex(z)
    t["cvap_adj"] = t.cvap - t.corr_adult
    t["split_exposure"] = split.reindex(z).fillna(0.0)
    t = t.join(score.derived(t))

    t = t.join(flags.apply_flags(t))
    t["P1"] = score.p1(t)
    t["P1_rank"] = score.rank_pct(t)
    t = t.join(score.rank_interval(t))
    if side is not None:
        t = t.join(side.reindex(z))
    t.insert(0, "ZCTA", t.index.astype(str))
    t.index.name = None  # ZCTA is a column; the index just carries the same labels
    return t


def build(raw: Path) -> tuple[pd.DataFrame, dict]:
    """The 240-row region table and the side-column QA facts (``irs.civic_counts`` qa plus
    ``context_blank``, the count of blank estimates per context column); reads ``raw``."""
    raw = Path(raw)
    blocks = crosswalk.build_blocks(raw, write=False)
    weights = crosswalk.vtd_weights(blocks)
    zpop = crosswalk.zcta_pop(blocks)
    vd22 = turnout.load_vd(raw_path(raw, "tlc_vd22"))
    vd24 = turnout.load_vd(raw_path(raw, "tlc_vd24"))
    rel = frame.read_rel(raw_path(raw, "zcta_county"))
    region = frame.region_zctas(rel)
    b05003 = acs.read_acs(raw_path(raw, "acs_b05003"), region.index)
    inreg = frame.inreg_share20(
        frame.read_rel(raw_path(raw, "zcta_tract")), frame.read_blocks_pop(raw), zpop.total)
    ctx = context.read_context(raw, region.index)
    civic, qa = irs.civic_counts(
        raw_path(raw, "irs_eo_tx"), region.index, set(rel.GEOID_ZCTA5_20))
    qa["context_blank"] = {k: int(ctx[k].isna().sum()) for k in context.SPECS}
    t = assemble(
        region, acs.cvap(b05003), zpop,
        turnout.allocate(weights, vd22, 22), turnout.allocate(weights, vd24, 24),
        turnout.split_exposure(weights, vd24), inreg, side=ctx.join(civic))
    return t, qa


def build_table(raw: Path) -> pd.DataFrame:
    """The 240-row region table (see ``assemble``); reads everything under ``raw``."""
    return build(raw)[0]
