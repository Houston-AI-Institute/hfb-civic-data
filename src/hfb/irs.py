"""IRS Exempt Organizations Business Master File (Texas): civic-organization counts by ZCTA.

Counts only. ``NAME`` is read to spot homeowner-association-like names and is never returned;
no name, street or ICO leaves this module.
"""

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CIVIC_PREFIXES = {"S20", "S21", "S22", "S80", "S81", "S82", "W24"}  # NTEE_CD[:3], any subsection
HOA_NAME = re.compile(
    r"HOMEOWNER|HOME OWNER|PROPERTY OWNER|OWNERS ASSOC|\bPOA\b|\bHOA\b|TOWNHOME|CONDO", re.I)
HOA_NTEE = "L5"
ACTIVE = "01"
COLS = ["NAME", "ZIP", "STATUS", "NTEE_CD"]


def band(n: int) -> str:
    """'0', '1-2', '3-5' or '6+'."""
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else "6+"


def _read(path) -> tuple[pd.DataFrame, str]:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=COLS,
                               encoding=enc), enc
        except UnicodeDecodeError:
            log.warning("%s is not valid %s", Path(path).name, enc)
    raise UnicodeDecodeError("latin-1", b"", 0, 1, f"cannot decode {path}")  # unreachable


def civic_counts(irs_csv, zctas, zcta_all) -> tuple[pd.DataFrame, dict]:
    """``(DataFrame[civic_orgs_n, civic_orgs_band, hoa_like_n] by zctas, qa dict)``.

    Active (STATUS 01) rows only; ZIP5 is ``ZIP[:5]``. ``R40`` and ``R6*`` are excluded from
    both counts. ``qa`` counts, over the whole Texas extract: ``irs_total_active``; in the
    region, ``civic_region``, ``hoa_like_region`` and ``overlap`` (rows in both);
    ``lost_irs_rows`` (civic rows whose ZIP5 is in no ZCTA but whose ZIP3 is a region ZIP3, so
    no ZCTA can hold them) and ``out_of_region`` (the other civic rows outside the region);
    plus the ``encoding`` that read the file.
    """
    zctas = pd.Index(zctas, name="ZCTA")
    region = set(zctas)
    region_zip3 = {z[:3] for z in region}
    b, enc = _read(irs_csv)
    b = b[b.STATUS == ACTIVE]
    ntee = b.NTEE_CD
    keep = ~((ntee.str[:3] == "R40") | (ntee.str[:2] == "R6"))
    civic = ntee.str[:3].isin(CIVIC_PREFIXES) & keep
    hoa = (b.NAME.str.contains(HOA_NAME) | ntee.str.startswith(HOA_NTEE)) & keep
    zip5 = b.ZIP.str[:5]
    in_region = zip5.isin(region)
    lost = civic & ~in_region & ~zip5.isin(set(zcta_all)) & b.ZIP.str[:3].isin(region_zip3)
    n_civic = zip5[civic & in_region].value_counts().reindex(zctas, fill_value=0)
    n_hoa = zip5[hoa & in_region].value_counts().reindex(zctas, fill_value=0)
    df = pd.DataFrame({"civic_orgs_n": n_civic.astype(int)})
    df["civic_orgs_band"] = df.civic_orgs_n.map(band)
    df["hoa_like_n"] = n_hoa.astype(int)
    qa = {
        "irs_total_active": len(b),
        "civic_region": int(n_civic.sum()),
        "hoa_like_region": int(n_hoa.sum()),
        "overlap": int((civic & hoa & in_region).sum()),
        "lost_irs_rows": int(lost.sum()),
        "out_of_region": int((civic & ~in_region).sum() - lost.sum()),
        "encoding": enc,
    }
    log.info("IRS civic counts: %s", qa)
    return df, qa
