"""The starter notebook runs on the built outputs and carries no ZIP codes or stored outputs."""

import json
import re
from pathlib import Path

import pytest
from test_county import synth_built, synth_county  # noqa: F401  (fixtures)

NB = Path(__file__).resolve().parent.parent / "starter" / "starter.ipynb"


def cells(kind):
    return ["".join(c["source"]) for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == kind]


def run(out_dir, monkeypatch):
    monkeypatch.chdir(out_dir)
    ns = {}
    for src in cells("code"):
        exec(compile(src, str(NB), "exec"), ns)
    return ns


def test_starter_has_no_outputs():
    nb = json.loads(NB.read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_starter_names_no_zip_codes():
    text = "".join(cells("code") + cells("markdown"))
    assert not re.search(r"(?<!\d)\d{5}(?!\d)", text)


def test_starter_score_leaves_conditions_out():
    from hfb.assemble import COLUMNS

    conditions = {c for c in COLUMNS[24:34] if not c.endswith("_moe")}
    assert len(conditions) == 5
    weights_cell = next(s for s in cells("code") if s.startswith("WEIGHTS"))
    assert not any(c in weights_cell for c in conditions)


def check_run(ns, n_rows):
    zcta, ranked, shortlist = ns["zcta"], ns["ranked"], ns["shortlist"]
    assert len(zcta) == n_rows
    assert not ranked["not_ranked"].any()
    assert shortlist.index.isin(ranked.index).all()
    assert not shortlist.index.duplicated().any()
    low = shortlist.loc[shortlist.group == "listen first", "score"]
    high = shortlist.loc[shortlist.group == "work with", "score"]
    assert len(low) > 0
    if len(high):
        assert low.max() <= high.min()
    assert zcta["ZCTA"].str.len().eq(5).all()


def test_starter_runs_on_synthetic_outputs(synth_built, monkeypatch):
    ns = run(synth_built.out, monkeypatch)
    check_run(ns, len(synth_built.table))
    assert (synth_built.out / "team_folder_shortlist.csv").exists()


@pytest.mark.real
def test_starter_runs_on_real_outputs(real_outputs, monkeypatch):
    out, _, _ = real_outputs
    ns = run(out, monkeypatch)
    check_run(ns, 240)
    assert len(ns["shortlist"]) == 8
    assert ns["shortlist"]["county_label"].nunique() >= 4
