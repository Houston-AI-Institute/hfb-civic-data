from pathlib import Path

import pytest

from hfb import build


def _stub(monkeypatch, qa_ok=True, fetch_exc=None):
    calls = []

    def fake_fetch(raw_dir, seed_root):
        calls.append(("fetch", raw_dir, seed_root))
        if fetch_exc:
            raise fetch_exc
        return Path(raw_dir) / "manifest.csv"

    def fake_main(raw, out_dir=Path("out"), qa_dir=Path("qa")):
        calls.append(("assemble", raw, out_dir, qa_dir))
        return qa_ok

    monkeypatch.setattr(build.fetch, "fetch_all", fake_fetch)
    monkeypatch.setattr(build.assemble, "main", fake_main)
    return calls


def test_build_runs_fetch_then_assemble_with_parsed_args(monkeypatch):
    calls = _stub(monkeypatch)
    rc = build.cli(["raw", "--seed", "seedroot", "--out", "o", "--qa", "q"])
    assert rc == 0
    assert calls == [
        ("fetch", Path("raw"), Path("seedroot")),
        ("assemble", Path("raw"), Path("o"), Path("q")),
    ]


def test_build_defaults(monkeypatch):
    calls = _stub(monkeypatch)
    assert build.cli(["raw"]) == 0
    assert calls[0] == ("fetch", Path("raw"), None)
    assert calls[1] == ("assemble", Path("raw"), Path("out"), Path("qa"))


def test_build_exits_nonzero_when_qa_fails(monkeypatch):
    _stub(monkeypatch, qa_ok=False)
    with pytest.raises(SystemExit) as e:
        build.cli(["raw"])
    assert e.value.code == 1


def test_build_fetch_failure_propagates_and_skips_assemble(monkeypatch):
    calls = _stub(monkeypatch, fetch_exc=FileNotFoundError("seed missing"))
    with pytest.raises(FileNotFoundError):
        build.cli(["raw", "--seed", "s"])
    assert [c[0] for c in calls] == ["fetch"]
