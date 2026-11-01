import numpy as np
import pandas as pd
import pytest

from hfb import flags, score


def frame(n=6, **over):
    """n ranked rows with increasing turnout; P1 increases with the index."""
    idx = [f"770{i:02d}" for i in range(n)]
    d = pd.DataFrame({
        "to22": np.arange(1, n + 1) * 100.0,
        "to24": np.arange(1, n + 1) * 120.0,
        "cvap": 3000.0, "cvap_moe": 300.0, "corr_adult": 0.0,
        "not_ranked": False, "not_ranked_reason": "",
    }, index=idx)
    for k, v in over.items():
        d[k] = v
    d["cvap_adj"] = d.cvap - d.corr_adult
    return d


def with_p1(d):
    d = d.copy()
    d["P1"] = score.p1(d)
    return d


# ---------- p1 ----------

def test_p1_formula():
    d = frame(2, cvap=1000.0, corr_adult=[0.0, 200.0])
    d["cvap_adj"] = d.cvap - d.corr_adult
    got = score.p1(d)
    assert got.iloc[0] == pytest.approx((100 + 120) / (2 * 1000))
    assert got.iloc[1] == pytest.approx((200 + 240) / (2 * 800))


def test_p1_blank_for_land_partial_nocvap():
    d = frame(5)
    d["not_ranked_reason"] = ["LAND", "PARTIAL", "NO_CVAP", "POP;PARTIAL", "POP"]
    p = score.p1(d)
    assert p.iloc[:4].isna().all()
    assert p.iloc[4] == pytest.approx((500 + 600) / 6000)  # POP alone keeps P1


def test_p1_kept_for_gq_and_correctional_rows():
    d = frame(2)
    d["not_ranked_reason"] = ["GQ", "CORRECTIONAL"]
    assert score.p1(d).notna().all()


def test_p1_blank_where_cvap_adj_not_positive_and_division_safe():
    d = frame(3, cvap=[100.0, 100.0, 100.0], corr_adult=[100.0, 150.0, 0.0])
    d["cvap_adj"] = d.cvap - d.corr_adult
    with np.errstate(all="raise"):
        p = score.p1(d)
    assert p.iloc[:2].isna().all() and p.iloc[2] == pytest.approx((300 + 360) / (2 * 100))
    assert not np.isinf(p).any()


def test_p1_over_one_is_not_capped():
    d = frame(1, cvap=100.0)
    d["to22"], d["to24"] = 150.0, 150.0
    assert score.p1(d).iloc[0] == pytest.approx(1.5)


def test_p1_does_not_mutate_input():
    d = frame(3)
    before = d.copy()
    score.p1(d)
    pd.testing.assert_frame_equal(d, before)


# ---------- rank_pct ----------

def test_rank_direction_100_is_highest():
    r = score.rank_pct(with_p1(frame(5)))
    assert r.idxmax() == "77004" and r.max() == 100.0
    assert r.idxmin() == "77000" and r.min() == pytest.approx(20.0)
    assert r.is_monotonic_increasing


def test_rank_ignores_not_ranked():
    d = frame(6)
    d.loc["77005", ["not_ranked", "not_ranked_reason"]] = [True, "GQ"]  # highest P1
    r = score.rank_pct(with_p1(d))
    assert np.isnan(r["77005"])
    assert r["77004"] == 100.0  # top of the ranked rows, not displaced by 77005
    assert r.dropna().shape[0] == 5


def test_rank_ties_use_average_method():
    d = frame(3)
    d["to22"], d["to24"] = 100.0, 100.0
    d.loc["77002", ["to22", "to24"]] = [200.0, 200.0]
    r = score.rank_pct(with_p1(d))
    # tied pair share ranks 1.5 of 3 -> 50; top is 100
    assert r["77000"] == r["77001"] == pytest.approx(50.0)
    assert r["77002"] == 100.0


def test_rank_pct_matches_pandas_definition_on_random_data():
    rng = np.random.default_rng(3)
    d = frame(40)
    d["to22"], d["to24"] = rng.uniform(50, 900, 40), rng.uniform(50, 900, 40)
    d = with_p1(d)
    assert np.allclose(score.rank_pct(d), d.P1.rank(pct=True) * 100)


# ---------- rank_interval ----------

def test_interval_brackets_point_rank():
    rng = np.random.default_rng(5)
    d = frame(30)
    d["to22"], d["to24"] = rng.uniform(100, 1200, 30), rng.uniform(100, 1200, 30)
    d["cvap"] = rng.uniform(2000, 6000, 30)
    d["cvap_moe"] = d.cvap * 0.10
    d["cvap_adj"] = d.cvap - d.corr_adult
    d = with_p1(d)
    r = score.rank_pct(d)
    iv = score.rank_interval(d, n=1000, seed=7)
    assert list(iv.columns) == ["rank_p05", "rank_p95"]
    assert (iv.rank_p05 <= iv.rank_p95).all()
    assert ((iv.rank_p05 <= r + 1e-9) & (r <= iv.rank_p95 + 1e-9)).mean() >= 0.9


def test_interval_deterministic_for_fixed_seed_and_varies_with_seed():
    d = with_p1(frame(20, cvap_moe=900.0))
    a = score.rank_interval(d, n=200, seed=7)
    b = score.rank_interval(d, n=200, seed=7)
    c = score.rank_interval(d, n=200, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_interval_zero_moe_collapses_to_point_rank():
    d = with_p1(frame(8, cvap_moe=0.0))
    iv = score.rank_interval(d, n=50, seed=7)
    r = score.rank_pct(d)
    assert np.allclose(iv.rank_p05, r) and np.allclose(iv.rank_p95, r)


def test_interval_not_ranked_rows_are_nan():
    d = frame(6)
    d.loc["77001", ["not_ranked", "not_ranked_reason"]] = [True, "POP"]
    iv = score.rank_interval(with_p1(d), n=50)
    assert iv.loc["77001"].isna().all()
    assert iv.drop("77001").notna().all().all()
    assert iv.index.equals(d.index)


def test_interval_ignores_nan_moe_on_not_ranked_rows_but_raises_on_ranked():
    d = frame(6)
    d.loc["77001", ["not_ranked", "not_ranked_reason", "cvap_moe"]] = [True, "NO_CVAP", np.nan]
    score.rank_interval(with_p1(d), n=20)  # fine
    d.loc["77002", "cvap_moe"] = np.nan
    with pytest.raises(ValueError, match="cvap_moe"):
        score.rank_interval(with_p1(d), n=20)


def test_interval_draw_formula_matches_manual_computation():
    d = with_p1(frame(5, cvap=1500.0, cvap_moe=400.0, corr_adult=[0.0, 0, 100, 0, 0]))
    iv = score.rank_interval(d, n=40, seed=11)
    rng = np.random.default_rng(11)
    P = np.empty((40, 5))
    for i in range(40):
        cv = np.maximum(rng.normal(d.cvap.values, d.cvap_moe.values / 1.645), 1)
        adj = np.maximum(cv - d.corr_adult.values, 1)
        P[i] = pd.Series((d.to22.values + d.to24.values) / (2 * adj)).rank(pct=True).values * 100
    assert np.allclose(iv.rank_p05, np.percentile(P, 5, axis=0))
    assert np.allclose(iv.rank_p95, np.percentile(P, 95, axis=0))


# ---------- derived columns ----------

def test_derived_columns():
    d = pd.DataFrame({"to24": [50.0, 5.0], "reg24": [100.0, 0.0], "gq": [10, 0],
                      "total": [100, 0], "college": [20, 0], "corr_adult": [5, 0],
                      "vap": [50, 0]}, index=["a", "b"])
    out = score.derived(d)
    assert out.to_reg24.iloc[0] == 0.5 and np.isnan(out.to_reg24.iloc[1])
    assert out.gq_share.iloc[0] == 0.1 and np.isnan(out.gq_share.iloc[1])
    assert out.college_share.iloc[0] == 0.2
    assert out.corr_share.iloc[0] == 0.1 and np.isnan(out.corr_share.iloc[1])
    assert not np.isinf(out.to_numpy(dtype=float)).any()
