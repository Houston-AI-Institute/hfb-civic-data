"""Task 8: P1 map PNG and GeoJSON.

Unit tests use a small synthetic table and square geometries. The five named brief tests run
on the real build (``-m real``) and write into a temp dir, never into the repo's ``out/``.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from hfb import assemble, map as hmap, safeguards
from hfb.copy import CREDITS, MAP_FOOTER, MISUSE, USE_LIMIT

from test_assemble import synth_table

NOT_RANKED_BLANK = ["P1", "P1_rank", "rank_p05", "rank_p95"]


# ---------- synthetic geometry ----------

def synth_geoms(table):
    """One unit square per ZCTA in a row (EPSG:4269), and two county squares around them."""
    z = gpd.GeoDataFrame(
        {"ZCTA5CE20": table.ZCTA.tolist()},
        geometry=[box(-95.5 + i * 0.1, 29.5, -95.4 + i * 0.1, 29.6) for i in range(len(table))],
        crs=4269)
    c = gpd.GeoDataFrame(
        {"county": ["201", "157"]},
        geometry=[box(-95.55, 29.45, -95.25, 29.65), box(-95.25, 29.45, -94.9, 29.65)],
        crs=4269)
    return z, c


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    t = synth_table()
    z, c = synth_geoms(t)
    out = tmp_path_factory.mktemp("map_synth")
    png, gj = hmap.write_files(t, z, c, out)
    return t, out, png, gj, json.loads(gj.read_text(encoding="utf-8"))


def _feat(fc, zcta):
    return next(f for f in fc["features"] if f["properties"]["ZCTA"] == zcta)


# ---------- unit tests ----------

def test_class_breaks_are_quantiles_of_ranked_only():
    p1 = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    edges = hmap.class_breaks(p1)
    assert len(edges) == hmap.N_CLASSES + 1
    assert np.allclose(edges, np.quantile(p1, [0, .2, .4, .6, .8, 1]))
    cls = hmap.assign_classes(p1, edges)
    assert cls.min() == 0 and cls.max() == hmap.N_CLASSES - 1
    assert cls.value_counts().eq(2).all()


def test_synth_geojson_shape(synth):
    t, _, _, _, fc = synth
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == len(t)
    for f in fc["features"]:
        assert list(f["properties"]) == assemble.COLUMNS
        assert f["geometry"]["type"] in {"Polygon", "MultiPolygon"}
        assert f["properties"]["ZCTA"] == str(f["properties"]["ZCTA"])
        assert isinstance(f["properties"]["county_main"], str)


def test_synth_leading_zero_zcta_kept_as_text(synth):
    assert _feat(synth[4], "00501")["properties"]["ZCTA"] == "00501"


def test_synth_not_ranked_blank_but_ranked_kept(synth):
    t, _, _, _, fc = synth
    nr = t.set_index("ZCTA").not_ranked.astype(bool)
    assert nr.any() and (~nr).any()
    for f in fc["features"]:
        p = f["properties"]
        if nr[p["ZCTA"]]:
            assert p["not_ranked"] is True
            assert all(p[c] is None for c in NOT_RANKED_BLANK), p["ZCTA"]
        else:
            assert p["not_ranked"] is False
            assert all(p[c] is not None for c in NOT_RANKED_BLANK), p["ZCTA"]


def test_synth_correctional_row_has_measured_p1_in_table_but_null_in_geojson(synth):
    t, _, _, _, fc = synth
    row = t[t.ZCTA == "77003"].iloc[0]
    assert bool(row.not_ranked) and row.P1 == row.P1  # measured in the table (not NaN)
    assert _feat(fc, "77003")["properties"]["P1"] is None


def test_synth_foreign_members(synth):
    fc = synth[4]
    assert fc["use_limit"] == f"{MISUSE} {USE_LIMIT}"
    assert fc["credits"] == list(CREDITS)
    assert "crs" not in fc  # RFC 7946: coordinates are WGS 84 lon/lat


def test_synth_coordinates_are_lonlat_4326(synth):
    for f in synth[4]["features"]:
        for ring in f["geometry"]["coordinates"]:
            for lon, lat in ring:
                assert -180 <= lon <= 180 and -90 <= lat <= 90
                assert abs(lon + 95) < 1 and abs(lat - 29.5) < 1


def test_synth_reprojects_from_projected_input(tmp_path):
    t = synth_table()
    z, c = synth_geoms(t)
    png, gj = hmap.write_files(t, z.to_crs(3083), c.to_crs(3083), tmp_path)
    fc = json.loads(gj.read_text(encoding="utf-8"))
    lon, lat = fc["features"][0]["geometry"]["coordinates"][0][0]
    assert abs(lon + 95.5) < 1e-3 and abs(lat - 29.5) < 1e-3


def test_synth_png_written_nonempty(synth):
    _, _, png, _, _ = synth
    assert png.name == "zcta_p1_map.png"
    data = png.read_bytes()
    assert len(data) > 5000 and data[:8] == b"\x89PNG\r\n\x1a\n"


def test_synth_geojson_safeguards(synth):
    _, out, _, gj, fc = synth
    assert safeguards.check_forbidden_columns(fc["features"][0]["properties"]) == []
    assert safeguards.check_banned_words({"geojson": gj.read_text(encoding="utf-8")}) == []
    assert safeguards.check_out_dir(out) == []


def test_missing_zcta_geometry_raises(tmp_path):
    t = synth_table()
    z, c = synth_geoms(t)
    with pytest.raises(ValueError, match="77004"):
        hmap.write_files(t, z[z.ZCTA5CE20 != "77004"], c, tmp_path)


def test_county_outlines_dissolve_and_filter():
    vtd = gpd.GeoDataFrame(
        {"CNTY": [201, 201, 15, 999]},  # 15 must match the padded "015", 999 is out of region
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1), box(3, 0, 4, 1)], crs=3083)
    c = hmap.county_outlines(vtd, ["201", "015"])
    assert sorted(c.county) == ["015", "201"]
    assert c[c.county == "201"].geometry.iloc[0].area == pytest.approx(2.0)


def test_run_all_scans_geojson(synth, tmp_path):
    # the safeguards group for the geojson is empty on a clean file and flags a banned word
    t, out, png, gj, fc = synth
    import shutil
    d = tmp_path / "o"
    shutil.copytree(out, d)
    assert safeguards.run_all(d)["geojson"] == []
    (d / "zcta_civic.geojson").write_text(gj.read_text(encoding="utf-8") + "\nHFB\n")
    assert safeguards.run_all(d)["geojson"]


# ---------- real data (the five brief tests) ----------

@pytest.fixture(scope="module")
def real_map(raw, table, tmp_path_factory):
    out = tmp_path_factory.mktemp("map_real")
    png, gj = hmap.write_map(table, raw, out)
    return table, png, gj, json.loads(gj.read_text(encoding="utf-8"))


@pytest.mark.real
def test_geojson_feature_count_240(real_map):
    _, _, _, fc = real_map
    assert len(fc["features"]) == 240
    assert len({f["properties"]["ZCTA"] for f in fc["features"]}) == 240
    assert list(fc["features"][0]["properties"]) == assemble.COLUMNS
    assert all(len(f["properties"]["county_main"]) == 5 for f in fc["features"])


@pytest.mark.real
def test_not_ranked_feature_has_no_rank(real_map):
    t, _, _, fc = real_map
    nr = [f["properties"] for f in fc["features"] if f["properties"]["not_ranked"]]
    assert len(nr) == int(t.not_ranked.astype(bool).sum()) > 0
    for p in nr:
        assert all(p[c] is None for c in NOT_RANKED_BLANK), p["ZCTA"]
    for p in (f["properties"] for f in fc["features"] if not f["properties"]["not_ranked"]):
        assert all(p[c] is not None for c in ["P1", "P1_rank", "rank_p05", "rank_p95"])


@pytest.mark.real
def test_geojson_has_use_limit_and_credits(real_map):
    fc = real_map[3]
    assert fc["use_limit"] == f"{MISUSE} {USE_LIMIT}"
    assert fc["credits"] == list(CREDITS)


@pytest.mark.real
def test_geojson_no_forbidden_properties(real_map):
    _, _, gj, fc = real_map
    assert safeguards.check_forbidden_columns(fc["features"][0]["properties"]) == []
    assert safeguards.check_banned_words({"geojson": gj.read_text(encoding="utf-8")}) == []


@pytest.mark.real
def test_png_written_nonempty(real_map):
    png = real_map[1]
    assert png.name == "zcta_p1_map.png" and png.stat().st_size > 50_000


@pytest.mark.real
def test_real_geojson_matches_csv_values(real_map, real_outputs):
    out, _, _ = real_outputs
    csv = pd.read_csv(out / "zcta_civic.csv", skiprows=1, dtype={"ZCTA": str, "county_main": str})
    fc = real_map[3]
    gj = pd.DataFrame([f["properties"] for f in fc["features"]])
    assert list(gj.columns) == list(csv.columns)
    assert gj.ZCTA.tolist() == csv.ZCTA.tolist()
    ranked = ~csv.not_ranked
    for c in ["P1", "P1_rank", "land_share", "cvap", "civic_orgs_n"]:
        assert np.allclose(gj.loc[ranked, c].astype(float), csv.loc[ranked, c].astype(float))


@pytest.mark.real
def test_real_outputs_include_map_files(real_outputs):
    out, _, ok = real_outputs
    assert ok
    assert {"zcta_civic.csv", "zcta_civic.xlsx", "zcta_p1_map.png", "zcta_civic.geojson"} == {
        p.name for p in out.iterdir()}


@pytest.mark.real
def test_real_county_outlines_are_the_ten_region_counties(raw):
    from hfb.config import COUNTIES

    _, counties = hmap.load_geometries(raw)
    assert sorted(counties.county) == sorted(COUNTIES)
    assert not counties.geometry.is_empty.any()
