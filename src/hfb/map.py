"""P1 map (PNG) and the GeoJSON of the hand-off table.

``write_map(table, raw, out_dir)`` writes ``zcta_p1_map.png`` and ``zcta_civic.geojson``.

* P1 is shown in five quantile classes computed over ranked rows only; NOT_RANKED rows are
  filled grey (``#bdbdbd``). No basemap tiles, no ZCTA labels.
* County outlines come from the TLC voting-district polygons (``tlc_vtds24``) dissolved on
  county (``CNTY``, zero-padded to three digits), kept to the region counties. There is no
  county shapefile in ``raw``.
* The PNG is drawn in EPSG:3083 (Texas Centric Albers); the GeoJSON is EPSG:4326 (RFC 7946, so
  no ``crs`` member).
* GeoJSON properties are the hand-off table columns in order (``assemble.COLUMNS``), rounded as
  in the CSV. Unlike the CSV, P1 is also null on every NOT_RANKED row, so a map reader never
  sees a measured value on a row the table leaves out of the ranking.
"""

import json
import logging
import textwrap
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")  # no display needed; set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from shapely.geometry import mapping  # noqa: E402

from . import assemble  # noqa: E402
from .config import COUNTIES  # noqa: E402
from .copy import CREDITS, MAP_FOOTER, MISUSE, USE_LIMIT  # noqa: E402
from .paths import raw_path  # noqa: E402

log = logging.getLogger(__name__)

PNG_NAME = "zcta_p1_map.png"
GEOJSON_NAME = "zcta_civic.geojson"
N_CLASSES = 5
NOT_RANKED_FILL = "#bdbdbd"
PLOT_CRS = 3083  # Texas Centric Albers
OUT_CRS = 4326
ZCTA_ID = "ZCTA5CE20"
CMAP = "Blues"
BLANK_WHEN_NOT_RANKED = ["P1", "P1_rank", "rank_p05", "rank_p95"]
COORD_GRID = 1e-6  # degrees, about 0.1 m


# ---------- classes ----------

def class_breaks(p1: pd.Series) -> np.ndarray:
    """The N_CLASSES + 1 quantile edges of ``p1`` (pass ranked rows only)."""
    return np.quantile(p1.dropna().to_numpy(float), np.linspace(0, 1, N_CLASSES + 1))


def assign_classes(p1: pd.Series, edges: np.ndarray) -> pd.Series:
    """Class 0..N_CLASSES-1 for each value; an edge value falls in the upper class, the maximum
    in the top class."""
    return pd.Series(np.searchsorted(edges[1:-1], p1.to_numpy(float), side="right"),
                     index=p1.index)


def class_colors() -> list[str]:
    """Hex colours, light to dark, from one sequential single-hue colormap."""
    return [matplotlib.colors.to_hex(plt.get_cmap(CMAP)(x)) for x in np.linspace(0.25, 0.95, N_CLASSES)]


def legend_labels(edges: np.ndarray) -> list[str]:
    return [f"{lo:.2f} to {hi:.2f}" for lo, hi in zip(edges[:-1], edges[1:])]


# ---------- geometry ----------

def county_outlines(vtd: gpd.GeoDataFrame, counties=COUNTIES) -> gpd.GeoDataFrame:
    """Dissolve voting-district polygons on county FIPS, region counties only.

    The shapefile's ``CNTYVTD`` is not zero-padded (Austin is ``150001``), unlike the TLC CSV's
    (``0150001``), so ``CNTYVTD[:3]`` is wrong for counties below 100. ``CNTY`` is the integer
    FIPS; padded to 3 it equals the CSV's ``CNTYVTD[:3]`` on all 9,712 voting districts.
    """
    v = vtd[["CNTY", "geometry"]].copy()
    v["county"] = v.CNTY.astype(int).map("{:03d}".format)
    v = v[v.county.isin(counties)]
    return v.dissolve(by="county").reset_index()[["county", "geometry"]]


def _shp(folder: Path) -> Path:
    found = sorted(Path(folder).glob("*.shp"))
    if len(found) != 1:
        raise FileNotFoundError(f"expected one .shp in {folder}, found {len(found)}")
    return found[0]


def load_geometries(raw) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """(ZCTA polygons, county outlines) read from ``raw`` through the manifest paths."""
    raw = Path(raw)
    z = gpd.read_file(_shp(raw_path(raw, "cb_zcta")), columns=[ZCTA_ID])
    vtd = gpd.read_file(_shp(raw_path(raw, "tlc_vtds24")), columns=["CNTY"])
    return z, county_outlines(vtd)


# ---------- GeoJSON ----------

def _geometry(g):
    g = shapely.orient_polygons(shapely.set_precision(g, COORD_GRID))  # RFC 7946 ring order
    return mapping(g)


def build_geojson(table: pd.DataFrame, zcta: gpd.GeoDataFrame) -> dict:
    """FeatureCollection, one feature per table row, in the CSV's row order."""
    f = assemble.format_table(table)
    geoms = zcta.drop_duplicates(ZCTA_ID).set_index(ZCTA_ID).to_crs(OUT_CRS).geometry
    missing = sorted(set(f.ZCTA) - set(geoms.index))
    if missing:
        raise ValueError(f"{len(missing)} table ZCTAs have no polygon: {missing[:20]}")
    features = []
    for rec in f.to_dict("records"):
        props = {c: assemble._cell(rec[c]) for c in assemble.COLUMNS}
        if props["not_ranked"]:
            for c in BLANK_WHEN_NOT_RANKED:
                props[c] = None
        features.append({"type": "Feature", "properties": props,
                         "geometry": _geometry(geoms[props["ZCTA"]])})
    return {"type": "FeatureCollection",
            "use_limit": f"{MISUSE} {USE_LIMIT}",
            "credits": list(CREDITS),
            "features": features}


# ---------- PNG ----------

def render_png(table: pd.DataFrame, zcta: gpd.GeoDataFrame, counties: gpd.GeoDataFrame,
               path) -> np.ndarray:
    """Draw the map to ``path``; returns the class edges used."""
    f = assemble.format_table(table)
    nr = f.not_ranked.astype(bool)
    edges = class_breaks(f.P1[~nr])
    colors = class_colors()
    cls = assign_classes(f.P1[~nr], edges)
    fill = {zc: NOT_RANKED_FILL for zc in f.ZCTA}
    fill.update({zc: colors[i] for zc, i in zip(f.ZCTA[~nr], cls)})

    z = zcta.drop_duplicates(ZCTA_ID).set_index(ZCTA_ID).to_crs(PLOT_CRS)
    z = z.loc[sorted(set(f.ZCTA) & set(z.index))]
    c = counties.to_crs(PLOT_CRS)

    fig, ax = plt.subplots(figsize=(10, 9.4))
    z.plot(ax=ax, color=[fill[i] for i in z.index], edgecolor="white", linewidth=0.3)
    c.boundary.plot(ax=ax, color="#222222", linewidth=1.1)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title("P1 reference measure by ZCTA, ten-county region", fontsize=13, loc="left")

    handles = [Patch(facecolor=colors[i], edgecolor="#777777", linewidth=0.4, label=lab)
               for i, lab in enumerate(legend_labels(edges))]
    handles.append(Patch(facecolor=NOT_RANKED_FILL, edgecolor="#777777", linewidth=0.4,
                         label="Not ranked"))
    handles.append(plt.Line2D([], [], color="#222222", linewidth=1.1, label="County outline"))
    ax.legend(handles=handles, title="P1, quintiles of ranked ZCTAs", loc="lower left",
              fontsize=9, title_fontsize=9, frameon=False)

    footer = "\n".join(textwrap.wrap(MAP_FOOTER, 120) + textwrap.wrap(CREDITS[0], 120))
    fig.text(0.02, 0.012, footer, fontsize=7.5, va="bottom", ha="left", color="#333333")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.11)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    return edges


# ---------- entry points ----------

def write_files(table: pd.DataFrame, zcta: gpd.GeoDataFrame, counties: gpd.GeoDataFrame,
                out_dir) -> tuple[Path, Path]:
    """Write the PNG and the GeoJSON from geometries already in memory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png, gj = out / PNG_NAME, out / GEOJSON_NAME
    fc = build_geojson(table, zcta)  # first: fails before anything is written if a polygon is missing
    edges = render_png(table, zcta, counties, png)
    gj.write_text(json.dumps(fc, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                  encoding="utf-8")
    log.info("P1 class breaks: %s", ", ".join(f"{e:.4f}" for e in edges))
    return png, gj


def write_map(table: pd.DataFrame, raw, out_dir) -> tuple[Path, Path]:
    """Read the ZCTA and VTD shapefiles under ``raw`` and write (PNG, GeoJSON) into ``out_dir``."""
    zcta, counties = load_geometries(raw)
    return write_files(table, zcta, counties, out_dir)
