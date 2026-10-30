import numpy as np
import pandas as pd
import pytest

from hfb import acs

COLS = [f"B05003_{k}{i:03d}" for i in range(1, 24) for k in "EM"]


def write_dat(path, rows):
    """Pipe-delimited ACS file with all 46 columns; rows = {geo_id: {col: value}}."""
    with open(path, "w") as f:
        f.write("|".join(["GEO_ID"] + COLS) + "\n")
        for geo, vals in rows.items():
            f.write("|".join([geo] + [str(vals.get(c, 0)) for c in COLS]) + "\n")


def test_read_acs_filters_zctas_and_flags_sentinels(tmp_path):
    p = tmp_path / "b05003.dat"
    write_dat(p, {
        "0100000US": {"B05003_E001": 999},
        "860Z200US77001": {"B05003_E001": 1000, "B05003_M001": 50, "B05003_M009": -555555555},
        "860Z200US77002": {"B05003_E001": 2000, "B05003_E009": -666666666},
        "860Z200US99999": {"B05003_E001": 7},  # not requested
    })
    a = acs.read_acs(p, ["77001", "77002", "00000"])
    assert a.index.tolist() == ["77001", "77002"]  # absent ZCTA is not an error here
    assert a.index.name == "ZCTA"
    assert list(a.columns) == COLS
    assert all(dt == float for dt in a.dtypes)
    assert a.loc["77001", "B05003_E001"] == 1000
    assert np.isnan(a.loc["77001", "B05003_M009"])
    assert np.isnan(a.loc["77002", "B05003_E009"])
    assert a.loc["77001", "B05003_E009"] == 0  # zero stays zero


def test_cvap_formula():
    b = pd.DataFrame(
        {"B05003_E001": [1000, 500], "B05003_E008": [300, 200], "B05003_E019": [290, 150],
         "B05003_E009": [100, 10], "B05003_E011": [50, 20], "B05003_E020": [80, 30],
         "B05003_E022": [40, 40]}, index=pd.Index(["A", "B"], name="ZCTA")).astype(float)
    for c in ["M009", "M011", "M020", "M022"]:
        b[f"B05003_{c}"] = 0.0
    out = acs.cvap(b)
    assert list(out.columns) == ["acs_pop", "cvap", "cvap_moe", "acs_vap"]
    assert out.loc["A"].to_dict() == {"acs_pop": 1000, "cvap": 270, "cvap_moe": 0, "acs_vap": 590}
    assert out.loc["B"].to_dict() == {"acs_pop": 500, "cvap": 100, "cvap_moe": 0, "acs_vap": 350}


def test_moe_root_sum_square():
    b = pd.DataFrame({c: [0.0] for c in COLS}, index=pd.Index(["A"], name="ZCTA"))
    b["B05003_M009"], b["B05003_M011"], b["B05003_M020"], b["B05003_M022"] = 3.0, 4.0, 12.0, 0.0
    b["B05003_M001"] = 1000.0  # not part of the CVAP MOE
    assert acs.cvap(b).loc["A", "cvap_moe"] == pytest.approx(13.0)


def test_cvap_nan_propagates():
    b = pd.DataFrame({c: [1.0] for c in COLS}, index=pd.Index(["A"], name="ZCTA"))
    b["B05003_M020"] = np.nan
    out = acs.cvap(b)
    assert np.isnan(out.loc["A", "cvap_moe"])
    assert out.loc["A", "cvap"] == 4
