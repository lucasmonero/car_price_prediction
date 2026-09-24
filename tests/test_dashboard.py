import json
import re

import numpy as np
import pandas as pd
import pytest

from car_price.dashboard import MIN_CELL, build_cube, render


@pytest.fixture(scope="module")
def preds(clean_df):
    rng = np.random.default_rng(0)
    df = clean_df[["Brand", "FuelType", "gearboxType", "MatriculationYear", "km", "cv", "price"]].copy()
    df["pred"] = df["price"] * rng.lognormal(0, 0.1, len(df))
    return df


@pytest.fixture(scope="module")
def cube(preds):
    return build_cube(preds)


def test_no_cell_below_the_privacy_threshold(cube):
    assert cube["cells"], "cube is empty"
    assert all(cell[6] >= MIN_CELL for cell in cube["cells"])


def test_histograms_add_up_to_the_cell_count(cube):
    for cell in cube["cells"]:
        n, hy, hp = cell[6], cell[11], cell[12]
        assert sum(hy[1::2]) == n and sum(hp[1::2]) == n
        assert len(hy) % 2 == 0                                 # flat [bin, count, ...] pairs


def test_accounting_of_shown_and_hidden_listings(cube, preds):
    t = cube["meta"]["totals"]
    assert t["n_total"] == len(preds)
    assert t["n_shown"] == sum(c[6] for c in cube["cells"]) == cube["meta"]["reference"]["n"]
    assert 0 < t["coverage"] <= 1
    assert t["cells_hidden"] >= 0


def test_reference_values_match_the_cells(cube):
    ref, cells = cube["meta"]["reference"], cube["cells"]
    n = sum(c[6] for c in cells)
    assert sum(c[7] for c in cells) / n == pytest.approx(ref["mean_actual"], rel=1e-3)
    assert sum(c[9] for c in cells) / n == pytest.approx(ref["mae"], rel=1e-3)


def test_dimension_codes_stay_inside_their_labels(cube):
    dims = cube["meta"]["dims"]
    for cell in cube["cells"]:
        for code, name in zip(cell[:6], ["brand", "fuel", "gear", "age", "km", "power"]):
            assert 0 <= code < len(dims[name])


def test_higher_threshold_hides_more(preds):
    assert build_cube(preds, min_cell=50)["meta"]["totals"]["n_shown"] <= build_cube(preds)["meta"]["totals"]["n_shown"]


def _payload(html: str) -> dict:
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    return json.loads(blob.replace("<\\/", "</"))


def test_rendered_page_embeds_only_aggregates(cube):
    lb = pd.DataFrame({"model": ["lgbm", "dummy"], "tuned": [True, False], "trials": [20, 0],
                       **{f"{m}_{s}": [0.1, 0.2] for m in ["mae", "mape", "r2", "mae_log"] for s in ["mean", "std"]}})
    test = pd.DataFrame([{"model": "lgbm", "mae": 1.0, "rmse": 2.0, "mape": 0.1, "r2": 0.8, "mae_log": 0.1}])
    seg = pd.DataFrame({"segment": ["Ordinary", "Tiny"], "n": [500, 2], "MAPE": [0.1, 0.9], "MedAPE": [0.1, 0.9], "MAE": [1.0, 2.0]})
    html = render(cube, lb, test, seg)
    assert "__DATA__" not in html
    data = _payload(html)
    assert set(data) == {"meta", "cells", "models", "test", "segments"}      # no per-listing table
    assert all(len(c) == 13 for c in data["cells"])
    assert len(data["cells"]) < cube["meta"]["totals"]["n_total"]           # aggregated, not one row per car
    assert [s["segment"] for s in data["segments"]] == ["Ordinary"]         # tiny segment suppressed
    assert "</" not in html.split('type="application/json">')[1].split("</script>")[0]
