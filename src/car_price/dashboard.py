"""Build the static, aggregate-only HTML dashboard (``docs/index.html``).

    python -m car_price.dashboard             # uses the cached out-of-sample predictions
    python -m car_price.dashboard --refresh   # recompute them (5 model refits)

Privacy: the raw dataset is private and an HTML file carries its data, so the page embeds only
counts / sums / histograms per bucketed filter cell (cells with fewer than ``MIN_CELL`` listings
are dropped). Individual listings are never written into the page.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data import TARGET, clean, load_raw
from .evaluate import make_folds, oof_predict, split_train_test
from .pipelines import build_pipeline

MIN_CELL = 5                                   # k-anonymity threshold for a filter cell
TEMPLATE = Path(__file__).with_name("dashboard_template.html")
PRED_COLUMNS = ["Brand", "FuelType", "gearboxType", "MatriculationYear", "km", "cv", TARGET]


# --------------------------------------------------------------- predictions
def out_of_sample_predictions(cfg: dict, sample: bool = False, force: bool = False) -> pd.DataFrame:
    """Honest prediction for every listing: out-of-fold for the train rows, final model for test rows."""
    cache = resolve(cfg, "interim") / ("oof_sample.csv" if sample else "oof_predictions.csv")
    if cache.exists() and not force:
        return pd.read_csv(cache)

    lb = pd.read_csv(resolve(cfg, "results") / "cv_leaderboard.csv")
    best = lb.sort_values("mae_log_mean").iloc[0]            # same rule as train.run_final
    params = json.loads(best["params"])

    df = clean(load_raw(resolve(cfg, "sample" if sample else "raw")), cfg)
    train, test = split_train_test(df, cfg)
    Xtr, ytr = train.drop(columns=TARGET), train[TARGET]
    pipe = build_pipeline(best["model"], params, cfg["seed"], cfg["cleaning"]["reference_year"])

    print(f"out-of-fold predictions with {best['model']} on {len(train):,} train rows ...")
    train = train.assign(pred=oof_predict(pipe, Xtr, ytr, make_folds(ytr, cfg)), split="oof")

    final = joblib.load(resolve(cfg, "models") / "final_model.joblib")
    test = test.assign(pred=final.predict(test.drop(columns=TARGET)), split="test")

    out = pd.concat([train, test], ignore_index=True)[PRED_COLUMNS + ["pred", "split"]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


# ---------------------------------------------------------------- aggregation
N_BINS = 60
BIN_EDGES = np.geomspace(500, 250_000, N_BINS + 1)            # shared price grid, log-spaced
FUEL = {"Diesel": "Diesel", "Benzina": "Petrol", "Gas": "Gas (LPG/CNG)", "Alternative": "Hybrid / electric"}
GEAR = {"Cambio manuale": "Manual", "Cambio automatico": "Automatic", "Cambio Semiautomatico": "Automatic"}
AGE_EDGES, AGE_LABELS = [0, 3, 6, 10, 15, np.inf], ["1–3 yrs", "4–6 yrs", "7–10 yrs", "11–15 yrs", "16+ yrs"]
KM_EDGES, KM_LABELS = [0, 30e3, 80e3, 150e3, 250e3, np.inf], ["<30k km", "30–80k km", "80–150k km", "150–250k km", "250k+ km"]
CV_EDGES, CV_LABELS = [0, 70, 100, 140, 200, np.inf], ["<70 cv", "70–99 cv", "100–139 cv", "140–199 cv", "200+ cv"]
DIMS = ["brand", "fuel", "gear", "age", "km", "power"]         # order of the codes inside each cell
MODEL_NAMES = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "Random forest", "catboost": "CatBoost",
               "poly_ridge": "Polynomial Ridge", "ridge": "Ridge", "elasticnet": "ElasticNet",
               "dummy": "Median baseline"}


def _pretty_brand(name: str) -> str:
    return name if len(name) <= 3 else name.title()


def _price_bin(x) -> np.ndarray:
    """Bin index on the shared grid; out-of-range prices are clamped into the end bins."""
    return np.clip(np.searchsorted(BIN_EDGES, np.asarray(x, float), side="right") - 1, 0, N_BINS - 1)


def _bucket(x: pd.Series, edges, labels) -> np.ndarray:
    return pd.cut(x, bins=edges, labels=False, right=False).to_numpy()


def build_cube(df: pd.DataFrame, reference_year: int = 2019, min_cell: int = MIN_CELL,
               n_brands: int = 20) -> dict:
    """Aggregate listings (with ``price`` and ``pred``) into bucketed filter cells.

    Pure function: cells with fewer than ``min_cell`` listings are dropped, and nothing per-listing
    survives the aggregation.
    """
    d = df.copy()
    top = d["Brand"].value_counts().index[:n_brands].tolist()
    brand_labels = [_pretty_brand(b) for b in top] + ["Other brands"]
    d["brand"] = d["Brand"].map({b: i for i, b in enumerate(top)}).fillna(n_brands).astype(int)
    fuel_labels = list(dict.fromkeys(FUEL.values())) + ["Other"]
    d["fuel"] = d["FuelType"].map(lambda v: fuel_labels.index(FUEL.get(v, "Other")))
    gear_labels = ["Manual", "Automatic"]
    d["gear"] = d["gearboxType"].map(lambda v: gear_labels.index(GEAR.get(v, "Manual")))
    age = (reference_year - d["MatriculationYear"]).clip(lower=1)
    d["age"] = _bucket(age, AGE_EDGES, AGE_LABELS)
    d["km"] = _bucket(d["km"], KM_EDGES, KM_LABELS)
    d["power"] = _bucket(d["cv"], CV_EDGES, CV_LABELS)

    y, p = d[TARGET].to_numpy(float), d["pred"].to_numpy(float)
    d["y"], d["p"] = y, p
    d["abs"], d["ape"] = np.abs(y - p), np.abs(y - p) / y
    d["by"], d["bp"] = _price_bin(y), _price_bin(p)
    d["cell"] = d.groupby(DIMS, sort=True).ngroup()

    cells = d.groupby("cell").agg(**{k: (k, "first") for k in DIMS}, n=("y", "size"), sy=("y", "sum"),
                                  sp=("p", "sum"), sabs=("abs", "sum"), sape=("ape", "sum"))
    keep = cells[cells["n"] >= min_cell]

    def hist(col: str) -> dict:
        h = d[d["cell"].isin(keep.index)].groupby(["cell", col]).size()
        out: dict = {}
        for (c, b), n in h.items():
            out.setdefault(c, []).extend([int(b), int(n)])       # flat [bin, count, bin, count, ...]
        return out

    hy, hp = hist("by"), hist("bp")
    rows = [[int(r[k]) for k in DIMS] + [int(r.n), round(r.sy), round(r.sp, 1), round(r.sabs, 1),
                                          round(r.sape, 4), hy[c], hp[c]] for c, r in keep.iterrows()]
    n_shown = int(keep["n"].sum())
    kept = d[d["cell"].isin(keep.index)]                     # computed from rows, independently of the cell sums
    reference = {
        "n": len(kept), "mean_actual": float(kept["y"].mean()), "mean_pred": float(kept["p"].mean()),
        "mae": float(kept["abs"].mean()), "mape": float(kept["ape"].mean()),
        "median_actual": float(kept["y"].median()), "median_pred": float(kept["p"].median()),
        "brand_n": [int(v) for v in kept["brand"].value_counts().reindex(range(len(brand_labels)), fill_value=0)],
    }
    return {
        "meta": {
            "bins": [round(float(e), 1) for e in BIN_EDGES],
            "dims": {"brand": brand_labels, "fuel": fuel_labels, "gear": gear_labels, "age": AGE_LABELS,
                     "km": KM_LABELS, "power": CV_LABELS},
            "totals": {"n_total": len(d), "n_shown": n_shown, "coverage": n_shown / len(d),
                       "cells_shown": len(keep), "cells_hidden": int(len(cells) - len(keep)),
                       "min_cell": min_cell, "mape_all": float(d["ape"].mean()),
                       "mape_shown": float(kept["ape"].mean()),
                       "mape_hidden": float(d.loc[~d["cell"].isin(keep.index), "ape"].mean())
                       if n_shown < len(d) else None},
            "reference": reference,
        },
        "cells": rows,
    }


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def render(cube: dict, leaderboard: pd.DataFrame, test_metrics: pd.DataFrame, segments: pd.DataFrame,
           min_cell: int = MIN_CELL) -> str:
    lb = leaderboard.assign(label=leaderboard["model"].map(MODEL_NAMES)).sort_values("mae_log_mean")
    cols = ["model", "label", "tuned", "trials", "mae_mean", "mae_std", "mape_mean", "mape_std",
            "r2_mean", "r2_std", "mae_log_mean", "mae_log_std"]
    payload = {
        **cube,
        "models": _records(lb[cols].round(5)),
        "test": _records(test_metrics.round(5))[0],
        "segments": _records(segments[segments["n"] >= min_cell].round(4)),
    }
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob)


def write_dashboard(cfg: dict, preds: pd.DataFrame) -> Path:
    res = resolve(cfg, "results")
    cube = build_cube(preds, cfg["cleaning"]["reference_year"])
    html = render(cube, pd.read_csv(res / "cv_leaderboard.csv"), pd.read_csv(res / "test_metrics.csv"),
                  pd.read_csv(res / "test_segment_report.csv"))
    out = resolve(cfg, "dashboard")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    t = cube["meta"]["totals"]
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) | {t['cells_shown']:,} cells, "
          f"coverage {t['coverage']:.1%}, {t['cells_hidden']:,} small cells hidden")
    return out


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="recompute the cached predictions")
    ap.add_argument("--sample", action="store_true", help="use the synthetic sample dataset")
    args = ap.parse_args(argv)
    cfg = load_config()
    preds = out_of_sample_predictions(cfg, sample=args.sample, force=args.refresh)
    print(f"{len(preds):,} predictions ready")
    write_dashboard(cfg, preds)


if __name__ == "__main__":
    main()
