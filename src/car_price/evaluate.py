"""Splitting, cross-validation, metrics and reporting helpers."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import TARGET

METRICS = ["mae", "rmse", "mape", "r2", "mae_log"]


# ------------------------------------------------------------------ splitting
def _price_bins(y: pd.Series, n_bins: int) -> pd.Series:
    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")


def split_train_test(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single hold-out test set, stratified on price quantile bins."""
    s = cfg["split"]
    bins = _price_bins(df[TARGET], s["n_price_bins"])
    train, test = train_test_split(df, test_size=s["test_size"], stratify=bins,
                                   random_state=cfg["seed"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_folds(y: pd.Series, cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds stratified on price bins, so the luxury tail is spread across folds."""
    s = cfg["split"]
    bins = _price_bins(y, s["n_price_bins"])
    skf = StratifiedKFold(n_splits=s["n_folds"], shuffle=True, random_state=cfg["seed"])
    return list(skf.split(np.zeros(len(y)), bins))


# -------------------------------------------------------------------- metrics
def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.clip(np.asarray(y_pred, float), 1, None)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mae_log": mean_absolute_error(np.log1p(y_true), np.log1p(y_pred)),
    }


def cv_evaluate(pipeline, X: pd.DataFrame, y: pd.Series, folds, trial=None) -> pd.DataFrame:
    """Fit a fresh clone per fold; returns one row of metrics per fold.

    If an Optuna ``trial`` is given, intermediate values are reported after each fold so
    the pruner can stop unpromising configurations early.
    """
    rows = []
    for i, (tr, va) in enumerate(folds):
        model = clone(pipeline)
        t0 = time.perf_counter()
        model.fit(X.iloc[tr], y.iloc[tr])
        fit_time = time.perf_counter() - t0
        pred = model.predict(X.iloc[va])
        rows.append({"fold": i, "fit_time": fit_time, **metrics(y.iloc[va], pred)})
        if trial is not None:
            import optuna
            trial.report(float(np.mean([r["mae_log"] for r in rows])), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return pd.DataFrame(rows)


def oof_predict(pipeline, X: pd.DataFrame, y: pd.Series, folds) -> np.ndarray:
    """Out-of-fold predictions: every row is predicted by a model that never saw it."""
    pred = np.full(len(y), np.nan)
    for tr, va in folds:
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        pred[va] = model.predict(X.iloc[va])
    return pred


def summarise_cv(cv: pd.DataFrame) -> dict:
    out = {}
    for m in METRICS + ["fit_time"]:
        out[f"{m}_mean"], out[f"{m}_std"] = cv[m].mean(), cv[m].std(ddof=1)
    return out


# --------------------------------------------------------------------- reports
def segment_report(model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Error by brand segment; shows where the model is weak (e.g. rare luxury brands)."""
    builder = model.regressor_.named_steps["features"]
    seg = builder._segment(X["Brand"], X["Seats"])
    pred = np.clip(model.predict(X), 1, None)
    df = pd.DataFrame({"segment": seg.values, "y": y.values, "pred": pred})
    df["ape"] = (df["y"] - df["pred"]).abs() / df["y"]
    df["ae"] = (df["y"] - df["pred"]).abs()
    g = df.groupby("segment")
    return (pd.DataFrame({"n": g.size(), "MAPE": g["ape"].mean(), "MedAPE": g["ape"].median(),
                          "MAE": g["ae"].mean()})
            .sort_values("MAPE", ascending=False))


def plot_pred_vs_actual(y_true, y_pred, title: str = "Predicted vs actual price", ax=None):
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    ax.scatter(y_true, y_pred, s=6, alpha=0.3)
    lim = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lim, lim, "--", color="crimson", label="ideal")
    ax.set(xscale="log", yscale="log", xlabel="Actual price (€)", ylabel="Predicted price (€)",
           title=title)
    ax.legend()
    return ax
