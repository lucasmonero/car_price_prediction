"""Command line entry point.

    python -m car_price.train --model lgbm --trials 40      # tune one model with CV
    python -m car_price.train --model all --trials 0        # untuned baselines for every model
    python -m car_price.train --final                       # refit best model, score test set ONCE

The hold-out test set is only read by ``--final``.
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import pandas as pd

from .config import load_config, resolve
from .data import TARGET, clean, load_raw
from .evaluate import (cv_evaluate, make_folds, metrics, segment_report, split_train_test,
                       summarise_cv)
from .pipelines import MODELS, build_pipeline
from .tuning import TUNABLE, tune

warnings.filterwarnings("ignore")


def load_train_test(cfg: dict, sample: bool = False):
    path = resolve(cfg, "sample" if sample else "raw")
    df = clean(load_raw(path), cfg)
    return split_train_test(df, cfg)


def _leaderboard_path(cfg):
    out = resolve(cfg, "results")
    out.mkdir(parents=True, exist_ok=True)
    return out / "cv_leaderboard.csv"


def _update_leaderboard(path, row: dict):
    lb = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(lb):
        lb = lb[~((lb["model"] == row["model"]) & (lb["tuned"] == row["tuned"]))]
    lb = pd.concat([lb, pd.DataFrame([row])], ignore_index=True)
    lb.sort_values("mae_log_mean").to_csv(path, index=False)


def run_model(name, cfg, X, y, folds, n_trials, storage):
    lb_path = _leaderboard_path(cfg)
    params, tuned = {}, False
    if n_trials > 0 and name in TUNABLE:
        study = tune(name, X, y, folds, cfg, n_trials, storage)
        params, tuned = dict(study.best_params), True
        if name == "lgbm":
            params["subsample_freq"] = 1
        print(f"[{name}] best CV mae_log={study.best_value:.4f} after {len(study.trials)} trials")
    pipe = build_pipeline(name, params, cfg["seed"], cfg["cleaning"]["reference_year"])
    cv = cv_evaluate(pipe, X, y, folds)
    row = {"model": name, "tuned": tuned, "trials": n_trials if tuned else 0,
           **summarise_cv(cv), "params": json.dumps(params)}
    _update_leaderboard(lb_path, row)
    print(f"[{name}] CV MAE {row['mae_mean']:.0f} +/- {row['mae_std']:.0f}  "
          f"MAPE {row['mape_mean']:.2%}  R2 {row['r2_mean']:.3f}")


def run_final(cfg, sample=False):
    lb_path = _leaderboard_path(cfg)
    best = pd.read_csv(lb_path).sort_values("mae_log_mean").iloc[0]
    params = json.loads(best["params"])
    print(f"Best CV model: {best['model']} (tuned={best['tuned']})")

    train, test = load_train_test(cfg, sample)
    Xtr, ytr = train.drop(columns=TARGET), train[TARGET]
    Xte, yte = test.drop(columns=TARGET), test[TARGET]
    model = build_pipeline(best["model"], params, cfg["seed"],
                           cfg["cleaning"]["reference_year"]).fit(Xtr, ytr)

    res = resolve(cfg, "results")
    test_metrics = {"model": best["model"], **metrics(yte, model.predict(Xte))}
    pd.DataFrame([test_metrics]).to_csv(res / "test_metrics.csv", index=False)
    segment_report(model, Xte, yte).to_csv(res / "test_segment_report.csv")
    models_dir = resolve(cfg, "models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "final_model.joblib", compress=3)
    print("Test metrics:", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in test_metrics.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--model", default="all", help=f"one of {MODELS} or 'all'")
    ap.add_argument("--trials", type=int, default=None, help="Optuna trials (0 = defaults, no tuning)")
    ap.add_argument("--final", action="store_true", help="refit best model and score the test set")
    ap.add_argument("--sample", action="store_true", help="use the synthetic sample dataset")
    ap.add_argument("--no-storage", action="store_true", help="do not persist Optuna studies")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.final:
        return run_final(cfg, args.sample)

    train, _ = load_train_test(cfg, args.sample)   # the test split is discarded here on purpose
    X, y = train.drop(columns=TARGET), train[TARGET]
    folds = make_folds(y, cfg)

    storage = None
    if not args.no_storage:
        db = resolve(cfg, "optuna_db")
        db.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{db.as_posix()}"

    for name in (MODELS if args.model == "all" else [args.model]):
        n_trials = cfg["tuning"]["n_trials"].get(name, 0) if args.trials is None else args.trials
        run_model(name, cfg, X, y, folds, n_trials, storage)


if __name__ == "__main__":
    main()
