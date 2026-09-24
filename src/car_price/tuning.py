"""Optuna hyper-parameter search; the objective is the mean K-fold CV error on log-price."""
from __future__ import annotations

import optuna

from .evaluate import cv_evaluate
from .pipelines import build_pipeline

TUNABLE = ["ridge", "elasticnet", "poly_ridge", "rf", "xgb", "lgbm", "catboost"]


def suggest(name: str, trial: optuna.Trial) -> dict:
    """Search space per model family (log scales for rates / regularisation strengths)."""
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-2, 1e3, log=True)}
    if name == "poly_ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-1, 1e4, log=True)}
    if name == "elasticnet":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 1.0)}
    if name == "rf":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "max_depth": trial.suggest_int("max_depth", 8, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0)}
    if name == "xgb":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True)}
    if name == "lgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True)}
    if name == "catboost":
        return {"iterations": trial.suggest_int("iterations", 300, 800, step=100),
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True)}
    raise ValueError(f"no search space for '{name}'")


def tune(name: str, X, y, folds, cfg: dict, n_trials: int, storage: str | None = None,
         study_name: str | None = None) -> optuna.Study:
    """TPE search with median pruning; resumable when ``storage`` is a sqlite URL."""
    seed = cfg["seed"]

    def objective(trial: optuna.Trial) -> float:
        params = suggest(name, trial)
        pipe = build_pipeline(name, params, seed, cfg["cleaning"]["reference_year"])
        cv = cv_evaluate(pipe, X, y, folds, trial=trial)
        for m in ["mae", "rmse", "mape", "r2"]:
            trial.set_user_attr(m, float(cv[m].mean()))
        return float(cv["mae_log"].mean())

    study = optuna.create_study(
        study_name=study_name or f"car_price_{name}", storage=storage, load_if_exists=True,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    return study
