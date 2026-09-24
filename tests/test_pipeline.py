import numpy as np
import pytest

from car_price.evaluate import cv_evaluate, make_folds, metrics, split_train_test, summarise_cv
from car_price.pipelines import MODELS, build_pipeline


@pytest.mark.parametrize("name", MODELS)
def test_every_model_fits_and_predicts(name, Xy):
    X, y = Xy
    params = {"rf": {"n_estimators": 20}, "xgb": {"n_estimators": 30},
              "lgbm": {"n_estimators": 30}, "catboost": {"iterations": 30}}.get(name, {})
    pipe = build_pipeline(name, params).fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all() and (pred > 0).all()   # predictions are in currency units


def test_boosting_beats_dummy_in_cv(Xy, cfg):
    X, y = Xy
    folds = make_folds(y, {**cfg, "split": {**cfg["split"], "n_folds": 2}})
    dummy = summarise_cv(cv_evaluate(build_pipeline("dummy"), X, y, folds))
    lgbm = summarise_cv(cv_evaluate(build_pipeline("lgbm", {"n_estimators": 60}), X, y, folds))
    assert lgbm["mae_log_mean"] < dummy["mae_log_mean"]


def test_folds_partition_the_training_data(Xy, cfg):
    _, y = Xy
    folds = make_folds(y, cfg)
    val_idx = np.concatenate([va for _, va in folds])
    assert len(folds) == cfg["split"]["n_folds"]
    assert sorted(val_idx) == list(range(len(y)))          # every row validated exactly once
    for tr, va in folds:
        assert not set(tr) & set(va)


def test_holdout_split_is_disjoint_and_stratified(clean_df, cfg):
    train, test = split_train_test(clean_df, cfg)
    assert len(train) + len(test) == len(clean_df)
    assert abs(len(test) / len(clean_df) - cfg["split"]["test_size"]) < 0.01
    assert abs(np.median(train["price"]) - np.median(test["price"])) / np.median(train["price"]) < 0.15


def test_metrics_perfect_prediction():
    m = metrics([1000, 2000, 3000], [1000, 2000, 3000])
    assert m["mae"] == 0 and m["r2"] == 1
