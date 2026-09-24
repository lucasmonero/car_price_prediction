"""Pipeline factory: raw cleaned DataFrame in, price (currency units) out.

Every model is wrapped in ``TransformedTargetRegressor(log1p / expm1)`` so training
happens on log-price for all families (relative-error friendly), and predictions come
back in currency units.
"""
from __future__ import annotations

import numpy as np
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, PolynomialFeatures,
                                   StandardScaler, TargetEncoder)
from xgboost import XGBRegressor

from .features import HIGH_CARD, LOW_CARD, NUMERIC, FeatureBuilder

class CatBoostReg(BaseEstimator, RegressorMixin):
    """sklearn-clone-safe CatBoost wrapper; categorical columns = the non-numeric ones."""

    def __init__(self, iterations=800, depth=8, learning_rate=0.08, l2_leaf_reg=3.0,
                 random_strength=1.0, random_seed=17):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.random_seed = random_seed

    def fit(self, X, y):
        cats = [c for c in X.columns if c in CATEGORICAL]
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg, random_strength=self.random_strength,
            random_seed=self.random_seed, verbose=0, thread_count=-1, cat_features=cats,
            allow_writing_files=False)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


MODELS = ["dummy", "ridge", "elasticnet", "poly_ridge", "rf", "xgb", "lgbm", "catboost"]
CATEGORICAL = set(LOW_CARD + HIGH_CARD)
LINEAR = {"ridge", "elasticnet", "poly_ridge"}


def _estimator(name: str, params: dict, seed: int):
    p = dict(params)
    if name == "dummy":
        return DummyRegressor(strategy="median")
    if name == "ridge" or name == "poly_ridge":
        return Ridge(**{"alpha": 10.0, **p})
    if name == "elasticnet":
        return ElasticNet(**{"alpha": 1e-3, "l1_ratio": 0.5, "max_iter": 5000, **p})
    if name == "rf":
        return RandomForestRegressor(**{"n_estimators": 200, "min_samples_leaf": 3,
                                        "max_features": 0.5, "n_jobs": -1,
                                        "random_state": seed, **p})
    if name == "xgb":
        return XGBRegressor(**{"n_estimators": 600, "learning_rate": 0.05, "max_depth": 8,
                               "tree_method": "hist", "n_jobs": -1, "random_state": seed, **p})
    if name == "lgbm":
        return LGBMRegressor(**{"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63,
                                "n_jobs": -1, "random_state": seed, "verbose": -1, **p})
    if name == "catboost":
        return CatBoostReg(**{"random_seed": seed, **p})
    raise ValueError(f"unknown model '{name}', choose from {MODELS}")


def _preprocessor(name: str, seed: int) -> ColumnTransformer:
    te = lambda: TargetEncoder(target_type="continuous",  # noqa: E731
                               cv=KFold(5, shuffle=True, random_state=seed))

    if name == "catboost":  # consumes raw categoricals natively
        transformers = [("num", "passthrough", NUMERIC),
                        ("cat", "passthrough", LOW_CARD + HIGH_CARD)]
    elif name in LINEAR:
        num = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        if name == "poly_ridge":
            num.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
        transformers = [
            ("num", Pipeline(num), NUMERIC),
            ("low", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                  sparse_output=False), LOW_CARD),
            ("high", Pipeline([("te", te()), ("scale", StandardScaler())]), HIGH_CARD),
        ]
    else:  # dummy + tree ensembles (NaN-aware in numeric columns)
        transformers = [
            ("num", "passthrough", NUMERIC),
            ("low", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), LOW_CARD),
            ("high", te(), HIGH_CARD),
        ]
    ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    return ct


def build_pipeline(name: str, params: dict | None = None, seed: int = 17,
                   reference_year: int = 2019) -> TransformedTargetRegressor:
    steps = [
        ("features", FeatureBuilder(reference_year=reference_year)),
        ("prep", _preprocessor(name, seed)),
        ("model", _estimator(name, params or {}, seed)),
    ]
    return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log1p,
                                      inverse_func=np.expm1, check_inverse=False)
