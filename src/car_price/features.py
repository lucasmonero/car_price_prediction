"""Feature engineering as a scikit-learn transformer.

Everything data-dependent (99th percentile of km/year, most frequent brands, the
cv -> Engine imputation model) is learned in ``fit`` so that, inside cross-validation,
it only ever sees the training folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

LUXURY = {"BMW", "MERCEDES-BENZ", "AUDI", "LEXUS", "JAGUAR", "LAND ROVER"}
HIGH_LUXURY = {"LAMBORGHINI", "MCLAREN", "CADILLAC", "FERRARI", "BENTLEY",
               "ROLLS-ROYCE", "PORSCHE", "MASERATI"}
SPECIAL_TERMS = ["business", "sport", "lounge", "advantage", "coup", "titanium",
                 "plus", "msport", "luxury"]

NUMERIC = ["km", "cv", "car_age", "avg_km_per_year", "km_vs_expected", "Engine",
           "Seats", "ConsumeFuel", "Emissions", "specialModel"]
LOW_CARD = ["FuelType", "gearboxType", "Airbag", "EmissionClass", "AirConditioning",
            "NumberDoors", "brandSegment"]
HIGH_CARD = ["Brand", "Model", "province"]
RAW_COLUMNS = ["km", "cv", "MatriculationYear", "Engine", "Seats", "ConsumeFuel", "Emissions",
               "Preparation", "Brand", "Model", "province", "FuelType", "gearboxType",
               "Airbag", "EmissionClass", "AirConditioning", "NumberDoors"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Raw cleaned listings -> model-ready DataFrame (NUMERIC + LOW_CARD + HIGH_CARD)."""

    def __init__(self, reference_year: int = 2019, top_n_brands: int = 30,
                 clip_quantiles: tuple[float, float] = (0.001, 0.999)):
        self.reference_year = reference_year
        self.top_n_brands = top_n_brands
        self.clip_quantiles = clip_quantiles

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y=None):
        age = self._age(X)
        self.km_per_year_q99_ = float((X["km"] / age).quantile(0.99))
        self.top_brands_ = set(X["Brand"].value_counts().head(self.top_n_brands).index)

        known = X["Engine"].notna() & X["cv"].notna()
        self.engine_model_ = LinearRegression().fit(X.loc[known, ["cv"]], X.loc[known, "Engine"])

        # winsorisation bounds (learned on training data only) protect linear models
        # from extrapolating on rare extreme values
        raw = self._numeric(X)
        self.clip_lo_ = raw.quantile(self.clip_quantiles[0])
        self.clip_hi_ = raw.quantile(self.clip_quantiles[1])
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._numeric(X).clip(self.clip_lo_, self.clip_hi_, axis=1)
        out["specialModel"] = self._special(X)

        for col in ["FuelType", "gearboxType", "Airbag", "EmissionClass",
                    "AirConditioning", "NumberDoors"]:
            out[col] = X[col].astype(object).where(X[col].notna(), "Unknown")

        out["brandSegment"] = self._segment(X["Brand"], X["Seats"])

        for col in HIGH_CARD:
            out[col] = X[col].astype(object).where(X[col].notna(), "Unknown")
        return out[NUMERIC + LOW_CARD + HIGH_CARD]

    def get_feature_names_out(self, input_features=None):
        return np.array(NUMERIC + LOW_CARD + HIGH_CARD)

    # -------------------------------------------------------------- helpers
    def _numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        """Continuous features (before winsorisation)."""
        age = self._age(X)
        out = pd.DataFrame(index=X.index)
        out["km"] = X["km"].astype(float)
        out["cv"] = X["cv"].astype(float)
        out["car_age"] = age.astype(float)
        out["avg_km_per_year"] = out["km"] / age
        out["km_vs_expected"] = out["km"] / (age * self.km_per_year_q99_)  # ~[0, 1]

        engine = X["Engine"].astype(float)
        missing = engine.isna() & X["cv"].notna()
        if missing.any():
            engine = engine.copy()
            engine[missing] = self.engine_model_.predict(X.loc[missing, ["cv"]])
        out["Engine"] = engine
        out["Seats"] = X["Seats"].astype(float)
        out["ConsumeFuel"] = X["ConsumeFuel"].astype(float)
        out["Emissions"] = X["Emissions"].astype(float)
        return out

    @staticmethod
    def _special(X: pd.DataFrame) -> pd.Series:
        text = X["Preparation"].fillna("").astype(str).str.lower()
        text = text.str.replace(r"\S*\d\S*[.,/]?", "", regex=True)  # drop tokens containing digits
        return text.str.contains("|".join(SPECIAL_TERMS), na=False).astype(float)

    def _age(self, X: pd.DataFrame) -> pd.Series:
        # clip: a car registered in the reference year has age 0 -> avoid division by zero
        return (self.reference_year - X["MatriculationYear"]).clip(lower=1)

    def _segment(self, brand: pd.Series, seats: pd.Series) -> pd.Series:
        seg = pd.Series("Niche", index=brand.index, dtype=object)
        seg[brand.isin(self.top_brands_)] = "Ordinary"
        seg[brand.isin(LUXURY)] = "Luxury"
        seg[brand.isin(HIGH_LUXURY)] = "High Luxury"
        seg[seats > 9] = "Utility Vehicle"
        return seg
