import numpy as np
import pandas as pd

from car_price.data import clean
from car_price.features import HIGH_CARD, LOW_CARD, NUMERIC, FeatureBuilder


def test_cleaning_removes_implausible_values(cfg, clean_df):
    assert len(clean_df) > 1000
    assert clean_df["ConsumeFuel"].max() <= cfg["cleaning"]["max_consume"]
    assert clean_df["price"].min() > 0
    assert not clean_df.duplicated().any()


def test_output_schema_and_no_inf(Xy):
    X, _ = Xy
    out = FeatureBuilder().fit(X).transform(X)
    assert list(out.columns) == NUMERIC + LOW_CARD + HIGH_CARD
    assert not np.isinf(out[NUMERIC].to_numpy(float)).any()
    assert out["Engine"].notna().all()           # imputed from cv
    assert out[LOW_CARD + HIGH_CARD].notna().all().all()


def test_age_zero_is_safe(Xy):
    """A car registered in the reference year must not produce inf / NaN mileage features."""
    X, _ = Xy
    row = X.iloc[[0]].copy()
    row["MatriculationYear"] = 2019
    out = FeatureBuilder(reference_year=2019).fit(X).transform(row)
    assert np.isfinite(out[["car_age", "avg_km_per_year", "km_vs_expected"]].to_numpy(float)).all()
    assert out["car_age"].iloc[0] == 1


def test_statistics_are_learned_from_fit_data_only(Xy):
    """Leakage guard: the km/year quantile and brand list must come from the fitted subset."""
    X, _ = Xy
    part = X.iloc[: len(X) // 2]
    fb = FeatureBuilder().fit(part)
    age = (2019 - part["MatriculationYear"]).clip(lower=1)
    assert np.isclose(fb.km_per_year_q99_, (part["km"] / age).quantile(0.99))
    assert fb.top_brands_ <= set(part["Brand"])
    # transform must not refit on new data
    before = fb.km_per_year_q99_
    fb.transform(X)
    assert fb.km_per_year_q99_ == before


def test_luxury_segment_and_unknown_brand(Xy):
    X, _ = Xy
    fb = FeatureBuilder().fit(X)
    probe = X.iloc[[0]].copy()
    probe["Brand"] = "FERRARI"
    assert fb.transform(probe)["brandSegment"].iloc[0] == "High Luxury"
    probe["Brand"] = "NEVER_SEEN_BRAND"
    assert fb.transform(probe)["brandSegment"].iloc[0] == "Niche"
