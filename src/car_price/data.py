"""Loading and deterministic, row-level cleaning of the raw listings.

Nothing here learns statistics from the data that could leak into validation
folds: every rule is a fixed plausibility constant taken from ``configs/default.yaml``.
Data-dependent steps (imputation, brand segments, encodings) live in ``features.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "price"

# Rows missing any of these are dropped (all have <1% missing in the raw data).
REQUIRED = ["km", "cv", "FuelType", "gearboxType", "Brand", "Model", "Preparation"]

FUEL_GROUPS = {
    "Ibrida (benzina/elettrica)": "Alternative",
    "Ibrida (diesel/elettrica)": "Alternative",
    "Elettrico": "Alternative",
    "Idrogeno": "Alternative",
    "Metano": "Gas",
    "GPL": "Gas",
}


def load_raw(path) -> pd.DataFrame:
    # The source file mixes encodings (e.g. 'Sì', 'Coupé'); replace bad bytes instead of failing.
    return pd.read_csv(path, encoding="utf-8", encoding_errors="replace")


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, str]:
    """Raw data if present, otherwise the synthetic sample (so notebooks run on a fresh clone)."""
    from .config import resolve

    raw = resolve(cfg, "raw")
    if raw.exists():
        return load_raw(raw), "raw"
    print(f"{raw} not found: falling back to the synthetic sample (results will differ).")
    return load_raw(resolve(cfg, "sample")), "sample"


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    c = cfg["cleaning"]
    df = df.copy()

    obj = df.select_dtypes(include=["object", "string"]).columns
    for col in obj:
        df[col] = df[col].str.strip()

    df = df.drop_duplicates()

    df["FuelType"] = df["FuelType"].replace(FUEL_GROUPS)
    df["Brand"] = df["Brand"].replace({"MERCEDES BENZ": "MERCEDES-BENZ"})
    df["Model"] = df["Model"].replace({"Cinquecento": "500"})

    # --- year / km ---------------------------------------------------------
    df = df[df["MatriculationYear"] >= c["min_year"]]
    df = df[df["km"] <= c["max_km"]]
    age = (c["reference_year"] - df["MatriculationYear"]).clip(lower=1)  # age 0 must not zero the cap
    df = df[df["km"] <= age * c["max_km_per_year"]]

    # --- power -------------------------------------------------------------
    lo, hi = c["cv_range"]
    df = df[df["cv"].between(lo, hi)]
    small = df["Brand"].isin(["FIAT", "DAEWOO"]) & (df["cv"] > c["small_brand_max_cv"])
    df = df[~small]

    # --- engine / seats / consumption: implausible values become missing, imputed later ----
    e_lo, e_hi = c["engine_range"]
    bad_engine = (df["Engine"] < e_lo) | ((df["Engine"] > e_hi) & (df["Brand"] != "IVECO"))
    df.loc[bad_engine, "Engine"] = np.nan
    df.loc[df["Seats"] > c["max_seats"], "Seats"] = np.nan
    df.loc[df["ConsumeFuel"] > c["max_consume"], "ConsumeFuel"] = np.nan
    df.loc[df["Emissions"] > c["max_emissions"], "Emissions"] = np.nan

    df = df.dropna(subset=REQUIRED)
    df = df[df[TARGET] > 0]
    return df.reset_index(drop=True)
