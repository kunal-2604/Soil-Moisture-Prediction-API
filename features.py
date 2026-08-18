"""
features.py — Feature engineering layer for soil moisture prediction.

Domain knowledge encoded here:
- Soil water-holding capacity (WHC) from SoilGrids / FAO literature
- Crop coefficients (Kc) from FAO Irrigation Paper 56, Table 12
- Evapotranspiration index via simplified Hargreaves equation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Union


# ─── Domain Lookup Tables ───────────────────────────────────────────────────

# Field capacity (%) by soil type — source: FAO / SoilGrids
# Range: (min_WHC, max_WHC, typical_WHC)
SOIL_WHC: dict[str, float] = {
    "clay":        52.0,
    "loam":        35.0,
    "sandy loam":  25.0,
    "sandy":       14.0,
    "silt":        40.0,
    "peat":        68.0,
    "clay loam":   42.0,
    "silt loam":   38.0,
}

# Permanent wilting point (%) — soil can't release water below this
SOIL_PWP: dict[str, float] = {
    "clay":        27.0,
    "loam":        14.0,
    "sandy loam":  10.0,
    "sandy":        5.0,
    "silt":        18.0,
    "peat":        30.0,
    "clay loam":   21.0,
    "silt loam":   17.0,
}

# FAO-56 Kc (crop coefficient) — mid-season values
# Source: FAO Irrigation Paper 56, Table 12
CROP_KC: dict[str, float] = {
    "wheat":       1.15,
    "rice":        1.20,
    "maize":       1.20,
    "corn":        1.20,
    "cotton":      1.15,
    "soybean":     1.15,
    "potato":      1.10,
    "tomato":      1.15,
    "sugarcane":   1.25,
    "sunflower":   1.10,
    "barley":      1.15,
    "chickpea":    1.00,
    "groundnut":   1.15,
    "millet":      1.00,
    "sorghum":     1.05,
    "default":     1.10,
}

# Default fallbacks
DEFAULT_WHC = 30.0
DEFAULT_PWP = 12.0
DEFAULT_KC  = 1.10


# ─── Lookup Functions ─────────────────────────────────────────────────────────

def get_soil_whc(soil_type: Union[str, pd.Series]) -> Union[float, pd.Series]:
    """Return field capacity (%) for given soil type(s). Case-insensitive."""
    if isinstance(soil_type, pd.Series):
        return soil_type.str.lower().map(SOIL_WHC).fillna(DEFAULT_WHC)
    return SOIL_WHC.get(str(soil_type).lower(), DEFAULT_WHC)


def get_soil_pwp(soil_type: Union[str, pd.Series]) -> Union[float, pd.Series]:
    """Return permanent wilting point (%) for given soil type(s)."""
    if isinstance(soil_type, pd.Series):
        return soil_type.str.lower().map(SOIL_PWP).fillna(DEFAULT_PWP)
    return SOIL_PWP.get(str(soil_type).lower(), DEFAULT_PWP)


def get_crop_kc(crop_type: Union[str, pd.Series]) -> Union[float, pd.Series]:
    """Return FAO-56 crop coefficient for given crop type(s)."""
    if isinstance(crop_type, pd.Series):
        return crop_type.str.lower().map(CROP_KC).fillna(DEFAULT_KC)
    return CROP_KC.get(str(crop_type).lower(), DEFAULT_KC)


# ─── Feature Engineering Functions ───────────────────────────────────────────

def et0_hargreaves(temp: Union[float, np.ndarray, pd.Series],
                   humidity: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """
    Simplified reference evapotranspiration (ET₀) via Hargreaves approximation.

    ET₀ ≈ 0.0023 × (T + 17.8) × (1 - RH/100) × 0.408
    Units: mm/day equivalent (dimensionless proxy here)

    Args:
        temp:     Air temperature in °C
        humidity: Relative humidity in %
    Returns:
        ET₀ index (dimensionless, higher = more evaporation)
    """
    return 0.0023 * (temp + 17.8) * (1.0 - humidity / 100.0) * 0.408


def et_crop(temp, humidity, crop_type: Union[str, pd.Series]):
    """
    Crop-adjusted evapotranspiration (ETc = Kc × ET₀).
    """
    kc = get_crop_kc(crop_type)
    return kc * et0_hargreaves(temp, humidity)


def moisture_stress_index(days: Union[float, np.ndarray, pd.Series],
                          temp: Union[float, np.ndarray, pd.Series],
                          humidity: Union[float, np.ndarray, pd.Series],
                          crop_type: Union[str, pd.Series] = "default") -> Union[float, np.ndarray, pd.Series]:
    """
    Core moisture depletion signal: ETc × days_since_watering.
    Higher value → more water lost since last watering.
    """
    return et_crop(temp, humidity, crop_type) * days


def available_water_capacity(soil_type: Union[str, pd.Series]) -> Union[float, pd.Series]:
    """
    Available Water Capacity = WHC - PWP
    Represents the plant-accessible water range.
    """
    return get_soil_whc(soil_type) - get_soil_pwp(soil_type)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering to a dataframe with columns:
        crop_type, soil_type, temperature, humidity, days_since_watering

    Returns dataframe with original columns + engineered features.
    """
    df = df.copy()

    # Domain lookups
    df["soil_whc"]    = get_soil_whc(df["soil_type"])
    df["soil_pwp"]    = get_soil_pwp(df["soil_type"])
    df["crop_kc"]     = get_crop_kc(df["crop_type"])
    df["soil_awc"]    = df["soil_whc"] - df["soil_pwp"]

    # ET features
    df["et0"]             = et0_hargreaves(df["temperature"], df["humidity"])
    df["etc"]             = df["crop_kc"] * df["et0"]
    df["moisture_stress"] = df["etc"] * df["days_since_watering"]

    # Nonlinear interactions
    df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1e-6)
    df["days_x_et0"]          = df["days_since_watering"] * df["et0"]
    df["stress_per_whc"]      = df["moisture_stress"] / (df["soil_whc"] + 1e-6)

    return df


# ─── Feature Column Lists ─────────────────────────────────────────────────────

CATEGORICAL_FEATURES = ["crop_type", "soil_type"]
NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "days_since_watering",
    "soil_whc",
    "soil_pwp",
    "crop_kc",
    "soil_awc",
    "et0",
    "etc",
    "moisture_stress",
    "temp_humidity_ratio",
    "days_x_et0",
    "stress_per_whc",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "moisture_pct"

# Exported sets for API metadata (used by main.py /model-info endpoint)
SOIL_TYPES: list[str] = [k for k in SOIL_WHC if k != "default"]
CROP_TYPES: list[str] = [k for k in CROP_KC  if k != "default"]
