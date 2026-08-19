"""
predict.py — Inference wrapper for soil moisture model.

Loads models once at module import, exposes a clean predict() function
that handles feature engineering → preprocessing → model → status + recommendation.

Usage:
    from src.predict import predict
    result = predict(
        crop_type="wheat",
        soil_type="loam",
        temperature=28.0,
        humidity=55.0,
        days_since_watering=7.0,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features import engineer_features
from thresholds import get_status_badge
from recommendation import get_full_advice

# ─── Model Loading ────────────────────────────────────────────────────────────

MODELS_DIR = Path("models")
_xgb_pipeline = None


def _load_models():
    """Lazy-load XGBoost model once, cache in module globals."""
    global _xgb_pipeline
    if _xgb_pipeline is None:
        xgb_path = MODELS_DIR / "xgb_model.pkl"

        if not xgb_path.exists():
            raise FileNotFoundError(
                f"Model not found at {xgb_path}. "
                "Ensure models/xgb_model.pkl is present."
            )
        _xgb_pipeline = joblib.load(xgb_path)
    return _xgb_pipeline


# ─── Core Predict Function ────────────────────────────────────────────────────

def predict(
    crop_type:           str,
    soil_type:           str,
    temperature:         float,
    humidity:            float,
    days_since_watering: float,
    use_model:           str = "xgboost",  # retained for endpoint compatibility
    return_rf_too:       bool = False,     # retained for endpoint compatibility
) -> dict:
    """
    Predict soil moisture % and return full structured result using XGBoost model.

    Args:
        crop_type:            Crop name (e.g. "wheat", "rice", "tomato")
        soil_type:            Soil name (e.g. "clay", "loam", "sandy")
        temperature:          Air temperature in °C  [0–60]
        humidity:             Relative humidity in % [0–100]
        days_since_watering:  Days since last irrigation [0–30]
        use_model:            Retained for backwards compatibility (defaults to "xgboost")
        return_rf_too:        Retained for backwards compatibility

    Returns:
        dict with:
            moisture_pct       (float)
            badge              (dict: status, emoji, color, label, description)
            recommendation     (str)
            alt_recommendation (str)
            next_check_hours   (int)
            crop_group         (str)
            model_used         (str)
    """
    xgb_pipe = _load_models()

    # Build single-row DataFrame
    row = pd.DataFrame([{
        "crop_type":           crop_type.lower().strip(),
        "soil_type":           soil_type.lower().strip(),
        "temperature":         float(temperature),
        "humidity":            float(humidity),
        "days_since_watering": float(days_since_watering),
    }])

    # Feature engineering
    row_fe = engineer_features(row)

    # Primary prediction with XGBoost model
    raw_pred = xgb_pipe.predict(row_fe)
    moisture_pct = float(np.clip(raw_pred[0], 0.0, 100.0))

    # Build full advice
    advice = get_full_advice(moisture_pct, crop_type, soil_type, days_since_watering)

    result = {
        "moisture_pct":       round(moisture_pct, 1),
        "model_used":         "xgboost",
        **advice,
    }

    return result


# ─── Batch Predict ────────────────────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, use_model: str = "xgboost") -> pd.DataFrame:
    """
    Batch prediction for a DataFrame with columns:
        crop_type, soil_type, temperature, humidity, days_since_watering

    Returns input DataFrame with added columns:
        moisture_pct, status, status_emoji, status_color, recommendation
    """
    pipe = _load_models()

    df_fe = engineer_features(df.copy())
    raw = pipe.predict(df_fe)
    pcts = np.clip(raw, 0.0, 100.0)

    from thresholds import get_status
    from recommendation import get_recommendation

    statuses = [get_status(p) for p in pcts]

    df = df.copy()
    df["moisture_pct"]   = np.round(pcts, 1)
    df["status"]         = [s.status.value for s in statuses]
    df["status_emoji"]   = [s.emoji for s in statuses]
    df["status_color"]   = [s.color_hex for s in statuses]
    df["recommendation"] = [
        get_recommendation(p, row["crop_type"], row["days_since_watering"])
        for p, (_, row) in zip(pcts, df.iterrows())
    ]
    return df


if __name__ == "__main__":
    # Quick smoke-test
    print("Running smoke test...")
    result = predict(
        crop_type="wheat",
        soil_type="loam",
        temperature=28.0,
        humidity=55.0,
        days_since_watering=7.0,
    )
    import json
    print(json.dumps(result, indent=2, default=str))
