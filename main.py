"""
main.py — FastAPI application for soil moisture prediction.

Endpoints:
    GET  /health          - Health check
    GET  /model-info      - Model metadata and supported values
    POST /predict         - Main prediction endpoint
    POST /predict/batch   - Batch CSV prediction

Start server:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from predict import predict, predict_batch, _load_models
from features import SOIL_TYPES, CROP_TYPES
from schemas import (
    PredictRequest, PredictResponse, StatusBadge,
    HealthResponse, ModelInfoResponse,
)

APP_VERSION = "1.0.0"
MODELS_DIR  = Path("models")


# ─── Lifespan (pre-load models) ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load models at startup so first request isn't slow."""
    print("Loading models...")
    try:
        _load_models()
        print("✓ Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print("  Predictions will fail until models are trained.")
    yield
    print("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Soil Moisture Prediction API",
    description=(
        "Predicts soil moisture percentage (0–100%) from crop type, soil type, "
        "temperature, humidity, and days since last watering. "
        "Returns moisture %, status badge, and actionable recommendations."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

# CORS — allow Flutter/web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Health check endpoint."""
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Meta"])
async def model_info():
    """Return model metadata, supported crops/soils, and test metrics."""
    meta_path = MODELS_DIR / "training_metadata.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    xgb_meta = meta.get("xgboost", {})
    return ModelInfoResponse(
        model_name      = "XGBoost Regressor",
        rmse_test       = xgb_meta.get("rmse"),
        mae_test        = xgb_meta.get("mae"),
        r2_test         = xgb_meta.get("r2"),
        n_train         = meta.get("n_train"),
        supported_crops = sorted(CROP_TYPES),
        supported_soils = sorted(SOIL_TYPES),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_endpoint(request: PredictRequest):
    """
    Predict soil moisture % from 5 inputs and return status + recommendation.

    - **crop_type**: e.g. wheat, rice, tomato, maize, soybean
    - **soil_type**: e.g. clay, loam, sandy, silt
    - **temperature**: °C (0–60)
    - **humidity**: % (0–100)
    - **days_since_watering**: days (0–60)
    """
    t0 = time.perf_counter()
    try:
        result = predict(
            crop_type            = request.crop_type,
            soil_type            = request.soil_type,
            temperature          = request.temperature,
            humidity             = request.humidity,
            days_since_watering  = request.days_since_watering,
            use_model            = request.model.value,
            return_rf_too        = request.compare_rf,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Add latency header
    response = PredictResponse(
        moisture_pct       = result["moisture_pct"],
        badge              = StatusBadge(**result["badge"]),
        recommendation     = result["recommendation"],
        alt_recommendation = result["alt_recommendation"],
        next_check_hours   = result["next_check_hours"],
        crop_group         = result["crop_group"],
        model_used         = result["model_used"],
        rf_moisture_pct    = result.get("rf_moisture_pct"),
    )

    return JSONResponse(
        content=response.model_dump(),
        headers={"X-Inference-Ms": f"{elapsed_ms:.1f}"},
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """
    Batch prediction from uploaded CSV file.

    Required CSV columns: crop_type, soil_type, temperature, humidity, days_since_watering
    Returns enriched CSV with moisture_pct, status, recommendation columns added.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    try:
        contents = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    required_cols = {"crop_type", "soil_type", "temperature", "humidity", "days_since_watering"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    try:
        result_df = predict_batch(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")

    return JSONResponse(content=result_df.to_dict(orient="records"))
