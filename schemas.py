"""
schemas.py — Pydantic v2 request/response schemas for the FastAPI soil moisture API.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ModelChoice(str, Enum):
    xgboost      = "xgboost"
    random_forest = "random_forest"


# ─── Request ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    crop_type:           str   = Field(..., examples=["wheat"],  description="Crop type (e.g. wheat, rice, tomato)")
    soil_type:           str   = Field(..., examples=["loam"],   description="Soil type (e.g. clay, loam, sandy)")
    temperature:         float = Field(..., ge=-10, le=60,       description="Air temperature in °C")
    humidity:            float = Field(..., ge=0,   le=100,      description="Relative humidity in %")
    days_since_watering: float = Field(..., ge=0,   le=60,       description="Days since last irrigation")
    model:               ModelChoice = Field(ModelChoice.xgboost, description="Model to use for prediction")
    compare_rf:          bool  = Field(False, description="If true, also return Random Forest prediction")

    @field_validator("crop_type", "soil_type", mode="before")
    @classmethod
    def to_lower(cls, v: str) -> str:
        return str(v).strip().lower()


# ─── Response components ──────────────────────────────────────────────────────

class StatusBadge(BaseModel):
    status:      str = Field(..., description="Critical | Low | Optimal | Saturated")
    emoji:       str
    color:       str = Field(..., description="Hex color code")
    label:       str
    description: str
    moisture_pct: float


class PredictResponse(BaseModel):
    moisture_pct:       float  = Field(..., description="Predicted soil moisture percentage (0–100)")
    badge:              StatusBadge
    recommendation:     str    = Field(..., description="Primary actionable recommendation")
    alt_recommendation: str    = Field(..., description="Alternate phrasing")
    next_check_hours:   int    = Field(..., description="Suggested hours until next monitoring check")
    crop_group:         str    = Field(..., description="Crop water demand group")
    model_used:         str
    rf_moisture_pct:    Optional[float] = Field(None, description="Random Forest prediction (if compare_rf=true)")


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = "ok"
    version: str


class ModelInfoResponse(BaseModel):
    model_name:     str
    rmse_test:      Optional[float] = None
    mae_test:       Optional[float] = None
    r2_test:        Optional[float] = None
    n_train:        Optional[int]   = None
    supported_crops: list[str]
    supported_soils: list[str]
