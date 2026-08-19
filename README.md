# 🌱 Soil Moisture Prediction API

A production-ready **tabular regression pipeline** that predicts soil moisture percentage (0–100%) from 5 structured agronomic inputs, served via **FastAPI**. Combines XGBoost and Random Forest models with physics-based feature engineering (FAO-56 ET₀, Kc coefficients, soil WHC/PWP tables).

---

## 📌 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Inputs & Outputs](#inputs--outputs)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Supported Crops & Soils](#supported-crops--soils)
- [Dataset Strategy](#dataset-strategy)
- [Limitations](#limitations)

---

## ✨ Features

- **XGBoost model**: High-performance XGBoost Regressor model optimized for fast, accurate inference
- **Physics-grounded features**: Hargreaves ET₀, FAO-56 Kc, soil field capacity & wilting point
- **Rich response**: moisture %, status badge (Critical/Low/Optimal/Saturated), actionable recommendation, next-check interval
- **Batch prediction**: Upload a CSV, get enriched results back
- **CORS-enabled**: Ready to connect to Flutter/web frontends
- **Fast**: Typical inference latency < 50ms (reported via `X-Inference-Ms` header)

---

## 🏗️ Architecture

```
Raw Inputs (5 fields)
        ↓
Feature Engineering (ET₀, ETc, moisture stress, soil WHC/PWP, nonlinear interactions)
        ↓
    ┌───────────────────────┐
    │  XGBoost Regressor    │  (production model)
    └───────────────────────┘
        ↓
moisture_pct ∈ [0, 100]   (clipped)
        ↓
Threshold Layer  →  Critical | Low | Optimal | Saturated
        ↓
Rule-Based Recommendation Engine  →  text advice + next-check hours
```

---

## 📥 Inputs & Outputs

### Request Inputs

| Field | Type | Range / Values | Description |
|---|---|---|---|
| `crop_type` | `string` | wheat, rice, maize, tomato, … | Crop being cultivated |
| `soil_type` | `string` | clay, loam, sandy, silt, … | Soil classification |
| `temperature` | `float` | −10 – 60 °C | Air temperature |
| `humidity` | `float` | 0 – 100 % | Relative humidity |
| `days_since_watering` | `float` | 0 – 60 days | Days since last irrigation |

### Response Output

| Field | Type | Description |
|---|---|---|
| `moisture_pct` | `float` | Predicted soil moisture (0–100%) |
| `badge.status` | `string` | Critical / Low / Optimal / Saturated |
| `badge.emoji` | `string` | Status emoji |
| `badge.color` | `string` | Hex color code |
| `badge.label` | `string` | Short label |
| `badge.description` | `string` | Status description |
| `recommendation` | `string` | Primary actionable advice |
| `alt_recommendation` | `string` | Alternate phrasing |
| `next_check_hours` | `int` | Hours until next suggested check |
| `crop_group` | `string` | Crop water demand group |
| `model_used` | `string` | Which model was used |

---

## 📂 Project Structure

```
soil_moisture_prediction_dep/
├── main.py                  # FastAPI application (endpoints, CORS, lifespan)
├── predict.py               # Inference wrapper: load XGBoost model, run predict/predict_batch
├── features.py              # Feature engineering layer (ET₀, Kc, WHC, PWP tables)
├── schemas.py               # Pydantic v2 request/response schemas
├── requirements.txt         # Python dependencies
├── models/
│   ├── xgb_model.pkl        # Trained XGBoost pipeline (production model tracked in git)
│   └── training_metadata.json  # Model metrics & hyperparameters
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd soil_moisture_prediction_dep
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API docs will be available at **http://localhost:8000/docs**.

---

## 🌐 API Reference

### `GET /health`
Health check.

```json
{ "status": "ok", "version": "1.0.0" }
```

---

### `GET /model-info`
Returns model metadata, test metrics, and supported crop/soil values.

```json
{
  "model_name": "XGBoost Regressor (primary) + Random Forest (baseline)",
  "rmse_test": 1.72,
  "mae_test": 1.28,
  "r2_test": 0.9873,
  "n_train": 56000,
  "supported_crops": ["barley", "chickpea", "..."],
  "supported_soils": ["clay", "clay loam", "..."]
}
```

---

### `POST /predict`
Single prediction from JSON body.

**Request:**
```json
{
  "crop_type": "wheat",
  "soil_type": "loam",
  "temperature": 28.0,
  "humidity": 55.0,
  "days_since_watering": 7.0,
  "model": "xgboost",
  "compare_rf": false
}
```

**Response:**
```json
{
  "moisture_pct": 42.3,
  "badge": {
    "status": "Low",
    "emoji": "🟡",
    "color": "#f5a623",
    "label": "Low Moisture",
    "description": "Soil is drying out",
    "moisture_pct": 42.3
  },
  "recommendation": "Schedule irrigation within 24 hours.",
  "alt_recommendation": "Consider drip irrigation to conserve water.",
  "next_check_hours": 12,
  "crop_group": "medium_demand",
  "model_used": "xgboost"
}
```

Response header includes: `X-Inference-Ms: 18.4`

---

### `POST /predict/batch`
Batch prediction from an uploaded CSV file.

**Required CSV columns:** `crop_type`, `soil_type`, `temperature`, `humidity`, `days_since_watering`

**Added output columns:** `moisture_pct`, `status`, `status_emoji`, `status_color`, `recommendation`

```bash
curl -X POST http://localhost:8000/predict/batch \
  -F "file=@my_fields.csv"
```

---

## ⚙️ Feature Engineering

Features computed in `features.py` for each prediction:

| Engineered Feature | Formula / Source | Description |
|---|---|---|
| `soil_whc` | FAO / SoilGrids tables | Field capacity (water-holding capacity) |
| `soil_pwp` | FAO tables | Permanent wilting point |
| `crop_kc` | FAO-56 Paper 56, Table 12 | Crop coefficient (mid-season) |
| `soil_awc` | `WHC − PWP` | Available water capacity |
| `et0` | Hargreaves: `0.0023 × (T+17.8) × (1−RH/100) × 0.408` | Reference evapotranspiration proxy |
| `etc` | `Kc × ET₀` | Crop-adjusted evapotranspiration |
| `moisture_stress` | `ETc × days_since_watering` | Cumulative moisture depletion signal |
| `temp_humidity_ratio` | `T / (RH + ε)` | Nonlinear interaction |
| `days_x_et0` | `days × ET₀` | Time-weighted evaporation |
| `stress_per_whc` | `moisture_stress / WHC` | Relative stress normalised by soil capacity |

---

## 📊 Model Performance

Evaluated on a held-out test set of **12,000 samples**:

| Metric | XGBoost | Random Forest |
|---|---|---|
| **RMSE** | **1.72%** | 1.73% |
| **MAE** | **1.28%** | 1.29% |
| **R²** | **0.9873** | 0.9872 |

> Training set: 56,000 · Validation set: 12,000 · Test set: 12,000

**XGBoost Hyperparameters (tuned via Optuna):**

| Parameter | Value |
|---|---|
| `n_estimators` | 800 |
| `max_depth` | 6 |
| `learning_rate` | 0.05 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 3 |
| `gamma` | 0.1 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |

---

## 🌾 Supported Crops & Soils

### Crops
`wheat` · `rice` · `maize` · `corn` · `cotton` · `soybean` · `potato` · `tomato` · `sugarcane` · `sunflower` · `barley` · `chickpea` · `groundnut` · `millet` · `sorghum`

> Unlisted crops fall back to a default Kc of **1.10**.

### Soil Types
`clay` · `loam` · `sandy loam` · `sandy` · `silt` · `peat` · `clay loam` · `silt loam`

> Unlisted soils fall back to WHC = **30%**, PWP = **12%**.

---

## 🗃️ Dataset Strategy

No single public dataset provides all 5 inputs → labeled moisture output. This project uses a **3-layer strategy**:

1. **Synthetic data (primary):** Physics-grounded simulation using Hargreaves ET formula + FAO-56 Kc crop coefficients + soil field capacity tables
2. **Public datasets (distribution calibration):** Kaggle Crop Recommendation dataset, NASA SMAP soil moisture satellite data
3. **Real sensor data (optional calibration):** ESP32 + capacitive soil moisture sensor rig for local ground-truth correction

---

## ⚠️ Limitations

- Training data is **synthetically generated** using physics-based models. Real-world accuracy improves significantly with real sensor calibration data.
- FAO-56 Kc values are **mid-season** values; actual Kc varies by crop growth stage.
- Model does **not** account for rainfall events, irrigation amount, or soil drainage rate.
- Temperature range validated for **−10 °C to 60 °C** only.

---

## 📦 Dependencies

Key packages (see `requirements.txt` for pinned versions):

| Package | Purpose |
|---|---|
| `xgboost` | Primary regression model |
| `lightgbm` | Alternative gradient boosting |
| `scikit-learn` | Random Forest, pipelines, preprocessing |
| `optuna` | Hyperparameter tuning |
| `shap` | Model explainability |
| `fastapi` + `uvicorn` | REST API server |
| `pydantic` | Request/response validation |
| `pandas` + `numpy` | Data manipulation |
| `joblib` | Model serialisation |
| `category_encoders` | Categorical feature encoding |

---

## 📄 License

This project is provided for educational and research purposes.
