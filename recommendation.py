"""
recommendation.py — Rule-based recommendation engine for soil moisture.

Deterministic template system keyed on (moisture_bucket, crop_type, days_range).
Zero LLM cost, 100% predictable output, <1ms latency.

Template hierarchy (most specific wins):
    (status, crop)  →  crop-specific message
    (status, *)     →  generic status message
"""

from __future__ import annotations
from thresholds import MoistureStatus, get_status


# ─── Crop Groups (for template matching) ─────────────────────────────────────

# Group crops with similar water needs for template reuse
CROP_GROUPS: dict[str, str] = {
    "rice":      "high_water",
    "sugarcane": "high_water",
    "wheat":     "moderate_water",
    "maize":     "moderate_water",
    "corn":      "moderate_water",
    "barley":    "moderate_water",
    "sorghum":   "moderate_water",
    "millet":    "moderate_water",
    "cotton":    "moderate_water",
    "soybean":   "moderate_water",
    "groundnut": "moderate_water",
    "chickpea":  "low_water",
    "potato":    "low_water",
    "tomato":    "low_water",
    "sunflower": "low_water",
}

DEFAULT_GROUP = "moderate_water"


def _get_crop_group(crop_type: str) -> str:
    return CROP_GROUPS.get(crop_type.lower(), DEFAULT_GROUP)


# ─── Template Definitions ─────────────────────────────────────────────────────
# Key: (MoistureStatus, crop_group)  →  list of templates (one is picked by days bucket)

TEMPLATES: dict[tuple[MoistureStatus, str], list[str]] = {

    # ── CRITICAL ──────────────────────────────────────────────────────────────
    (MoistureStatus.CRITICAL, "high_water"): [
        "Moisture is critically low at {moisture:.0f}%. {crop} requires high water demand — irrigate immediately to prevent crop failure.",
        "Only {moisture:.0f}% moisture remains. {crop} is at severe water stress. Apply a deep irrigation now and monitor recovery.",
    ],
    (MoistureStatus.CRITICAL, "moderate_water"): [
        "Soil moisture at {moisture:.0f}% — critical threshold crossed. {crop} needs water urgently; {days:.0f} days without irrigation has caused significant depletion.",
        "At {moisture:.0f}%, your {crop} field is severely moisture-stressed. Irrigate as soon as possible to avoid yield loss.",
    ],
    (MoistureStatus.CRITICAL, "low_water"): [
        "{crop} is water-stressed at {moisture:.0f}%. Although this crop is drought-tolerant, immediate light irrigation is recommended.",
        "Soil moisture ({moisture:.0f}%) has fallen below safe levels even for {crop}. Irrigate promptly.",
    ],

    # ── LOW ───────────────────────────────────────────────────────────────────
    (MoistureStatus.LOW, "high_water"): [
        "Moisture is {moisture:.0f}% — getting low for {crop}. Plan irrigation within the next 12–24 hours to maintain optimal growth.",
        "{crop} prefers consistently moist soil. At {moisture:.0f}%, schedule watering soon — {days:.0f} days of depletion is approaching its limit.",
    ],
    (MoistureStatus.LOW, "moderate_water"): [
        "Soil moisture is {moisture:.0f}% — below optimal for {crop}. Consider irrigating in the next 24 hours.",
        "After {days:.0f} days, moisture is at {moisture:.0f}%. {crop} will benefit from irrigation soon to support healthy root development.",
    ],
    (MoistureStatus.LOW, "low_water"): [
        "Moisture is {moisture:.0f}%. {crop} can tolerate mild dryness, but watering in the next 1–2 days is advisable.",
        "At {moisture:.0f}%, conditions are slightly dry for {crop}. A light irrigation is recommended.",
    ],

    # ── OPTIMAL ───────────────────────────────────────────────────────────────
    (MoistureStatus.OPTIMAL, "high_water"): [
        "Soil moisture is optimal at {moisture:.0f}%. {crop} is in ideal growing conditions — no action needed.",
        "Excellent — {moisture:.0f}% moisture is perfect for {crop}. Continue your current irrigation schedule.",
    ],
    (MoistureStatus.OPTIMAL, "moderate_water"): [
        "Moisture at {moisture:.0f}% is in the optimal range for {crop}. No irrigation required right now.",
        "Your {crop} field has {moisture:.0f}% moisture — conditions are ideal. Next check recommended in {days:.0f} days.",
    ],
    (MoistureStatus.OPTIMAL, "low_water"): [
        "Moisture at {moisture:.0f}% is well within optimal range for {crop}. No action needed.",
        "{crop} is thriving at {moisture:.0f}% moisture. Hold off on irrigation.",
    ],

    # ── SATURATED ─────────────────────────────────────────────────────────────
    (MoistureStatus.SATURATED, "high_water"): [
        "Soil is saturated at {moisture:.0f}%. Even for water-intensive {crop}, excess moisture risks root rot and anaerobic conditions. Skip irrigation.",
        "At {moisture:.0f}%, the field is over-saturated. Avoid watering for at least 2–3 days and ensure proper drainage.",
    ],
    (MoistureStatus.SATURATED, "moderate_water"): [
        "Soil moisture is {moisture:.0f}% — saturated. Do not irrigate {crop} until levels drop below 70%. Check drainage.",
        "At {moisture:.0f}%, excess water may cause nutrient leaching and root rot in {crop}. Allow natural drying.",
    ],
    (MoistureStatus.SATURATED, "low_water"): [
        "Critically over-watered at {moisture:.0f}%. {crop} is drought-tolerant and highly susceptible to root rot under saturated conditions. Stop all irrigation immediately.",
        "Soil is waterlogged at {moisture:.0f}%. {crop} requires well-drained conditions — improve drainage and do not water.",
    ],
}


# ─── Recommendation Function ─────────────────────────────────────────────────

def get_recommendation(
    moisture_pct:       float,
    crop_type:          str,
    days_since_watering: float,
    variant: int | None = None,
) -> str:
    """
    Return a recommendation string for given conditions.

    Args:
        moisture_pct:         Current predicted moisture (0–100)
        crop_type:            Crop name string
        days_since_watering:  Number of days since last irrigation
        variant:              Optional fixed template index (0 or 1). Defaults to
                              deterministic choice based on days_since_watering.

    Returns:
        Formatted recommendation string
    """
    status_info = get_status(moisture_pct)
    status      = status_info.status
    crop_group  = _get_crop_group(crop_type)

    # Get templates — try specific, then generic group
    key = (status, crop_group)
    templates = TEMPLATES.get(key, [])

    if not templates:
        # Absolute fallback
        return f"Soil moisture is {moisture_pct:.0f}% ({status.value}). {status_info.description}."

    # Pick variant deterministically by days bucket if not specified
    if variant is None:
        variant = 0 if days_since_watering < 7 else 1
    variant = variant % len(templates)

    template = templates[variant]
    return template.format(
        moisture=moisture_pct,
        crop=crop_type.capitalize(),
        days=days_since_watering,
    )


def get_full_advice(
    moisture_pct:        float,
    crop_type:           str,
    soil_type:           str,
    days_since_watering: float,
) -> dict:
    """
    Return full structured advice dict for API response.
    """
    from thresholds import get_status_badge

    badge       = get_status_badge(moisture_pct)
    primary_rec = get_recommendation(moisture_pct, crop_type, days_since_watering, variant=0)
    alt_rec     = get_recommendation(moisture_pct, crop_type, days_since_watering, variant=1)

    # Next check interval (heuristic)
    status = badge["status"]
    if status == "Critical":
        next_check_hours = 6
    elif status == "Low":
        next_check_hours = 24
    elif status == "Optimal":
        next_check_hours = 48
    else:  # Saturated
        next_check_hours = 72

    return {
        "badge":              badge,
        "recommendation":     primary_rec,
        "alt_recommendation": alt_rec,
        "next_check_hours":   next_check_hours,
        "crop_group":         _get_crop_group(crop_type),
    }
