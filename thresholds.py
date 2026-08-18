"""
thresholds.py — Deterministic moisture status classification.

The model predicts moisture_pct (0–100). Status badges are
derived deterministically from thresholds — NOT from ML — for
100% consistency and predictability.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class MoistureStatus(str, Enum):
    CRITICAL  = "Critical"
    LOW       = "Low"
    OPTIMAL   = "Optimal"
    SATURATED = "Saturated"


@dataclass(frozen=True)
class StatusInfo:
    status:      MoistureStatus
    emoji:       str
    color_hex:   str
    label:       str
    description: str


# ─── Threshold Definitions ───────────────────────────────────────────────────

THRESHOLDS = [
    (0,   20,  StatusInfo(MoistureStatus.CRITICAL,  "🔴", "#EF4444", "Critical",  "Immediate watering required")),
    (20,  40,  StatusInfo(MoistureStatus.LOW,       "🟡", "#F59E0B", "Low",       "Water soon")),
    (40,  70,  StatusInfo(MoistureStatus.OPTIMAL,   "🟢", "#22C55E", "Optimal",   "Moisture levels are ideal")),
    (70,  101, StatusInfo(MoistureStatus.SATURATED, "💧", "#3B82F6", "Saturated", "No watering needed; risk of root rot")),
]


def get_status(moisture_pct: float) -> StatusInfo:
    """
    Return the StatusInfo for a given moisture percentage.

    Args:
        moisture_pct: float in [0, 100]

    Returns:
        StatusInfo dataclass with status, emoji, color, label, description
    """
    moisture_pct = float(moisture_pct)
    for lo, hi, info in THRESHOLDS:
        if lo <= moisture_pct < hi:
            return info
    # Edge case: exactly 100
    return THRESHOLDS[-1][2]


def get_status_badge(moisture_pct: float) -> dict:
    """Return status as a dict suitable for JSON serialization."""
    info = get_status(moisture_pct)
    return {
        "status":      info.status.value,
        "emoji":       info.emoji,
        "color":       info.color_hex,
        "label":       info.label,
        "description": info.description,
        "moisture_pct": round(moisture_pct, 1),
    }
