"""
sensors/calibration.py
----------------------
Low-cost methane sensor calibration and drift correction.
Based on AMMMU/SABER project research at CSU METEC.

Supports:
- MIRA methane sensor (used in AMMMU unit)
- Figaro TGS2611
- Alphasense OB3
- Generic low-cost sensors

Methods:
- Linear regression (with reference instrument)
- Drift correction (deployment-day based)
- Temperature/humidity compensation
- OilGasAI Model Alpha-assisted interpretation
"""

import numpy as np
from loguru import logger
from inference.chat import chat


# ── Calibration coefficients (defaults from SABER project) ───────────────────

SENSOR_DEFAULTS = {
    "MIRA_Methane": {
        "baseline_ppm": 1.87,         # atmospheric CH4 background
        "sensitivity_mv_ppm": 2.3,
        "temp_coefficient": -0.015,   # fractional change per °C
        "humidity_coefficient": 0.008,
        "drift_rate_pct_per_day": 0.12,
    },
    "Figaro_TGS2611": {
        "baseline_ppm": 1.87,
        "sensitivity_mv_ppm": 1.8,
        "temp_coefficient": -0.022,
        "humidity_coefficient": 0.011,
        "drift_rate_pct_per_day": 0.18,
    },
    "Alphasense_OB3": {
        "baseline_ppm": 1.87,
        "sensitivity_mv_ppm": 3.1,
        "temp_coefficient": -0.010,
        "humidity_coefficient": 0.006,
        "drift_rate_pct_per_day": 0.09,
    },
    "Other_Low_Cost": {
        "baseline_ppm": 1.87,
        "sensitivity_mv_ppm": 2.0,
        "temp_coefficient": -0.018,
        "humidity_coefficient": 0.009,
        "drift_rate_pct_per_day": 0.15,
    },
}


def calibrate_sensor(
    sensor_type: str,
    raw_readings: list[float],
    reference_readings: list[float] | None = None,
    temperature_c: list[float] | None = None,
    humidity_pct: list[float] | None = None,
    deployment_days: int | None = None,
    backend: dict | None = None,
) -> dict:
    """
    Apply calibration pipeline to raw sensor readings.

    Steps:
      1. Drift correction (if deployment_days provided)
      2. Temperature/humidity compensation (if env data provided)
      3. Linear regression alignment to reference (if reference provided)
      4. OilGasAI Model Alpha-assisted interpretation note

    Returns:
        dict with calibrated_readings, method, drift_correction_applied, notes
    """
    params = SENSOR_DEFAULTS.get(sensor_type, SENSOR_DEFAULTS["Other_Low_Cost"])
    readings = np.array(raw_readings, dtype=float)
    methods_applied = []

    # ── Step 1: Drift correction ──────────────────────────────────────────────
    drift_applied = False
    if deployment_days and deployment_days > 0:
        drift_factor = 1.0 + (params["drift_rate_pct_per_day"] * deployment_days / 100)
        readings = readings / drift_factor
        drift_applied = True
        methods_applied.append(f"Drift correction ({deployment_days}d, factor={drift_factor:.4f})")
        logger.info(f"Drift correction applied: factor={drift_factor:.4f}")

    # ── Step 2: Temperature compensation ─────────────────────────────────────
    if temperature_c is not None:
        temps = np.array(temperature_c, dtype=float)
        ref_temp = 20.0  # reference temperature °C
        temp_correction = 1.0 + params["temp_coefficient"] * (temps - ref_temp)
        readings = readings / temp_correction
        methods_applied.append("Temperature compensation (20°C reference)")

    # ── Step 3: Humidity compensation ────────────────────────────────────────
    if humidity_pct is not None:
        humidity = np.array(humidity_pct, dtype=float)
        ref_humidity = 50.0  # reference humidity %
        hum_correction = 1.0 + params["humidity_coefficient"] * (humidity - ref_humidity) / 100
        readings = readings / hum_correction
        methods_applied.append("Humidity compensation (50% RH reference)")

    # ── Step 4: Reference alignment (supervised calibration) ─────────────────
    if reference_readings is not None and len(reference_readings) >= 3:
        ref = np.array(reference_readings, dtype=float)
        # Ordinary least squares: ref = a * sensor + b
        A = np.vstack([readings, np.ones(len(readings))]).T
        slope, intercept = np.linalg.lstsq(A, ref, rcond=None)[0]
        readings = slope * readings + intercept
        methods_applied.append(
            f"Reference alignment (OLS: slope={slope:.4f}, intercept={intercept:.4f})"
        )
        logger.info(f"Reference alignment: slope={slope:.4f}, intercept={intercept:.4f}")

    calibrated = readings.tolist()

    # ── Step 5: OilGasAI Model Alpha interpretation note ──────────────────────────────────
    notes = _generate_calibration_note(
        sensor_type, calibrated, methods_applied, backend
    )

    return {
        "calibrated_readings": calibrated,
        "drift_correction_applied": drift_applied,
        "calibration_method": " | ".join(methods_applied) if methods_applied else "Raw (no corrections)",
        "notes": notes,
        "model": "OilGasAI-Model-Alpha",
    }


def _generate_calibration_note(
    sensor_type: str,
    calibrated: list[float],
    methods: list[str],
    backend: dict | None,
) -> str:
    """Generate a brief OilGasAI Model Alpha interpretation of the calibration results."""
    if backend is None:
        return "Calibration complete. Verify against co-located reference instrument for regulatory use."

    mean_val = float(np.mean(calibrated))
    max_val = float(np.max(calibrated))
    prompt = (
        f"I calibrated a {sensor_type} methane sensor using: {'; '.join(methods)}. "
        f"Calibrated readings: mean={mean_val:.2f} ppm, max={max_val:.2f} ppm, "
        f"n={len(calibrated)} readings. "
        f"In 2-3 sentences, interpret these results for EPA LDAR compliance purposes "
        f"and note any flags or verification steps required."
    )

    try:
        note = chat(prompt, backend)
    except Exception as e:
        logger.warning(f"OilGasAI Model Alpha note generation failed: {e}")
        note = "Calibration complete. Verify results against co-located reference instrument before regulatory submission."

    return note
