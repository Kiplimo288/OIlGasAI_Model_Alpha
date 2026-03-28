"""
test_sensors.py
---------------
Unit tests for sensor calibration and LDAR analysis.
"""

import pytest
import numpy as np
from sensors.calibration import calibrate_sensor
from sensors.ldar import analyze_ldar_survey


class TestCalibration:
    def test_basic_calibration_returns_correct_shape(self):
        """Calibration returns same number of readings as input."""
        raw = [1.2, 1.5, 1.8, 2.1, 2.4]
        result = calibrate_sensor("MIRA_Methane", raw)
        assert len(result["calibrated_readings"]) == len(raw)

    def test_drift_correction_reduces_readings(self):
        """Drift correction should bring down inflated readings."""
        raw = [5.0, 5.5, 6.0]
        result_no_drift = calibrate_sensor("MIRA_Methane", raw)
        result_with_drift = calibrate_sensor("MIRA_Methane", raw, deployment_days=90)

        no_drift_mean = np.mean(result_no_drift["calibrated_readings"])
        with_drift_mean = np.mean(result_with_drift["calibrated_readings"])
        assert with_drift_mean < no_drift_mean
        assert result_with_drift["drift_correction_applied"] is True

    def test_reference_alignment(self):
        """Reference alignment should bring sensor closer to reference values."""
        raw = [2.0, 3.0, 4.0, 5.0]
        reference = [2.2, 3.3, 4.4, 5.5]  # ~1.1x scaling
        result = calibrate_sensor("MIRA_Methane", raw, reference_readings=reference)
        calibrated = result["calibrated_readings"]
        # Calibrated values should be close to reference
        for cal, ref in zip(calibrated, reference):
            assert abs(cal - ref) < 0.5

    def test_all_sensor_types_supported(self):
        """All sensor types should calibrate without error."""
        sensor_types = ["MIRA_Methane", "Figaro_TGS2611", "Alphasense_OB3", "Other_Low_Cost"]
        raw = [1.0, 2.0, 3.0]
        for st in sensor_types:
            result = calibrate_sensor(st, raw)
            assert "calibrated_readings" in result

    def test_no_corrections_without_optional_params(self):
        """Without optional params, readings pass through with no corrections."""
        raw = [1.87, 1.87, 1.87]
        result = calibrate_sensor("MIRA_Methane", raw)
        assert result["drift_correction_applied"] is False
        assert "Raw" in result["calibration_method"]


class TestLDAR:
    SURVEY_DATA = [
        {"component": "valve", "reading_ppm": 600, "location": "Well A-1"},
        {"component": "connector", "reading_ppm": 200, "location": "Well A-2"},
        {"component": "pump", "reading_ppm": 2500, "location": "Well B-1"},
        {"component": "valve", "reading_ppm": 100, "location": "Well B-2"},  # below threshold
    ]

    def test_leaks_detected_correctly(self):
        """Only components above threshold flagged as leaks."""
        result = analyze_ldar_survey(
            self.SURVEY_DATA, "Test Facility", "NSPS_OOOOa", "2026-01-15"
        )
        assert result["leak_count"] == 2  # valve@600, pump@2500
        assert result["total_components_surveyed"] == 4

    def test_high_priority_flagging(self):
        """Components >3x threshold should be HIGH priority."""
        result = analyze_ldar_survey(
            self.SURVEY_DATA, "Test Facility", "NSPS_OOOOa", "2026-01-15"
        )
        leaks = result["leaks_detected"]
        pump_leak = next(l for l in leaks if "pump" in l["component"])
        assert pump_leak["priority"] == "HIGH"

    def test_repair_deadlines_present(self):
        """Repair deadlines should be included in response."""
        result = analyze_ldar_survey(
            self.SURVEY_DATA, "Test Facility", "NSPS_OOOOa", "2026-01-15"
        )
        assert "first_attempt_days" in result["repair_deadlines"]
        assert "final_repair_days" in result["repair_deadlines"]

    def test_no_leaks_scenario(self):
        """Zero leaks should return empty list."""
        clean_data = [
            {"component": "valve", "reading_ppm": 50, "location": "Site A"},
            {"component": "connector", "reading_ppm": 30, "location": "Site B"},
        ]
        result = analyze_ldar_survey(
            clean_data, "Clean Facility", "NSPS_OOOOa", "2026-01-15"
        )
        assert result["leak_count"] == 0
        assert result["leaks_detected"] == []

    def test_stricter_nsps_oooob_thresholds(self):
        """NSPS OOOOb (200 ppm) should flag more leaks than OOOOa (500 ppm)."""
        result_a = analyze_ldar_survey(
            self.SURVEY_DATA, "Facility", "NSPS_OOOOa", "2026-01-15"
        )
        result_b = analyze_ldar_survey(
            self.SURVEY_DATA, "Facility", "NSPS_OOOOb", "2026-01-15"
        )
        assert result_b["leak_count"] >= result_a["leak_count"]
