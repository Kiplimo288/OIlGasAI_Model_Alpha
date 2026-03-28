"""
routes/sensors.py
-----------------
Low-cost sensor calibration and LDAR analysis endpoints.
Integrates the sensor layer (sensors/) with OilGasAI Model Alpha's domain knowledge.
"""

from fastapi import APIRouter, Depends
from api.schemas import (
    SensorCalibrationRequest,
    CalibrationResponse,
    LDARRequest,
    LDARResponse,
)
from api.main import get_backend
from sensors.calibration import calibrate_sensor
from sensors.ldar import analyze_ldar_survey

router = APIRouter()


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate(
    request: SensorCalibrationRequest,
    backend: dict = Depends(get_backend),
):
    """
    Calibrate low-cost sensor readings.
    Applies drift correction and reference-method alignment.
    Supports MIRA, Figaro TGS2611, Alphasense OB3, and generic sensors.
    """
    result = calibrate_sensor(
        sensor_type=request.sensor_type.value,
        raw_readings=request.raw_readings,
        reference_readings=request.reference_readings,
        temperature_c=request.temperature_c,
        humidity_pct=request.humidity_pct,
        deployment_days=request.deployment_days,
        backend=backend,
    )
    return CalibrationResponse(**result)


@router.post("/ldar", response_model=LDARResponse)
async def ldar_analysis(
    request: LDARRequest,
    backend: dict = Depends(get_backend),
):
    """
    Analyze LDAR survey data and identify leaks requiring repair.
    Returns prioritized repair list with regulatory deadlines.
    """
    result = analyze_ldar_survey(
        survey_data=request.survey_data,
        facility_name=request.facility_name,
        regulation=request.regulation.value,
        survey_date=request.survey_date,
        backend=backend,
    )
    return LDARResponse(**result)
