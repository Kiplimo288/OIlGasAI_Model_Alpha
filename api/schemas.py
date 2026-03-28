"""
schemas.py
----------
Pydantic request/response models for the OilGasAI FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Chat ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    history: list[ChatMessage] = Field(default_factory=list)
    use_rag: bool = Field(default=True, description="Use RAG pipeline for grounded response")
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list[dict]] = None
    model: str = "OilGasAI-Model-Alpha"


# ── Compliance ───────────────────────────────────────────────────────────────

class RegulationFramework(str, Enum):
    subpart_w = "EPA_GHGRP_Subpart_W"
    nsps_oooo = "NSPS_OOOOa"
    nsps_oooob = "NSPS_OOOOb"
    nsps_ooooc = "NSPS_OOOOc"
    ogmp_2 = "OGMP_2.0"
    state = "State_Regulation"


class ComplianceReportRequest(BaseModel):
    facility_name: str
    facility_id: Optional[str] = None
    regulation: RegulationFramework
    reporting_year: int = Field(..., ge=2010, le=2100)
    equipment_list: list[str] = Field(
        ...,
        description="List of equipment types, e.g. ['centrifugal compressor', 'pneumatic controller']"
    )
    additional_context: Optional[str] = None


class ComplianceCheckRequest(BaseModel):
    question: str
    regulation: Optional[RegulationFramework] = None
    facility_type: Optional[str] = None


class ComplianceResponse(BaseModel):
    report: str
    regulation: str
    facility_name: str
    warnings: list[str] = Field(default_factory=list)
    model: str = "OilGasAI-Model-Alpha"


# ── Sensors ──────────────────────────────────────────────────────────────────

class SensorType(str, Enum):
    mira = "MIRA_Methane"
    figaro = "Figaro_TGS2611"
    alphasense = "Alphasense_OB3"
    other = "Other_Low_Cost"


class SensorCalibrationRequest(BaseModel):
    sensor_type: SensorType
    raw_readings: list[float] = Field(..., description="Raw sensor voltage or ADC readings")
    reference_readings: Optional[list[float]] = Field(
        None,
        description="Co-located reference instrument readings (ppm) for supervised calibration"
    )
    temperature_c: Optional[list[float]] = None
    humidity_pct: Optional[list[float]] = None
    deployment_days: Optional[int] = Field(None, description="Days since last calibration")


class CalibrationResponse(BaseModel):
    calibrated_readings: list[float]
    drift_correction_applied: bool
    calibration_method: str
    notes: str
    model: str = "OilGasAI-Model-Alpha"


class LDARRequest(BaseModel):
    survey_data: list[dict] = Field(
        ...,
        description="List of survey readings: [{'component': str, 'reading_ppm': float, 'location': str}]"
    )
    facility_name: str
    regulation: RegulationFramework = RegulationFramework.nsps_oooo
    survey_date: str


class LDARResponse(BaseModel):
    leaks_detected: list[dict]
    total_components_surveyed: int
    leak_count: int
    repair_deadlines: dict
    summary: str
    model: str = "OilGasAI-Model-Alpha"
