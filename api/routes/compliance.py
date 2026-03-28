"""
routes/compliance.py
--------------------
Compliance report auto-generation and regulatory check endpoints.
"""

from fastapi import APIRouter, Depends
from api.schemas import (
    ComplianceReportRequest,
    ComplianceCheckRequest,
    ComplianceResponse,
    ChatResponse,
)
from api.main import get_backend, get_rag
from inference.chat import chat

router = APIRouter()


def _build_report_prompt(req: ComplianceReportRequest) -> str:
    equipment_str = "\n".join(f"  - {e}" for e in req.equipment_list)
    context = f"\nAdditional context: {req.additional_context}" if req.additional_context else ""
    return f"""Generate a structured {req.regulation.value} compliance report for the following facility:

Facility Name: {req.facility_name}
Facility ID: {req.facility_id or 'Not provided'}
Reporting Year: {req.reporting_year}
Regulation: {req.regulation.value}

Equipment List:
{equipment_str}
{context}

The report should include:
1. Applicability determination
2. Required monitoring/measurement methods for each equipment type
3. Calculation methodology and emission factors
4. Reporting deadlines and submission requirements
5. Key compliance action items with priority flags
6. Any regulatory warnings or recent updates relevant to this facility

Format the report clearly with numbered sections."""


def _build_check_prompt(req: ComplianceCheckRequest) -> str:
    reg_str = f" under {req.regulation.value}" if req.regulation else ""
    facility_str = f" for a {req.facility_type} facility" if req.facility_type else ""
    return f"Compliance question{reg_str}{facility_str}: {req.question}"


@router.post("/report", response_model=ComplianceResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
    backend: dict = Depends(get_backend),
    rag: object = Depends(get_rag),
):
    """
    Auto-generate a structured compliance report for a facility.
    Uses RAG pipeline to ground the report in current regulations.
    """
    prompt = _build_report_prompt(request)
    result = rag.query(question=prompt, backend=backend, top_k=8)

    return ComplianceResponse(
        report=result["answer"],
        regulation=request.regulation.value,
        facility_name=request.facility_name,
        warnings=[
            "Always verify against current EPA regulations before submission.",
            "This report was AI-generated — review by a qualified engineer is required.",
        ],
    )


@router.post("/check", response_model=ChatResponse)
async def compliance_check(
    request: ComplianceCheckRequest,
    backend: dict = Depends(get_backend),
    rag: object = Depends(get_rag),
):
    """
    Ask a targeted compliance question and get a grounded answer.
    """
    prompt = _build_check_prompt(request)
    result = rag.query(question=prompt, backend=backend, top_k=5, return_sources=True)

    return ChatResponse(answer=result["answer"], sources=result.get("sources"))
