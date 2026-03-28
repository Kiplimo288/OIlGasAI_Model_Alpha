"""
test_api.py
-----------
Integration tests for FastAPI endpoints.
Uses TestClient with mocked model backend.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


def make_mock_backend():
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = "This is a OilGasAI Model Alpha test response about EPA compliance."
    client.chat_completion.return_value = response
    return {"mode": "api", "client": client}


def make_mock_rag():
    rag = MagicMock()
    rag.query.return_value = {
        "answer": "RAG-grounded answer about Subpart W.",
        "sources": [{"source": "EPA_SubpartW.pdf", "score": 0.85, "excerpt": "..."}],
    }
    return rag


@pytest.fixture
def client():
    from api.main import app, app_state
    app_state["backend"] = make_mock_backend()
    app_state["rag"] = make_mock_rag()
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model"] == "OilGasAI-Model-Alpha"

    def test_root_returns_welcome(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "OilGasAI" in response.json()["message"]


class TestChatEndpoint:
    def test_basic_chat(self, client):
        response = client.post("/chat", json={"message": "What is Subpart W?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_chat_with_rag_disabled(self, client):
        response = client.post("/chat", json={
            "message": "Explain LDAR",
            "use_rag": False
        })
        assert response.status_code == 200

    def test_chat_empty_message_rejected(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422


class TestComplianceEndpoints:
    def test_generate_report(self, client):
        response = client.post("/compliance/report", json={
            "facility_name": "Permian Basin Site A",
            "regulation": "EPA_GHGRP_Subpart_W",
            "reporting_year": 2025,
            "equipment_list": ["centrifugal compressor", "pneumatic controller"],
        })
        assert response.status_code == 200
        data = response.json()
        assert "report" in data
        assert data["facility_name"] == "Permian Basin Site A"
        assert len(data["warnings"]) > 0

    def test_compliance_check(self, client):
        response = client.post("/compliance/check", json={
            "question": "Do I need to report pneumatic controllers under Subpart W?",
            "regulation": "EPA_GHGRP_Subpart_W",
        })
        assert response.status_code == 200
        assert "answer" in response.json()


class TestSensorEndpoints:
    def test_calibrate_sensor(self, client):
        response = client.post("/sensors/calibrate", json={
            "sensor_type": "MIRA_Methane",
            "raw_readings": [1.2, 1.5, 1.8, 2.1],
            "deployment_days": 30,
        })
        assert response.status_code == 200
        data = response.json()
        assert "calibrated_readings" in data
        assert len(data["calibrated_readings"]) == 4

    def test_ldar_analysis(self, client):
        response = client.post("/sensors/ldar", json={
            "survey_data": [
                {"component": "valve", "reading_ppm": 750, "location": "Well A"},
                {"component": "connector", "reading_ppm": 100, "location": "Well B"},
            ],
            "facility_name": "Test Facility",
            "regulation": "NSPS_OOOOa",
            "survey_date": "2026-03-01",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["leak_count"] == 1
        assert data["total_components_surveyed"] == 2
