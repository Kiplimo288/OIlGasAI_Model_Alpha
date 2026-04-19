
# OilGasAI API Documentation

Base URL: `http://localhost:8000` (local) or your deployed endpoint.

Interactive Swagger docs available at `/docs` when the server is running.

---

## Authentication

If `API_KEY` is set in `.env`, include it as a Bearer token:

```
Authorization: Bearer your_api_key_here
```

---

## Endpoints

### `GET /health`
Returns server status and model info.

**Response:**
```json
{
  "status": "ok",
  "model": "OilGasAI-Model-Alpha",
  "backend": "api",
  "platform": "https://innovations1-spec.github.io/oilgasai/"
}
```

---

### `POST /chat`
General-purpose domain Q&A with optional RAG grounding.

**Request:**
```json
{
  "message": "What are EPA Subpart W reporting thresholds for compressors?",
  "use_rag": true,
  "top_k": 5,
  "history": []
}
```

**Response:**
```json
{
  "answer": "Under EPA GHGRP Subpart W...",
  "sources": [
    {
      "source": "EPA_SubpartW_2024.pdf",
      "score": 0.847,
      "excerpt": "Centrifugal compressors must..."
    }
  ],
  "model": "OilGasAI-Model-Alpha"
}
```

---

### `POST /compliance/report`
Auto-generate a structured compliance report.

**Request:**
```json
{
  "facility_name": "Permian Basin Site A",
  "facility_id": "EPA-12345",
  "regulation": "EPA_GHGRP_Subpart_W",
  "reporting_year": 2025,
  "equipment_list": [
    "centrifugal compressor",
    "pneumatic controller",
    "storage tank"
  ],
  "additional_context": "Facility operates 12 months/year, gas-to-oil ratio > 300"
}
```

**Regulation options:**
- `EPA_GHGRP_Subpart_W`
- `NSPS_OOOOa`
- `NSPS_OOOOb`
- `NSPS_OOOOc`
- `OGMP_2.0`
- `State_Regulation`

---

### `POST /compliance/check`
Quick compliance question with regulatory grounding.

**Request:**
```json
{
  "question": "Do pneumatic controllers need to be reported if installed before 2016?",
  "regulation": "NSPS_OOOOa",
  "facility_type": "natural gas processing plant"
}
```

---

### `POST /sensors/calibrate`
Calibrate low-cost methane sensor readings.

**Request:**
```json
{
  "sensor_type": "MIRA_Methane",
  "raw_readings": [1.92, 2.15, 1.88, 2.34, 5.67, 2.01],
  "reference_readings": [1.95, 2.20, 1.90, 2.40, 5.80, 2.05],
  "temperature_c": [22.1, 22.3, 21.8, 23.0, 24.5, 22.7],
  "humidity_pct": [48, 51, 47, 53, 55, 49],
  "deployment_days": 45
}
```

**Sensor type options:**
- `MIRA_Methane`
- `Figaro_TGS2611`
- `Alphasense_OB3`
- `Other_Low_Cost`

**Response:**
```json
{
  "calibrated_readings": [1.94, 2.18, 1.89, 2.38, 5.75, 2.03],
  "drift_correction_applied": true,
  "calibration_method": "Drift correction (45d, factor=1.054) | Temperature compensation | Reference alignment (OLS: slope=1.023, intercept=0.012)",
  "notes": "Calibrated readings are within acceptable range for EPA LDAR screening...",
  "model": "OilGasAI-Model-Alpha"
}
```

---

### `POST /sensors/ldar`
Analyze LDAR survey data and generate repair action list.

**Request:**
```json
{
  "survey_data": [
    {"component": "valve", "reading_ppm": 750, "location": "Wellpad A, position V-12"},
    {"component": "connector", "reading_ppm": 2100, "location": "Wellpad A, flange F-03"},
    {"component": "pump", "reading_ppm": 180, "location": "Wellpad B, pump P-01"}
  ],
  "facility_name": "Eagle Ford Midstream Site B",
  "regulation": "NSPS_OOOOa",
  "survey_date": "2026-03-15"
}
```

**Response:**
```json
{
  "leaks_detected": [
    {
      "component": "valve",
      "location": "Wellpad A, position V-12",
      "reading_ppm": 750,
      "threshold_ppm": 500,
      "excess_ppm": 250.0,
      "first_repair_deadline": "2026-03-20",
      "final_repair_deadline": "2026-03-30",
      "priority": "STANDARD"
    }
  ],
  "total_components_surveyed": 3,
  "leak_count": 1,
  "repair_deadlines": {
    "first_attempt_days": 5,
    "final_repair_days": 15,
    "regulation": "NSPS_OOOOa"
  },
  "summary": "One leak detected at Eagle Ford Midstream Site B...",
  "model": "OilGasAI-Model-Alpha"
}
```

---

## Error Responses

| Code | Meaning |
|------|---------|
| 401 | Invalid or missing API key |
| 422 | Request validation error (check field types/values) |
| 500 | Internal server error (check logs) |

---

## Python Client Example

```python
import httpx

BASE_URL = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer your_api_key"}

# Simple chat
response = httpx.post(f"{BASE_URL}/chat", headers=HEADERS, json={
    "message": "What are the LDAR requirements for my gas processing facility?",
    "use_rag": True,
})
print(response.json()["answer"])

# Compliance report
response = httpx.post(f"{BASE_URL}/compliance/report", headers=HEADERS, json={
    "facility_name": "My Facility",
    "regulation": "EPA_GHGRP_Subpart_W",
    "reporting_year": 2025,
    "equipment_list": ["compressor", "valve"],
})
print(response.json()["report"])
```
