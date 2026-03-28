"""
sensors/ldar.py
---------------
Leak Detection and Repair (LDAR) survey analysis.
Applies NSPS OOOOa/b/c and EPA Method 21 leak thresholds.
Integrates with OilGasAI Model Alpha for regulatory interpretation.
"""

import json
from datetime import datetime, timedelta
from loguru import logger
from inference.chat import chat


# ── Leak thresholds (ppm) by regulation ──────────────────────────────────────

LEAK_THRESHOLDS = {
    "NSPS_OOOOa": {
        "default": 500,       # ppm by volume, Method 21
        "connectors": 500,
        "valves": 500,
        "pumps": 2000,
        "compressor_seals": 500,
    },
    "NSPS_OOOOb": {
        "default": 200,
        "connectors": 200,
        "valves": 200,
        "pumps": 1000,
        "compressor_seals": 200,
    },
    "NSPS_OOOOc": {
        "default": 200,
        "connectors": 200,
        "valves": 200,
        "pumps": 1000,
        "compressor_seals": 200,
    },
    "EPA_GHGRP_Subpart_W": {
        "default": 500,
        "connectors": 500,
        "valves": 500,
        "pumps": 2000,
        "compressor_seals": 500,
    },
    "OGMP_2.0": {
        "default": 500,
        "connectors": 500,
        "valves": 500,
        "pumps": 2000,
        "compressor_seals": 500,
    },
}

# Repair deadlines in days by regulation
REPAIR_DEADLINES = {
    "NSPS_OOOOa": {"first_attempt": 5, "final": 15},
    "NSPS_OOOOb": {"first_attempt": 5, "final": 15},
    "NSPS_OOOOc": {"first_attempt": 5, "final": 15},
    "EPA_GHGRP_Subpart_W": {"first_attempt": 5, "final": 15},
    "OGMP_2.0": {"first_attempt": 7, "final": 30},
}


def analyze_ldar_survey(
    survey_data: list[dict],
    facility_name: str,
    regulation: str,
    survey_date: str,
    backend: dict | None = None,
) -> dict:
    """
    Analyze LDAR survey data and identify components exceeding leak thresholds.

    Args:
        survey_data: List of {"component": str, "reading_ppm": float, "location": str}
        facility_name: Facility name for report
        regulation: Regulation key (e.g. 'NSPS_OOOOa')
        survey_date: Survey date string (YYYY-MM-DD)
        backend: OilGasAI Model Alpha backend for interpretation

    Returns:
        dict with leaks_detected, repair_deadlines, summary
    """
    thresholds = LEAK_THRESHOLDS.get(regulation, LEAK_THRESHOLDS["NSPS_OOOOa"])
    deadlines_config = REPAIR_DEADLINES.get(regulation, REPAIR_DEADLINES["NSPS_OOOOa"])

    try:
        survey_dt = datetime.strptime(survey_date, "%Y-%m-%d")
    except ValueError:
        survey_dt = datetime.now()

    leaks = []
    for component in survey_data:
        comp_type = component.get("component", "default").lower()
        reading = float(component.get("reading_ppm", 0))
        location = component.get("location", "Unknown")

        # Match threshold key
        threshold_key = "default"
        for key in thresholds:
            if key in comp_type:
                threshold_key = key
                break

        threshold = thresholds[threshold_key]

        if reading >= threshold:
            first_repair = survey_dt + timedelta(days=deadlines_config["first_attempt"])
            final_repair = survey_dt + timedelta(days=deadlines_config["final"])

            leaks.append({
                "component": component.get("component"),
                "location": location,
                "reading_ppm": reading,
                "threshold_ppm": threshold,
                "excess_ppm": round(reading - threshold, 1),
                "first_repair_deadline": first_repair.strftime("%Y-%m-%d"),
                "final_repair_deadline": final_repair.strftime("%Y-%m-%d"),
                "priority": "HIGH" if reading > threshold * 3 else "STANDARD",
            })

    leak_count = len(leaks)
    total_surveyed = len(survey_data)

    repair_deadlines_summary = {
        "first_attempt_days": deadlines_config["first_attempt"],
        "final_repair_days": deadlines_config["final"],
        "regulation": regulation,
    }

    # OilGasAI Model Alpha summary
    summary = _generate_ldar_summary(
        facility_name, regulation, total_surveyed, leaks, backend
    )

    return {
        "leaks_detected": leaks,
        "total_components_surveyed": total_surveyed,
        "leak_count": leak_count,
        "repair_deadlines": repair_deadlines_summary,
        "summary": summary,
        "model": "OilGasAI-Model-Alpha",
    }


def _generate_ldar_summary(
    facility_name: str,
    regulation: str,
    total_surveyed: int,
    leaks: list[dict],
    backend: dict | None,
) -> str:
    """Generate OilGasAI Model Alpha regulatory interpretation of LDAR results."""
    if backend is None or not leaks:
        return (
            f"LDAR survey complete for {facility_name}. "
            f"{len(leaks)} leak(s) detected out of {total_surveyed} components surveyed under {regulation}."
        )

    high_priority = [l for l in leaks if l["priority"] == "HIGH"]
    leak_summary = json.dumps(
        [{"component": l["component"], "reading_ppm": l["reading_ppm"], "location": l["location"]}
         for l in leaks[:10]],  # limit to first 10
        indent=2,
    )

    prompt = (
        f"LDAR survey results for {facility_name} under {regulation}:\n"
        f"- Components surveyed: {total_surveyed}\n"
        f"- Leaks detected: {len(leaks)}\n"
        f"- High-priority leaks: {len(high_priority)}\n"
        f"- Leak details (first 10):\n{leak_summary}\n\n"
        f"In 3-4 sentences, summarize the compliance status, prioritize repair actions, "
        f"and flag any regulatory reporting obligations triggered by these results."
    )

    try:
        return chat(prompt, backend)
    except Exception as e:
        logger.warning(f"OilGasAI Model Alpha LDAR summary failed: {e}")
        return (
            f"{len(leaks)} leak(s) detected at {facility_name}. "
            f"Initiate repairs per {regulation} deadlines. "
            f"Document all repairs and re-survey as required."
        )
