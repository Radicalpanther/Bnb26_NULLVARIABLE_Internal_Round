# test_api.py - Unit and integration tests for FastAPI /diagnose and /health endpoints
import json
import requests
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app import app

client = TestClient(app)

SAMPLE_3STEP_RUN = {
    "run_id": "test_run_001",
    "steps": [
        {
            "run_id": "test_run_001",
            "step_index": 0,
            "step_type": "plan",
            "tool_name": "none",
            "input_text": "User Goal: Calculate final price with 10% tax on $250 item.",
            "output_text": "1. Compute base price. 2. Calculate tax. 3. Format answer.",
            "latency_ms": 180.0,
            "error_flag": 0,
            "retry_count": 0,
            "output_length": 60,
            "state_snapshot": {"goal": "Calculate price", "task_type": "math_lookup", "extracted_values": {}},
            "depends_on": ""
        },
        {
            "run_id": "test_run_001",
            "step_index": 1,
            "step_type": "call_tool",
            "tool_name": "calculator",
            "input_text": "250 * 1.10",
            "output_text": "Error 500: syntax error in expression evaluator failed",
            "latency_ms": 1450.0,
            "error_flag": 1,
            "retry_count": 1,
            "output_length": 54,
            "state_snapshot": {"goal": "Calculate price", "task_type": "math_lookup", "extracted_values": {}},
            "depends_on": "0"
        },
        {
            "run_id": "test_run_001",
            "step_index": 2,
            "step_type": "write_answer",
            "tool_name": "none",
            "input_text": "Synthesize response",
            "output_text": "Calculation failed due to evaluator error.",
            "latency_ms": 200.0,
            "error_flag": 0,
            "retry_count": 0,
            "output_length": 42,
            "state_snapshot": {"goal": "Calculate price", "task_type": "math_lookup", "extracted_values": {}},
            "depends_on": "1"
        }
    ]
}


def test_health_check():
    """Verify /health returns 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_diagnose_response_shape():
    """Verify /diagnose with a fake 3-step run returns 200 and exact expected response shape."""
    response = client.post("/diagnose?llm=false", json=SAMPLE_3STEP_RUN)
    assert response.status_code == 200
    
    data = response.json()
    assert "run_id" in data
    assert data["run_id"] == "test_run_001"
    
    assert "ranking" in data
    assert isinstance(data["ranking"], list)
    assert len(data["ranking"]) == 3

    # Verify ranking item fields and sorting
    prev_score = float("inf")
    for item in data["ranking"]:
        assert "step_index" in item
        assert "step_type" in item
        assert "score" in item
        assert "evidence" in item
        assert isinstance(item["step_index"], int)
        assert isinstance(item["step_type"], str)
        assert isinstance(item["score"], (int, float))
        assert isinstance(item["evidence"], list)
        assert item["score"] <= prev_score
        prev_score = item["score"]

    # Top step should have SHAP evidence
    assert len(data["ranking"][0]["evidence"]) > 0
    assert isinstance(data["ranking"][0]["evidence"][0], str)

    assert "explanation" in data
    assert isinstance(data["explanation"], str)
    assert len(data["explanation"]) > 0


def test_diagnose_ollama_unavailable_fallback():
    """Verify that when Ollama is unavailable/fails, /diagnose gracefully returns fallback sentence."""
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Ollama connection refused")):
        response = client.post("/diagnose?llm=true", json=SAMPLE_3STEP_RUN)
        assert response.status_code == 200
        
        data = response.json()
        assert "explanation" in data
        explanation = data["explanation"]
        
        # Check that the explanation contains the fallback format
        assert "identified as the root cause" in explanation
        assert "Replaying execution from step" in explanation


def test_diagnose_raw_list_input():
    """Verify that /diagnose also accepts a raw list of step dictionaries."""
    response = client.post("/diagnose?llm=false", json=SAMPLE_3STEP_RUN["steps"])
    assert response.status_code == 200
    data = response.json()
    assert len(data["ranking"]) == 3
    assert data["run_id"] == "test_run_001"
