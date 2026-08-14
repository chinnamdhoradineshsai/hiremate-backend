import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.api.interview import LOCAL_SESSIONS, LOCAL_QUESTIONS, LOCAL_ANSWERS
from app.api.deps import get_current_user, UserProfileContext

client = TestClient(app)

# Override auth dependency for automated testing
def mock_get_current_user():
    return UserProfileContext(
        id="test-user-id-123",
        email="testcandidate@hiremate.ai",
        name="Test Candidate"
    )

app.dependency_overrides[get_current_user] = mock_get_current_user

def test_prepare_interview_creates_finite_stages():
    """Verify prepare_interview initializes finite stage configuration and state Machine."""
    payload = {
        "company": "TCS",
        "role": "Software Engineer",
        "mode": "Standard"
    }
    response = client.post("/api/v1/interview/prepare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "stage_configuration" in data
    stages = data["stage_configuration"]
    assert isinstance(stages, list)
    assert len(stages) >= 2
    for stage in stages:
        assert "name" in stage
        assert "type" in stage
        assert "question_count" in stage
        assert stage["question_count"] > 0

def test_idempotency_duplicate_answer_submission():
    """Verify repeated answer submissions return existing evaluation without duplicate processing."""
    # 1. Prepare session
    prep_resp = client.post("/api/v1/interview/prepare", json={
        "company": "Google",
        "role": "Site Reliability Engineer",
        "mode": "Standard"
    })
    session_id = prep_resp.json()["session_id"]
    
    # Fetch initial question
    sess_resp = client.get(f"/api/v1/interview/session/{session_id}")
    questions = sess_resp.json()["questions"]
    q_id = questions[0]["id"]

    answer_payload = {
        "question_id": q_id,
        "answer_text": "I would use Prometheus for metric scraping and Grafana for dashboarding with alertmanager.",
        "code_submission": None,
        "selected_option_index": None
    }

    # First Submit
    resp1 = client.post("/api/v1/interview/submit-answer", json=answer_payload)
    assert resp1.status_code == 200
    eval1 = resp1.json()

    # Second Submit (Duplicate)
    resp2 = client.post("/api/v1/interview/submit-answer", json=answer_payload)
    assert resp2.status_code == 200
    eval2 = resp2.json()

    assert eval1["score"] == eval2["score"]
    assert eval1["feedback"] == eval2["feedback"]

def test_idempotency_duplicate_next_question():
    """Verify duplicate /next-question requests do not create duplicate questions."""
    prep_resp = client.post("/api/v1/interview/prepare", json={
        "company": "Amazon",
        "role": "Backend Engineer",
        "mode": "Standard"
    })
    session_id = prep_resp.json()["session_id"]
    
    sess_resp = client.get(f"/api/v1/interview/session/{session_id}")
    q_id = sess_resp.json()["questions"][0]["id"]

    # Submit answer to first question
    client.post("/api/v1/interview/submit-answer", json={
        "question_id": q_id,
        "answer_text": "Used DynamoDB for low latency key-value storage.",
    })

    # Call /next-question twice
    next_req = {"session_id": session_id}
    res1 = client.post("/api/v1/interview/next-question", json=next_req)
    res2 = client.post("/api/v1/interview/next-question", json=next_req)

    assert res1.status_code == 200
    assert res2.status_code == 200

    q1 = res1.json()
    q2 = res2.json()

    # Should return identical question object or state
    if q1.get("id") and q2.get("id"):
        assert q1["id"] == q2["id"]

def test_finite_interview_completion():
    """Verify interview reaches completed state after questions for all stages are answered."""
    prep_resp = client.post("/api/v1/interview/prepare", json={
        "company": "Apple",
        "role": "iOS Developer",
        "mode": "Standard"
    })
    session_id = prep_resp.json()["session_id"]
    
    # Force single stage with 1 question for fast deterministic completion testing
    LOCAL_SESSIONS[session_id]["stage_configuration"] = [
        {"name": "Swift Fundamentals", "type": "Technical", "question_count": 1, "status": "official"}
    ]

    sess_resp = client.get(f"/api/v1/interview/session/{session_id}")
    q_id = sess_resp.json()["questions"][0]["id"]

    # Answer single stage question
    client.post("/api/v1/interview/submit-answer", json={
        "question_id": q_id,
        "answer_text": "ARC handles memory management via strong, weak, and unowned references.",
    })

    # Call next-question -> Should trigger interview completion
    next_res = client.post("/api/v1/interview/next-question", json={"session_id": session_id})
    assert next_res.status_code == 200
    data = next_res.json()
    assert data.get("interview_completed") is True

    # Finish interview
    finish_res = client.post(f"/api/v1/interview/finish/{session_id}")
    assert finish_res.status_code == 200
    report = finish_res.json()
    assert report["session_id"] == session_id
    assert report["overall_score"] >= 0.0

def test_idempotent_finish_interview():
    """Verify calling finish multiple times does not throw error and returns valid report."""
    prep_resp = client.post("/api/v1/interview/prepare", json={
        "company": "Microsoft",
        "role": "Software Engineer",
        "mode": "Standard"
    })
    session_id = prep_resp.json()["session_id"]

    res1 = client.post(f"/api/v1/interview/finish/{session_id}")
    res2 = client.post(f"/api/v1/interview/finish/{session_id}")

    assert res1.status_code == 200
    assert res2.status_code == 200
    assert res1.json()["session_id"] == res2.json()["session_id"]
