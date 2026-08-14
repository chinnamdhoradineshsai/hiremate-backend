import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import json
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.api.interview import LOCAL_SESSIONS, LOCAL_QUESTIONS, LOCAL_ANSWERS
from app.api.deps import get_current_user, UserProfileContext

def mock_get_current_user():
    return UserProfileContext(
        id="a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        email="testcandidate@hiremate.ai",
        name="Test Candidate"
    )

app.dependency_overrides[get_current_user] = mock_get_current_user

from unittest.mock import patch, AsyncMock

MOCK_AI_QUESTIONS = [
    {"round_type": "Aptitude", "question_text": "Sample Aptitude Question 1", "topic": "Math", "difficulty": "Medium", "options": ["A", "B", "C", "D"], "correct_option_index": 0},
    {"round_type": "Aptitude", "question_text": "Sample Aptitude Question 2", "topic": "Math", "difficulty": "Medium", "options": ["A", "B", "C", "D"], "correct_option_index": 1},
    {"round_type": "Aptitude", "question_text": "Sample Aptitude Question 3", "topic": "Math", "difficulty": "Medium", "options": ["A", "B", "C", "D"], "correct_option_index": 2},
    {"round_type": "Technical", "question_text": "Sample Tech Question 1", "topic": "Architecture", "difficulty": "Hard"},
    {"round_type": "Technical", "question_text": "Sample Tech Question 2", "topic": "Databases", "difficulty": "Hard"},
    {"round_type": "Technical", "question_text": "Sample Tech Question 3", "topic": "Networking", "difficulty": "Medium"},
    {"round_type": "Technical", "question_text": "Sample Tech Question 4", "topic": "OS", "difficulty": "Medium"},
    {"round_type": "HR", "question_text": "Sample HR Question 1", "topic": "Behavioral", "difficulty": "Medium"},
    {"round_type": "HR", "question_text": "Sample HR Question 2", "topic": "Leadership", "difficulty": "Medium"},
]

async def test_prepare_interview_creates_finite_stages():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"company": "TCS", "role": "Software Engineer", "mode": "Standard"}
        with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=MOCK_AI_QUESTIONS)):
            response = await client.post("/api/v1/interview/prepare", json=payload)
        assert response.status_code == 200, f"Status: {response.status_code}, Body: {response.text}"
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
        print("[PASS] test_prepare_interview_creates_finite_stages")

async def test_idempotency_duplicate_answer_submission():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=MOCK_AI_QUESTIONS)):
            prep_resp = await client.post("/api/v1/interview/prepare", json={"company": "Google", "role": "SRE", "mode": "Standard"})
        session_id = prep_resp.json()["session_id"]
        
        sess_resp = await client.get(f"/api/v1/interview/session/{session_id}")
        questions = sess_resp.json()["questions"]
        q_id = questions[0]["id"]

        answer_payload = {
            "question_id": q_id,
            "answer_text": "I would use Prometheus for metrics and Grafana for dashboards.",
            "code_submission": None,
            "selected_option_index": None
        }

        resp1 = await client.post("/api/v1/interview/submit-answer", json=answer_payload)
        assert resp1.status_code == 200, f"Error: {resp1.text}"
        eval1 = resp1.json()

        # Duplicate submission
        resp2 = await client.post("/api/v1/interview/submit-answer", json=answer_payload)
        assert resp2.status_code == 200, f"Error: {resp2.text}"
        eval2 = resp2.json()

        assert eval1["score"] == eval2["score"]
        assert eval1["feedback"] == eval2["feedback"]
        print("[PASS] test_idempotency_duplicate_answer_submission")

async def test_idempotency_duplicate_next_question():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=MOCK_AI_QUESTIONS)):
            prep_resp = await client.post("/api/v1/interview/prepare", json={"company": "Amazon", "role": "Backend Engineer", "mode": "Standard"})
        session_id = prep_resp.json()["session_id"]
        
        sess_resp = await client.get(f"/api/v1/interview/session/{session_id}")
        q_id = sess_resp.json()["questions"][0]["id"]

        await client.post("/api/v1/interview/submit-answer", json={"question_id": q_id, "answer_text": "DynamoDB key-value store."})

        # Double /next-question
        next_req = {"session_id": session_id}
        res1 = await client.post("/api/v1/interview/next-question", json=next_req)
        res2 = await client.post("/api/v1/interview/next-question", json=next_req)

        assert res1.status_code == 200, f"Error: {res1.text}"
        assert res2.status_code == 200, f"Error: {res2.text}"
        
        q1 = res1.json()
        q2 = res2.json()

        if q1.get("id") and q2.get("id"):
            assert q1["id"] == q2["id"]
        print("[PASS] test_idempotency_duplicate_next_question")

async def test_finite_interview_completion():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=MOCK_AI_QUESTIONS)):
            prep_resp = await client.post("/api/v1/interview/prepare", json={"company": "Apple", "role": "iOS Developer", "mode": "Standard"})
        session_id = prep_resp.json()["session_id"]
        
        # Override stage config for deterministic single-question test
        LOCAL_SESSIONS[session_id]["stage_configuration"] = [
            {"name": "Swift Fundamentals", "type": "Technical", "question_count": 1, "status": "official"}
        ]

        sess_resp = await client.get(f"/api/v1/interview/session/{session_id}")
        q_id = sess_resp.json()["questions"][0]["id"]

        await client.post("/api/v1/interview/submit-answer", json={"question_id": q_id, "answer_text": "ARC memory management."})

        next_res = await client.post("/api/v1/interview/next-question", json={"session_id": session_id})
        assert next_res.status_code == 200
        assert next_res.json().get("interview_completed") is True

        finish_res = await client.post(f"/api/v1/interview/finish/{session_id}")
        assert finish_res.status_code == 200
        report = finish_res.json()
        assert report["session_id"] == session_id
        assert report["overall_score"] >= 0.0
        print("[PASS] test_finite_interview_completion")

async def run_all_tests():
    print("Executing HireMate Interview Engine Test Suite...")
    try:
        await test_prepare_interview_creates_finite_stages()
        await test_idempotency_duplicate_answer_submission()
        await test_idempotency_duplicate_next_question()
        await test_finite_interview_completion()
        print("\nALL TEST SUITE CHECKS PASSED SUCCESSFULLY!")
    except Exception as e:
        print(f"\nTEST FAILURE: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_all_tests())
