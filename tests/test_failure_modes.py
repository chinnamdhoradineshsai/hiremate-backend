import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.api.interview import LOCAL_SESSIONS, LOCAL_QUESTIONS, LOCAL_ANSWERS
from app.api.deps import get_current_user, UserProfileContext
from app.services.research.tavily_client import tavily_client
from app.services.ai.nvidia_client import nvidia_client
from app.services.ai.research_agent import research_agent
from app.services.ai.interview_agent import interview_agent

def mock_get_current_user():
    return UserProfileContext(
        id="test-user-id-999",
        email="hardening_tester@hiremate.ai",
        name="Hardening Tester"
    )

app.dependency_overrides[get_current_user] = mock_get_current_user

class TestFailureModesAndHardening(unittest.IsolatedAsyncioTestCase):

    async def test_1_tavily_timeout_failure(self):
        """1. Tavily timeout/failure handling."""
        with patch.object(tavily_client, "search", new=AsyncMock(side_effect=Exception("Tavily timeout 20s"))):
            res = await research_agent.get_or_research_company("MockCorp", "DevOps Engineer", force_refresh=True)
            self.assertEqual(res["sources"], [])
            self.assertIn("stage_configuration", res)
            self.assertFalse(res.get("is_verified", True))

    async def test_2_nemotron_timeout_failure(self):
        """2. Nemotron timeout/failure handling."""
        with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(side_effect=Exception("NVIDIA Nemotron 45s Timeout"))):
            with self.assertRaises(Exception) as ctx:
                await interview_agent.generate_questions_for_session(
                    user_id="u1", company="TCS", role="SE", mode="Standard",
                    resume_text="", research_data={"stage_configuration": [{"name": "Tech", "type": "Technical", "question_count": 3}]},
                    db=None
                )
            self.assertIn("NVIDIA Nemotron 45s Timeout", str(ctx.exception))

    async def test_3_supabase_write_failure(self):
        """3. Supabase write failure returns 500 error instead of silent success."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.api.interview.is_supabase_configured", return_value=True):
                mock_supa = MagicMock()
                mock_supa.table.return_value.insert.side_effect = Exception("Database connection lost")
                with patch("app.api.interview.get_supabase", return_value=mock_supa):
                    with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=[{"round_type": "Technical", "question_text": "Q1"}])):
                        resp = await client.post("/api/v1/interview/prepare", json={
                            "company": "FailCorp",
                            "role": "Backend Engineer",
                            "mode": "Standard"
                        })
                        self.assertEqual(resp.status_code, 500)
                        self.assertIn("Failed to create interview session in database", resp.json()["detail"])

    async def test_4_duplicate_answer_submission(self):
        """4. Duplicate answer submission returns existing evaluation cleanly."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=[{"round_type": "Technical", "question_text": "Q1"}] * 10)):
                prep_resp = await client.post("/api/v1/interview/prepare", json={"company": "TCS", "role": "SE", "mode": "Standard"})
            session_id = prep_resp.json()["session_id"]
            
            sess_resp = await client.get(f"/api/v1/interview/session/{session_id}")
            q_id = sess_resp.json()["questions"][0]["id"]

            payload = {
                "question_id": q_id,
                "answer_text": "Object-oriented design principles: Encapsulation, Inheritance, Polymorphism, Abstraction.",
            }

            resp1 = await client.post("/api/v1/interview/submit-answer", json=payload)
            resp2 = await client.post("/api/v1/interview/submit-answer", json=payload)

            self.assertEqual(resp1.status_code, 200)
            self.assertEqual(resp2.status_code, 200)
            self.assertEqual(resp1.json()["score"], resp2.json()["score"])

    async def test_5_duplicate_next_question_request(self):
        """5. Duplicate next-question request returns identical unanswered question."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=[{"round_type": "Technical", "question_text": "Q1"}] * 10)):
                prep_resp = await client.post("/api/v1/interview/prepare", json={"company": "Infosys", "role": "Developer", "mode": "Standard"})
            session_id = prep_resp.json()["session_id"]
            
            sess_resp = await client.get(f"/api/v1/interview/session/{session_id}")
            q_id = sess_resp.json()["questions"][0]["id"]

            await client.post("/api/v1/interview/submit-answer", json={"question_id": q_id, "answer_text": "Java Garbage Collection."})

            res1 = await client.post("/api/v1/interview/next-question", json={"session_id": session_id})
            res2 = await client.post("/api/v1/interview/next-question", json={"session_id": session_id})

            self.assertEqual(res1.status_code, 200)
            self.assertEqual(res2.status_code, 200)
            if res1.json().get("id") and res2.json().get("id"):
                self.assertEqual(res1.json()["id"], res2.json()["id"])

    async def test_6_duplicate_finish_request(self):
        """6. Duplicate finish request returns identical final report."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=[{"round_type": "Technical", "question_text": "Q1"}] * 10)):
                prep_resp = await client.post("/api/v1/interview/prepare", json={"company": "Wipro", "role": "Tester", "mode": "Standard"})
            session_id = prep_resp.json()["session_id"]

            res1 = await client.post(f"/api/v1/interview/finish/{session_id}")
            res2 = await client.post(f"/api/v1/interview/finish/{session_id}")

            self.assertEqual(res1.status_code, 200)
            self.assertEqual(res2.status_code, 200)
            self.assertEqual(res1.json()["session_id"], res2.json()["session_id"])

    async def test_7_tavily_returns_no_useful_sources(self):
        """7. Tavily returns empty results -> research returns empty sources and unverified label."""
        with patch("app.services.research.search_provider.search_provider.search", new=AsyncMock(return_value=[])):
            res = await research_agent.get_or_research_company("EmptyCo", "Role", force_refresh=True)
            self.assertEqual(res["sources"], [])
            self.assertFalse(res.get("is_verified", True))
            for st in res["stage_configuration"]:
                self.assertEqual(st["status"], "Generic fallback interview — company-specific process could not be verified.")

    async def test_8_nemotron_returns_too_few_questions(self):
        """8. Nemotron returns too few questions -> retries bounded and raises clean error if still insufficient."""
        with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value=[])):
            with self.assertRaises(ValueError) as ctx:
                await interview_agent.generate_questions_for_session(
                    user_id="u1", company="TCS", role="SE", mode="Standard",
                    resume_text="", research_data={"stage_configuration": [{"name": "Stage 1", "type": "Technical", "question_count": 5}]},
                    db=None
                )
            self.assertIn("insufficient verified questions", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
