import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import unittest
from unittest.mock import AsyncMock, patch
from app.services.ai.resume_agent import resume_agent
from app.services.ai.research_agent import research_agent
from app.core.supabase import check_supabase_connection
from app.core.rate_limiter import request_deduplicator

class TestResilienceAndFailureModes(unittest.IsolatedAsyncioTestCase):

    async def test_nvidia_failure_raises_clean_exception(self):
        """Verify that NVIDIA failure raises a clean Exception without fake scores."""
        with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(side_effect=Exception("NVIDIA API timeout/error"))):
            with self.assertRaises(Exception) as context:
                await resume_agent.analyze_ats(
                    resume_text="Experienced Software Engineer with Python skills",
                    target_company="Google",
                    target_role="Software Engineer"
                )
            self.assertIn("NVIDIA", str(context.exception))

    async def test_tavily_failure_returns_no_fake_urls(self):
        """Verify that Tavily failure returns empty sources without fake URLs."""
        with patch("app.services.research.search_provider.search_provider.search", new=AsyncMock(return_value=[])):
            with patch("app.services.ai.ai_router.ai_router.execute_json_task", new=AsyncMock(return_value={
                "interview_stages": ["Technical Round"],
                "stage_configuration": [{"name": "Technical Round", "type": "Technical", "question_count": 3, "status": "likely/common"}],
                "common_topics": ["Python"],
                "public_questions": [],
                "role_requirements": ["Problem Solving"]
            })):
                res = await research_agent._perform_real_research("Google", "Software Engineer", "")
                self.assertEqual(res["sources"], [])
                for st in res["stage_configuration"]:
                    self.assertIn(st["status"], ["official", "likely/common", "Generic fallback interview — company-specific process could not be verified."])

    async def test_supabase_health_check_returns_bool(self):
        """Verify check_supabase_connection returns a boolean."""
        connected = check_supabase_connection()
        self.assertIsInstance(connected, bool)

    async def test_request_deduplication(self):
        """Verify parallel requests for the same key await the same single call."""
        call_count = 0

        async def slow_ai_call():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return {"result": "ok"}

        t1 = request_deduplicator.execute_or_await("test_key", slow_ai_call)
        t2 = request_deduplicator.execute_or_await("test_key", slow_ai_call)

        res1, res2 = await asyncio.gather(t1, t2)
        self.assertEqual(res1, {"result": "ok"})
        self.assertEqual(res2, {"result": "ok"})
        self.assertEqual(call_count, 1)

if __name__ == "__main__":
    unittest.main()
