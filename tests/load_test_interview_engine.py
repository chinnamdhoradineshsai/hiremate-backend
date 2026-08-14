import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import time
import httpx

BASE_URL = "http://localhost:8000/api/v1"

from app.main import app
from app.api.deps import get_current_user, UserProfileContext

def mock_get_current_user():
    return UserProfileContext(
        id="test-load-user-123",
        email="loadtest@hiremate.ai",
        name="Load Test Candidate"
    )

app.dependency_overrides[get_current_user] = mock_get_current_user

async def simulate_candidate_session(client: httpx.AsyncClient, session_num: int):
    """Simulates a candidate running through an interview session from prep to finish."""
    headers = {"Authorization": "Bearer mock-token-test"}
    
    # 1. Prepare Interview
    prep_resp = await client.post("/api/v1/interview/prepare", json={
        "company": "TCS" if session_num % 2 == 0 else "Google",
        "role": "Software Engineer",
        "mode": "Standard"
    }, headers=headers)
    
    if prep_resp.status_code != 200:
        return {"session": session_num, "status": "FAILED_PREP", "code": prep_resp.status_code}
    
    session_data = prep_resp.json()
    session_id = session_data["session_id"]

    # 2. Fetch session details & initial questions
    sess_resp = await client.get(f"/api/v1/interview/session/{session_id}", headers=headers)
    if sess_resp.status_code != 200:
        return {"session": session_num, "status": "FAILED_FETCH_SESSION", "code": sess_resp.status_code}
    
    questions = sess_resp.json().get("questions", [])
    if not questions:
        return {"session": session_num, "status": "NO_QUESTIONS", "code": 500}

    # 3. Simulate answer submission for initial question
    first_q = questions[0]
    ans_resp = await client.post("/api/v1/interview/submit-answer", json={
        "question_id": first_q["id"],
        "answer_text": f"Concurrent load test response from session {session_num}",
        "selected_option_index": 1 if first_q.get("round_type") == "Aptitude" else None
    }, headers=headers)

    if ans_resp.status_code != 200:
        return {"session": session_num, "status": "FAILED_ANSWER", "code": ans_resp.status_code}

    # 4. Duplicate Submit Test (Idempotency under concurrency)
    dup_ans_resp = await client.post("/api/v1/interview/submit-answer", json={
        "question_id": first_q["id"],
        "answer_text": f"Concurrent load test response from session {session_num}",
        "selected_option_index": 1 if first_q.get("round_type") == "Aptitude" else None
    }, headers=headers)

    if dup_ans_resp.status_code != 200:
        return {"session": session_num, "status": "FAILED_IDEMPOTENT_SUBMIT", "code": dup_ans_resp.status_code}

    # 5. Concurrent Next-Question Request
    next_resp = await client.post("/api/v1/interview/next-question", json={"session_id": session_id}, headers=headers)
    if next_resp.status_code != 200:
        return {"session": session_num, "status": "FAILED_NEXT_QUESTION", "code": next_resp.status_code}

    # 6. Finish Session
    finish_resp = await client.post(f"/api/v1/interview/finish/{session_id}", headers=headers)
    if finish_resp.status_code != 200:
        return {"session": session_num, "status": "FAILED_FINISH", "code": finish_resp.status_code}

    return {"session": session_num, "status": "SUCCESS"}

async def run_load_test(concurrent_users: int = 50):
    print(f"Starting HireMate Scalability & Load Test simulating {concurrent_users} concurrent sessions...")
    start_time = time.time()
    
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        tasks = [simulate_candidate_session(client, i) for i in range(1, concurrent_users + 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time
    
    successes = [r for r in results if isinstance(r, dict) and r.get("status") == "SUCCESS"]
    failures = [r for r in results if not isinstance(r, dict) or r.get("status") != "SUCCESS"]

    print("\n==========================================")
    print("LOAD TEST EXECUTION SUMMARY")
    print("==========================================")
    print(f"Total Concurrent Sessions Attempted: {concurrent_users}")
    print(f"Successful Completed Sessions: {len(successes)}")
    print(f"Failed Sessions: {len(failures)}")
    print(f"Total Execution Duration: {elapsed:.2f} seconds")
    print(f"Throughput: {len(successes) / max(0.1, elapsed):.2f} sessions/sec")
    
    if failures:
        print("\nFailure Sample Details:")
        for f in failures[:5]:
            print(f" - {f}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    asyncio.run(run_load_test(count))
