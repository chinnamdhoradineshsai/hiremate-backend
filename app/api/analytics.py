from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user, UserProfileContext
from app.core.supabase import get_supabase, is_supabase_configured

router = APIRouter(prefix="/analytics", tags=["Dashboard Analytics"])

@router.get("/dashboard")
async def get_dashboard_analytics(
    current_user: UserProfileContext = Depends(get_current_user)
):
    """
    Supabase Real-Time Dashboard Analytics.
    Queries Supabase tables for ATS scores, interview readiness, score trends, and weak skills dynamically.
    Returns None/empty lists when the user has not completed actions or Supabase is unconfigured.
    """
    if not is_supabase_configured():
        return {
            "user_profile": {
                "name": current_user.name,
                "email": current_user.email,
                "avatar_url": current_user.avatar_url
            },
            "ats_card": {
                "score": None,
                "max": 100,
                "delta_text": None,
                "breakdown": None
            },
            "interview_card": {
                "readiness": None,
                "max": 100,
                "delta_text": None,
                "breakdown": None
            },
            "recent_interview": None,
            "total_questions_answered": 0,
            "progress_trends": {
                "labels": [],
                "ats_progress": [],
                "interview_progress": [],
                "technical_progress": [],
                "aptitude_progress": [],
                "hr_progress": []
            },
            "weak_skills": [],
            "total_interviews_taken": 0
        }

    supabase = get_supabase()

    # 1. Fetch user ATS Analyses from Supabase
    ats_res = supabase.table("ats_analyses").select("*").eq("user_id", current_user.id).order("created_at", desc=False).execute()
    ats_records = ats_res.data if ats_res and ats_res.data else []

    latest_ats = ats_records[-1] if ats_records else None
    ats_score: Optional[int] = latest_ats["overall_score"] if latest_ats else None
    ats_history: List[int] = [a["overall_score"] for a in ats_records] if ats_records else []

    ats_delta_text: Optional[str] = None
    if len(ats_records) == 1:
        ats_delta_text = "First analysis"
    elif len(ats_records) > 1:
        diff = ats_records[-1]["overall_score"] - ats_records[-2]["overall_score"]
        ats_delta_text = f"{'+' if diff >= 0 else ''}{diff}% since previous edit"

    # 2. Fetch completed interview sessions from Supabase
    int_res = supabase.table("interview_sessions").select("*").eq("user_id", current_user.id).order("created_at", desc=False).execute()
    sessions = int_res.data if int_res and int_res.data else []

    completed_sessions = [s for s in sessions if s.get("status") == "completed" and s.get("overall_score") is not None]

    interview_readiness: Optional[int] = int(completed_sessions[-1]["overall_score"]) if completed_sessions else None
    interview_history: List[int] = [int(s["overall_score"]) for s in completed_sessions] if completed_sessions else []
    technical_history: List[int] = [int(s.get("technical_score") or 0) for s in completed_sessions] if completed_sessions else []
    aptitude_history: List[int] = [int(s.get("aptitude_score") or 0) for s in completed_sessions] if completed_sessions else []
    hr_history: List[int] = [int(s.get("hr_score") or 0) for s in completed_sessions] if completed_sessions else []

    interview_delta_text: Optional[str] = None
    if len(completed_sessions) == 1:
        interview_delta_text = "First interview"
    elif len(completed_sessions) > 1:
        diff = int(completed_sessions[-1]["overall_score"] - completed_sessions[-2]["overall_score"])
        interview_delta_text = f"{'+' if diff >= 0 else ''}{diff}% since previous session"

    recent_interview = None
    if completed_sessions:
        s0 = completed_sessions[-1]
        recent_interview = {
            "session_id": s0["id"],
            "company": s0["company"],
            "role": s0["role"],
            "score": s0["overall_score"],
            "aptitude_score": s0.get("aptitude_score", 0.0),
            "technical_score": s0.get("technical_score", 0.0),
            "coding_score": s0.get("coding_score", 0.0),
            "hr_score": s0.get("hr_score", 0.0),
            "date": s0.get("created_at", "")[:10]
        }

    # 3. Compute answered questions count from Supabase
    ans_res = supabase.table("interview_answers").select("id", count="exact").eq("user_id", current_user.id).execute()
    total_answers = ans_res.count if ans_res and ans_res.count is not None else len(ans_res.data if ans_res and ans_res.data else [])

    # 4. Extract weak skills
    weak_skills_set = set()
    if latest_ats and latest_ats.get("missing_skills"):
        for s in latest_ats.get("missing_skills", []):
            if isinstance(s, dict) and s.get("skill_name"):
                weak_skills_set.add(s.get("skill_name"))

    low_ans_res = supabase.table("interview_answers").select("question_id, score").eq("user_id", current_user.id).lt("score", 70.0).execute()
    if low_ans_res and low_ans_res.data:
        low_q_ids = [a["question_id"] for a in low_ans_res.data if a.get("question_id")]
        if low_q_ids:
            q_res = supabase.table("interview_questions").select("topic").in_("id", low_q_ids).execute()
            if q_res and q_res.data:
                for q in q_res.data:
                    if q.get("topic"): weak_skills_set.add(q.get("topic"))

    # 5. Trend Labels
    max_attempts = max(len(ats_history), len(interview_history))
    trend_labels = [f"Attempt {i+1}" for i in range(max_attempts)]

    return {
        "user_profile": {
            "name": current_user.name,
            "email": current_user.email,
            "avatar_url": current_user.avatar_url
        },
        "ats_card": {
            "score": ats_score,
            "max": 100,
            "delta_text": ats_delta_text,
            "breakdown": latest_ats.get("breakdown") if latest_ats else None
        },
        "interview_card": {
            "readiness": interview_readiness,
            "max": 100,
            "delta_text": interview_delta_text,
            "breakdown": {
                "aptitude": int(completed_sessions[-1].get("aptitude_score") or 0) if completed_sessions else None,
                "technical": int(completed_sessions[-1].get("technical_score") or 0) if completed_sessions else None,
                "hr": int(completed_sessions[-1].get("hr_score") or 0) if completed_sessions else None
            } if completed_sessions else None
        },
        "recent_interview": recent_interview,
        "total_questions_answered": total_answers,
        "progress_trends": {
            "labels": trend_labels,
            "ats_progress": ats_history,
            "interview_progress": interview_history,
            "technical_progress": technical_history,
            "aptitude_progress": aptitude_history,
            "hr_progress": hr_history
        },
        "weak_skills": list(weak_skills_set),
        "total_interviews_taken": len(completed_sessions)
    }
