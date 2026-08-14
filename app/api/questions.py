from typing import Optional
from fastapi import APIRouter, Depends, Query
from app.api.deps import get_current_user, UserProfileContext
from app.core.supabase import get_supabase, is_supabase_configured

router = APIRouter(prefix="/questions", tags=["Question History Vault"])

@router.get("/vault")
async def get_question_vault(
    company: Optional[str] = Query(None),
    round_type: Optional[str] = Query(None),
    topic: Optional[str] = Query(None),
    current_user: UserProfileContext = Depends(get_current_user)
):
    if not is_supabase_configured():
        return {
            "total_questions": 0,
            "questions": []
        }

    supabase = get_supabase()
    q_res = supabase.table("interview_questions").select("*").eq("user_id", current_user.id).order("created_at", desc=True).execute()
    questions = q_res.data if q_res and q_res.data else []

    ans_res = supabase.table("interview_answers").select("*").eq("user_id", current_user.id).execute()
    answers = ans_res.data if ans_res and ans_res.data else []
    answers_map = {a["question_id"]: a for a in answers}

    vault_items = []
    for q in questions:
        q_comp = q.get("company", "")
        q_round = q.get("round_type", "")
        q_topic = q.get("topic", "")

        if company and company.lower() not in q_comp.lower():
            continue
        if round_type and round_type.lower() != q_round.lower():
            continue
        if topic and topic.lower() not in q_topic.lower():
            continue

        a = answers_map.get(q["id"])
        vault_items.append({
            "id": q["id"],
            "company": q_comp,
            "role": q.get("role", ""),
            "round_type": q_round,
            "question_text": q.get("question_text", ""),
            "topic": q_topic,
            "difficulty": q.get("difficulty", "Medium"),
            "source_type": q.get("source_type", "AI Generated 🟣"),
            "source_url": q.get("source_url"),
            "user_answer": a["answer_text"] if a else None,
            "code_submission": a["code_submission"] if a else None,
            "score": a["score"] if a else None,
            "evaluation": a["evaluation"] if a else None,
            "date": q.get("created_at", "")[:16]
        })

    return {
        "total_questions": len(vault_items),
        "questions": vault_items
    }
