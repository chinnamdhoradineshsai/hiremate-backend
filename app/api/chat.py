import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_current_user, UserProfileContext
from app.services.ai.ai_gateway import ai_gateway
from app.core.supabase import get_supabase, is_supabase_configured
from app.core.rate_limiter import rate_limit_expensive_ai
from app.schemas.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["AI Career Assistant"])

@router.post("", response_model=ChatResponse, dependencies=[Depends(rate_limit_expensive_ai)])
async def chat_with_career_assistant(
    req: ChatRequest,
    current_user: UserProfileContext = Depends(get_current_user)
):
    supabase = get_supabase() if is_supabase_configured() else None

    # 1. Fetch user resume & ATS analysis context from Supabase
    ats_score = "N/A"
    missing_skills = []
    target_company = req.current_company or "N/A"
    target_role = req.current_role or "N/A"
    latest_session = None

    if supabase:
        try:
            res = supabase.table("ats_analyses").select("*").eq("user_id", current_user.id).order("created_at", desc=True).limit(1).execute()
            if res and res.data:
                latest_ats = res.data[0]
                ats_score = latest_ats.get("overall_score", "N/A")
                missing_skills = [s.get("skill_name") for s in latest_ats.get("missing_skills", []) if isinstance(s, dict)]
                if not req.current_company and latest_ats.get("company"):
                    target_company = latest_ats.get("company")
                if not req.current_role and latest_ats.get("role"):
                    target_role = latest_ats.get("role")

            sess_res = supabase.table("interview_sessions").select("*").eq("user_id", current_user.id).order("created_at", desc=True).limit(1).execute()
            latest_session = sess_res.data[0] if sess_res and sess_res.data else None
        except Exception as e:
            print(f"[Supabase Chat Context Warning]: {e}")

    user_context = {
        "name": current_user.name,
        "ats_score": ats_score,
        "target_company": target_company,
        "target_role": target_role,
        "missing_skills": missing_skills,
        "interview_readiness": int(latest_session["overall_score"]) if (latest_session and latest_session.get("overall_score") is not None) else "N/A"
    }

    history_dicts = [m.model_dump() for m in req.history] if req.history else []
    chat_result = await ai_gateway.chat_career_assistant(
        message=req.message,
        history=history_dicts,
        user_context=user_context
    )

    reply_text = chat_result.get("reply", "I am here to assist with your interview preparation.")

    # 2. Persist Chat History in Supabase 'career_chat_history' table if configured
    if supabase:
        try:
            supabase.table("career_chat_history").insert([
                {"id": str(uuid.uuid4()), "user_id": current_user.id, "role": "user", "content": req.message, "created_at": datetime.utcnow().isoformat()},
                {"id": str(uuid.uuid4()), "user_id": current_user.id, "role": "assistant", "content": reply_text, "created_at": datetime.utcnow().isoformat()}
            ]).execute()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist chat history in database: {str(e)}"
            )

    return ChatResponse(
        reply=reply_text,
        suggested_actions=chat_result.get("suggested_actions", [])
    )
