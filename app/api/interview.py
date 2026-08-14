import uuid
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from app.api.deps import get_current_user, UserProfileContext
from app.services.ai.ai_gateway import ai_gateway
from app.core.supabase import get_supabase, is_supabase_configured
from app.schemas.schemas import (
    InterviewPrepRequest, SubmitAnswerRequest, AnswerEvaluationOut, FinalReportOut
)

from app.core.rate_limiter import rate_limit_expensive_ai

router = APIRouter(prefix="/interview", tags=["Interview Preparation"])

# Local persistence store for Demo Mode and unconfigured Supabase environments
LOCAL_SESSIONS: Dict[str, Dict[str, Any]] = {}
LOCAL_QUESTIONS: Dict[str, List[Dict[str, Any]]] = {}
LOCAL_ANSWERS: Dict[str, List[Dict[str, Any]]] = {}
LOCAL_DEMO_PROFILES: Dict[str, Dict[str, Any]] = {}

# Session lock registry for race-condition prevention during concurrent requests
SESSION_LOCKS: Dict[str, asyncio.Lock] = {}

def get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in SESSION_LOCKS:
        SESSION_LOCKS[session_id] = asyncio.Lock()
    return SESSION_LOCKS[session_id]

class AdaptiveNextQuestionRequest(BaseModel):
    session_id: str

@router.get("/demo-status")
async def get_demo_status(
    current_user: UserProfileContext = Depends(get_current_user)
):
    """Checks whether the one-time demo has been completed for the user."""
    demo_used = False
    if is_supabase_configured():
        try:
            supabase = get_supabase()
            prof_res = supabase.table("profiles").select("demo_used").eq("user_id", current_user.id).execute()
            if prof_res and prof_res.data:
                demo_used = bool(prof_res.data[0].get("demo_used"))
        except Exception as e:
            print(f"[Supabase Demo Status Query Warning]: {e}")

    if not demo_used:
        demo_used = LOCAL_DEMO_PROFILES.get(current_user.id, {}).get("demo_used", False)

    return {"demo_used": demo_used}

@router.post("/prepare", dependencies=[Depends(rate_limit_expensive_ai)])
async def prepare_interview(
    req: InterviewPrepRequest,
    current_user: UserProfileContext = Depends(get_current_user)
):
    supabase = get_supabase() if is_supabase_configured() else None
    is_demo = (req.mode == "Demo" or req.mode == "demo")

    # Check Demo eligibility if demo mode requested
    if is_demo:
        demo_already_done = False
        if supabase:
            try:
                prof_res = supabase.table("profiles").select("demo_used").eq("user_id", current_user.id).execute()
                if prof_res and prof_res.data and prof_res.data[0].get("demo_used"):
                    demo_already_done = True
            except Exception as e:
                print(f"[Demo Eligibility Check Warning]: {e}")

        if not demo_already_done:
            demo_already_done = LOCAL_DEMO_PROFILES.get(current_user.id, {}).get("demo_used", False)

        if demo_already_done:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Demo mode has already been completed for this account. Please sign in with Google for full access."
            )

    company_target = "TCS" if is_demo else req.company
    role_target = "Software Engineer" if is_demo else req.role

    t_prepare_start = time.monotonic()
    print(f"[Interview Timing] prepare start: {company_target} / {role_target}")

    # 1. Fetch user's latest resume if not explicitly passed
    resume_text = ""
    if supabase:
        try:
            if req.resume_id:
                res = supabase.table("resumes").select("*").eq("id", req.resume_id).eq("user_id", current_user.id).execute()
                if res and res.data:
                    resume_text = res.data[0].get("extracted_text", "")
            else:
                res = supabase.table("resumes").select("*").eq("user_id", current_user.id).order("uploaded_at", desc=True).limit(1).execute()
                if res and res.data:
                    resume_text = res.data[0].get("extracted_text", "")
        except Exception as e:
            print(f"[Supabase Resume Fetch Warning]: {e}")

    # 2. Perform or fetch Company Research (hits in-process cache if /research was called first)
    t_research_start = time.monotonic()
    research_data = await ai_gateway.research_company(
        company=company_target,
        role=role_target,
        db=None,
        job_description=req.job_description or "",
        force_refresh=False
    )
    print(f"[Interview Timing] research: {(time.monotonic() - t_research_start)*1000:.0f} ms")

    # 3. Extract researched Stage Configuration as Source of Truth
    stage_config = research_data.get("stage_configuration", [])
    is_verified = research_data.get("is_verified", True)

    if not is_verified or not stage_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Company-specific interview process could not be verified."
        )

    # Sanitize and clamp question_count values (1-10 per stage)
    sanitized_config = []
    for st in stage_config:
        raw_cnt = st.get("question_count", 3)
        try:
            cnt = int(raw_cnt)
            if cnt <= 0:
                cnt = 3
            elif cnt > 10:
                cnt = 10
        except (ValueError, TypeError):
            cnt = 3
        st_clean = dict(st)
        st_clean["question_count"] = cnt
        sanitized_config.append(st_clean)

    stage_config = sanitized_config
    total_planned = sum(st.get("question_count", 3) for st in stage_config)

    # 4. Create Interview Session with Finite State Tracking
    session_id = str(uuid.uuid4())
    session_row = {
        "id": session_id,
        "user_id": current_user.id,
        "company": company_target,
        "role": role_target,
        "mode": "Demo" if is_demo else (req.mode or "Standard"),
        "status": "in_progress",
        "is_demo": is_demo,
        "stage_configuration": stage_config,
        "current_stage_index": 0,
        "questions_completed_in_stage": 0,
        "total_questions_completed": 0,
        "question_count": 0,
        "answered_count": 0,
        "correct_count": 0,
        "created_at": datetime.utcnow().isoformat(),
        "last_activity_at": datetime.utcnow().isoformat()
    }

    LOCAL_SESSIONS[session_id] = session_row

    if is_demo:
        LOCAL_DEMO_PROFILES[current_user.id] = {
            "demo_used": False,
            "demo_started_at": datetime.utcnow().isoformat(),
            "demo_session_id": session_id
        }

    if supabase:
        try:
            supabase.table("interview_sessions").insert(session_row).execute()
            if is_demo:
                supabase.table("profiles").update({
                    "demo_started_at": datetime.utcnow().isoformat(),
                    "demo_session_id": session_id
                }).eq("user_id", current_user.id).execute()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create interview session in database: {str(e)}"
            )

    # 5. Generate initial questions for Stage 1
    t_questions_start = time.monotonic()
    raw_questions = await ai_gateway.generate_interview_questions(
        user_id=current_user.id,
        company=company_target,
        role=role_target,
        mode="Demo" if is_demo else (req.mode or "Standard"),
        resume_text=resume_text,
        research_data=research_data,
        db=None
    )

    if not raw_questions:
        print(f"[Prepare Session Warning]: Question generation returned empty list. Retrying generation for session {session_id}...")
        raw_questions = await ai_gateway.generate_interview_questions(
            user_id=current_user.id,
            company=company_target,
            role=role_target,
            mode="Demo" if is_demo else (req.mode or "Standard"),
            resume_text=resume_text,
            research_data=research_data,
            db=None
        )

    print(f"[Interview Timing] question generation: {(time.monotonic() - t_questions_start)*1000:.0f} ms (got {len(raw_questions or [])} questions)")

    if not raw_questions:
        print(f"[Prepare Session Error]: Failed to generate questions after retry for {company_target} ({role_target}).")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to generate interview questions. Please retry."
        )

    question_rows = []
    for idx, q in enumerate(raw_questions):
        q_id = str(uuid.uuid4())
        row = {
            "id": q_id,
            "session_id": session_id,
            "user_id": current_user.id,
            "company": company_target,
            "role": role_target,
            "round_type": q.get("round_type", stage_config[0].get("type", "Technical")),
            "question_text": q.get("question_text", ""),
            "topic": q.get("topic", "General"),
            "difficulty": q.get("difficulty", "Medium"),
            "source_type": q.get("source_type", "AI Generated 🟣"),
            "source_url": q.get("source_url"),
            "code_template": q.get("code_template"),
            "coding_constraints": q.get("coding_constraints"),
            "options": q.get("options"),
            "correct_option_index": q.get("correct_option_index"),
            "stage_name": q.get("stage_name", stage_config[0].get("name", "Stage 1")),
            "stage_index": q.get("stage_index", 0),
            "stage_question_count": q.get("stage_question_count", stage_config[0].get("question_count", 3)),
            "questions_completed_in_stage": q.get("questions_completed_in_stage", idx),
            "order_index": idx + 1,
            "created_at": datetime.utcnow().isoformat()
        }
        question_rows.append(row)

    LOCAL_QUESTIONS[session_id] = question_rows
    session_row["question_count"] = len(question_rows)

    # [QUESTION DEBUG] Log metadata (NOT answers or sensitive data) for uniqueness verification
    for _qi, _q in enumerate(question_rows):
        print(f"[QUESTION DEBUG] index={_qi} question_id={_q.get('id','?')} question_text_preview={_q.get('question_text','')[:60]!r} stage={_q.get('stage_name','?')}")

    print(f"[Prepare Session Success]: Created session {session_id} for user {current_user.id} ({company_target} - {role_target}) with {len(question_rows)} questions across {len(stage_config)} stages.")

    if question_rows and supabase:
        try:
            supabase.table("interview_questions").insert(question_rows).execute()
            supabase.table("interview_sessions").update({"question_count": len(question_rows)}).eq("id", session_id).execute()
        except Exception as e:
            print(f"[Supabase Question Persist Warning]: {e}")

    return {
        "session_id": session_id,
        "company": company_target,
        "role": role_target,
        "mode": "Demo" if is_demo else (req.mode or "Standard"),
        "stage_configuration": stage_config,
        "current_stage_index": 0,
        "questions_completed_in_stage": 0,
        "total_questions_completed": 0,
        "total_planned_questions": total_planned,
        "total_questions": len(question_rows),
        "research": research_data
    }

@router.get("/session/{session_id}")
async def get_interview_session(
    session_id: str,
    current_user: UserProfileContext = Depends(get_current_user)
):
    session_obj = None
    questions = []
    answers = []

    if is_supabase_configured():
        try:
            supabase = get_supabase()
            s_res = supabase.table("interview_sessions").select("*").eq("id", session_id).eq("user_id", current_user.id).execute()
            if s_res and s_res.data:
                session_obj = s_res.data[0]

            q_res = supabase.table("interview_questions").select("*").eq("session_id", session_id).order("order_index", desc=False).execute()
            if q_res and q_res.data:
                questions = q_res.data

            ans_res = supabase.table("interview_answers").select("*").eq("session_id", session_id).eq("user_id", current_user.id).execute()
            if ans_res and ans_res.data:
                answers = ans_res.data
        except Exception as e:
            print(f"[Supabase Get Session Warning]: {e}")

    # Fallback to local persistence store if unconfigured or not found in Supabase
    if not session_obj:
        session_obj = LOCAL_SESSIONS.get(session_id)

    if not session_obj:
        raise HTTPException(status_code=404, detail="Interview session not found or access denied.")

    if not questions:
        questions = LOCAL_QUESTIONS.get(session_id, [])

    if not answers:
        answers = LOCAL_ANSWERS.get(session_id, [])

    answers_map = {a["question_id"]: a for a in answers}
    stage_config = session_obj.get("stage_configuration", [])

    current_st_idx = 0
    questions_count_in_st = 0

    questions_out = []
    for idx, q in enumerate(questions):
        if stage_config:
            while (current_st_idx < len(stage_config) - 1 and 
                   questions_count_in_st >= stage_config[current_st_idx].get("question_count", 3)):
                current_st_idx += 1
                questions_count_in_st = 0
            
            st = stage_config[current_st_idx]
            st_name = q.get("stage_name") or st.get("name", f"Stage {current_st_idx + 1}")
            st_idx = current_st_idx
            st_q_count = st.get("question_count", 3)
            q_comp_in_st = questions_count_in_st
            questions_count_in_st += 1
        else:
            st_name = q.get("stage_name") or q.get("round_type", "Technical")
            st_idx = 0
            st_q_count = len(questions)
            q_comp_in_st = idx

        q_id = q["id"]
        ans = answers_map.get(q_id)
        questions_out.append({
            "id": q_id,
            "session_id": q["session_id"],
            "round_type": q.get("round_type", "Technical"),
            "question_text": q["question_text"],
            "topic": q["topic"],
            "difficulty": q["difficulty"],
            "source_type": q["source_type"],
            "source_url": q.get("source_url"),
            "code_template": q.get("code_template"),
            "coding_constraints": q.get("coding_constraints"),
            "options": q.get("options"),
            "correct_option_index": q.get("correct_option_index"),
            "order_index": q.get("order_index", idx + 1),
            "current_stage_name": st_name,
            "stage_name": st_name,
            "stage_index": st_idx,
            "stage_question_count": st_q_count,
            "questions_completed_in_stage": q_comp_in_st,
            "answered": ans is not None,
            "user_answer": ans.get("answer_text") if ans else None,
            "score": ans.get("score") if ans else None,
            "evaluation": ans.get("evaluation") if ans else None
        })

    stage_config = session_obj.get("stage_configuration", [])
    curr_stage_idx = session_obj.get("current_stage_index", 0)
    curr_stage = stage_config[curr_stage_idx] if stage_config and curr_stage_idx < len(stage_config) else {}

    return {
        "session": session_obj,
        "questions": questions_out,
        "stage_configuration": stage_config,
        "current_stage": curr_stage,
        "current_stage_index": curr_stage_idx,
        "questions_completed_in_stage": session_obj.get("questions_completed_in_stage", len(answers)),
        "total_questions_completed": session_obj.get("total_questions_completed", len(answers)),
        "interview_completed": session_obj.get("status") == "completed" or (stage_config and curr_stage_idx >= len(stage_config))
    }

@router.post("/submit-answer", response_model=AnswerEvaluationOut, dependencies=[Depends(rate_limit_expensive_ai)])
async def submit_answer(
    req: SubmitAnswerRequest,
    current_user: UserProfileContext = Depends(get_current_user)
):
    supabase = get_supabase() if is_supabase_configured() else None

    question = None
    if supabase:
        try:
            q_res = supabase.table("interview_questions").select("*").eq("id", req.question_id).eq("user_id", current_user.id).execute()
            if q_res and q_res.data:
                question = q_res.data[0]
        except Exception as e:
            print(f"[Supabase Question Query Warning]: {e}")

    # Fallback to local store for question lookup
    if not question:
        for sess_qs in LOCAL_QUESTIONS.values():
            for q in sess_qs:
                if q.get("id") == req.question_id:
                    question = q
                    break

    if not question:
        raise HTTPException(status_code=404, detail="Interview question not found.")

    session_id = question.get("session_id", "default_session")

    # Acquire session lock for concurrency safety & idempotency
    async with get_session_lock(session_id):
        # 1. IDEMPOTENCY CHECK: If answer for this question already submitted, return existing evaluation
        existing_answer = None
        if supabase:
            try:
                ans_res = supabase.table("interview_answers").select("*").eq("question_id", req.question_id).eq("user_id", current_user.id).execute()
                if ans_res and ans_res.data:
                    existing_answer = ans_res.data[0]
            except Exception as e:
                print(f"[Supabase Existing Answer Check Warning]: {e}")

        if not existing_answer and session_id in LOCAL_ANSWERS:
            existing_answer = next((a for a in LOCAL_ANSWERS[session_id] if a.get("question_id") == req.question_id), None)

        if existing_answer:
            eval_dict = existing_answer.get("evaluation", {})
            return AnswerEvaluationOut(
                score=float(existing_answer.get("score", 0.0)),
                correctness=eval_dict.get("correctness", ""),
                relevance=eval_dict.get("relevance", ""),
                technical_depth=eval_dict.get("technical_depth", ""),
                feedback=eval_dict.get("feedback", ""),
                suggestions=eval_dict.get("suggestions", ""),
                follow_up_question=eval_dict.get("follow_up_question")
            )

        # 2. Instant Local & DB Save (No AI evaluation during interview flow)
        question_text = question["question_text"]
        round_type = question["round_type"]
        topic = question["topic"]
        options = question.get("options")

        user_text = (req.answer_text or "").strip()
        code_text = (req.code_submission or "").strip()
        sel_opt = req.selected_option_index

        is_answered = bool(user_text or code_text or sel_opt is not None)
        answer_status = "answered" if is_answered else "unanswered"

        ans_row = {
            "id": str(uuid.uuid4()),
            "question_id": req.question_id,
            "session_id": session_id,
            "user_id": current_user.id,
            "answer_text": user_text or (options[sel_opt] if options and sel_opt is not None and sel_opt < len(options) else code_text or "No answer provided"),
            "code_submission": req.code_submission,
            "selected_option": req.selected_option_index,
            "status": answer_status,
            "score": 0.0,
            "evaluation": {
                "correctness": "Saved",
                "relevance": "Submitted",
                "technical_depth": "Pending Batch Evaluation",
                "feedback": "Answer recorded successfully.",
                "suggestions": "Full AI feedback will be generated upon final submission."
            },
            "created_at": datetime.utcnow().isoformat()
        }

        if session_id not in LOCAL_ANSWERS:
            LOCAL_ANSWERS[session_id] = []
        
        # Replace existing answer if re-attempted or append
        existing_idx = next((i for i, a in enumerate(LOCAL_ANSWERS[session_id]) if a.get("question_id") == req.question_id), -1)
        if existing_idx >= 0:
            LOCAL_ANSWERS[session_id][existing_idx] = ans_row
        else:
            LOCAL_ANSWERS[session_id].append(ans_row)

        if supabase:
            try:
                supabase.table("interview_answers").upsert(ans_row).execute()
            except Exception as e:
                print(f"[Supabase Answer Save Warning]: {e}")

        # 3. Update Session Progress Counters
        session_obj = LOCAL_SESSIONS.get(session_id)
        if session_obj:
            session_obj["questions_completed_in_stage"] = session_obj.get("questions_completed_in_stage", 0) + 1
            session_obj["total_questions_completed"] = session_obj.get("total_questions_completed", 0) + 1
            if is_answered:
                session_obj["answered_count"] = session_obj.get("answered_count", 0) + 1
            session_obj["last_activity_at"] = datetime.utcnow().isoformat()

            if supabase:
                try:
                    supabase.table("interview_sessions").update({
                        "questions_completed_in_stage": session_obj["questions_completed_in_stage"],
                        "total_questions_completed": session_obj["total_questions_completed"],
                        "answered_count": session_obj["answered_count"],
                        "last_activity_at": session_obj["last_activity_at"]
                    }).eq("id", session_id).execute()
                except Exception as e:
                    print(f"[Supabase Session Update Warning]: {e}")

        return AnswerEvaluationOut(
            score=0.0,
            correctness="Saved",
            relevance="Submitted",
            technical_depth="Pending Batch Evaluation",
            feedback="Answer recorded. Continuing to next question...",
            suggestions="Full batch AI evaluation will run at the end of the interview.",
            follow_up_question=None
        )

@router.post("/next-question", dependencies=[Depends(rate_limit_expensive_ai)])
async def generate_next_adaptive_question(
    req: AdaptiveNextQuestionRequest,
    current_user: UserProfileContext = Depends(get_current_user)
):
    session_id = req.session_id

    async def _process_next_question():
        session = LOCAL_SESSIONS.get(session_id)
        questions = LOCAL_QUESTIONS.get(session_id, [])
        answers = LOCAL_ANSWERS.get(session_id, [])

        if is_supabase_configured():
            try:
                supabase = get_supabase()
                s_res = supabase.table("interview_sessions").select("*").eq("id", session_id).eq("user_id", current_user.id).execute()
                if s_res and s_res.data:
                    session = s_res.data[0]

                q_res = supabase.table("interview_questions").select("*").eq("session_id", session_id).order("order_index", desc=False).execute()
                if q_res and q_res.data:
                    questions = q_res.data

                ans_res = supabase.table("interview_answers").select("*").eq("session_id", session_id).order("created_at", desc=False).execute()
                if ans_res and ans_res.data:
                    answers = ans_res.data
            except Exception as e:
                print(f"[Supabase Adaptive Fetch Warning]: {e}")

        if not session:
            raise HTTPException(status_code=404, detail="Interview session not found.")

        # Check if session is already completed
        if session.get("status") == "completed":
            return {
                "id": None,
                "session_id": session_id,
                "interview_completed": True,
                "message": "Interview session has been completed."
            }

        company = session.get("company", "TCS")
        role = session.get("role", "Software Engineer")
        stage_config = session.get("stage_configuration", [])
        if not stage_config:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Company-specific interview process could not be verified."
            )

        current_stage_idx = session.get("current_stage_index", 0)
        completed_in_stage = session.get("questions_completed_in_stage", len(answers))
        total_completed = session.get("total_questions_completed", len(answers))

        # HARD ENFORCE: Total planned question limit strictly calculated from stage_configuration
        total_planned = sum(st.get("question_count", 0) for st in stage_config)

        if total_completed >= total_planned and total_planned > 0:
            session["status"] = "completed"
            session["completed_at"] = datetime.utcnow().isoformat()
            if is_supabase_configured():
                try:
                    get_supabase().table("interview_sessions").update({
                        "status": "completed",
                        "completed_at": session["completed_at"],
                        "last_activity_at": datetime.utcnow().isoformat()
                    }).eq("id", session_id).execute()
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to update session completion in database: {str(e)}"
                    )

            return {
                "id": None,
                "session_id": session_id,
                "interview_completed": True,
                "current_stage_index": current_stage_idx,
                "message": "Configured total interview question limit reached. Interview completed."
            }

        # Check stage completion transition
        if current_stage_idx < len(stage_config):
            required_count = stage_config[current_stage_idx].get("question_count", 3)
            if completed_in_stage >= required_count:
                # Stage is COMPLETE -> Advance stage
                current_stage_idx += 1
                completed_in_stage = 0
                session["current_stage_index"] = current_stage_idx
                session["questions_completed_in_stage"] = 0
                
                if is_supabase_configured():
                    try:
                        get_supabase().table("interview_sessions").update({
                            "current_stage_index": current_stage_idx,
                            "questions_completed_in_stage": 0,
                            "last_activity_at": datetime.utcnow().isoformat()
                        }).eq("id", session_id).execute()
                    except Exception as e:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to advance interview stage in database: {str(e)}"
                        )

        # Final Stopping Condition: If all stages complete -> Finish Interview
        if current_stage_idx >= len(stage_config):
            session["status"] = "completed"
            session["completed_at"] = datetime.utcnow().isoformat()
            if is_supabase_configured():
                try:
                    get_supabase().table("interview_sessions").update({
                        "status": "completed",
                        "completed_at": session["completed_at"],
                        "last_activity_at": datetime.utcnow().isoformat()
                    }).eq("id", session_id).execute()
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to mark interview completion in database: {str(e)}"
                    )

            return {
                "id": None,
                "session_id": session_id,
                "interview_completed": True,
                "current_stage_index": current_stage_idx,
                "message": "All interview stages are completed."
            }

        active_stage = stage_config[current_stage_idx]
        active_stage_name = active_stage.get("name", f"Stage {current_stage_idx + 1}")
        active_stage_type = active_stage.get("type", "Technical")
        required_stage_count = active_stage.get("question_count", 3)

        # IDEMPOTENCY CHECK FOR NEXT-QUESTION:
        # Return existing unanswered question instead of generating a duplicate!
        answered_q_ids = set(a["question_id"] for a in answers)
        unanswered_qs = [q for q in questions if q["id"] not in answered_q_ids]

        if unanswered_qs:
            existing_q = unanswered_qs[0]
            existing_q["current_stage_index"] = current_stage_idx
            existing_q["current_stage_name"] = active_stage_name
            existing_q["current_stage_type"] = active_stage_type
            existing_q["questions_completed_in_stage"] = completed_in_stage
            existing_q["stage_question_count"] = required_stage_count
            existing_q["total_questions_completed"] = total_completed
            existing_q["total_planned_questions"] = total_planned
            existing_q["interview_completed"] = False
            return existing_q

        # Extract weak topics from low-scoring past answers
        weak_topics = []
        q_map = {q["id"]: q.get("topic") for q in questions}
        for a in answers:
            if float(a.get("score", 0.0)) < 70.0:
                q_id = a.get("question_id")
                if q_id in q_map and q_map[q_id] not in weak_topics:
                    weak_topics.append(q_map[q_id])

        latest_eval = answers[-1].get("evaluation", {}) if answers else {}

        # Generate Stage-Aware Adaptive Next Question
        adaptive_q = await ai_gateway.generate_adaptive_next_question(
            session_id=session_id,
            user_id=current_user.id,
            company=company,
            role=role,
            current_evaluation=latest_eval,
            session_questions=questions,
            current_stage_name=active_stage_name,
            current_stage_type=active_stage_type,
            stage_question_count=required_stage_count,
            questions_completed_in_stage=completed_in_stage,
            weak_topics=weak_topics
        )

        q_id = str(uuid.uuid4())
        row = {
            "id": q_id,
            "session_id": session_id,
            "user_id": current_user.id,
            "company": company,
            "role": role,
            "round_type": active_stage_type,
            "question_text": adaptive_q.get("question_text", ""),
            "topic": adaptive_q.get("topic", f"{active_stage_type} Topic"),
            "difficulty": adaptive_q.get("difficulty", "Medium"),
            "source_type": adaptive_q.get("source_type", "Adaptive Follow-up 🟣"),
            "source_url": adaptive_q.get("source_url"),
            "code_template": adaptive_q.get("code_template"),
            "coding_constraints": adaptive_q.get("coding_constraints"),
            "options": adaptive_q.get("options"),
            "correct_option_index": adaptive_q.get("correct_option_index"),
            "order_index": len(questions) + 1,
            "created_at": datetime.utcnow().isoformat()
        }

        if session_id not in LOCAL_QUESTIONS:
            LOCAL_QUESTIONS[session_id] = []
        LOCAL_QUESTIONS[session_id].append(row)

        if is_supabase_configured():
            try:
                get_supabase().table("interview_questions").insert(row).execute()
                get_supabase().table("interview_sessions").update({"question_count": len(questions) + 1}).eq("id", session_id).execute()
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to persist adaptive question in database: {str(e)}"
                )

        # Attach stage progress contract metadata
        row["current_stage_index"] = current_stage_idx
        row["current_stage_name"] = active_stage_name
        row["current_stage_type"] = active_stage_type
        row["questions_completed_in_stage"] = completed_in_stage
        row["stage_question_count"] = required_stage_count
        row["total_questions_completed"] = total_completed
        row["total_planned_questions"] = total_planned
        row["interview_completed"] = False

        return row

    from app.core.rate_limiter import request_deduplicator
    dedup_key = f"next_q:{session_id}"
    return await request_deduplicator.execute_or_await(dedup_key, _process_next_question)


@router.post("/finish/{session_id}", response_model=FinalReportOut)
async def finish_interview(
    session_id: str,
    current_user: UserProfileContext = Depends(get_current_user)
):
    async with get_session_lock(session_id):
        session = LOCAL_SESSIONS.get(session_id)
        questions = LOCAL_QUESTIONS.get(session_id, [])
        answers = LOCAL_ANSWERS.get(session_id, [])

        if is_supabase_configured():
            try:
                supabase = get_supabase()
                s_res = supabase.table("interview_sessions").select("*").eq("id", session_id).eq("user_id", current_user.id).execute()
                if s_res and s_res.data:
                    session = s_res.data[0]

                q_res = supabase.table("interview_questions").select("*").eq("session_id", session_id).execute()
                if q_res and q_res.data:
                    questions = q_res.data

                ans_res = supabase.table("interview_answers").select("*").eq("session_id", session_id).eq("user_id", current_user.id).execute()
                if ans_res and ans_res.data:
                    answers = ans_res.data
            except Exception as e:
                print(f"[Supabase Finish Fetch Warning]: {e}")

        company = session.get("company", "TCS") if session else "TCS"
        role = session.get("role", "Software Engineer") if session else "Software Engineer"
        is_demo = session.get("is_demo", False) if session else True
        stage_config = session.get("stage_configuration", []) if session else []

        # Perform Batch AI Evaluation across all collected dynamic stage questions & answers
        from app.services.ai.evaluator import evaluator
        batch_res = await evaluator.batch_evaluate_session(
            company=company,
            role=role,
            stage_config=stage_config,
            questions=questions,
            answers=answers
        )

        overall_score = float(batch_res.get("overall_score", 0.0))
        round_scores = batch_res.get("round_scores", {})
        qa_list = batch_res.get("questions_with_answers", [])

        weak_topics = [qa.get("topic") for qa in qa_list if float(qa.get("score", 0.0)) < 70.0 and qa.get("topic")]

        update_data = {
            "overall_score": overall_score,
            "aptitude_score": round_scores.get("Aptitude", 0.0),
            "technical_score": round_scores.get("Technical", 0.0),
            "coding_score": round_scores.get("Coding", 0.0),
            "hr_score": round_scores.get("HR", 0.0),
            "weak_topics": list(set(weak_topics)) if weak_topics else [],
            "status": "completed",
            "answered_count": batch_res.get("answered_count", 0),
            "last_activity_at": datetime.utcnow().isoformat(),
            "completed_at": session.get("completed_at") or datetime.utcnow().isoformat()
        }

        if session:
            session.update(update_data)

        if is_demo:
            LOCAL_DEMO_PROFILES[current_user.id] = {
                "demo_used": True,
                "demo_completed_at": datetime.utcnow().isoformat()
            }

        if is_supabase_configured():
            try:
                supabase = get_supabase()
                supabase.table("interview_sessions").update(update_data).eq("id", session_id).execute()
                
                # Persist evaluated scores back to interview_answers
                for qa in qa_list:
                    q_id = qa.get("question_id")
                    if q_id:
                        supabase.table("interview_answers").update({
                            "score": qa.get("score", 0.0),
                            "evaluation": {
                                "label": qa.get("evaluation_label"),
                                "feedback": qa.get("feedback"),
                                "suggestions": qa.get("suggestions"),
                                "status": qa.get("status")
                            }
                        }).eq("question_id", q_id).eq("session_id", session_id).execute()

                if is_demo:
                    supabase.table("profiles").update({
                        "demo_used": True,
                        "demo_completed_at": datetime.utcnow().isoformat()
                    }).eq("user_id", current_user.id).execute()
            except Exception as e:
                print(f"[Supabase Record Completed Session Warning]: {e}")

        demo_skill_gap = list(set(weak_topics)) if weak_topics else ["System Architecture", "Edge Case Diagnostics", "Technical Explanations"]
        demo_roadmap = [
            f"1. Practice {demo_skill_gap[0]} fundamentals and production failure scenarios.",
            f"2. Solve targeted problem sets in {demo_skill_gap[1] if len(demo_skill_gap) > 1 else 'Distributed Systems'}.",
            f"3. Refine communication structure for {company} technical rounds."
        ]

        return FinalReportOut(
            session_id=session_id,
            company=company,
            role=role,
            overall_score=overall_score,
            round_scores=round_scores,
            strengths=batch_res.get("strengths", []),
            weaknesses=batch_res.get("weaknesses", []),
            struggled_questions=batch_res.get("struggled_questions", []),
            resume_vulnerabilities=batch_res.get("resume_vulnerabilities", []),
            readiness_level=batch_res.get("readiness_level", "Role Ready 🚀"),
            recommended_resources=batch_res.get("recommended_resources", []),
            is_demo=is_demo,
            questions_with_answers=qa_list,
            correct_answers_count=batch_res.get("correct_answers_count", 0),
            incorrect_answers_count=batch_res.get("incorrect_answers_count", 0),
            total_questions_count=batch_res.get("total_questions_count", len(questions)),
            answered_count=batch_res.get("answered_count", 0),
            unanswered_count=batch_res.get("unanswered_count", 0),
            strong_answers_count=batch_res.get("strong_answers_count", 0),
            acceptable_answers_count=batch_res.get("acceptable_answers_count", 0),
            weak_answers_count=batch_res.get("weak_answers_count", 0),
            stage_breakdown=batch_res.get("stage_breakdown", []),
            demo_roadmap=demo_roadmap,
            demo_skill_gap=demo_skill_gap
        )

@router.get("/history")
async def get_interview_history(
    current_user: UserProfileContext = Depends(get_current_user)
):
    if not is_supabase_configured():
        return []

    supabase = get_supabase()
    res = supabase.table("interview_sessions").select("*").eq("user_id", current_user.id).order("created_at", desc=True).execute()
    sessions = res.data if res and res.data else []

    out = []
    for s in sessions:
        # Exclude demo sessions from real interview history
        if s.get("is_demo") or s.get("mode") == "Demo":
            continue
        out.append({
            "session_id": s["id"],
            "company": s["company"],
            "role": s["role"],
            "mode": s.get("mode", "Standard"),
            "status": s["status"],
            "created_at": s["created_at"],
            "overall_score": s.get("overall_score"),
            "aptitude_score": s.get("aptitude_score"),
            "technical_score": s.get("technical_score"),
            "coding_score": s.get("coding_score"),
            "hr_score": s.get("hr_score")
        })
    return out
