from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_current_admin, UserProfileContext
from app.core.config import settings
from app.core.supabase import get_supabase, is_supabase_configured

router = APIRouter(prefix="/admin", tags=["Admin Portal"])

@router.get("/overview")
async def get_admin_overview(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin Overview Statistics.
    Queries Supabase tables if connected, otherwise returns is_supabase_connected = False.
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "message": "Database not connected",
            "metrics": None
        }

    try:
        supabase = get_supabase()
        
        # Query total users & averages
        prof_res = supabase.table("profiles").select("user_id", count="exact").execute()
        total_users = prof_res.count if prof_res and prof_res.count is not None else 0

        int_res = supabase.table("interview_sessions").select("*").execute()
        sessions = int_res.data if int_res and int_res.data else []
        completed_sessions = [s for s in sessions if s.get("status") == "completed" and s.get("overall_score") is not None]
        avg_int_score = (sum(float(s["overall_score"]) for s in completed_sessions) / len(completed_sessions)) if completed_sessions else None

        ats_res = supabase.table("ats_analyses").select("*").execute()
        ats_records = ats_res.data if ats_res and ats_res.data else []
        avg_ats_score = (sum(float(a["overall_score"]) for a in ats_records) / len(ats_records)) if ats_records else None

        res_res = supabase.table("company_research").select("id", count="exact").execute()
        total_research = res_res.count if res_res and res_res.count is not None else (len(res_res.data) if res_res and res_res.data else 0)

        roadmap_res = supabase.table("learning_items").select("id", count="exact").execute()
        total_roadmaps = roadmap_res.count if roadmap_res and roadmap_res.count is not None else (len(roadmap_res.data) if roadmap_res and roadmap_res.data else 0)

        chat_res = supabase.table("career_chat_history").select("id", count="exact").execute()
        total_chatbot = chat_res.count if chat_res and chat_res.count is not None else (len(chat_res.data) if chat_res and chat_res.data else 0)

        return {
            "is_supabase_connected": True,
            "message": "Connected",
            "metrics": {
                "total_users": total_users,
                "total_interviews": len(sessions),
                "total_ats_analyses": len(ats_records),
                "total_research_sessions": total_research,
                "total_roadmaps": total_roadmaps,
                "total_chatbot_conversations": total_chatbot,
                "average_ats_score": round(avg_ats_score, 1) if avg_ats_score is not None else None,
                "average_interview_score": round(avg_int_score, 1) if avg_int_score is not None else None
            }
        }
    except Exception as e:
        print(f"[Admin Overview Error]: {e}")
        return {
            "is_supabase_connected": False,
            "message": f"Database query warning: {str(e)}",
            "metrics": None
        }

@router.get("/users")
async def get_admin_users(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin Users Monitoring.
    Returns list of registered users and non-sensitive activity metadata.
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "users": []
        }

    try:
        supabase = get_supabase()
        prof_res = supabase.table("profiles").select("user_id, email, name, created_at, last_login, demo_used").execute()
        profiles = prof_res.data if prof_res and prof_res.data else []

        out = []
        for p in profiles:
            u_id = p.get("user_id")
            
            # Count user sessions
            int_cnt = 0
            ats_cnt = 0
            road_cnt = 0
            
            try:
                i_res = supabase.table("interview_sessions").select("id", count="exact").eq("user_id", u_id).execute()
                int_cnt = i_res.count if i_res and i_res.count is not None else len(i_res.data or [])
                a_res = supabase.table("ats_analyses").select("id", count="exact").eq("user_id", u_id).execute()
                ats_cnt = a_res.count if a_res and a_res.count is not None else len(a_res.data or [])
                r_res = supabase.table("learning_items").select("id", count="exact").eq("user_id", u_id).execute()
                road_cnt = r_res.count if r_res and r_res.count is not None else len(r_res.data or [])
            except Exception:
                pass

            out.append({
                "user_id": u_id,
                "email": p.get("email"),
                "name": p.get("name"),
                "created_at": p.get("created_at"),
                "last_activity": p.get("last_login") or p.get("created_at"),
                "demo_used": p.get("demo_used", False),
                "interviews_count": int_cnt,
                "ats_count": ats_cnt,
                "roadmap_count": road_cnt
            })

        return {
            "is_supabase_connected": True,
            "users": out
        }
    except Exception as e:
        print(f"[Admin Users Query Error]: {e}")
        return {
            "is_supabase_connected": False,
            "users": []
        }

@router.get("/activity")
async def get_admin_user_activity(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin User Activity Stream.
    Returns recent candidate actions (Research, Interview, ATS, Roadmap).
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "activities": []
        }

    try:
        supabase = get_supabase()
        activities = []

        # User profile email mapping
        prof_res = supabase.table("profiles").select("user_id, email").execute()
        user_map = {p["user_id"]: p.get("email") for p in (prof_res.data or [])}

        # Recent User Research Activities
        res_act = supabase.table("research_activity").select("user_id, company, role, created_at").order("created_at", desc=True).limit(10).execute()
        if res_act and res_act.data:
            for r in res_act.data:
                u_id = r.get("user_id")
                activities.append({
                    "user_id": u_id,
                    "email": user_map.get(u_id, "user@candidate.com"),
                    "activity_type": "Company Research",
                    "details": f"Target: {r.get('company', 'N/A')} - {r.get('role', 'N/A')}",
                    "timestamp": r.get("created_at")
                })

        # Recent ATS analyses
        ats_res = supabase.table("ats_analyses").select("user_id, company, role, overall_score, created_at").order("created_at", desc=True).limit(10).execute()
        if ats_res and ats_res.data:
            for a in ats_res.data:
                u_id = a.get("user_id")
                activities.append({
                    "user_id": u_id,
                    "email": user_map.get(u_id, "user@candidate.com"),
                    "activity_type": "Completed ATS Analysis",
                    "details": f"Target: {a.get('company', 'N/A')} - {a.get('role', 'N/A')} (Score: {a.get('overall_score')}/100)",
                    "timestamp": a.get("created_at")
                })

        # Recent Interviews
        int_res = supabase.table("interview_sessions").select("user_id, company, role, mode, status, overall_score, created_at").order("created_at", desc=True).limit(10).execute()
        if int_res and int_res.data:
            for s in int_res.data:
                u_id = s.get("user_id")
                activities.append({
                    "user_id": u_id,
                    "email": user_map.get(u_id, "user@candidate.com"),
                    "activity_type": "Started / Completed Interview",
                    "details": f"Company: {s.get('company')} | Role: {s.get('role')} | Mode: {s.get('mode', 'Standard')} | Status: {s.get('status')}",
                    "timestamp": s.get("created_at")
                })

        # Sort activities by timestamp desc
        activities.sort(key=lambda x: x.get("timestamp") or "", reverse=True)

        return {
            "is_supabase_connected": True,
            "activities": activities[:25]
        }
    except Exception as e:
        print(f"[Admin Activity Query Error]: {e}")
        return {
            "is_supabase_connected": False,
            "activities": []
        }

@router.get("/companies")
async def get_admin_companies(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin Companies / Research Monitoring.
    Returns global company research cache and user-specific research activities.
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "researched_companies": [],
            "user_research_activities": []
        }

    try:
        supabase = get_supabase()
        res = supabase.table("company_research").select("company, role, is_fresh, updated_at").order("updated_at", desc=True).limit(20).execute()
        records = res.data if res and res.data else []

        prof_res = supabase.table("profiles").select("user_id, email").execute()
        user_map = {p["user_id"]: p.get("email") for p in (prof_res.data or [])}

        act_res = supabase.table("research_activity").select("*").order("created_at", desc=True).limit(20).execute()
        user_activities = []
        if act_res and act_res.data:
            for r in act_res.data:
                u_id = r.get("user_id")
                user_activities.append({
                    "id": r.get("id"),
                    "user_id": u_id,
                    "email": user_map.get(u_id, "user@candidate.com"),
                    "company": r.get("company"),
                    "role": r.get("role"),
                    "timestamp": r.get("created_at")
                })

        return {
            "is_supabase_connected": True,
            "researched_companies": records,
            "user_research_activities": user_activities
        }
    except Exception as e:
        print(f"[Admin Companies Query Error]: {e}")
        return {
            "is_supabase_connected": False,
            "researched_companies": [],
            "user_research_activities": []
        }

@router.get("/research-activity")
async def get_admin_research_activity(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin Candidate Research Activity Monitoring.
    Retrieves candidate research logs from public.research_activity table.
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "activities": []
        }

    try:
        supabase = get_supabase()
        prof_res = supabase.table("profiles").select("user_id, email").execute()
        user_map = {p["user_id"]: p.get("email") for p in (prof_res.data or [])}

        act_res = supabase.table("research_activity").select("*").order("created_at", desc=True).limit(50).execute()
        records = act_res.data if act_res and act_res.data else []

        out = []
        for r in records:
            u_id = r.get("user_id")
            out.append({
                "id": r.get("id"),
                "user_id": u_id,
                "email": user_map.get(u_id, "user@candidate.com"),
                "company": r.get("company"),
                "role": r.get("role"),
                "query": r.get("query"),
                "timestamp": r.get("created_at")
            })

        return {
            "is_supabase_connected": True,
            "activities": out
        }
    except Exception as e:
        print(f"[Admin Research Activity Error]: {e}")
        return {
            "is_supabase_connected": False,
            "activities": []
        }

@router.get("/interviews")
async def get_admin_interviews(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin Interview Monitoring.
    Returns interview sessions, score averages, and dynamically calculated weak skill clusters.
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "interviews": [],
            "average_score": None,
            "top_weak_skills": []
        }

    try:
        supabase = get_supabase()
        s_res = supabase.table("interview_sessions").select("*").order("created_at", desc=True).limit(50).execute()
        sessions = s_res.data if s_res and s_res.data else []

        completed = [s for s in sessions if s.get("status") == "completed" and s.get("overall_score") is not None]
        avg_score = sum(float(s["overall_score"]) for s in completed) / len(completed) if completed else None

        # Dynamically calculate weak topics across completed and active sessions
        weak_topic_counts: Dict[str, int] = {}
        for s in sessions:
            wt_list = s.get("weak_topics")
            if isinstance(wt_list, list):
                for wt in wt_list:
                    if wt and isinstance(wt, str):
                        weak_topic_counts[wt] = weak_topic_counts.get(wt, 0) + 1

        # Also query low-scoring answers in DB for additional weak topics
        try:
            low_ans_res = supabase.table("interview_answers").select("question_id, score").lt("score", 70.0).limit(100).execute()
            if low_ans_res and low_ans_res.data:
                q_ids = [a["question_id"] for a in low_ans_res.data if "question_id" in a]
                if q_ids:
                    q_res = supabase.table("interview_questions").select("id, topic").in_("id", q_ids).execute()
                    if q_res and q_res.data:
                        for q in q_res.data:
                            top = q.get("topic")
                            if top and isinstance(top, str):
                                weak_topic_counts[top] = weak_topic_counts.get(top, 0) + 1
        except Exception:
            pass

        top_weak_skills = sorted(weak_topic_counts.keys(), key=lambda k: weak_topic_counts[k], reverse=True)[:5] if weak_topic_counts else []

        return {
            "is_supabase_connected": True,
            "interviews": sessions,
            "average_score": round(avg_score, 1) if avg_score is not None else None,
            "top_weak_skills": top_weak_skills
        }
    except Exception as e:
        print(f"[Admin Interviews Query Error]: {e}")
        return {
            "is_supabase_connected": False,
            "interviews": [],
            "average_score": None,
            "top_weak_skills": []
        }


@router.get("/ats")
async def get_admin_ats(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin ATS Monitoring.
    Returns ATS analysis statistics and common missing skill trends.
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "ats_analyses": [],
            "stats": None
        }

    try:
        supabase = get_supabase()
        ats_res = supabase.table("ats_analyses").select("*").order("created_at", desc=True).limit(25).execute()
        records = ats_res.data if ats_res and ats_res.data else []

        scores = [float(r.get("overall_score", 0)) for r in records if r.get("overall_score") is not None]
        avg_score = (sum(scores) / len(scores)) if scores else None
        high_score = max(scores) if scores else None
        low_score = min(scores) if scores else None

        missing_counts: Dict[str, int] = {}
        for r in records:
            ms_list = r.get("missing_skills")
            if isinstance(ms_list, list):
                for ms in ms_list:
                    s_name = ms.get("skill_name") if isinstance(ms, dict) else (ms if isinstance(ms, str) else None)
                    if s_name:
                        missing_counts[s_name] = missing_counts.get(s_name, 0) + 1

        top_missing_skills = sorted(missing_counts.keys(), key=lambda k: missing_counts[k], reverse=True)[:5] if missing_counts else []

        return {
            "is_supabase_connected": True,
            "ats_analyses": records,
            "stats": {
                "total": len(records),
                "average_score": round(avg_score, 1) if avg_score is not None else None,
                "highest_score": high_score,
                "lowest_score": low_score,
                "common_missing_skills": top_missing_skills
            }
        }
    except Exception as e:
        print(f"[Admin ATS Query Error]: {e}")
        return {
            "is_supabase_connected": False,
            "ats_analyses": [],
            "stats": None
        }

@router.get("/roadmaps")
async def get_admin_roadmaps(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin Roadmap Monitoring.
    Returns targeted career roles and learning roadmap generation stats.
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "roadmaps": []
        }

    try:
        supabase = get_supabase()
        res = supabase.table("learning_items").select("*").order("created_at", desc=True).limit(25).execute()
        records = res.data if res and res.data else []

        return {
            "is_supabase_connected": True,
            "roadmaps": records
        }
    except Exception as e:
        print(f"[Admin Roadmaps Query Error]: {e}")
        return {
            "is_supabase_connected": False,
            "roadmaps": []
        }

@router.get("/chatbot")
async def get_admin_chatbot(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin Chatbot Monitoring (Privacy Preserved).
    Returns conversation & question counts without leaking raw private message contents.
    """
    if not is_supabase_configured():
        return {
            "is_supabase_connected": False,
            "total_conversations": 0,
            "popular_topics": []
        }

    try:
        supabase = get_supabase()
        chat_res = supabase.table("career_chat_history").select("id, role, created_at").order("created_at", desc=True).limit(50).execute()
        records = chat_res.data if chat_res and chat_res.data else []

        return {
            "is_supabase_connected": True,
            "total_conversations": len(records),
            "popular_topics": []
        }
    except Exception as e:
        print(f"[Admin Chatbot Query Error]: {e}")
        return {
            "is_supabase_connected": False,
            "total_conversations": 0,
            "popular_topics": []
        }


@router.get("/system-status")
async def get_admin_system_status(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin System / API Status Monitor.
    Checks configuration status for core integrations WITHOUT exposing actual API keys.
    """
    supabase_active = is_supabase_configured()
    nemotron_active = bool(settings.NVIDIA_API_KEY and not "placeholder" in settings.NVIDIA_API_KEY)
    tavily_active = bool(settings.TAVILY_API_KEY and not "placeholder" in settings.TAVILY_API_KEY)
    google_auth_active = bool(settings.GOOGLE_CLIENT_ID and not "placeholder" in settings.GOOGLE_CLIENT_ID)

    return {
        "services": [
            {
                "name": "FastAPI Backend",
                "status": "CONNECTED",
                "version": settings.VERSION,
                "is_active": True
            },
            {
                "name": "Supabase PostgreSQL",
                "status": "CONNECTED" if supabase_active else "NOT CONFIGURED",
                "is_active": supabase_active
            },
            {
                "name": "NVIDIA Nemotron AI",
                "status": "CONNECTED" if nemotron_active else "NOT CONFIGURED",
                "model": settings.NVIDIA_MODEL,
                "is_active": nemotron_active
            },
            {
                "name": "Tavily Research Provider",
                "status": "CONNECTED" if tavily_active else "NOT CONFIGURED",
                "is_active": tavily_active
            },
            {
                "name": "Google Authentication",
                "status": "CONNECTED" if google_auth_active else "NOT CONFIGURED",
                "is_active": google_auth_active
            }
        ]
    }

@router.post("/test-pipeline")
async def admin_test_research_pipeline(
    company: str = "Google",
    role: str = "Senior Software Engineer",
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Admin Real-Application Testing Mode Endpoint.
    Executes live search & reasoning using the EXACT same FastAPI service layer,
    NVIDIA Nemotron model, and Tavily credentials as regular users.
    """
    from app.services.ai.ai_gateway import ai_gateway
    from app.services.research.tavily_client import tavily_client

    # Test Tavily health
    tavily_test = await tavily_client.search(f"{company} career requirements", max_results=2)
    
    # Execute full pipeline
    result = await ai_gateway.research_company(
        company=company,
        role=role,
        force_refresh=True
    )

    return {
        "admin_testing_mode": True,
        "service_layer": "Shared FastAPI Core",
        "pipeline_status": "SUCCESS",
        "tavily_status": {
            "configured": bool(settings.TAVILY_API_KEY),
            "results_count": len(tavily_test),
            "sample": tavily_test[0] if tavily_test else None
        },
        "nemotron_status": {
            "model": settings.NVIDIA_MODEL,
            "configured": bool(settings.NVIDIA_API_KEY)
        },
        "pipeline_result": result
    }
