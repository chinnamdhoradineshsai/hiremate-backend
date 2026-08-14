from fastapi import APIRouter, Depends, Query, HTTPException, status
from app.api.deps import get_current_user, UserProfileContext
from app.services.ai.ai_gateway import ai_gateway
from app.schemas.schemas import CompanyResearchOut
from app.core.rate_limiter import rate_limit_expensive_ai
from app.core.supabase import get_supabase, is_supabase_configured

router = APIRouter(prefix="/research", tags=["Company Research"])

@router.get("", response_model=CompanyResearchOut, dependencies=[Depends(rate_limit_expensive_ai)])
@router.post("", response_model=CompanyResearchOut, dependencies=[Depends(rate_limit_expensive_ai)])
async def get_company_research(
    company: str = Query(...),
    role: str = Query(...),
    force_refresh: bool = Query(False),
    current_user: UserProfileContext = Depends(get_current_user)
):
    research = await ai_gateway.research_company(
        company=company,
        role=role,
        db=None,
        force_refresh=force_refresh
    )

    # Record individual candidate research activity for Admin activity tracking
    if is_supabase_configured():
        try:
            supabase = get_supabase()
            supabase.table("research_activity").insert({
                "user_id": current_user.id,
                "company": company.strip(),
                "role": role.strip(),
                "query": f"{company.strip()} - {role.strip()}"
            }).execute()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to record research activity in database: {str(e)}"
            )

    return CompanyResearchOut(
        company=company,
        role=role,
        interview_stages=research.get("interview_stages", []),
        stage_configuration=research.get("stage_configuration", []),
        common_topics=research.get("common_topics", []),
        public_questions=research.get("public_questions", []),
        role_requirements=research.get("role_requirements", []),
        sources=research.get("sources", []),
        updated_at=research.get("updated_at", "2026-08-10T00:00:00Z"),
        is_fresh=research.get("is_fresh", True)
    )
