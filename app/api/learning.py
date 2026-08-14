from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_current_user, UserProfileContext
from app.core.supabase import get_supabase, is_supabase_configured

router = APIRouter(prefix="/learning", tags=["Learning Roadmap"])

@router.get("")
async def get_learning_roadmap(
    current_user: UserProfileContext = Depends(get_current_user)
):
    if not is_supabase_configured():
        return {"roadmap": []}

    supabase = get_supabase()
    res = supabase.table("learning_items").select("*").eq("user_id", current_user.id).order("roadmap_week", desc=False).execute()
    items = res.data if res and res.data else []

    roadmap_list = []
    for item in items:
        roadmap_list.append({
            "id": item["id"],
            "skill_name": item["skill_name"],
            "category": item["category"],
            "priority": item.get("priority", "High"),
            "resource_title": item["resource_title"],
            "resource_url": item["resource_url"],
            "difficulty": item.get("difficulty", "Intermediate"),
            "source_name": item.get("source_name", "Verified Resource"),
            "status": item.get("status", "Pending"),
            "roadmap_week": item.get("roadmap_week", 1),
            "target_company": item.get("target_company"),
            "target_role": item.get("target_role"),
            "roadmap_type": item.get("roadmap_type", "personalized"),
            "progress": item.get("progress", 100 if item.get("status") == "Completed" else (50 if item.get("status") == "In Progress" else 0))
        })

    return {"roadmap": roadmap_list}

@router.post("/toggle/{item_id}")
async def toggle_learning_status(
    item_id: str,
    current_user: UserProfileContext = Depends(get_current_user)
):
    if not is_supabase_configured():
        return {"id": item_id, "status": "Pending", "progress": 0}

    supabase = get_supabase()
    res = supabase.table("learning_items").select("*").eq("id", item_id).eq("user_id", current_user.id).execute()
    if not res or not res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Learning item not found or access denied.")

    item = res.data[0]
    current_status = item.get("status", "Pending")

    if current_status == "Pending":
        new_status = "In Progress"
        new_progress = 50
    elif current_status == "In Progress":
        new_status = "Completed"
        new_progress = 100
    else:
        new_status = "Pending"
        new_progress = 0

    try:
        supabase.table("learning_items").update({"status": new_status, "progress": new_progress}).eq("id", item_id).execute()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update learning item status in database: {str(e)}"
        )

    return {"id": item_id, "status": new_status, "progress": new_progress}
