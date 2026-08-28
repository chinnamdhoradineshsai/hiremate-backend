```python
import httpx
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Header, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional

from app.core.security import create_access_token, resolve_user_name
from app.core.config import settings
from app.core.supabase import get_supabase, is_supabase_configured
from app.schemas.schemas import Token, UserOut
from app.api.deps import get_current_user, get_current_admin, UserProfileContext


router = APIRouter(prefix="/auth", tags=["Authentication"])


class GoogleLoginRequest(BaseModel):
    id_token: Optional[str] = None
    access_token: Optional[str] = None
    demo_session_id: Optional[str] = None


class AdminLoginRequest(BaseModel):
    username: str
    password: str


class DemoLoginRequest(BaseModel):
    email: str = "demo.candidate@hiremate.ai"


class AssociateDemoRequest(BaseModel):
    demo_session_id: Optional[str] = None


@router.post("/admin-login", response_model=Token)
async def admin_login(req: AdminLoginRequest):
    """
    Backend-validated Admin Login endpoint.
    Checks credentials against environment variables ADMIN_USERNAME & ADMIN_PASSWORD.
    Validates SUPABASE_ADMIN_USER_ID against Supabase auth.users.
    Returns access token with admin privileges.
    """

    admin_user = (settings.ADMIN_USERNAME or "").strip()
    admin_pass = (settings.ADMIN_PASSWORD or "").strip()
    admin_id = (settings.SUPABASE_ADMIN_USER_ID or "").strip()

    if not admin_user or not admin_pass:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin credentials not configured in backend environment."
        )

    if req.username.strip() != admin_user or req.password.strip() != admin_pass:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access denied. Invalid admin credentials."
        )

    if not admin_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin configuration error: SUPABASE_ADMIN_USER_ID is not configured in backend environment."
        )

    try:
        uuid.UUID(admin_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Admin configuration error: SUPABASE_ADMIN_USER_ID '{admin_id}' is not a valid UUID string."
        )

    if is_supabase_configured():
        supabase = get_supabase()
        valid_user = False

        try:
            admin_user_res = supabase.auth.admin.get_user_by_id(admin_id)

            if admin_user_res and getattr(admin_user_res, "user", None):
                valid_user = True

        except Exception as ex:
            print(f"[Admin Auth Check Warning]: {ex}")

        if not valid_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Admin identity validation failed: SUPABASE_ADMIN_USER_ID '{admin_id}' does not exist in Supabase auth.users."
            )

        try:
            upsert_res = supabase.table("profiles").upsert({
                "user_id": admin_id,
                "email": f"{admin_user}@hiremate.ai",
                "name": "Rayn",
                "avatar_url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150",
                "role": "admin",
                "last_login": datetime.utcnow().isoformat()
            }).execute()

            if not upsert_res or (
                hasattr(upsert_res, "data")
                and upsert_res.data is None
            ):
                raise Exception(
                    "Database returned null response for profile upsert."
                )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Admin profile persistence failed: Database error during upsert - {str(e)}"
            )

    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin profile persistence failed: Database connection not configured."
        )

    token = create_access_token(
        admin_id,
        extra_claims={
            "is_admin": True,
            "role": "admin",
            "email": f"{admin_user}@hiremate.ai"
        }
    )

    return Token(
        access_token=token,
        user=UserOut(
            id=admin_id,
            email=f"{admin_user}@hiremate.ai",
            name="Rayn",
            avatar_url="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150",
            google_id=f"admin_{admin_id}"
        )
    )


@router.get("/admin/stats")
async def get_admin_dashboard_stats(
    current_admin: UserProfileContext = Depends(get_current_admin)
):
    """
    Protected Admin Dashboard Statistics Endpoint.
    Requires valid Admin Token.
    Queries Supabase database tables for real application statistics.
    Returns None for unavailable metrics so frontend displays "Not available yet".
    """

    if not is_supabase_configured():
        return {
            "total_users": None,
            "demo_users": None,
            "registered_users": None,
            "interviews_completed": None,
            "ats_analyses": None,
            "questions_used": None
        }

    supabase = get_supabase()

    stats = {
        "total_users": None,
        "demo_users": None,
        "registered_users": None,
        "interviews_completed": None,
        "ats_analyses": None,
        "questions_used": None
    }

    try:
        profiles_res = (
            supabase
            .table("profiles")
            .select("user_id, demo_used, email")
            .execute()
        )

        if profiles_res and profiles_res.data:
            data = profiles_res.data

            stats["total_users"] = len(data)

            stats["demo_users"] = sum(
                1 for p in data
                if p.get("demo_used")
            )

            stats["registered_users"] = sum(
                1
                for p in data
                if p.get("email")
                and "demo" not in p.get("email", "").lower()
            )

    except Exception as e:
        print(f"[Admin Stats Profiles Query Error]: {e}")

    try:
        sessions_res = (
            supabase
            .table("interview_sessions")
            .select("id")
            .eq("status", "completed")
            .execute()
        )

        if sessions_res and sessions_res.data is not None:
            stats["interviews_completed"] = len(
                sessions_res.data
            )

    except Exception as e:
        print(f"[Admin Stats Sessions Query Error]: {e}")

    try:
        ats_res = (
            supabase
            .table("ats_analyses")
            .select("id")
            .execute()
        )

        if ats_res and ats_res.data is not None:
            stats["ats_analyses"] = len(
                ats_res.data
            )

    except Exception as e:
        print(f"[Admin Stats ATS Query Error]: {e}")

    try:
        q_res = (
            supabase
            .table("interview_questions")
            .select("id")
            .execute()
        )

        if q_res and q_res.data is not None:
            stats["questions_used"] = len(
                q_res.data
            )

    except Exception as e:
        print(f"[Admin Stats Questions Query Error]: {e}")

    return stats


@router.post("/associate-demo")
async def associate_demo_session(
    req: AssociateDemoRequest,
    current_user: UserProfileContext = Depends(get_current_user)
):
    """
    Associates an anonymous Demo session with an authenticated Google user profile.

    Sets demo_used = True in database.

    The frontend may send a custom demo session ID such as:
    demo_session_1786780335901_oy2p5

    If the Supabase demo_session_id column is UUID,
    non-UUID values are ignored to prevent PostgreSQL error 22P02.
    """

    now_iso = datetime.utcnow().isoformat()

    if is_supabase_configured():
        try:
            supabase = get_supabase()

            update_data = {
                "demo_used": True,
                "demo_completed_at": now_iso
            }

            # Only save demo_session_id when it is a valid UUID.
            if req.demo_session_id:
                try:
                    uuid.UUID(req.demo_session_id)

                    update_data["demo_session_id"] = (
                        req.demo_session_id
                    )

                except ValueError:
                    print(
                        "[Associate Demo] Ignoring non-UUID "
                        f"demo_session_id: {req.demo_session_id}"
                    )

            (
                supabase
                .table("profiles")
                .update(update_data)
                .eq("user_id", current_user.id)
                .execute()
            )

        except Exception as e:
            print(
                f"[Associate Demo Supabase Error]: {e}"
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Failed to record demo completion state "
                    f"in database: {str(e)}"
                )
            )

    return {
        "status": "success",
        "demo_used": True
    }


@router.post("/google", response_model=Token)
async def google_login(
    req: GoogleLoginRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Google OAuth / Supabase Auth Authentication endpoint.

    Verifies Supabase session access token against Supabase Auth API,
    upserts profile in Supabase profiles table using auth.users.id
    as canonical user_id,
    and returns access token and profile metadata.
    """

    token_str = req.access_token or req.id_token

    if (
        not token_str
        and authorization
        and authorization.startswith("Bearer ")
    ):
        token_str = authorization.split(" ")[1]

    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supabase session access token is required."
        )

    user_id = None
    email = None
    name = None
    avatar_url = None
    google_user_id = None

    if is_supabase_configured():
        try:
            supabase = get_supabase()

            user_res = supabase.auth.get_user(
                token_str
            )

            if user_res and user_res.user:
                u = user_res.user

                user_id = str(u.id)
                email = str(u.email or "")

                meta = u.user_metadata or {}

                name = resolve_user_name(
                    meta,
                    email
                )

                avatar_url = (
                    meta.get("avatar_url")
                    or meta.get("picture")
                    or "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150"
                )

                google_user_id = (
                    meta.get("sub")
                    or meta.get("provider_id")
                    or f"google_{user_id}"
                )

        except Exception as e:
            print(
                f"[Supabase Auth Verification Error]: {e}"
            )

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Supabase authentication verification "
                    f"failed: {str(e)}"
                )
            )

    if not user_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Invalid or expired Supabase "
                "authentication session."
            )
        )

    now_iso = datetime.utcnow().isoformat()

    if is_supabase_configured():
        supabase = get_supabase()

        profile_data = {
            "user_id": user_id,
            "google_user_id": (
                google_user_id
                or f"google_{user_id}"
            ),
            "email": str(email),
            "name": name,
            "avatar_url": avatar_url,
            "role": "user",
            "updated_at": now_iso,
            "last_login": now_iso
        }

        # If Google login itself contains a demo session ID,
        # only save it when it is a valid UUID.
        if req.demo_session_id:
            try:
                uuid.UUID(req.demo_session_id)

                profile_data["demo_used"] = True
                profile_data["demo_completed_at"] = now_iso
                profile_data["demo_session_id"] = (
                    req.demo_session_id
                )

            except ValueError:
                print(
                    "[Google Login] Ignoring non-UUID "
                    f"demo_session_id: {req.demo_session_id}"
                )

                profile_data["demo_used"] = True
                profile_data["demo_completed_at"] = now_iso

        try:
            upsert_res = (
                supabase
                .table("profiles")
                .upsert(
                    profile_data,
                    on_conflict="user_id"
                )
                .execute()
            )

            if not upsert_res or (
                hasattr(upsert_res, "data")
                and upsert_res.data is None
            ):
                raise Exception(
                    "Database returned null data "
                    "for profile upsert."
                )

        except Exception as e:
            print(
                f"[Supabase Profile Sync Error]: {e}"
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Profile synchronization error: "
                    "Failed to save user profile to "
                    f"database - {str(e)}"
                )
            )

    token = (
        token_str
        if is_supabase_configured()
        else create_access_token(user_id)
    )

    return Token(
        access_token=token,
        user=UserOut(
            id=user_id,
            email=email,
            name=name,
            avatar_url=avatar_url,
            google_id=(
                google_user_id
                or f"google_{user_id}"
            )
        )
    )
```
