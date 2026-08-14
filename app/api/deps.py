from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from typing import Optional
import jwt
from datetime import datetime
from app.core.security import resolve_user_name
from app.core.config import settings
from app.core.supabase import get_supabase, is_supabase_configured

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/google", auto_error=True)

class UserProfileContext(BaseModel):
    id: str
    email: EmailStr
    name: str
    avatar_url: Optional[str] = None
    role: Optional[str] = "user"

    class Config:
        from_attributes = True

async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> UserProfileContext:
    """
    FastAPI Supabase Authentication Dependency.
    Validates Supabase Bearer JWT / session token, derives authenticated user_id,
    and returns authenticated user profile context from Supabase profiles table.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_meta: Optional[dict] = None
    is_admin: bool = False

    # 1. Try Supabase Auth API verification if Supabase is configured
    if is_supabase_configured():
        try:
            supabase = get_supabase()
            res = supabase.auth.get_user(token)
            if res and res.user:
                user_id = str(res.user.id)
                user_email = res.user.email
                user_meta = res.user.user_metadata or {}
        except Exception:
            pass

    # 2. Fallback to local JWT decode if custom HireMate token issued (e.g. Admin Session)
    if not user_id:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            user_id = payload.get("sub")
            is_admin = bool(payload.get("is_admin"))
            if user_id and is_admin:
                user_email = payload.get("email") or "admin@hiremate.ai"
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to resolve authenticated user identity from token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Retrieve or upsert profile in Supabase profiles table
    if is_supabase_configured():
        try:
            supabase = get_supabase()
            response = supabase.table("profiles").select("*").eq("user_id", user_id).execute()
            profiles = response.data if response and response.data else []

            if profiles:
                p = profiles[0]
                resolved_role = p.get("role") or ("admin" if is_admin else "user")
                resolved_name = "Rayn" if resolved_role == "admin" else (p.get("name") or resolve_user_name(user_meta, p.get("email") or user_email))
                return UserProfileContext(
                    id=str(p.get("user_id")),
                    email=p.get("email") or user_email or "candidate@hiremate.ai",
                    name=resolved_name,
                    avatar_url=p.get("avatar_url"),
                    role=resolved_role
                )
            elif is_admin:
                admin_prof = {
                    "user_id": user_id,
                    "email": user_email or "admin@hiremate.ai",
                    "name": "Rayn",
                    "avatar_url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150",
                    "role": "admin",
                    "updated_at": datetime.utcnow().isoformat(),
                    "last_login": datetime.utcnow().isoformat()
                }
                try:
                    supabase.table("profiles").upsert(admin_prof, on_conflict="user_id").execute()
                except Exception as ex:
                    print(f"[Supabase Admin Profile Upsert Warning]: {ex}")
                return UserProfileContext(
                    id=user_id,
                    email=admin_prof["email"],
                    name=admin_prof["name"],
                    avatar_url=admin_prof["avatar_url"],
                    role="admin"
                )
            else:
                # If non-admin profile missing in public.profiles, perform automatic metadata priority upsert
                resolved_name = resolve_user_name(user_meta, user_email)
                now_iso = datetime.utcnow().isoformat()
                new_profile = {
                    "user_id": user_id,
                    "google_user_id": (user_meta or {}).get("sub") or (user_meta or {}).get("provider_id") or f"google_{user_id}",
                    "email": user_email or "candidate@hiremate.ai",
                    "name": resolved_name,
                    "avatar_url": (user_meta or {}).get("avatar_url") or (user_meta or {}).get("picture") or "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
                    "role": "user",
                    "updated_at": now_iso,
                    "last_login": now_iso
                }
                try:
                    supabase.table("profiles").upsert(new_profile, on_conflict="user_id").execute()
                except Exception as ex:
                    print(f"[Supabase Fallback User Profile Upsert Error]: {ex}")

                return UserProfileContext(
                    id=user_id,
                    email=new_profile["email"],
                    name=new_profile["name"],
                    avatar_url=new_profile["avatar_url"],
                    role="user"
                )
        except Exception as e:
            print(f"[Supabase Profile Query Warning]: {e}")

    return UserProfileContext(
        id=user_id,
        email=user_email or ("admin@hiremate.ai" if is_admin else "demo.candidate@hiremate.ai"),
        name="Rayn" if is_admin else resolve_user_name(user_meta, user_email),
        avatar_url="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
        role="admin" if is_admin else "user"
    )

async def get_current_admin(
    token: str = Depends(oauth2_scheme)
) -> UserProfileContext:
    """
    FastAPI Admin Authentication Dependency.
    Verifies that the provided token belongs to an authenticated administrator session.
    Denies access to non-admin users.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin access denied. Missing token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: Optional[str] = None
    user_email: Optional[str] = None

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        user_id = str(payload.get("sub", ""))
        is_admin = payload.get("is_admin", False)
        user_role = payload.get("role", "user")

        admin_env_id = (settings.SUPABASE_ADMIN_USER_ID or "").strip()

        if is_admin or user_role == "admin" or (admin_env_id and user_id == admin_env_id):
            return UserProfileContext(
                id=user_id,
                email=payload.get("email") or "dinesh@hiremate.ai",
                name="Rayn",
                avatar_url="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150",
                role="admin"
            )
    except jwt.PyJWTError:
        pass

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access denied. Invalid admin credentials or missing admin privileges."
    )
