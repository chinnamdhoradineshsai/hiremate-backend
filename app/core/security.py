from datetime import datetime, timedelta, timezone
from typing import Optional, Any
import jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(subject: str | Any, expires_delta: Optional[timedelta] = None, extra_claims: Optional[dict] = None) -> str:
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {"exp": expire, "sub": str(subject)}
    if extra_claims:
        to_encode.update(extra_claims)
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    try:
        decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return decoded_token.get("sub")
    except jwt.PyJWTError:
        return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def resolve_user_name(meta: Optional[dict], email: Optional[str]) -> str:
    """
    Metadata Priority Name Resolution:
    1. user.user_metadata.full_name
    2. user.user_metadata.name
    3. email-derived name
    4. "Candidate" only as the final fallback
    """
    if meta and isinstance(meta, dict):
        full_name = meta.get("full_name")
        if isinstance(full_name, str) and full_name.strip():
            return full_name.strip()

        name = meta.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    if email and isinstance(email, str) and email.strip():
        local_part = email.split("@")[0].strip()
        if local_part:
            derived = local_part.replace(".", " ").replace("_", " ").strip().title()
            if derived:
                return derived

    return "Candidate"

