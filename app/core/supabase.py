import threading
from typing import Optional
from supabase import create_client, Client
from app.core.config import settings

_supabase_client: Optional[Client] = None
_client_lock = threading.Lock()

class SupabaseNotConfiguredError(Exception):
    """Raised when Supabase credentials are not set in environment."""
    pass

def get_supabase() -> Client:
    """
    Returns a thread-safe singleton Supabase Client instance.
    Raises SupabaseNotConfiguredError if credentials are missing.
    """
    global _supabase_client
    if _supabase_client is None:
        with _client_lock:
            if _supabase_client is None:
                url = (settings.SUPABASE_URL or "").strip()
                key = (settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY or "").strip()
                if not url or not key or url.startswith("https://placeholder"):
                    raise SupabaseNotConfiguredError(
                        "Supabase not configured. Set SUPABASE_URL, SUPABASE_ANON_KEY, "
                        "and SUPABASE_SERVICE_ROLE_KEY in your backend .env file."
                    )
                _supabase_client = create_client(url, key)
    return _supabase_client

def is_supabase_configured() -> bool:
    """Check if Supabase credentials are present (without creating client)."""
    url = (settings.SUPABASE_URL or "").strip()
    key = (settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY or "").strip()
    return bool(url and key and not url.startswith("https://placeholder"))

def check_supabase_connection() -> bool:
    """
    Safely tests Supabase DB connection by issuing a lightweight query.
    Returns True if connected, False otherwise. Does not expose keys.
    """
    if not is_supabase_configured():
        return False
    try:
        client = get_supabase()
        res = client.table("profiles").select("user_id", count="exact").limit(1).execute()
        return res is not None
    except Exception as e:
        print(f"[Supabase Connection Health Warning]: {e}")
        return False

class SupabaseStorageService:
    @staticmethod
    async def upload_resume_file(user_id: str, resume_id: str, filename: str, file_bytes: bytes, content_type: str = "application/pdf") -> str:
        """
        Uploads a resume file to the Supabase Storage 'resumes' bucket.
        Returns the storage path.
        """
        supabase = get_supabase()
        storage_path = f"{user_id}/{resume_id}/{filename}"
        
        try:
            supabase.storage.from_("resumes").upload(
                file=file_bytes,
                path=storage_path,
                file_options={"content-type": content_type, "upsert": "true"}
            )
        except Exception as e:
            print(f"[Supabase Storage Upload Warning]: {e}")

        return storage_path

    @staticmethod
    async def download_resume_file(storage_path: str) -> bytes:
        """Downloads resume file from Supabase Storage 'resumes' bucket."""
        supabase = get_supabase()
        return supabase.storage.from_("resumes").download(storage_path)
