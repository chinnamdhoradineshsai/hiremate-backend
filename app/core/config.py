import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "HireMate AI Career Platform"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Supabase Architecture Credentials
    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # NVIDIA Nemotron AI (Primary Reasoning Engine)
    NVIDIA_API_KEY: str = ""
    NVIDIA_API_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b"
    NVIDIA_REASONING_ENABLED: bool = True
    NVIDIA_REASONING_BUDGET: int = 16384
    AI_PROVIDER: str = "NVIDIA Nemotron 3 Ultra + Tavily"
    AI_ROUTER_MODE: str = "nemotron"

    # Tavily Research Provider
    TAVILY_API_KEY: str = ""
    
    # Admin Credentials for Demo Management
    ADMIN_USERNAME: str = ""
    ADMIN_PASSWORD: str = ""
    SUPABASE_ADMIN_USER_ID: str = ""

    # Auth & OAuth Security
    SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200  # 30 days
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # Web Research Provider Configuration
    RESEARCH_CACHE_EXPIRY_HOURS: int = 168  # 7 days
    
    # CORS
    FRONTEND_URL: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
