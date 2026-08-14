from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.supabase import get_supabase
from app.api import auth, resume, interview, research, chat, analytics, learning, questions, admin

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Supabase client connection check
    try:
        supabase = get_supabase()
        print(f"[Supabase Architecture Active]: Connected to {settings.SUPABASE_URL or 'Default Supabase'}")
    except Exception as e:
        print(f"[Supabase Warning]: {e}")
    yield

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    errors = exc.errors()
    missing_fields = []
    invalid_fields = []
    for err in errors:
        loc = err.get("loc", [])
        field_name = str(loc[-1]) if loc else "field"
        err_type = err.get("type", "")
        if "missing" in err_type or err.get("msg") == "Field required":
            missing_fields.append(field_name)
        else:
            invalid_fields.append(f"{field_name}: {err.get('msg')}")
    
    msg_parts = []
    if missing_fields:
        msg_parts.append(f"Missing required field(s): {', '.join(missing_fields)}")
    if invalid_fields:
        msg_parts.append(f"Invalid field(s): {', '.join(invalid_fields)}")
    
    detail_str = ". ".join(msg_parts) if msg_parts else str(exc)
    return JSONResponse(
        status_code=422,
        content={"detail": detail_str}
    )

# Enable CORS for Vite frontend
origins = [
    settings.FRONTEND_URL,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(admin.router, prefix=settings.API_V1_STR)
app.include_router(resume.router, prefix=settings.API_V1_STR)
app.include_router(interview.router, prefix=settings.API_V1_STR)
app.include_router(research.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(analytics.router, prefix=settings.API_V1_STR)
app.include_router(learning.router, prefix=settings.API_V1_STR)
app.include_router(questions.router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "status": "online",
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": "Supabase PostgreSQL",
        "ai_provider": settings.AI_PROVIDER,
        "ai_router_mode": settings.AI_ROUTER_MODE,
        "docs_url": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "Supabase"}

@app.get(f"{settings.API_V1_STR}/health/supabase")
@app.get("/health/supabase")
async def supabase_health():
    """
    Safe Supabase health check endpoint.
    Returns connected: true/false without exposing any keys or credentials.
    """
    from app.core.supabase import check_supabase_connection
    connected = check_supabase_connection()
    return {
        "connected": connected
    }

