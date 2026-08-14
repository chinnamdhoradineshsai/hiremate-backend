from typing import List, Optional, Any, Dict
from pydantic import BaseModel, EmailStr

# User Schemas
class UserBase(BaseModel):
    email: str
    name: str
    avatar_url: Optional[str] = None

class UserCreate(UserBase):
    google_id: Optional[str] = None

class UserOut(UserBase):
    id: str
    google_id: Optional[str] = None
    created_at: Optional[Any] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

# Resume & ATS Schemas
class ResumeCreate(BaseModel):
    target_company: Optional[str] = "TCS"
    target_role: Optional[str] = "Software Engineer"
    job_description: Optional[str] = None

class MissingSkillItem(BaseModel):
    skill_name: str
    importance: str  # Critical, High, Medium, Low
    classification: str  # Required, Preferred
    why_it_matters: str
    where_it_appears: str
    how_to_improve: str

class WritingImprovementItem(BaseModel):
    section: str
    original: str
    improved: str
    reason: str

class FreeResourceItem(BaseModel):
    skill_name: str
    why_needed: str
    resource_title: str
    resource_url: str
    difficulty: str
    source_name: str

class ATSBreakdown(BaseModel):
    keyword_match: int
    required_skills: int
    role_relevance: int
    experience: int
    projects: int
    education: int
    formatting: int

class ATSResponse(BaseModel):
    id: str
    resume_id: str
    overall_score: int
    breakdown: ATSBreakdown
    missing_skills: List[MissingSkillItem]
    missing_keywords: List[str]
    writing_improvements: List[WritingImprovementItem]
    free_resources: List[FreeResourceItem]
    created_at: Any

# Interview Schemas
class InterviewPrepRequest(BaseModel):
    resume_id: Optional[str] = None
    company: str
    role: str
    job_description: Optional[str] = None
    mode: Optional[str] = "Standard"  # Standard, New Challenge, Hard Mode, Weakness Training, Surprise Interview

class InterviewSessionOut(BaseModel):
    id: str
    company: str
    role: str
    mode: str
    status: str
    current_round: int
    overall_score: Optional[float] = None
    aptitude_score: Optional[float] = None
    technical_score: Optional[float] = None
    coding_score: Optional[float] = None
    hr_score: Optional[float] = None
    created_at: Any

class QuestionOut(BaseModel):
    id: str
    session_id: str
    round_type: str
    question_text: str
    topic: str
    difficulty: str
    source_type: str  # Official 🟢, Publicly Reported 🔵, AI Generated 🟣
    source_url: Optional[str] = None
    code_template: Optional[str] = None
    coding_constraints: Optional[str] = None
    options: Optional[List[str]] = None
    order_index: int

class SubmitAnswerRequest(BaseModel):
    question_id: str
    answer_text: Optional[str] = None
    code_submission: Optional[str] = None
    selected_option_index: Optional[int] = None
    time_spent_seconds: Optional[int] = 0

class AnswerEvaluationOut(BaseModel):
    score: float
    correctness: str
    relevance: str
    technical_depth: str
    feedback: str
    suggestions: str
    follow_up_question: Optional[str] = None

class FinalReportOut(BaseModel):
    session_id: str
    company: str
    role: str
    overall_score: float
    round_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    struggled_questions: List[Dict[str, Any]]
    resume_vulnerabilities: List[str]
    readiness_level: str
    recommended_resources: List[FreeResourceItem]
    is_demo: Optional[bool] = False
    questions_with_answers: Optional[List[Dict[str, Any]]] = None
    correct_answers_count: Optional[int] = 0
    incorrect_answers_count: Optional[int] = 0
    total_questions_count: Optional[int] = 0
    answered_count: Optional[int] = 0
    unanswered_count: Optional[int] = 0
    strong_answers_count: Optional[int] = 0
    acceptable_answers_count: Optional[int] = 0
    weak_answers_count: Optional[int] = 0
    stage_breakdown: Optional[List[Dict[str, Any]]] = None
    demo_roadmap: Optional[List[str]] = None
    demo_skill_gap: Optional[List[str]] = None

# Research Schemas
class CompanyResearchOut(BaseModel):
    company: str
    role: str
    interview_stages: List[str]
    stage_configuration: Optional[List[Dict[str, Any]]] = None
    common_topics: List[str]
    public_questions: List[Dict[str, Any]]
    role_requirements: List[str]
    sources: List[Dict[str, Any]]
    updated_at: Any
    is_fresh: bool

# Chatbot Schemas
class ChatMessage(BaseModel):
    role: str  # user or assistant
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    current_company: Optional[str] = None
    current_role: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    suggested_actions: Optional[List[str]] = []
