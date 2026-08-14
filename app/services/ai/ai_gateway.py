from typing import Dict, Any, List
from app.services.ai.ai_router import ai_router
from app.services.ai.research_agent import research_agent
from app.services.ai.resume_agent import resume_agent
from app.services.ai.interview_agent import interview_agent
from app.services.ai.evaluator import evaluator

class AIGateway:
    """
    Centralized Gateway for ALL AI capabilities across HireMate.
    Routes through NVIDIA Nemotron as the primary reasoning engine.
    """

    async def analyze_resume_ats(
        self, resume_text: str, target_company: str, target_role: str, job_description: str = ""
    ) -> Dict[str, Any]:
        return await resume_agent.analyze_ats(resume_text, target_company, target_role, job_description)

    async def research_company(
        self, company: str, role: str, db: Any = None, job_description: str = "", force_refresh: bool = False
    ) -> Dict[str, Any]:
        return await research_agent.get_or_research_company(company, role, db, job_description, force_refresh)

    async def generate_interview_questions(
        self,
        user_id: str,
        company: str,
        role: str,
        mode: str,
        resume_text: str,
        research_data: Dict[str, Any],
        db: Any = None
    ) -> List[Dict[str, Any]]:
        return await interview_agent.generate_questions_for_session(
            user_id, company, role, mode, resume_text, research_data, db
        )

    async def evaluate_answer(
        self,
        question_text: str,
        round_type: str,
        topic: str,
        user_answer: str,
        code_submission: str = None,
        selected_option_index: int = None,
        correct_option_index: int = None,
        options: List[str] = None
    ) -> Dict[str, Any]:
        return await evaluator.evaluate_answer(
            question_text, round_type, topic, user_answer,
            code_submission, selected_option_index, correct_option_index, options
        )

    async def generate_final_interview_report(
        self, company: str, role: str, questions_with_answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return await evaluator.generate_final_report(company, role, questions_with_answers)

    async def generate_adaptive_next_question(
        self,
        session_id: str,
        user_id: str,
        company: str,
        role: str,
        current_evaluation: Dict[str, Any],
        session_questions: List[Dict[str, Any]],
        current_stage_name: str = "Technical Round",
        current_stage_type: str = "Technical",
        stage_question_count: int = 5,
        questions_completed_in_stage: int = 1,
        weak_topics: List[str] = None,
        resume_text: str = ""
    ) -> Dict[str, Any]:
        """Generate the next adaptive question based on current stage & performance."""
        return await interview_agent.generate_adaptive_question(
            session_id=session_id,
            user_id=user_id,
            company=company,
            role=role,
            current_evaluation=current_evaluation,
            session_questions=session_questions,
            current_stage_name=current_stage_name,
            current_stage_type=current_stage_type,
            stage_question_count=stage_question_count,
            questions_completed_in_stage=questions_completed_in_stage,
            weak_topics=weak_topics,
            resume_text=resume_text
        )

    async def chat_career_assistant(
        self,
        message: str,
        history: List[Dict[str, str]],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        from app.services.research.search_provider import search_provider

        msg_lower = message.lower()
        target_company = user_context.get('target_company', 'N/A')
        
        # Determine if current/company-specific web research is needed
        needs_web_research = any(kw in msg_lower for kw in [
            "interview", "salary", "process", "round", "stage", "hiring", 
            "recent", "current", "news", "requirement", "glassdoor", "experience"
        ]) or (target_company != 'N/A' and target_company.lower() in msg_lower)

        web_context_str = ""
        sources = []
        if needs_web_research:
            search_query = f"{target_company if target_company != 'N/A' else ''} {message}".strip()
            try:
                research_results = await search_provider.search(search_query, max_results=3, deep_research=False)
                if research_results:
                    context_items = []
                    for idx, item in enumerate(research_results):
                        sources.append({"title": item.get("title"), "url": item.get("url")})
                        context_items.append(f"[{idx+1}] {item.get('title')}: {item.get('snippet')} (URL: {item.get('url')})")
                    web_context_str = "\nRelevant Live Web Context (Tavily Research):\n" + "\n".join(context_items)
            except Exception as e:
                print(f"[Chatbot Tavily Research Warning]: {e}")

        system_prompt = f"""
You are the HireMate AI Career Assistant — a personal career coach and interview preparation advisor.
User Profile Context:
- Name: {user_context.get('name', 'Candidate')}
- Recent ATS Score: {user_context.get('ats_score', 'N/A')}/100
- Recent Target Company: {target_company}
- Recent Target Role: {user_context.get('target_role', 'N/A')}
- Identified Skill Gaps: {user_context.get('missing_skills', [])}
- Past Interview Readiness Score: {user_context.get('interview_readiness', 'N/A')}%
{web_context_str}

Provide helpful, specific, non-generic career guidance grounded in the user's actual data and provided web evidence.
If referencing web evidence, cite the source URLs provided.
Never fabricate data, scores, company names, or make false claims.
"""
        history_formatted = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in history[-6:]])
        prompt = f"{history_formatted}\nuser: {message}\nassistant:"

        response_text = await ai_router.execute_task("local", prompt, system_prompt)
        if not response_text:
            response_text = (
                "I'm sorry, the AI service is currently unavailable. "
                "Please check that your NVIDIA API key is configured in the backend .env file and try again."
            )

        # Dynamic suggested actions based on actual user context
        suggested_actions = []
        if user_context.get('ats_score') == 'N/A':
            suggested_actions.append("Analyze my resume first")
        else:
            suggested_actions.append("How can I improve my ATS score?")
        
        if target_company != 'N/A':
            suggested_actions.append(f"How can I prepare for {target_company}?")
        else:
            suggested_actions.append("Help me choose a target company")
        
        if user_context.get('missing_skills'):
            suggested_actions.append("What skills should I learn first?")
        
        suggested_actions.append("Generate a learning roadmap for me")

        res_dict = {
            "reply": response_text,
            "suggested_actions": suggested_actions
        }
        if sources:
            res_dict["sources"] = sources
        return res_dict

ai_gateway = AIGateway()
