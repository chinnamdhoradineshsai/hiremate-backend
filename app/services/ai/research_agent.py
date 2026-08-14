import datetime
import json
import uuid
import time
from typing import Dict, Any, List
from app.services.ai.ai_router import ai_router
from app.services.research.search_provider import search_provider
from app.services.research.web_fetcher import web_fetcher
from app.services.research.evidence import classify_source_type, EvidenceRecord
from app.core.config import settings
from app.core.supabase import get_supabase, is_supabase_configured

# ── In-process research cache ──────────────────────────────────────────────
# Eliminates the duplicate research call that occurs when the frontend first
# calls /research (to show stage cards) and then immediately calls
# /interview/prepare (which re-runs research for the same company+role).
# TTL: 30 minutes.  Key: (company_lower, role_lower)
_RESEARCH_CACHE: Dict[str, Dict[str, Any]] = {}
_RESEARCH_CACHE_TTL_SECONDS = 1800  # 30 minutes

def _cache_key(company: str, role: str) -> str:
    return f"{company.strip().lower()}::{role.strip().lower()}"

def _get_cached_research(company: str, role: str) -> Dict[str, Any] | None:
    key = _cache_key(company, role)
    entry = _RESEARCH_CACHE.get(key)
    if entry and (time.monotonic() - entry["_cached_at"]) < _RESEARCH_CACHE_TTL_SECONDS:
        return entry["data"]
    return None

def _set_cached_research(company: str, role: str, data: Dict[str, Any]) -> None:
    key = _cache_key(company, role)
    _RESEARCH_CACHE[key] = {"data": data, "_cached_at": time.monotonic()}
# ───────────────────────────────────────────────────────────────────────────

class ResearchAgent:
    async def get_or_research_company(
        self,
        company: str,
        role: str,
        db: Any = None,
        job_description: str = "",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieves company research.
        Priority order (fastest → slowest):
          1. In-process memory cache (sub-millisecond, 30-min TTL)
          2. Supabase research cache (fast DB read)
          3. Full Tavily + NVIDIA research pipeline (slow, only on cache miss)
        """
        t_start = time.monotonic()
        company_clean = company.strip()
        role_clean = role.strip()
        supabase = get_supabase() if is_supabase_configured() else None

        # 1. In-process cache hit (eliminates duplicate /prepare calls)
        if not force_refresh:
            cached = _get_cached_research(company_clean, role_clean)
            if cached:
                elapsed = (time.monotonic() - t_start) * 1000
                print(f"[Interview Timing] research (in-process cache): {elapsed:.0f} ms")
                cached["is_verified"] = cached.get("is_verified", True)
                return cached

        # 2. Supabase cache
        if not force_refresh and supabase:
            try:
                res = supabase.table("company_research").select("*").eq("company", company_clean).eq("role", role_clean).limit(1).execute()
                db_cached = res.data[0] if res and res.data else None

                if db_cached:
                    res_data = {
                        "company": db_cached["company"],
                        "role": db_cached["role"],
                        "interview_stages": db_cached.get("interview_stages", []),
                        "stage_configuration": db_cached.get("stage_configuration", []),
                        "common_topics": db_cached.get("common_topics", []),
                        "public_questions": db_cached.get("public_questions", []),
                        "role_requirements": db_cached.get("role_requirements", []),
                        "sources": db_cached.get("sources", []),
                        "updated_at": db_cached.get("updated_at", datetime.datetime.utcnow().isoformat()),
                        "is_fresh": True,
                        "is_verified": True
                    }
                    # Populate in-process cache so subsequent calls are instant
                    _set_cached_research(company_clean, role_clean, res_data)
                    elapsed = (time.monotonic() - t_start) * 1000
                    print(f"[Interview Timing] research (Supabase cache): {elapsed:.0f} ms")
                    return res_data
            except Exception as e:
                print(f"[Supabase Research Cache Warning]: {e}")

        # 3. Full research pipeline (Tavily + NVIDIA)
        print(f"[Interview Timing] research: cache miss — running full research pipeline...")
        research_data = await self._perform_real_research(company_clean, role_clean, job_description)
        elapsed = (time.monotonic() - t_start) * 1000
        print(f"[Interview Timing] research (full pipeline): {elapsed:.0f} ms")

        # Save to Supabase
        if supabase:
            cache_row = {
                "id": str(uuid.uuid4()),
                "company": company_clean,
                "role": role_clean,
                "interview_stages": research_data.get("interview_stages", []),
                "stage_configuration": research_data.get("stage_configuration", []),
                "common_topics": research_data.get("common_topics", []),
                "public_questions": research_data.get("public_questions", []),
                "role_requirements": research_data.get("role_requirements", []),
                "sources": research_data.get("sources", []),
                "updated_at": datetime.datetime.utcnow().isoformat(),
                "is_fresh": True
            }
            try:
                supabase.table("company_research").upsert(cache_row, on_conflict="company,role").execute()
            except Exception as e:
                print(f"[Supabase Research Cache Save Warning]: {e}")

        research_data["is_fresh"] = False
        research_data["is_verified"] = bool(research_data.get("stage_configuration"))
        # Populate in-process cache
        _set_cached_research(company_clean, role_clean, research_data)
        return research_data

    async def _perform_real_research(self, company: str, role: str, jd: str) -> Dict[str, Any]:
        """
        Executes the multi-stage research architecture:
        1. Tavily (Web Research Layer) -> Search initial web evidence & candidate experiences
        2. Web Fetcher -> Extract body excerpts
        3. NVIDIA Nemotron (AI Reasoning Engine) -> Analyze evidence & extract insights
        4. Tavily (Deep Research Layer) -> Triggered when deeper context or extra questions are required
        5. Nemotron -> Final synthesis
        """
        q_interview = f"{company} {role} interview process questions experiences"
        q_careers = f"{company} careers {role} job requirements skills"

        # Step 1: Tavily Primary Web Research
        tavily_results_1 = await search_provider.search(q_interview, max_results=4, deep_research=False)
        tavily_results_2 = await search_provider.search(q_careers, max_results=3, deep_research=False)
        combined_search = tavily_results_1 + tavily_results_2

        evidence_records: List[EvidenceRecord] = []
        sources_list: List[Dict[str, Any]] = []
        seen_urls = set()

        for item in combined_search:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            stype = classify_source_type(url, company)
            domain = url.split("/")[2] if "//" in url else url

            page_text = ""
            if len(evidence_records) < 3:
                page_text = await web_fetcher.fetch_page_content(url, max_chars=1200)

            rec = EvidenceRecord(
                title=title,
                url=url,
                snippet=snippet,
                domain=domain,
                source_type=stype,
                content=page_text
            )
            evidence_records.append(rec)
            
            sources_list.append({
                "title": title,
                "url": url,
                "source_type": stype,
                "engine": "tavily"
            })

        system_prompt = (
            "You are an expert AI Career & Corporate Research Analyst powering HireMate. "
            "Analyze the provided web evidence records to extract authentic interview process stages, "
            "common interview topics, candidate reported questions, and role requirements for the target company. "
            "CRITICAL RULES:\n"
            "1. Do not assume a fixed number of interview stages. Determine the stage structure from the retrieved evidence for this company, role, location and experience level. Stage configurations can range from 2 to 5+ stages based on actual evidence.\n"
            "2. Determine stage order, stage names, stage types (Aptitude, Technical, Coding, HR), question counts, and stage completion statuses from the retrieved evidence.\n"
            "3. Every stage status MUST be strictly classified as either 'official' or 'likely/common'. Do not falsely claim an inferred stage is official.\n"
            "4. Do NOT fabricate URLs or domains. Use ONLY genuine URLs from evidence records or set source_url to null. Never create fake URLs like https://hiremate.ai/research."
        )

        evidence_prompt_text = "\n\n".join([
            f"Source [{i+1}] Title: {rec.title}\nURL: {rec.url}\nType: {rec.source_type}\nEngine: tavily\nSnippet: {rec.snippet}\nContent Excerpt: {rec.content[:600]}"
            for i, rec in enumerate(evidence_records)
        ])

        prompt = f"""
Target Company: {company}
Target Role: {role}
Job Description: {jd or 'Standard role expectations'}

Gathered Web Evidence Records (Tavily Web Research Layer):
{evidence_prompt_text if evidence_prompt_text else 'No verified public search evidence retrieved.'}

Generate a structured JSON response with this exact schema:
{{
  "interview_stages": [
    "Stage Name 1",
    "Stage Name 2"
  ],
  "stage_configuration": [
    {{
      "name": "Stage Name 1",
      "type": "Aptitude",
      "question_count": 5,
      "status": "official"
    }},
    {{
      "name": "Stage Name 2",
      "type": "Technical",
      "question_count": 4,
      "status": "likely/common"
    }}
  ],
  "common_topics": [
    "Topic 1", "Topic 2"
  ],
  "public_questions": [
    {{
      "round": "Technical",
      "question": "Candidate reported question extracted from evidence for {company}",
      "topic": "DBMS",
      "source_type": "Publicly Reported 🔵",
      "source_url": "{evidence_records[0].url if evidence_records else None}"
    }}
  ],
  "role_requirements": [
    "Skill requirement 1"
  ],
  "requires_deep_research": false
}}
"""
        # Step 2: Nemotron Initial Reasoning & Analysis
        res = await ai_router.execute_json_task("research", prompt, system_prompt)

        # Step 3: Tavily Deep Research Trigger (if initial public questions sparse or explicit deep context needed)
        needs_deep_research = False
        if isinstance(res, dict):
            public_qs = res.get("public_questions", [])
            needs_deep_research = res.get("requires_deep_research", False) or len(public_qs) == 0 or len(evidence_records) < 2

        if needs_deep_research:
            deep_query = f"{company} {role} technical interview coding interview questions past experiences glassdoor geeksforgeeks"
            deep_results = await search_provider.search(deep_query, max_results=5, deep_research=True)

            for item in deep_results:
                url = item.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                stype = classify_source_type(url, company)
                domain = url.split("/")[2] if "//" in url else url
                page_text = await web_fetcher.fetch_page_content(url, max_chars=1000)

                rec = EvidenceRecord(
                    title=item.get("title", ""),
                    url=url,
                    snippet=item.get("snippet", ""),
                    domain=domain,
                    source_type=stype,
                    content=page_text
                )
                evidence_records.append(rec)
                sources_list.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "source_type": stype,
                    "engine": "tavily"
                })

            # Nemotron Final Re-synthesis with Deep Research evidence
            deep_evidence_prompt_text = "\n\n".join([
                f"Source [{i+1}] Title: {rec.title}\nURL: {rec.url}\nType: {rec.source_type}\nSnippet: {rec.snippet}\nContent Excerpt: {rec.content[:600]}"
                for i, rec in enumerate(evidence_records)
            ])

            deep_prompt = f"""
Target Company: {company}
Target Role: {role}
Job Description: {jd or 'Standard role expectations'}

Enriched Multi-Source Web Evidence (Tavily Deep Research):
{deep_evidence_prompt_text if deep_evidence_prompt_text else 'No verified public search evidence retrieved.'}

Generate the final JSON response incorporating Tavily deep research findings:
{{
  "interview_stages": [
    "Stage Name 1",
    "Stage Name 2"
  ],
  "stage_configuration": [
    {{
      "name": "Stage Name 1",
      "type": "Aptitude",
      "question_count": 4,
      "status": "official"
    }},
    {{
      "name": "Stage Name 2",
      "type": "Technical",
      "question_count": 4,
      "status": "official"
    }}
  ],
  "common_topics": [
    "Topic 1", "Topic 2"
  ],
  "public_questions": [
    {{
      "round": "Technical",
      "question": "Candidate reported question extracted from evidence for {company}",
      "topic": "DBMS",
      "source_type": "Publicly Reported 🔵",
      "source_url": "{evidence_records[0].url if evidence_records else None}"
    }}
  ],
  "role_requirements": [
    "Requirement 1"
  ]
}}
"""
            res = await ai_router.execute_json_task("research", deep_prompt, system_prompt)

        has_evidence = len(evidence_records) > 0

        if not isinstance(res, dict) or "interview_stages" not in res or not has_evidence:
            return {
                "company": company,
                "role": role,
                "interview_stages": [],
                "stage_configuration": [],
                "common_topics": [],
                "public_questions": [],
                "role_requirements": [],
                "sources": [],
                "is_verified": False,
                "verification_message": "Company-specific interview process could not be verified."
            }

        # Ensure every stage status and question_count is strictly sanitized
        if "stage_configuration" in res and isinstance(res["stage_configuration"], list):
            for st in res["stage_configuration"]:
                if st.get("status") not in ["official", "likely/common"]:
                    st["status"] = "likely/common"
                try:
                    q_cnt = int(st.get("question_count", 3))
                    st["question_count"] = q_cnt if q_cnt > 0 else 3
                except (ValueError, TypeError):
                    st["question_count"] = 3

        # If stage_configuration missing, build it from interview_stages
        if "stage_configuration" not in res or not res["stage_configuration"]:
            stages = res.get("interview_stages", [])
            configs = []
            for i, st in enumerate(stages):
                st_type = "Technical"
                st_lower = st.lower()
                if "aptitude" in st_lower or "assessment" in st_lower or "screening" in st_lower or "online" in st_lower:
                    st_type = "Aptitude"
                elif "hr" in st_lower or "managerial" in st_lower or "culture" in st_lower or "behavioral" in st_lower:
                    st_type = "HR"
                elif "coding" in st_lower or "algorithm" in st_lower:
                    st_type = "Coding"
                
                configs.append({
                    "name": st,
                    "type": st_type,
                    "question_count": 3 if st_type in ["Aptitude", "Technical"] else 2,
                    "status": "likely/common"
                })
            res["stage_configuration"] = configs

        res["sources"] = sources_list if has_evidence else []
        res["is_verified"] = has_evidence
        return res


research_agent = ResearchAgent()
