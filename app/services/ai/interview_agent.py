import json
import re
from typing import Dict, Any, List, Optional
from app.services.ai.ai_router import ai_router
from app.core.supabase import get_supabase, is_supabase_configured

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words)

def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(normalize_text(text1).split())
    set2 = set(normalize_text(text2).split())
    if not set1 or not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

class InterviewAgent:
    async def generate_questions_for_session(
        self,
        user_id: str,
        company: str,
        role: str,
        mode: str,
        resume_text: str,
        research_data: Dict[str, Any],
        db: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Generates custom multi-round questions with strong deduplication & weak-topic progression using Supabase history.
        """
        prev_questions: List[str] = []
        weak_topics: List[str] = []

        if is_supabase_configured():
            try:
                supabase = get_supabase()
                res = supabase.table("interview_questions").select("id, question_text, topic").eq("user_id", user_id).eq("company", company).eq("role", role).execute()
                if res and res.data:
                    prev_questions = [r["question_text"] for r in res.data if r.get("question_text")]
                    q_map = {r["id"]: r["topic"] for r in res.data if "id" in r and "topic" in r}
                    
                    # Fetch answers with score < 70 for weak topics
                    ans_res = supabase.table("interview_answers").select("question_id, score").eq("user_id", user_id).lt("score", 70.0).execute()
                    if ans_res and ans_res.data:
                        for ans in ans_res.data:
                            q_id = ans.get("question_id")
                            if q_id in q_map and q_map[q_id] not in weak_topics:
                                weak_topics.append(q_map[q_id])
            except Exception as e:
                print(f"[Supabase Questions & Weak Topics Query Warning]: {e}")
        
        previously_asked_texts = prev_questions

        system_prompt = (
            "You are the Lead Technical Interviewer conducting a multi-round dynamic interview. "
            "Generate questions according to the research stage configuration provided. "
            "Your questions must be highly realistic, company-tailored, and directly aligned with the specified stage names and round types. "
            "CRITICAL: Do NOT duplicate questions previously asked to this candidate. "
            "If the candidate previously struggled on a topic, ask a fresh variation of that topic."
        )

        stage_config = research_data.get("stage_configuration", [])
        if not stage_config and research_data.get("interview_stages"):
            stages = research_data.get("interview_stages", [])
            stage_config = [{"name": st, "type": "Technical", "question_count": 3, "status": "likely/common"} for st in stages]

        prompt = f"""
Target Company: {company}
Target Role: {role}
Interview Mode: {mode} (Modes: Standard, New Challenge, Hard Mode, Weakness Training)

Stage Configuration (Generate questions for each stage according to specified types and counts):
{json.dumps(stage_config, indent=2)}

Company Research Context:
- Public Topics: {research_data.get("common_topics", [])}
- Reported Questions: {[q.get("question") for q in research_data.get("public_questions", []) if q.get("question")][:5]}

Resume Excerpt:
{resume_text[:1500] if resume_text else 'Standard software engineering candidate'}

PREVIOUSLY ASKED QUESTIONS TO STRICTLY EXCLUDE:
{json.dumps(previously_asked_texts[:30])}

CANDIDATE WEAK TOPICS TO TARGET WITH FRESH VARIATIONS:
{json.dumps(weak_topics[:5])}

Generate the exact questions corresponding to the stage configuration above.

Return a JSON array of objects with this exact schema:
[
  {{
    "round_type": "Aptitude",
    "question_text": "Quantitative or logical reasoning problem",
    "topic": "Quantitative Reasoning",
    "difficulty": "Medium",
    "source_type": "Publicly Reported 🔵",
    "source_url": null,
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_option_index": 1
  }},
  {{
    "round_type": "Technical",
    "question_text": "Technical architectural or scenario question",
    "topic": "DBMS",
    "difficulty": "Hard",
    "source_type": "AI Generated 🟣",
    "source_url": null,
    "options": null,
    "correct_option_index": null
  }}
]
"""
        # Sanitize stage configuration question counts to ensure at least 1 question per stage
        sanitized_stages = []
        for st in stage_config:
            st_clean = dict(st)
            try:
                cnt = int(st_clean.get("question_count", 3))
                st_clean["question_count"] = cnt if cnt > 0 else 3
            except (ValueError, TypeError):
                st_clean["question_count"] = 3
            sanitized_stages.append(st_clean)
        stage_config = sanitized_stages

        total_planned = sum(st.get("question_count", 3) for st in stage_config)

        # Execute bounded retries if AI model output is insufficient
        retries = 0
        max_retries = 2
        filtered_questions = []

        while retries <= max_retries:
            raw_questions = await ai_router.execute_json_task("local", prompt, system_prompt)
            if isinstance(raw_questions, list) and len(raw_questions) > 0:
                filtered_questions = []
                for q in raw_questions:
                    if not isinstance(q, dict):
                        continue
                    q_text = q.get("question_text", "")
                    if not q_text:
                        continue
                    is_dup = False

                    for prev in previously_asked_texts:
                        if q_text.lower().strip() == prev.lower().strip() or jaccard_similarity(q_text, prev) > 0.60:
                            is_dup = True
                            break

                    if not is_dup:
                        filtered_questions.append(q)

                if len(filtered_questions) >= total_planned:
                    break

            retries += 1

        # ── Fallback question pools — defined here so they are always accessible
        # for both the fill-slots path and the final deduplication pass.
        APTITUDE_FALLBACKS = [
            {
                "round_type": "Aptitude",
                "question_text": "A project team of 6 engineers completes a task in 12 days. If 2 engineers are reassigned after 3 days, how many total days will it take to finish?",
                "topic": "Quantitative Reasoning",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "options": ["15 days", "16.5 days", "18 days", "14 days"],
                "correct_option_index": 1
            },
            {
                "round_type": "Aptitude",
                "question_text": "If a train travels 360 km at 90 km/h and returns at 60 km/h, what is the average speed for the entire journey?",
                "topic": "Speed & Distance",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "options": ["72 km/h", "75 km/h", "70 km/h", "80 km/h"],
                "correct_option_index": 0
            },
            {
                "round_type": "Aptitude",
                "question_text": "What is the next number in the series: 2, 6, 12, 20, 30, ?",
                "topic": "Number Series",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵",
                "options": ["40", "42", "38", "44"],
                "correct_option_index": 1
            },
            {
                "round_type": "Aptitude",
                "question_text": "A shopkeeper marks a product 40% above cost price and gives a 20% discount. What is the profit percentage?",
                "topic": "Profit & Loss",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "options": ["10%", "12%", "15%", "20%"],
                "correct_option_index": 1
            },
            {
                "round_type": "Aptitude",
                "question_text": "How many ways can 4 people be seated in a row if 2 specific people must always sit together?",
                "topic": "Permutations & Combinations",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "options": ["12", "24", "48", "6"],
                "correct_option_index": 1
            },
            {
                "round_type": "Aptitude",
                "question_text": "A pipe fills a tank in 6 hours, another drains it in 9 hours. If both are open together, how long to fill an empty tank?",
                "topic": "Pipes & Cisterns",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "options": ["15 hours", "18 hours", "12 hours", "10 hours"],
                "correct_option_index": 1
            },
            {
                "round_type": "Aptitude",
                "question_text": "If log(x) = 2 and log(y) = 3, what is log(x²y)?",
                "topic": "Logarithms",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "options": ["6", "7", "8", "5"],
                "correct_option_index": 1
            },
            {
                "round_type": "Aptitude",
                "question_text": "The average of 5 consecutive odd numbers is 21. What is the largest number?",
                "topic": "Averages",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵",
                "options": ["23", "25", "27", "21"],
                "correct_option_index": 1
            },
        ]

        CODING_FALLBACKS = [
            {
                "round_type": "Coding",
                "question_text": "Write a function `two_sum(nums: List[int], target: int) -> List[int]` that returns indices of two numbers in `nums` such that they add up to `target`.",
                "topic": "Data Structures & Algorithms",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "code_template": "def two_sum(nums, target):\n    # Write your solution here\n    pass",
                "coding_constraints": "Time Complexity: O(N), Space Complexity: O(N)"
            },
            {
                "round_type": "Coding",
                "question_text": "Implement `is_palindrome(s: str) -> bool` that checks if a string reads the same forwards and backwards, ignoring case and spaces.",
                "topic": "String Manipulation",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵",
                "code_template": "def is_palindrome(s: str) -> bool:\n    # Write your solution here\n    pass",
                "coding_constraints": "Time Complexity: O(N)"
            },
            {
                "round_type": "Coding",
                "question_text": "Write `max_subarray(nums: List[int]) -> int` that finds the contiguous subarray with the largest sum (Kadane's Algorithm).",
                "topic": "Dynamic Programming",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "code_template": "def max_subarray(nums):\n    # Write your solution here\n    pass",
                "coding_constraints": "Time Complexity: O(N), Space: O(1)"
            },
            {
                "round_type": "Coding",
                "question_text": "Implement `level_order(root) -> List[List[int]]` that returns the level-order traversal of a binary tree.",
                "topic": "Trees & BFS",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵",
                "code_template": "from collections import deque\ndef level_order(root):\n    # Write your BFS solution here\n    pass",
                "coding_constraints": "Time Complexity: O(N)"
            },
            {
                "round_type": "Coding",
                "question_text": "Write `merge_sorted(l1: List[int], l2: List[int]) -> List[int]` that merges two sorted arrays into one sorted array.",
                "topic": "Sorting & Merging",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵",
                "code_template": "def merge_sorted(l1, l2):\n    # Write your solution here\n    pass",
                "coding_constraints": "Time Complexity: O(N+M)"
            },
            {
                "round_type": "Coding",
                "question_text": "Implement `valid_parentheses(s: str) -> bool` that checks if a string of brackets is properly matched and nested.",
                "topic": "Stack & Queue",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵",
                "code_template": "def valid_parentheses(s: str) -> bool:\n    # Write your solution here\n    pass",
                "coding_constraints": "Time Complexity: O(N)"
            },
            {
                "round_type": "Coding",
                "question_text": "Write `longest_common_prefix(strs: List[str]) -> str` that finds the longest common prefix among a list of strings.",
                "topic": "String Processing",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵",
                "code_template": "def longest_common_prefix(strs):\n    # Write your solution here\n    pass",
                "coding_constraints": "Time Complexity: O(N*M)"
            },
            {
                "round_type": "Coding",
                "question_text": "Implement `find_duplicates(nums: List[int]) -> List[int]` that returns all numbers that appear more than once in the list.",
                "topic": "Hashing",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵",
                "code_template": "def find_duplicates(nums):\n    # Write your solution here\n    pass",
                "coding_constraints": "Time Complexity: O(N)"
            },
        ]

        HR_FALLBACKS = [
            {
                "round_type": "HR",
                "question_text": f"Describe a situation where you faced a significant technical roadblock or deadline pressure at a previous role. How did you handle communication with your team?",
                "topic": "Behavioral & Culture Fit",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "HR",
                "question_text": f"Tell me about a time you disagreed with a technical decision made by your team lead or manager. What did you do and what was the outcome?",
                "topic": "Conflict Resolution",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "HR",
                "question_text": f"Why do you specifically want to work at {company} for the {role} position? What attracts you to this company's culture and mission?",
                "topic": "Motivation & Fit",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "HR",
                "question_text": "Give an example of a time you had to learn a new technology or skill rapidly under pressure. How did you approach it?",
                "topic": "Learning Agility",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "HR",
                "question_text": "Describe a situation where you took ownership of a project that was failing. What actions did you take and what was the result?",
                "topic": "Ownership & Accountability",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "HR",
                "question_text": "How do you prioritize tasks when you have multiple urgent deadlines competing for your attention? Walk me through your process.",
                "topic": "Time Management",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "HR",
                "question_text": "Tell me about a time you mentored or helped a colleague overcome a technical challenge. What approach did you take?",
                "topic": "Collaboration & Mentorship",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "HR",
                "question_text": "Where do you see your career in 3-5 years, and how does this role at our company fit into that vision?",
                "topic": "Career Goals",
                "difficulty": "Easy",
                "source_type": "Company Standard 🔵"
            },
        ]

        TECHNICAL_FALLBACKS = [
            {
                "round_type": "Technical",
                "question_text": f"Explain the key architectural differences between monolithic and microservice architectures for {role} applications. When would you choose one over the other?",
                "topic": "System Architecture",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "Technical",
                "question_text": "What is the difference between a process and a thread? When would you use multi-threading versus multi-processing?",
                "topic": "Operating Systems",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "Technical",
                "question_text": "Explain the SOLID principles with a practical example from your experience. Which one do you find most challenging to apply?",
                "topic": "Object-Oriented Design",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "Technical",
                "question_text": "What is database normalization? Explain the first three normal forms (1NF, 2NF, 3NF) with examples.",
                "topic": "Database Design",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "Technical",
                "question_text": "What are the differences between SQL and NoSQL databases? When would you choose one over the other in a production system?",
                "topic": "Database Systems",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "Technical",
                "question_text": "Explain the CAP theorem. Can a distributed system guarantee all three properties simultaneously? Give a real-world example.",
                "topic": "Distributed Systems",
                "difficulty": "Hard",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "Technical",
                "question_text": "What is RESTful API design? Explain the key constraints and best practices for designing a robust REST API.",
                "topic": "API Design",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
            {
                "round_type": "Technical",
                "question_text": "Explain the concept of caching. What are different caching strategies (LRU, LFU, TTL) and when would you use each?",
                "topic": "Performance & Caching",
                "difficulty": "Medium",
                "source_type": "Company Standard 🔵"
            },
        ]

        FALLBACK_POOLS = {
            "Aptitude": APTITUDE_FALLBACKS,
            "Coding": CODING_FALLBACKS,
            "HR": HR_FALLBACKS,
        }

        def get_fallback_question(round_type: str, idx: int) -> Dict[str, Any]:
            pool = FALLBACK_POOLS.get(round_type, TECHNICAL_FALLBACKS)
            return dict(pool[idx % len(pool)])

        # If filtered_questions is less than total_planned, fill remaining slots with fallback questions
        if len(filtered_questions) < total_planned:
            print(f"[Interview Agent Warning]: AI generated {len(filtered_questions)} of {total_planned} requested questions. Filling remaining slots with stage-specific questions.")

            # Map available questions or fill with fallback
            filled_questions = list(filtered_questions)
            while len(filled_questions) < total_planned:
                missing_idx = len(filled_questions)
                # Determine which stage type this index belongs to
                cur_count = 0
                target_type = "Technical"
                for st in stage_config:
                    st_cnt = st.get("question_count", 3)
                    if missing_idx < cur_count + st_cnt:
                        target_type = st.get("type", "Technical")
                        break
                    cur_count += st_cnt

                filled_questions.append(get_fallback_question(target_type, missing_idx))

            filtered_questions = filled_questions


        final_questions = filtered_questions[:total_planned]

        # Attach stage metadata to each question
        running_idx = 0
        assigned_questions = []
        for st_idx, st in enumerate(stage_config):
            st_name = st.get("name", f"Stage {st_idx + 1}")
            st_type = st.get("type", "Technical")
            st_count = st.get("question_count", 3)

            for stage_q_idx in range(st_count):
                if running_idx < len(final_questions):
                    q = dict(final_questions[running_idx])
                    q["round_type"] = st_type
                    q["stage_name"] = st_name
                    q["stage_index"] = st_idx
                    q["stage_question_count"] = st_count
                    q["questions_completed_in_stage"] = stage_q_idx
                    q["order_index"] = running_idx + 1
                    assigned_questions.append(q)
                    running_idx += 1

        # ── FINAL SAFETY DEDUPLICATION PASS ────────────────────────────────
        # Guarantee no two assigned questions share the same question_text.
        # Replaces any duplicate with a fallback from the matching round_type pool.
        seen_texts: set = set()
        dedup_fallback_counters: dict = {}
        for i, q in enumerate(assigned_questions):
            text = (q.get("question_text") or "").strip().lower()
            if not text or text in seen_texts:
                # Choose fallback pool by the question's round_type
                rt = q.get("round_type", "Technical")
                fb_pool = FALLBACK_POOLS.get(rt, TECHNICAL_FALLBACKS)
                # Advance a per-round counter to avoid picking the same fallback twice
                dedup_fallback_counters[rt] = dedup_fallback_counters.get(rt, 0) + 1
                fb_q = dict(fb_pool[(dedup_fallback_counters[rt] + i) % len(fb_pool)])
                # Preserve stage metadata already set
                fb_q["round_type"] = q["round_type"]
                fb_q["stage_name"] = q.get("stage_name", "")
                fb_q["stage_index"] = q.get("stage_index", 0)
                fb_q["stage_question_count"] = q.get("stage_question_count", 3)
                fb_q["questions_completed_in_stage"] = q.get("questions_completed_in_stage", i)
                fb_q["order_index"] = q.get("order_index", i + 1)
                fb_text = (fb_q.get("question_text") or "").strip().lower()
                # Keep trying fallback offsets until we find one not already seen
                attempts = 0
                while fb_text in seen_texts and attempts < len(fb_pool):
                    attempts += 1
                    dedup_fallback_counters[rt] += 1
                    fb_q = dict(fb_pool[(dedup_fallback_counters[rt] + i) % len(fb_pool)])
                    fb_q["round_type"] = q["round_type"]
                    fb_q["stage_name"] = q.get("stage_name", "")
                    fb_q["stage_index"] = q.get("stage_index", 0)
                    fb_q["stage_question_count"] = q.get("stage_question_count", 3)
                    fb_q["questions_completed_in_stage"] = q.get("questions_completed_in_stage", i)
                    fb_q["order_index"] = q.get("order_index", i + 1)
                    fb_text = (fb_q.get("question_text") or "").strip().lower()
                assigned_questions[i] = fb_q
                seen_texts.add(fb_text)
                print(f"[Interview Agent DEDUP] Replaced duplicate question at index {i} with fallback (round_type={rt})")
            else:
                seen_texts.add(text)
        # ────────────────────────────────────────────────────────────────────

        return assigned_questions


    async def generate_adaptive_question(
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
        """Generates the next adaptive question based on active stage & candidate performance."""
        score = current_evaluation.get("score", 70.0)
        feedback = current_evaluation.get("feedback", "")
        asked_texts = [q.get("question_text", "") for q in session_questions]
        stage_type_clean = current_stage_type if current_stage_type in ["Aptitude", "Technical", "Coding", "HR"] else "Technical"

        if score >= 80:
            adaptation_instruction = "The candidate scored >= 80 (Strong Performance). Increase difficulty level to Hard and ask a deeper, challenging question suitable for this stage."
            target_diff = "Hard"
        elif score < 70:
            adaptation_instruction = "The candidate scored < 70 (Weakness Detected). Target the specific missing concept with a fundamental question appropriate for this stage."
            target_diff = "Medium"
        else:
            adaptation_instruction = "The candidate scored 70-79 (Moderate Performance). Continue at Medium difficulty and probe concepts in this stage more deeply."
            target_diff = "Medium"

        system_prompt = (
            f"You are a Senior Technical & Behavioral Interviewer conducting Stage: '{current_stage_name}' (Round Type: {stage_type_clean}). "
            f"Generate 1 highly targeted next question that belongs STRICTLY to the round type '{stage_type_clean}'. "
            f"CRITICAL: The question round_type MUST be exactly '{stage_type_clean}'."
        )

        prompt = f"""
Target Company: {company}
Target Role: {role}
Current Active Stage: {current_stage_name}
Required Round Type: {stage_type_clean}
Stage Progress: Question {questions_completed_in_stage + 1} of {stage_question_count}
Latest Answer Score: {score}/100
Latest Evaluator Feedback: {feedback}
Adaptation Rule: {adaptation_instruction}
Weak Topics: {weak_topics or []}
Candidate Resume Excerpt: {resume_text[:600] if resume_text else 'Standard candidate profile'}

Previously Asked Questions in Session (STRICTLY EXCLUDE):
{json.dumps(asked_texts)}

Generate 1 adaptive question for this exact stage.
Return a JSON object with schema:
{{
  "round_type": "{stage_type_clean}",
  "question_text": "Stage-appropriate adaptive question text",
  "topic": "Target Topic",
  "difficulty": "{target_diff}",
  "source_type": "Adaptive Follow-up 🟣",
  "source_url": null,
  "options": {"['Option A', 'Option B', 'Option C', 'Option D']" if stage_type_clean == 'Aptitude' else "null"},
  "correct_option_index": {1 if stage_type_clean == 'Aptitude' else "null"}
}}
"""
        res = await ai_router.execute_json_task("local", prompt, system_prompt, reasoning_budget=16384)
        if isinstance(res, dict) and "question_text" in res and res.get("question_text"):
            res["round_type"] = stage_type_clean  # Enforce backend guarantee
            return res

        raise RuntimeError("AI Adaptive Question service unavailable. NVIDIA Nemotron service did not return a valid adaptive question. Please retry.")

interview_agent = InterviewAgent()

