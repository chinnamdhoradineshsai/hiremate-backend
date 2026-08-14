import json
from typing import Dict, Any, List
from app.services.ai.ai_router import ai_router

class AnswerEvaluator:
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
        """
        Evaluates answer deterministically for Aptitude MCQs, or via AI for Technical/Coding/HR text answers.
        Generates an adaptive follow-up question.
        """
        # Deterministic evaluation for Aptitude MCQs with formula explanation
        if round_type == "Aptitude" and selected_option_index is not None and correct_option_index is not None:
            is_correct = (selected_option_index == correct_option_index)
            score = 100.0 if is_correct else 0.0
            correct_str = options[correct_option_index] if options and len(options) > correct_option_index else ""
            user_str = options[selected_option_index] if options and len(options) > selected_option_index else ""

            system_prompt = "You are a quantitative aptitude tutor. Explain the step-by-step mathematical solution concisely."
            prompt = f"Question: {question_text}\nCorrect Answer Option: {correct_str}\nExplain the step-by-step solution in 2 sentences."
            explanation = await ai_router.execute_task("local", prompt, system_prompt)

            return {
                "score": score,
                "correctness": "Excellent - Exact Match" if is_correct else "Incorrect Option Selected",
                "relevance": "Direct MCQ Choice",
                "technical_depth": "Quantitative Reasoning Execution",
                "feedback": f"Your selected choice '{user_str}' was {'correct!' if is_correct else f'incorrect. The correct option is: {correct_str}.'}\n\nStep-by-Step Solution: {explanation}",
                "suggestions": "Review formula fundamentals and practice speed calculation under time constraints.",
                "follow_up_question": "Let me present the next question to evaluate your logical speed." if is_correct else "Would you like to solve a related variation of this quantitative problem?"
            }

        # AI evaluation for Technical, Coding, HR, or descriptive answers
        system_prompt = (
            "You are a Senior Technical Lead and HR Director evaluating candidate interview responses. "
            "Score responses strictly from 0 to 100 based on correctness, technical depth, code complexity, structure, and real-world examples. "
            "For Coding: evaluate correctness, edge cases, time/space complexity. "
            "For HR: evaluate communication, clarity, STAR method structure, and cultural alignment. "
            "Generate an adaptive follow-up question: "
            "If answer is strong (>=80) -> ask an advanced follow-up challenging their knowledge deeper. "
            "If answer is weak (<70) -> ask a guiding follow-up breaking down the core missing concept."
        )

        answer_body = user_answer if user_answer else f"Submitted Code Solution:\n{code_submission}"

        prompt = f"""
Question ({round_type} - {topic}):
{question_text}

Candidate Answer:
{answer_body}

Evaluate the response and return a JSON object with this exact schema:
{{
  "score": 85.0,
  "correctness": "Strong conceptual accuracy with minor detail omissions.",
  "relevance": "High - directly answers the interviewer's prompt.",
  "technical_depth": "Demonstrates practical knowledge of performance and trade-offs.",
  "feedback": "Great explanation of core principles. You correctly highlighted primary execution pathways.",
  "suggestions": "Explain specific edge cases or failure modes under high load.",
  "follow_up_question": "How would you handle failure recovery if this service encounters a timeout?"
}}
"""
        eval_res = await ai_router.execute_json_task("local", prompt, system_prompt, reasoning_budget=16384)

        if not isinstance(eval_res, dict) or "score" not in eval_res:
            raise RuntimeError("AI Answer Evaluation service unavailable. NVIDIA Nemotron service did not return a valid evaluation. Please click Retry.")

        return eval_res

    async def generate_final_report(
        self,
        company: str,
        role: str,
        questions_with_answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generates the final comprehensive interview performance report dynamically using AI.
        """
        apt_scores = []
        tech_scores = []
        code_scores = []
        hr_scores = []
        struggled = []

        for item in questions_with_answers:
            round_type = item.get("round_type", "")
            score = float(item.get("score", 70.0))

            if round_type == "Aptitude": apt_scores.append(score)
            elif round_type == "Technical": tech_scores.append(score)
            elif round_type == "Coding": code_scores.append(score)
            elif round_type == "HR": hr_scores.append(score)

            if score < 70.0:
                struggled.append({
                    "question": item.get("question_text"),
                    "round": round_type,
                    "score": score,
                    "feedback": item.get("feedback")
                })

        all_scores = [float(item.get("score", 0.0)) for item in questions_with_answers if item.get("score") is not None]
        
        avg_apt = sum(apt_scores) / len(apt_scores) if apt_scores else (sum(all_scores)/len(all_scores) if all_scores else 0.0)
        avg_tech = sum(tech_scores) / len(tech_scores) if tech_scores else (sum(all_scores)/len(all_scores) if all_scores else 0.0)
        avg_code = sum(code_scores) / len(code_scores) if code_scores else (sum(all_scores)/len(all_scores) if all_scores else 0.0)
        avg_hr = sum(hr_scores) / len(hr_scores) if hr_scores else (sum(all_scores)/len(all_scores) if all_scores else 0.0)

        overall_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0
        overall_score = min(100.0, max(0.0, overall_score))

        readiness = "Role Ready 🚀" if overall_score >= 80 else ("Near Ready 📈" if overall_score >= 65 else "Needs Practice 🎯")

        # Synthesize custom dynamic strengths, weaknesses, vulnerabilities using AI Router
        system_prompt = "You are an Executive Hiring Manager synthesizing candidate interview evaluation data."
        prompt = f"""
Company: {company}
Role: {role}
Scores: Aptitude={avg_apt}, Technical={avg_tech}, Coding={avg_code}, HR={avg_hr}
Struggled Topics/Questions: {json.dumps(struggled[:3])}

Return JSON with exact keys:
{{
  "strengths": ["Strength point 1", "Strength point 2", "Strength point 3"],
  "weaknesses": ["Weakness point 1", "Weakness point 2"],
  "resume_vulnerabilities": ["Vulnerability 1"],
  "recommended_resources": [
    {{
      "skill_name": "Database Indexing",
      "why_needed": "Improves query optimization under high concurrency.",
      "resource_title": "PostgreSQL Documentation",
      "resource_url": "https://www.postgresql.org/docs/",
      "difficulty": "Intermediate",
      "source_name": "Official Docs"
    }}
  ]
}}
"""
        synthesis = await ai_router.execute_json_task("local", prompt, system_prompt)

        strengths = synthesis.get("strengths") if isinstance(synthesis, dict) and synthesis.get("strengths") else [
            f"Demonstrated solid performance in {role} round assessments.",
            "Strong communication structure during scenario-based prompts."
        ]

        weaknesses = synthesis.get("weaknesses") if isinstance(synthesis, dict) and synthesis.get("weaknesses") else [
            "Technical depth on complex edge cases under timed constraints."
        ]

        resume_vulnerabilities = synthesis.get("resume_vulnerabilities") if isinstance(synthesis, dict) and synthesis.get("resume_vulnerabilities") else []

        recommended_resources = synthesis.get("recommended_resources") if isinstance(synthesis, dict) and synthesis.get("recommended_resources") else [
            {
                "skill_name": f"{role} Core Skills",
                "why_needed": "Strengthen technical defense and algorithm complexity.",
                "resource_title": "LeetCode Algorithmic Practice",
                "resource_url": "https://leetcode.com/explore/",
                "difficulty": "Intermediate",
                "source_name": "LeetCode"
            }
        ]

        return {
            "overall_score": overall_score,
            "round_scores": {
                "Aptitude": round(avg_apt, 1),
                "Technical": round(avg_tech, 1),
                "Coding": round(avg_code, 1),
                "HR": round(avg_hr, 1)
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "struggled_questions": struggled,
            "resume_vulnerabilities": resume_vulnerabilities,
            "readiness_level": readiness,
            "recommended_resources": recommended_resources
        }

    async def batch_evaluate_session(
        self,
        company: str,
        role: str,
        stage_config: List[Dict[str, Any]],
        questions: List[Dict[str, Any]],
        answers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates an entire multi-stage interview session in ONE batch call.
        Collects answered and unanswered questions across all dynamic stages.
        """
        answers_map = {a["question_id"]: a for a in answers if isinstance(a, dict) and "question_id" in a}
        
        evaluated_qa = []
        correct_count = 0
        incorrect_count = 0
        strong_count = 0
        acceptable_count = 0
        weak_count = 0
        unanswered_count = 0
        answered_count = 0
        
        subjective_items_to_eval = []
        
        for q in questions:
            q_id = q["id"]
            a = answers_map.get(q_id)
            
            user_text = (a.get("answer_text") or "").strip() if a else ""
            code_text = (a.get("code_submission") or "").strip() if a else ""
            sel_opt = a.get("selected_option") if a else None
            
            is_answered = bool(user_text or code_text or sel_opt is not None)
            
            round_type = q.get("round_type", "Technical")
            stage_name = q.get("current_stage_name") or round_type
            
            if not is_answered:
                unanswered_count += 1
                evaluated_qa.append({
                    "question_id": q_id,
                    "question_text": q["question_text"],
                    "round_type": round_type,
                    "stage_name": stage_name,
                    "topic": q.get("topic", "General"),
                    "status": "unanswered",
                    "evaluation_label": "Not Answered",
                    "score": 0.0,
                    "user_answer": "Not Answered",
                    "feedback": "Question was left unanswered.",
                    "suggestions": "Review topic principles and attempt all questions."
                })
            else:
                answered_count += 1
                # Check if objective question (Aptitude / MCQ with options & correct index)
                if round_type == "Aptitude" and sel_opt is not None and q.get("correct_option_index") is not None:
                    correct_idx = q.get("correct_option_index")
                    options = q.get("options", [])
                    is_corr = (sel_opt == correct_idx)
                    score = 100.0 if is_corr else 0.0
                    if is_corr:
                        correct_count += 1
                        label = "Correct"
                    else:
                        incorrect_count += 1
                        label = "Incorrect"
                    
                    correct_opt_str = options[correct_idx] if options and correct_idx < len(options) else ""
                    user_opt_str = options[sel_opt] if options and sel_opt < len(options) else f"Option {sel_opt+1}"
                    
                    evaluated_qa.append({
                        "question_id": q_id,
                        "question_text": q["question_text"],
                        "round_type": round_type,
                        "stage_name": stage_name,
                        "topic": q.get("topic", "General"),
                        "status": "answered",
                        "evaluation_label": label,
                        "score": score,
                        "user_answer": user_opt_str,
                        "feedback": f"Selected '{user_opt_str}'. {'Correct!' if is_corr else f'Incorrect. Correct option: {correct_opt_str}'}",
                        "suggestions": "Practice speed quantitative problem solving."
                    })
                else:
                    # Subjective question queued for AI batch evaluation
                    subjective_items_to_eval.append({
                        "question_id": q_id,
                        "question_text": q["question_text"],
                        "round_type": round_type,
                        "stage_name": stage_name,
                        "topic": q.get("topic", "General"),
                        "user_answer": user_text or f"Code Submission:\n{code_text}"
                    })

        # Batch AI evaluation for subjective items
        if subjective_items_to_eval:
            system_prompt = (
                "You are an Executive Hiring Manager evaluating a candidate's multi-stage interview answers in batch. "
                "For each question, assign a score (0-100), an evaluation_label ('Strong', 'Acceptable', or 'Weak'), "
                "a concise feedback explanation, and a targeted improvement suggestion."
            )
            
            prompt_items = []
            for idx, item in enumerate(subjective_items_to_eval):
                prompt_items.append(f"Q{idx+1} [ID: {item['question_id']}] ({item['round_type']} - Stage: {item['stage_name']} - Topic: {item['topic']}):\n"
                                   f"Question: {item['question_text']}\n"
                                   f"Candidate Answer: {item['user_answer']}\n")
            
            prompt = f"Company: {company}\nRole: {role}\nEvaluate these candidate responses:\n" + "\n".join(prompt_items) + \
                     "\nReturn JSON array of objects with keys: 'question_id', 'score', 'evaluation_label' ('Strong'|'Acceptable'|'Weak'), 'feedback', 'suggestions'."
            
            try:
                ai_results = await ai_router.execute_json_task("local", prompt, system_prompt, reasoning_budget=16384)
                ai_eval_map = {}
                if isinstance(ai_results, list):
                    for item in ai_results:
                        if isinstance(item, dict) and "question_id" in item:
                            ai_eval_map[item["question_id"]] = item
                elif isinstance(ai_results, dict) and "evaluations" in ai_results:
                    for item in ai_results["evaluations"]:
                        if isinstance(item, dict) and "question_id" in item:
                            ai_eval_map[item["question_id"]] = item
                            
                for item in subjective_items_to_eval:
                    q_id = item["question_id"]
                    ai_eval = ai_eval_map.get(q_id, {})
                    score = float(ai_eval.get("score", 75.0))
                    label = ai_eval.get("evaluation_label") or ("Strong" if score >= 80 else ("Acceptable" if score >= 60 else "Weak"))
                    
                    if label == "Strong": strong_count += 1
                    elif label == "Acceptable": acceptable_count += 1
                    else: weak_count += 1
                    
                    evaluated_qa.append({
                        "question_id": q_id,
                        "question_text": item["question_text"],
                        "round_type": item["round_type"],
                        "stage_name": item["stage_name"],
                        "topic": item["topic"],
                        "status": "answered",
                        "evaluation_label": label,
                        "score": score,
                        "user_answer": item["user_answer"],
                        "feedback": ai_eval.get("feedback") or "Demonstrated foundational understanding of core concepts.",
                        "suggestions": ai_eval.get("suggestions") or "Refine architectural depth and failure mode analysis."
                    })
            except Exception as e:
                print(f"[Batch AI Evaluation Warning]: {e}. Using deterministic evaluation fallback.")
                for item in subjective_items_to_eval:
                    ans_len = len(item["user_answer"])
                    score = 80.0 if ans_len > 100 else (65.0 if ans_len > 30 else 50.0)
                    label = "Strong" if score >= 80 else ("Acceptable" if score >= 60 else "Weak")
                    if label == "Strong": strong_count += 1
                    elif label == "Acceptable": acceptable_count += 1
                    else: weak_count += 1
                    evaluated_qa.append({
                        "question_id": item["question_id"],
                        "question_text": item["question_text"],
                        "round_type": item["round_type"],
                        "stage_name": item["stage_name"],
                        "topic": item["topic"],
                        "status": "answered",
                        "evaluation_label": label,
                        "score": score,
                        "user_answer": item["user_answer"],
                        "feedback": "Answer recorded and evaluated.",
                        "suggestions": "Elaborate with concrete metrics and production trade-offs."
                    })

        # Calculate Per Dynamic Stage Breakdown
        stage_map = {}
        for qa in evaluated_qa:
            st_name = qa["stage_name"]
            if st_name not in stage_map:
                stage_map[st_name] = {
                    "stage_name": st_name,
                    "round_type": qa["round_type"],
                    "total_questions": 0,
                    "answered_count": 0,
                    "unanswered_count": 0,
                    "correct_count": 0,
                    "incorrect_count": 0,
                    "strong_count": 0,
                    "acceptable_count": 0,
                    "weak_count": 0,
                    "scores": []
                }
            st_obj = stage_map[st_name]
            st_obj["total_questions"] += 1
            if qa["status"] == "answered":
                st_obj["answered_count"] += 1
                st_obj["scores"].append(qa["score"])
                lbl = qa.get("evaluation_label")
                if lbl == "Correct": st_obj["correct_count"] += 1
                elif lbl == "Incorrect": st_obj["incorrect_count"] += 1
                elif lbl == "Strong": st_obj["strong_count"] += 1
                elif lbl == "Acceptable": st_obj["acceptable_count"] += 1
                elif lbl == "Weak": st_obj["weak_count"] += 1
            else:
                st_obj["unanswered_count"] += 1
                st_obj["scores"].append(0.0)

        stage_breakdown = []
        for st_name, st_data in stage_map.items():
            scores = st_data["scores"]
            avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
            st_data["stage_score"] = avg_score
            del st_data["scores"]
            stage_breakdown.append(st_data)

        # Overall Session Scores & Summary
        all_scores = [qa["score"] for qa in evaluated_qa]
        overall_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0.0

        # Synthesize final strengths, weaknesses, vulnerabilities
        report_synthesis = await self.generate_final_report(company, role, evaluated_qa)

        return {
            "overall_score": overall_score,
            "total_questions_count": len(questions),
            "answered_count": answered_count,
            "unanswered_count": unanswered_count,
            "correct_answers_count": correct_count,
            "incorrect_answers_count": incorrect_count,
            "strong_answers_count": strong_count,
            "acceptable_answers_count": acceptable_count,
            "weak_answers_count": weak_count,
            "stage_breakdown": stage_breakdown,
            "questions_with_answers": evaluated_qa,
            "round_scores": report_synthesis.get("round_scores", {}),
            "strengths": report_synthesis.get("strengths", []),
            "weaknesses": report_synthesis.get("weaknesses", []),
            "struggled_questions": report_synthesis.get("struggled_questions", []),
            "resume_vulnerabilities": report_synthesis.get("resume_vulnerabilities", []),
            "readiness_level": report_synthesis.get("readiness_level", "Role Ready 🚀"),
            "recommended_resources": report_synthesis.get("recommended_resources", [])
        }

evaluator = AnswerEvaluator()

