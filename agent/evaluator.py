# agent/evaluator.py
import google.generativeai as genai
import os
import re
import json
from typing import List, Dict

# --- NEW PROMPT: Softer, concept-focused grading ---
GRADER_SYSTEM_PROMPT = """
You are a helpful and encouraging Quiz Grader. You will receive a Question, a "Gold Answer" (the perfect factual answer), and a "User Answer". 
Your job is to evaluate if the User's Answer shows a conceptual understanding, even if the phrasing is different.

- 'Correct': The User Answer is semantically equivalent or a perfect match. It demonstrates they understand the core concept.
- 'Partial': The User Answer is on the right track but missed a key detail (like a specific number or year), or is too generic but correct. Be lenient if they have the main idea.
- 'Incorrect': The User Answer is factually wrong or completely unrelated.

You MUST return ONLY a single, valid JSON object with two keys:
1. "status": (a string, must be one of "Correct", "Partial", or "Incorrect")
2. "feedback": (a one-sentence constructive explanation for your grading. If wrong, gently provide the correct fact.)
"""

def evaluate_quiz(quiz_questions: List[Dict], user_answers: List[str]) -> Dict:
    """
    Grades the user's quiz using the Gemini API for robust semantic evaluation.
    """
    report_items = []
    correct_count = 0
    
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite",
            system_instruction=GRADER_SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"} 
        )
    except Exception as e:
        return {"summary": "Failed to load Grader Agent", "items": [], "score": 0}

    for i, qa_pair in enumerate(quiz_questions):
        question = qa_pair['question']
        gold_ans = qa_pair['answer_gold']
        user_ans = user_answers[i].strip() if user_answers[i] else "No answer provided."

        prompt = f"""
        Question: "{question}"
        Gold Answer: "{gold_ans}"
        User Answer: "{user_ans}"
        """
        
        try:
            response = model.generate_content(prompt)
            grade_json = json.loads(response.text)
            status = grade_json.get("status", "Incorrect")
            feedback = grade_json.get("feedback", "No feedback provided.")
        except Exception as e:
            print(f"Error grading with LLM: {e}")
            status = "Error"
            feedback = "Could not parse grading response from AI."

        if status == "Correct":
            correct_count += 1
            
        report_items.append({
            "question": question,
            "your_answer": user_ans,
            "status": status,
            "correction": feedback  
        })

    summary = f"You scored {correct_count} out of {len(report_items)}."
    return {"summary": summary, "items": report_items, "score": correct_count}