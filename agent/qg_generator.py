# agent/qg_generator.py
import google.generativeai as genai
import os
import re
import json
from typing import List, Dict

# --- NEW PROMPT: Asks for more insightful, core-concept questions ---
QG_SYSTEM_PROMPT = """
You are an expert Quiz Generation Agent. Your job is to generate exactly 5 high-quality, insightful questions based ONLY on the provided text from a single research paper page.

RULES:
1. Generate exactly 5 questions.
2. Do NOT generate trivial questions about text formatting or simple definitions.
3. Generate questions that test the *core concepts*, *implications*, *methodologies*, or *key data results* presented on the page.
4. The answer to each question MUST be found directly in the text.
5. You MUST return ONLY a valid JSON list of objects. Do not add any other text, markdown, or commentary.
6. Each object MUST have exactly two keys: "question" (the insightful question) and "answer_gold" (the concise, factual answer from the text).
"""

class QuestionGenerator:
    """
    This is the "Test Mode" QuizAgent. It calls the Gemini API.
    (This will be replaced by the local fine-tuned agent for the final submission).
    """
    
    def __init__(self, model_id: str = None, adapter_path: str = None):
        print("Initialized API-based (Mock) Question Generator.")
        pass # No models to load

    def _parse_llm_response(self, response_text: str) -> List[Dict]:
        """Safely parses the JSON list from the LLM's raw text response."""
        match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if not match:
            print("QG Parser Error: No JSON list found in response.")
            return []
            
        json_string = match.group(0)
        
        try:
            qa_list = json.loads(json_string)
            if isinstance(qa_list, list) and all(isinstance(item, dict) for item in qa_list):
                return [item for item in qa_list if 'question' in item and 'answer_gold' in item]
            else:
                return []
        except json.JSONDecodeError as e:
            print(f"QG Parser Error: Failed to decode JSON. Error: {e}")
            return []

    def generate_questions(self, page_data: Dict, num_questions: int = 5) -> List[Dict]:
        """
        Calls the Gemini API to generate a list of Q/A pairs for a given page.
        """
        page_text = page_data['text']
        page_num = page_data["page"]

        try:
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash-lite",
                system_instruction=QG_SYSTEM_PROMPT,
                generation_config={"response_mime_type": "application/json"} # Force JSON
            )
        except Exception as e:
            return [{"question": "Error: Could not configure Gemini API", "answer_gold": str(e), "page": page_num, "context_snippet": ""}]

        user_prompt = f"CONTEXT (from Page {page_num}):\n```{page_text[:100000]}```"

        try:
            response = model.generate_content(user_prompt)
            parsed_qas = self._parse_llm_response(response.text)
            
            if not parsed_qas:
                 return [{"question": "Error: Model returned an invalid quiz format.", "answer_gold": "", "page": page_num, "context_snippet": ""}]

            final_qa_list = []
            for qa in parsed_qas:
                final_qa_list.append({
                    "question": qa["question"],
                    "answer_gold": qa["answer_gold"],
                    "page": page_num, 
                    "context_snippet": qa["answer_gold"] 
                })
            
            return final_qa_list

        except Exception as e:
            print(f"Error calling Gemini for QG: {e}")
            return [{"question": f"Error: {e}", "answer_gold": "", "page": page_num, "context_snippet": ""}]