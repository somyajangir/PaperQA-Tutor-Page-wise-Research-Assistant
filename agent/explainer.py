# agent/explainer.py
import google.generativeai as genai
import os
import re
from typing import Dict, Tuple

# --- NEW PROMPT: Professional, structured, and clear ---
SYSTEM_PROMPT = """
You are an expert Academic Research Assistant. Your job is to clarify dense research paper text, page by page.
You will receive "Previous Context" (a bulleted summary of topics already covered) and the "Current Page Text".
Your tone should be clear, professional, and insightful. Assume the user is an undergraduate or graduate student who enjoys research.

You MUST return your response in three distinct parts, using the exact headers:

CONCEPTUAL_SUMMARY:
[Your 2-3 sentence high-level summary of this page's main argument or purpose.]

DETAILED_BREAKDOWN:
- [A bulleted list explaining the first key concept, finding, or method on this page, using a simple analogy or example if it helps clarity.]
- [A bulleted list for the second key concept...]
- [And so on for all major points on the page.]

KEY_TERMS:
- [Term 1]: [A concise, one-sentence definition as it relates to this paper.]
- [Term 2]: [A concise, one-sentence definition...]
- [Term 3]: [A concise, one-sentence definition...]

---SUMMARY---
- [Your new 3-bullet summary of THIS page's key takeaways, to be passed as context to the NEXT page.]
"""

FALLBACK_EXPLANATION = {
    "summary": "Could not parse explanation from model.",
    "breakdown": "The model response was not in the correct format. Please try again.",
    "terms": "- N/A"
}

def get_explanation(page_text: str, memory_bullets: str) -> Tuple[Dict, str]:
    """Calls Gemini API to get a structured explanation AND a new summary."""
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash-lite",
        system_instruction=SYSTEM_PROMPT,
        generation_config={"temperature": 0.3} # More factual
    )
    user_prompt = f"Previous Context:\n{memory_bullets}\n---\nCurrent Page Text:\n{page_text[:100000]}"

    try:
        response = model.generate_content(user_prompt)
        response_text = response.text
        
        summary = "No summary generated."
        if "---SUMMARY---" in response_text:
            main_content, summary = response_text.split("---SUMMARY---", 1)
            summary = summary.strip()
        else:
            main_content = response_text

        # Use new regex keys
        summary_block = re.search(r"CONCEPTUAL_SUMMARY:\n([\s\S]*?)(?=\nDETAILED_BREAKDOWN:|\nKEY_TERMS:|$)", main_content, re.IGNORECASE)
        breakdown_block = re.search(r"DETAILED_BREAKDOWN:\n([\s\S]*?)(?=\nKEY_TERMS:|$)", main_content, re.IGNORECASE)
        terms_block = re.search(r"KEY_TERMS:\n([\s\S]*)", main_content, re.IGNORECASE)

        explanation = {
            "summary": summary_block.group(1).strip() if summary_block else "N/A",
            "breakdown": breakdown_block.group(1).strip() if breakdown_block else "N/A",
            "terms": terms_block.group(1).strip() if terms_block else "N/A"
        }
        
        if explanation["summary"] == "N/A" and explanation["breakdown"] == "N/A":
             return FALLBACK_EXPLANATION, summary

        return explanation, summary
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        return {"summary": "API Error", "breakdown": f"Error: {e}", "terms": ""}, "Error."