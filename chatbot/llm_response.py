import os
from groq import Groq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# The old prompt was too generic ("diet advisor, 3 sentences").
# The new prompt explicitly:
#   - Forbids the LLM from contradicting or re-evaluating the ML verdict
#   - Lists the nutrients relevant to each condition so explanations are specific
#   - Prevents hallucination of food names ("Latino" class of bug)
#   - Keeps response concise but condition-specific
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical diet explanation assistant.

Your ONLY job is to explain WHY a food verdict was given for a patient.
You must NOT re-evaluate, contradict, or override the verdict already stated.
You must NOT invent or paraphrase food names — use the exact food name given.
You must NOT give general diet advice — focus on the specific condition and nutrients mentioned.

Condition-specific focus:
- kidney_disease: Focus on potassium, phosphorus, sodium, and protein. Mention CKD stage if given.
  For dialysis patients, note that protein needs are higher than non-dialysis CKD.
- diabetes: Focus on glycemic index, sugar content, fiber, and carbohydrates.
  For type 1, note that carb counting for insulin dosing is critical.
- hypertension: Focus on sodium and potassium (potassium is protective).
- healthy: Focus on overall calorie density, sugar, and sodium.

Rules:
- Respond in exactly 2 sentences.
- Be specific to the patient's condition and the nutrients provided.
- Do not start with "I" or "As a".
- Do not mention the ML model, confidence scores, or SHAP values.
- Do not repeat the verdict word (avoid/moderate/safe) — it is already shown separately.
"""


def generate_response(prompt: str) -> str:
    """
    Generate a 2-sentence explanation for the given fusion prompt.
    The prompt already contains the verdict, patient context, and nutrient values.
    """
    try:
        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': prompt},
            ],
            temperature=0.15,    # low — medical explanations should be consistent
            max_tokens=150,      # enough for 2 specific sentences
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return (
            "A detailed explanation could not be generated at this time. "
            f"Please consult your doctor or dietitian for personalized advice. (Error: {str(e)})"
        )