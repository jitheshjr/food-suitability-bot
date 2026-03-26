import json
import os
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EXTRACTION_PROMPT = """You are an entity extractor for a food health advisory system.
Extract exactly three fields from the user message:
1. age       — patient age as integer, or null
2. condition — exactly one of: "diabetes", "hypertension", "kidney_disease", "healthy", or null
3. food      — food item as short string, or null

Additional rules:
- If the message is gibberish, random characters, or has no health/food meaning → set gibberish: true, all fields null
- "healthy" means the person has no medical condition
- Age must be between 1 and 110, otherwise null
- Return ONLY the JSON object — no explanation, no markdown, no extra text

Output format:
{"age": <int|null>, "condition": <str|null>, "food": <str|null>, "gibberish": <bool>}

Examples:
Input: "I am 64 years old diabetic, can I eat ice cream?"
Output: {"age": 64, "condition": "diabetes", "food": "ice cream", "gibberish": false}

Input: "sdnbsdbf sdjkfhsdfs"
Output: {"age": null, "condition": null, "food": null, "gibberish": true}

Input: "55"
Output: {"age": 55, "condition": null, "food": null, "gibberish": false}

Input: "diabetes"
Output: {"age": null, "condition": "diabetes", "food": null, "gibberish": false}

Input: "rice"
Output: {"age": null, "condition": null, "food": "rice", "gibberish": false}

Input: "hello"
Output: {"age": null, "condition": null, "food": null, "gibberish": false}

Input: "I have high blood pressure. Is biryani ok?"
Output: {"age": null, "condition": "hypertension", "food": "biryani", "gibberish": false}

Input: "can i eat chicken curry? i am 50 with kidney problems"
Output: {"age": 50, "condition": "kidney_disease", "food": "chicken curry", "gibberish": false}

Input: "!!!! ??? ###"
Output: {"age": null, "condition": null, "food": null, "gibberish": true}

Now extract:
Input: "{TEXT}"
Output:"""


def extract_entities_llm(text: str) -> dict:
    try:
        prompt = EXTRACTION_PROMPT.replace("{TEXT}", text.replace('"', "'"))

        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=60,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'```json|```', '', raw).strip()

        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON in response: {raw[:100]}")

        parsed = json.loads(json_match.group())

        # Validate age
        age = parsed.get('age')
        if age is not None:
            try:
                age = int(age)
                if not (1 <= age <= 110):
                    age = None
            except (ValueError, TypeError):
                age = None

        # Validate condition
        condition = parsed.get('condition')
        if condition not in ['diabetes', 'hypertension', 'kidney_disease', 'healthy', None]:
            condition = None

        # Validate food
        food = parsed.get('food')
        if food:
            food = str(food).strip().lower()
            bad = ['it', 'this', 'that', 'food', 'something', 'anything', '']
            if len(food) < 2 or food in bad:
                food = None

        gibberish = bool(parsed.get('gibberish', False))

        return {
            'age':       age,
            'condition': condition,
            'food':      food,
            'gibberish': gibberish,
            'raw_text':  text,
            'error':     None,
        }

    except Exception as e:
        # Fallback regex for age only
        age = None
        m = re.search(r'(\d{1,3})\s*(?:year|yr)s?\s*old', text.lower())
        if m:
            candidate = int(m.group(1))
            if 1 <= candidate <= 110:
                age = candidate
        return {
            'age':       age,
            'condition': None,
            'food':      None,
            'gibberish': False,
            'raw_text':  text,
            'error':     str(e),
        }


if __name__ == "__main__":
    tests = [
        "I am a 64 years old diabetic patient. Can I eat ice cream?",
        "I have hypertension and I am 55. Can I eat salty chips?",
        "sdnbsdbf sdjkfhsdfs",
        "hello",
        "ice cream",
        "64",
        "diabetes",
        "!!!! ??? ###",
        "can i eat biryani? i am 50 with kidney disease",
        "I am 30 and healthy. Can I eat white rice?",
    ]

    print(f"LLM ENTITY EXTRACTOR — {GROQ_MODEL}")
    print("=" * 60)
    for t in tests:
        r = extract_entities_llm(t)
        print(f"\nInput : {t}")
        print(f"Result: age={r['age']} | condition={r['condition']} | food={r['food']} | gibberish={r['gibberish']}")
        if r.get('error'):
            print(f"Error : {r['error']}")