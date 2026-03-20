import re
import os

CONDITIONS = {
    'diabetes':       ['diabet', 'blood sugar', 'insulin', 'hyperglycemi'],
    'hypertension':   ['hypertens', 'high blood pressure', 'blood pressure'],
    'kidney_disease': ['kidney', 'renal', 'nephro', 'ckd'],
    'healthy':        ['healthy', 'normal', 'no condition', 'no disease'],
}

def extract_entities(text: str) -> dict:
    t = text.lower().strip()

    # ── Age ──────────────────────────────────────────
    age = None
    age_patterns = [
        r'(\d{1,3})\s*(?:year|yr)s?\s*old',
        r'(?:aged?|am)\s*(\d{1,3})',
        r'(\d{1,3})\s*(?:year|yr)s?\s*(?:of age)?',
    ]
    for pattern in age_patterns:
        m = re.search(pattern, t)
        if m:
            candidate = int(m.group(1))
            if 1 <= candidate <= 120:
                age = candidate
                break

    # ── Condition ─────────────────────────────────────
    condition = None
    for cond_key, keywords in CONDITIONS.items():
        for kw in keywords:
            if kw in t:
                condition = cond_key
                break
        if condition:
            break

    # ── Food item ─────────────────────────────────────
    food = None
    food_patterns = [
        r'(?:eat|eating|have|consume|drink|take|eating)\s+([a-z\s]+?)(?:\?|\.|\!|,|$)',
        r'(?:can i|is it ok|is it safe|should i).*?(?:have|eat|consume|drink)\s+([a-z\s]+?)(?:\?|\.|\!|,|$)',
        r'(?:about|regarding|for)\s+([a-z\s]+?)(?:\?|\.|\!|,|$)',
        r'(?:craving|want to eat|like to eat)\s+([a-z\s]+?)(?:\?|\.|\!|,|$)',
    ]
    for pattern in food_patterns:
        m = re.search(pattern, t)
        if m:
            candidate = m.group(1).strip()
            # Filter out non-food phrases
            skip = ['it', 'this', 'that', 'anything', 'everything', 'more']
            if candidate and candidate not in skip and len(candidate) > 2:
                food = candidate
                break

    return {
        'age':       age,
        'condition': condition,
        'food':      food,
        'raw_text':  text,
    }


if __name__ == "__main__":
    tests = [
        "I am a 64 years old diabetic patient, I am craving to eat ice cream. Is that ok?",
        "I have hypertension and I am 55 years old. Can I eat salty chips?",
        "I am 45 and have kidney disease. Is it safe to eat chicken?",
        "I am 30 years old and healthy. Can I consume white rice?",
    ]
    for t in tests:
        result = extract_entities(t)
        print(f"Input : {t[:60]}...")
        print(f"Result: {result}\n")
