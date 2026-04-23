import json
import os
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ──────────────────────────────────────────────────────────────────────────────
# EXTRACTION PROMPT
# Extended to capture all new patient profile fields the ML model now uses.
# Fields are split into REQUIRED (age, condition, food) and OPTIONAL (all others).
# The LLM only fills optional fields when the user explicitly mentions them —
# it never guesses or infers them.
# ──────────────────────────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """You are an entity extractor for a food health advisory system.
Extract fields from the user message. Return ONLY a JSON object — no explanation, no markdown.

REQUIRED fields (extract if mentioned):
- age        : integer 1-110, or null
- condition  : one of "diabetes", "hypertension", "kidney_disease", "healthy", or null
- food       : short food name string, or null

OPTIONAL fields (only fill if the user EXPLICITLY mentions them — never guess):
- gender          : "male" or "female", or null
- ckd_stage       : integer 1-5 (only if user says e.g. "stage 3 kidney disease"), or null
- activity_level  : one of "sedentary", "lightly_active", "moderately_active", "very_active", or null
                    Map: "not active/sit all day/lazy" → "sedentary"
                         "light walk/not very active" → "lightly_active"
                         "moderate exercise/gym sometimes" → "moderately_active"
                         "very active/athlete/workout daily" → "very_active"
- comorbidity     : one of "diabetes", "hypertension", "kidney_disease", "obesity", "anemia", "none", or null
                    Only fill if it is a SECONDARY condition (not the primary condition field)
- medication      : one of "ace_inhibitor", "arb", "phosphate_binder", "diuretic",
                    "beta_blocker", "insulin", "metformin", "sglt2_inhibitor", "none", or null
                    Map common names: "lisinopril/enalapril/ramipril" → "ace_inhibitor"
                                     "losartan/valsartan/olmesartan" → "arb"
                                     "furosemide/hydrochlorothiazide/HCTZ" → "diuretic"
                                     "atenolol/metoprolol/bisoprolol" → "beta_blocker"
- height_cm       : integer in cm, or null (convert feet/inches if mentioned)
- weight_kg       : float in kg, or null (convert lbs if mentioned: lbs * 0.453592)
- dialysis_type   : one of "hemodialysis", "peritoneal", "none", or null
- diabetes_type   : one of "type1", "type2", or null

Additional rules:
- If the message is gibberish, random characters, or has no health/food meaning → set gibberish: true, all fields null
- "healthy" means the person has no medical condition
- Return ONLY the JSON — no extra text whatsoever

Output format (include ALL keys, null for unknown):
{
  "age": <int|null>,
  "condition": <str|null>,
  "food": <str|null>,
  "gender": <str|null>,
  "ckd_stage": <int|null>,
  "activity_level": <str|null>,
  "comorbidity": <str|null>,
  "medication": <str|null>,
  "height_cm": <int|null>,
  "weight_kg": <float|null>,
  "dialysis_type": <str|null>,
  "diabetes_type": <str|null>,
  "gibberish": <bool>
}

Examples:

Input: "I am 64 years old diabetic, can I eat ice cream?"
Output: {"age": 64, "condition": "diabetes", "food": "ice cream", "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "I am a 55 year old male with stage 3 kidney disease. I take lisinopril. Can I eat chicken pasta?"
Output: {"age": 55, "condition": "kidney_disease", "food": "chicken pasta", "gender": "male", "ckd_stage": 3, "activity_level": null, "comorbidity": null, "medication": "ace_inhibitor", "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "I have type 2 diabetes and also high blood pressure. I am 62, female, moderately active. Can I eat biryani?"
Output: {"age": 62, "condition": "diabetes", "food": "biryani", "gender": "female", "ckd_stage": null, "activity_level": "moderately_active", "comorbidity": "hypertension", "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": "type2", "gibberish": false}

Input: "I am on hemodialysis, 70kg, 168cm tall. Can I eat salmon?"
Output: {"age": null, "condition": "kidney_disease", "food": "salmon", "gender": null, "ckd_stage": 5, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": 168, "weight_kg": 70.0, "dialysis_type": "hemodialysis", "diabetes_type": null, "gibberish": false}

Input: "I weigh 180 lbs and I am 5ft 10. I am 45 with hypertension and I take metoprolol. Can I eat pizza?"
Output: {"age": 45, "condition": "hypertension", "food": "pizza", "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": "beta_blocker", "height_cm": 178, "weight_kg": 81.6, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "sdnbsdbf sdjkfhsdfs"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": true}

Input: "55"
Output: {"age": 55, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "I don't exercise much. I just sit at home all day."
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": "sedentary", "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}


Input: "sedentary"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": "sedentary", "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "very active"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": "very_active", "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "moderately active"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": "moderately_active", "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "type 2"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": "type2", "gibberish": false}

Input: "type 1"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": "type1", "gibberish": false}

Input: "stage 3"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": 3, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "stage 4"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": 4, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "hemodialysis"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": 5, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": "hemodialysis", "diabetes_type": null, "gibberish": false}

Input: "peritoneal dialysis"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": 5, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": "peritoneal", "diabetes_type": null, "gibberish": false}

Input: "lisinopril"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": "ace_inhibitor", "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "metformin"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": "metformin", "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "none"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": "none", "medication": "none", "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "I also have high blood pressure"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": "hypertension", "medication": null, "height_cm": null, "weight_kg": null, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "80 kg and 170 cm"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": 170, "weight_kg": 80.0, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Input: "150 lbs, 5ft 6"
Output: {"age": null, "condition": null, "food": null, "gender": null, "ckd_stage": null, "activity_level": null, "comorbidity": null, "medication": null, "height_cm": 168, "weight_kg": 68.0, "dialysis_type": null, "diabetes_type": null, "gibberish": false}

Now extract:
Input: "{TEXT}"
Output:"""

# ──────────────────────────────────────────────────────────────────────────────
# VALID VALUES FOR VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

VALID_CONDITIONS    = {'diabetes', 'hypertension', 'kidney_disease', 'healthy'}
VALID_GENDERS       = {'male', 'female'}
VALID_ACTIVITY      = {'sedentary', 'lightly_active', 'moderately_active', 'very_active'}
VALID_COMORBIDITY   = {'diabetes', 'hypertension', 'kidney_disease', 'obesity', 'anemia', 'none'}
VALID_MEDICATION    = {
    'ace_inhibitor', 'arb', 'phosphate_binder', 'diuretic',
    'beta_blocker', 'insulin', 'metformin', 'sglt2_inhibitor', 'none'
}
VALID_DIALYSIS      = {'hemodialysis', 'peritoneal', 'none'}
VALID_DIABETES_TYPE = {'type1', 'type2'}

BAD_FOOD_WORDS = {'it', 'this', 'that', 'food', 'something', 'anything', 'thing', ''}


def _validate_age(raw) -> int | None:
    if raw is None:
        return None
    try:
        age = int(raw)
        return age if 1 <= age <= 110 else None
    except (ValueError, TypeError):
        return None


def _validate_enum(raw, valid_set: set) -> str | None:
    if raw is None:
        return None
    val = str(raw).strip().lower()
    return val if val in valid_set else None


def _validate_food(raw) -> str | None:
    if not raw:
        return None
    food = str(raw).strip().lower()
    if len(food) < 2 or food in BAD_FOOD_WORDS:
        return None
    return food


def _validate_positive_float(raw) -> float | None:
    if raw is None:
        return None
    try:
        val = float(raw)
        return val if val > 0 else None
    except (ValueError, TypeError):
        return None


def _validate_ckd_stage(raw, dialysis_type=None) -> int | None:
    """If dialysis is mentioned, infer stage 5."""
    if dialysis_type in ('hemodialysis', 'peritoneal'):
        return 5
    if raw is None:
        return None
    try:
        stage = int(raw)
        return stage if 1 <= stage <= 5 else None
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

def extract_entities_llm(text: str) -> dict:
    try:
        prompt = EXTRACTION_PROMPT.replace("{TEXT}", text.replace('"', "'"))

        response = _client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            max_tokens=200,    # increased — larger JSON response now
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r'```json|```', '', raw).strip()

        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON in response: {raw[:100]}")

        parsed = json.loads(json_match.group())

        # ── Validate every field ──────────────────────────────────
        dialysis_raw     = _validate_enum(parsed.get('dialysis_type'), VALID_DIALYSIS)
        ckd_stage        = _validate_ckd_stage(parsed.get('ckd_stage'), dialysis_raw)
        height_cm_raw    = _validate_positive_float(parsed.get('height_cm'))
        weight_kg_raw    = _validate_positive_float(parsed.get('weight_kg'))

        # Sanity bounds for height/weight
        height_cm = int(height_cm_raw) if height_cm_raw and 100 <= height_cm_raw <= 250 else None
        weight_kg = round(weight_kg_raw, 1) if weight_kg_raw and 20 <= weight_kg_raw <= 300 else None

        return {
            # Required
            'age':           _validate_age(parsed.get('age')),
            'condition':     _validate_enum(parsed.get('condition'), VALID_CONDITIONS),
            'food':          _validate_food(parsed.get('food')),
            # Optional patient fields
            'gender':        _validate_enum(parsed.get('gender'), VALID_GENDERS),
            'ckd_stage':     ckd_stage,
            'activity_level':_validate_enum(parsed.get('activity_level'), VALID_ACTIVITY),
            'comorbidity':   _validate_enum(parsed.get('comorbidity'), VALID_COMORBIDITY),
            'medication':    _validate_enum(parsed.get('medication'), VALID_MEDICATION),
            'height_cm':     height_cm,
            'weight_kg':     weight_kg,
            'dialysis_type': dialysis_raw,
            'diabetes_type': _validate_enum(parsed.get('diabetes_type'), VALID_DIABETES_TYPE),
            # Meta
            'gibberish':     bool(parsed.get('gibberish', False)),
            'raw_text':      text,
            'error':         None,
        }

    except Exception as e:
        # ── Fallback: regex-only extraction for required fields ────
        age = None
        m = re.search(r'(\d{1,3})\s*(?:year|yr)s?\s*old', text.lower())
        if m:
            candidate = int(m.group(1))
            if 1 <= candidate <= 110:
                age = candidate

        return {
            'age':           age,
            'condition':     None,
            'food':          None,
            'gender':        None,
            'ckd_stage':     None,
            'activity_level':None,
            'comorbidity':   None,
            'medication':    None,
            'height_cm':     None,
            'weight_kg':     None,
            'dialysis_type': None,
            'diabetes_type': None,
            'gibberish':     False,
            'raw_text':      text,
            'error':         str(e),
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "I am a 64 years old diabetic patient. Can I eat ice cream?",
        "I have hypertension and I am 55. Can I eat salty chips?",
        "I am a 55 year old male with stage 3 kidney disease on lisinopril. Can I eat chicken pasta?",
        "I have type 2 diabetes and also high blood pressure. 62 female, moderately active. Biryani ok?",
        "I am on hemodialysis, 70kg, 168cm. Can I eat salmon?",
        "I weigh 180 lbs, 5ft 10. 45 with hypertension, taking metoprolol. Can I eat pizza?",
        "I sit at home all day, don't exercise. 50, kidney disease stage 4. Can I eat banana?",
        "sdnbsdbf sdjkfhsdfs",
        "hello",
        "64",
        "!!!! ??? ###",
    ]

    print(f"LLM ENTITY EXTRACTOR — {GROQ_MODEL}")
    print("=" * 70)
    for t in tests:
        r = extract_entities_llm(t)
        print(f"\nInput  : {t}")
        print(f"Required : age={r['age']} | condition={r['condition']} | food={r['food']}")
        print(f"Optional : gender={r['gender']} | ckd_stage={r['ckd_stage']} | "
              f"activity={r['activity_level']} | comorbidity={r['comorbidity']}")
        print(f"           medication={r['medication']} | height={r['height_cm']}cm | "
              f"weight={r['weight_kg']}kg | dialysis={r['dialysis_type']} | "
              f"diabetes_type={r['diabetes_type']}")
        print(f"Meta     : gibberish={r['gibberish']}")
        if r.get('error'):
            print(f"Error    : {r['error']}")