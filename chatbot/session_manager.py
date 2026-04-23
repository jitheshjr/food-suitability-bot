import time
from chatbot.entity_extractor import extract_entities_llm

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# ENRICHMENT QUESTION DEFINITIONS
#
# After the 3 required fields are collected, the bot asks condition-specific
# follow-up questions to populate the optional ML features.
#
# Design rules:
#   - Max 2 enrichment questions per session — avoid survey fatigue
#   - Questions are ordered by ML feature importance for that condition
#   - Every question has a 'skip_phrases' list — if the user says "skip" or
#     "I don't know", the field stays None and the next question is asked
#   - Each question maps to one session field via 'field'
#   - 'condition_filter': None means asked for all conditions
# ──────────────────────────────────────────────────────────────────────────────

ENRICHMENT_QUESTIONS = [
    # ── Question 1: CKD stage (only for kidney_disease) ──────────────────────
    {
        'field':            'ckd_stage',
        'condition_filter': ['kidney_disease'],
        'check':            lambda s: s.ckd_stage is None and s.dialysis_type is None,
        'ask': (
            "To give you a more accurate answer — do you know your **CKD stage** "
            "(1 to 5)? Stage 5 means you are on dialysis. "
            "If unsure, just say **skip**."
        ),
    },
    # ── Question 2: Dialysis type (only for kidney_disease stage 5 or dialysis mentioned) ──
    {
        'field':            'dialysis_type',
        'condition_filter': ['kidney_disease'],
        'check':            lambda s: (
            s.dialysis_type is None and
            s.ckd_stage is not None and s.ckd_stage == 5
        ),
        'ask': (
            "Are you on **hemodialysis** or **peritoneal dialysis**? "
            "This affects your protein requirements significantly. "
            "Say **skip** if unsure."
        ),
    },
    # ── Question 3: Diabetes type (only for diabetes) ─────────────────────────
    {
        'field':            'diabetes_type',
        'condition_filter': ['diabetes'],
        'check':            lambda s: s.diabetes_type is None,
        'ask': (
            "Do you have **Type 1** or **Type 2** diabetes? "
            "Say **skip** if you're not sure."
        ),
    },
    # ── Question 4: Comorbidity (kidney_disease + diabetes are most impactful) ─
    {
        'field':            'comorbidity',
        'condition_filter': ['kidney_disease', 'diabetes', 'hypertension'],
        'check':            lambda s: s.comorbidity is None,
        'ask': (
            "Do you have any **other medical conditions** alongside your primary one? "
            "For example: diabetes, high blood pressure, or obesity. "
            "Say **none** if you don't, or **skip** if unsure."
        ),
    },
    # ── Question 5: Activity level (all conditions) ───────────────────────────
    {
        'field':            'activity_level',
        'condition_filter': None,   # asked for all conditions
        'check':            lambda s: s.activity_level is None,
        'ask': (
            "How physically active are you on a typical day?\n"
            "- **Sedentary** — mostly sitting, little movement\n"
            "- **Lightly active** — short walks, light housework\n"
            "- **Moderately active** — exercise 3–4 times a week\n"
            "- **Very active** — daily workouts or physical job\n"
            "Say **skip** if you'd rather not answer."
        ),
    },
    # ── Question 6: Weight (all conditions — used for BMI) ────────────────────
    {
        'field':            'weight_kg',
        'condition_filter': None,
        'check':            lambda s: s.weight_kg is None and s.height_cm is None,
        'ask': (
            "Could you share your **weight and height**? "
            "(e.g. \"70 kg and 170 cm\" or \"154 lbs and 5ft 7\") "
            "This helps us personalise the analysis. Say **skip** to continue without it."
        ),
    },
    # ── Question 7: Medication (kidney_disease + hypertension most impactful) ──
    {
        'field':            'medication',
        'condition_filter': ['kidney_disease', 'hypertension', 'diabetes'],
        'check':            lambda s: s.medication is None,
        'ask': (
            "Are you currently taking any **medications** for your condition? "
            "For example: lisinopril, metformin, insulin, furosemide, losartan. "
            "Say **none** if not, or **skip** to continue."
        ),
    },
]

# How many enrichment questions to ask before running the pipeline regardless
MAX_ENRICHMENT_QUESTIONS = 2

# Phrases that mean "I don't want to answer this question"
SKIP_PHRASES = {
    'skip', 'idk', "don't know", "dont know", "not sure", "no idea",
    'pass', 'next', 'ignore', 'na', 'n/a', 'unsure', 'unknown',
}


class SessionState:
    """
    Accumulates patient profile fields across multiple conversation turns.

    Collection phases:
      Phase 1 — Required fields: food → age → condition
      Phase 2 — Enrichment:     condition-specific follow-up questions (max 2)
      Phase 3 — Run pipeline

    The transition from Phase 1 → Phase 2 → Phase 3 is managed by
    process_turn() using enrichment_index to track which question is next.
    """

    def __init__(self):
        # ── Required ──────────────────────────────────────────────
        self.age       = None
        self.condition = None
        self.food      = None

        # ── Optional patient fields ───────────────────────────────
        self.gender         = None
        self.height_cm      = None
        self.weight_kg      = None
        self.activity_level = None
        self.ckd_stage      = None
        self.dialysis_type  = None
        self.diabetes_type  = None
        self.comorbidity    = None
        self.medication     = None

        # ── Enrichment tracking ───────────────────────────────────
        # enrichment_index: which question in ENRICHMENT_QUESTIONS was last asked
        # enrichment_asked: how many enrichment questions have been asked so far
        # enrichment_phase: True once required fields are done and we're asking extras
        self.enrichment_index = 0
        self.enrichment_asked = 0
        self.enrichment_phase = False

        # ── Session meta ──────────────────────────────────────────
        self.step       = 'idle'
        self.turns      = 0
        self.created_at = time.time()

    def update_from_extraction(self, extracted: dict):
        """Merge extracted fields — first mention of each field wins."""
        if extracted.get('age') and not self.age:
            self.age = extracted['age']
        if extracted.get('condition') and not self.condition:
            self.condition = extracted['condition']
        if extracted.get('food') and not self.food:
            self.food = extracted['food']

        if extracted.get('gender') and not self.gender:
            self.gender = extracted['gender']
        if extracted.get('height_cm') and not self.height_cm:
            self.height_cm = extracted['height_cm']
        if extracted.get('weight_kg') and not self.weight_kg:
            self.weight_kg = extracted['weight_kg']
        if extracted.get('activity_level') and not self.activity_level:
            self.activity_level = extracted['activity_level']
        if extracted.get('ckd_stage') and not self.ckd_stage:
            self.ckd_stage = extracted['ckd_stage']
        if extracted.get('dialysis_type') and not self.dialysis_type:
            self.dialysis_type = extracted['dialysis_type']
        if extracted.get('diabetes_type') and not self.diabetes_type:
            self.diabetes_type = extracted['diabetes_type']
        if extracted.get('comorbidity') and not self.comorbidity:
            self.comorbidity = extracted['comorbidity']
        if extracted.get('medication') and not self.medication:
            self.medication = extracted['medication']

    def required_missing(self) -> list:
        missing = []
        if not self.food:      missing.append('food')
        if not self.age:       missing.append('age')
        if not self.condition: missing.append('condition')
        return missing

    def required_complete(self) -> bool:
        return len(self.required_missing()) == 0

    def next_enrichment_question(self) -> dict | None:
        """
        Returns the next enrichment question that:
          1. Applies to the current condition (condition_filter matches)
          2. Its field is still None (check lambda returns True)
          3. We haven't exceeded MAX_ENRICHMENT_QUESTIONS
        Returns None if no more questions should be asked.
        """
        if self.enrichment_asked >= MAX_ENRICHMENT_QUESTIONS:
            return None

        for i in range(self.enrichment_index, len(ENRICHMENT_QUESTIONS)):
            q = ENRICHMENT_QUESTIONS[i]
            # Check condition filter
            if q['condition_filter'] is not None:
                if self.condition not in q['condition_filter']:
                    continue
            # Check if field still needs to be collected
            if not q['check'](self):
                continue
            # This question is applicable
            self.enrichment_index = i + 1   # advance past this one
            return q
        return None

    def to_patient_dict(self) -> dict:
        return {
            'age':            self.age,
            'condition':      self.condition,
            'gender':         self.gender,
            'height_cm':      self.height_cm,
            'weight_kg':      self.weight_kg,
            'activity_level': self.activity_level,
            'ckd_stage':      self.ckd_stage,
            'dialysis_type':  self.dialysis_type,
            'diabetes_type':  self.diabetes_type,
            'comorbidity':    self.comorbidity,
            'medication':     self.medication,
        }

    def to_dict(self) -> dict:
        return {
            'age':             self.age,
            'condition':       self.condition,
            'food':            self.food,
            'gender':          self.gender,
            'height_cm':       self.height_cm,
            'weight_kg':       self.weight_kg,
            'activity_level':  self.activity_level,
            'ckd_stage':       self.ckd_stage,
            'dialysis_type':   self.dialysis_type,
            'diabetes_type':   self.diabetes_type,
            'comorbidity':     self.comorbidity,
            'medication':      self.medication,
            'enrichment_asked':self.enrichment_asked,
            'step':            self.step,
            'turns':           self.turns,
        }

    def reset(self):
        self.__init__()


# ── Session management ────────────────────────────────────────────────────────

def get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    return _sessions[session_id]


def clear_session(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].reset()


# ── Required field questions ──────────────────────────────────────────────────

def _ask_for_required(session: SessionState, gibberish: bool = False) -> str:
    if gibberish:
        return (
            "I'm sorry, I couldn't understand that. To help you, I need:\n\n"
            "- The **food** you'd like to check\n"
            "- Your **age**\n"
            "- Your **medical condition** (diabetes, hypertension, kidney disease, or none)\n\n"
            "Example: \"I am 45 years old with diabetes, can I eat rice?\""
        )

    missing = session.required_missing()

    if 'food' in missing:
        return "I'd be happy to help! Which **food** would you like to check?"

    if 'age' in missing:
        return (
            f"Got it — you'd like to know about **{session.food}**. "
            "Could you share your **age**?"
        )

    if 'condition' in missing:
        return (
            "Almost there! Do you have any medical conditions such as "
            "**diabetes**, **hypertension**, or **kidney disease**? "
            "If you have none, just say **healthy**."
        )

    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def process_turn(session_id: str, user_text: str) -> dict:
    """
    Processes one conversation turn.

    Phase 1 — Collect required fields (food, age, condition).
    Phase 2 — Ask enrichment questions (condition-specific, max 2).
    Phase 3 — Run pipeline.

    Enrichment questions are skippable — if the user says "skip" or "I don't know",
    the field stays None and we move to the next question or run the pipeline.
    """
    session = get_session(session_id)
    session.turns += 1
    t  = user_text.strip()
    tl = t.lower()

    # ── Trivial inputs ────────────────────────────────────────────

    if not t:
        return {
            'action':      'reject',
            'message':     "Your message appears to be empty. Please tell me which food you'd like to check.",
            'session':     session,
            'input_class': 'empty',
        }

    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon', 'hiya', 'howdy']
    if any(tl == g or tl.startswith(g + ' ') for g in greetings):
        return {
            'action':      'greet',
            'message':     (
                "Hello! I'm your food health advisor.\n\n"
                "I can tell you whether a food is suitable for your health condition. "
                "To get started — which **food** would you like to check?"
            ),
            'session':     session,
            'input_class': 'greeting',
        }

    # Word-boundary thanks check — prevents "type 2" matching "ty"
    thanks = ['thank you', 'thanks', 'thx']
    thanks_exact = ['ty', 'thank']   # only match these as whole words
    is_thanks = (
        any(w in tl for w in thanks) or
        any(tl == w or tl.startswith(w + ' ') or tl.endswith(' ' + w)
            for w in thanks_exact)
    )
    if is_thanks and len(tl) < 30:
        return {
            'action':      'thanks',
            'message':     "You're welcome! Feel free to ask about another food anytime.",
            'session':     session,
            'input_class': 'thanks',
        }

    resets = ['reset', 'start over', 'restart', 'new question', 'clear', 'begin again']
    if any(w in tl for w in resets):
        clear_session(session_id)
        session = get_session(session_id)
        return {
            'action':      'reset',
            'message':     "Sure, let's start fresh! Which **food** would you like to check?",
            'session':     session,
            'input_class': 'reset',
        }

    # ── LLM extraction ────────────────────────────────────────────
    extracted = extract_entities_llm(user_text)
    session.update_from_extraction(extracted)
    session.step = 'collecting'

    # ── Phase 1: Required fields ──────────────────────────────────
    if not session.required_complete():
        message = _ask_for_required(session, gibberish=extracted.get('gibberish', False))
        return {
            'action':      'ask_missing',
            'message':     message,
            'session':     session,
            'input_class': 'partial',
        }

    # ── Phase 2: Enrichment questions ─────────────────────────────
    # Required fields are now complete. Enter enrichment phase.
    session.enrichment_phase = True

    # Check if user's reply was a skip phrase
    user_skipped = any(p in tl for p in SKIP_PHRASES) and len(tl) < 25

    if not user_skipped:
        # Normal answer — extraction already merged whatever was found
        # Additional check: if user said "none" for comorbidity, store it explicitly
        if 'none' in tl and session.comorbidity is None and len(tl) < 20:
            session.comorbidity = 'none'

    # Find the next enrichment question to ask
    next_q = session.next_enrichment_question()

    if next_q is not None:
        session.enrichment_asked += 1
        session.step = 'enriching'
        return {
            'action':      'ask_enrichment',
            'message':     next_q['ask'],
            'session':     session,
            'input_class': 'enrichment',
        }

    # ── Phase 3: All done — run pipeline ─────────────────────────
    session.step = 'ready'
    return {
        'action':      'run_pipeline',
        'message':     None,
        'session':     session,
        'input_class': 'complete',
    }