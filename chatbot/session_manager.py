import time
from chatbot.entity_extractor import extract_entities_llm

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# ONBOARDING MESSAGE
#
# Shown once when the user greets the bot. Asks ALL required fields upfront
# in one friendly, non-intimidating message. User can answer everything at
# once or spread it across multiple replies — the session merges whatever
# is provided each turn.
# ──────────────────────────────────────────────────────────────────────────────

ONBOARDING_MESSAGE = (
    "Hello! 👋 I'm your Food Health Advisor.\n\n"
    "I can tell you whether a food is suitable for your health condition. "
    "To get started, I need a few details about you:\n\n"
    "- 🍽️ **Food** — which food would you like to check?\n"
    "- 🎂 **Age** — how old are you?\n"
    "- ⚕️ **Medical condition** — do you have diabetes, hypertension, kidney disease, or none?\n"
    "- 🧍 **Gender** — male or female?\n"
    "- 🏃 **Activity level** — sedentary, lightly active, moderately active, or very active?\n"
    "- ⚖️ **Height & weight** — e.g. \"170 cm and 65 kg\" or \"5ft 7 and 143 lbs\"\n\n"
    "Feel free to share everything in one message or a few — and skip anything you're unsure about. "
    "I'll ask about anything that's still missing. 😊"
)


# ──────────────────────────────────────────────────────────────────────────────
# ENRICHMENT QUESTION DEFINITIONS
#
# After the required fields are collected, the bot asks condition-specific
# follow-up questions to populate the remaining optional ML features.
#
# Design rules:
#   - Max 2 enrichment questions per session — avoid survey fatigue
#   - Questions are ordered by ML feature importance for that condition
#   - Every question has a 'skip_phrases' list — if the user says "skip" or
#     "I don't know", the field stays None and the next question is asked
#   - Each question maps to one session field via 'field'
#   - 'condition_filter': None means asked for all conditions
#
# NOTE: activity_level and weight/height are now REQUIRED fields asked
# upfront in onboarding — they are intentionally removed from here.
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
    # ── Question 5: Medication (kidney_disease + hypertension most impactful) ──
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
      Phase 0 — Greeting: show onboarding message with all required fields listed
      Phase 1 — Required fields: food, age, condition, gender, activity_level,
                                 height_cm / weight_kg
      Phase 2 — Enrichment:     condition-specific follow-up questions (max 2)
      Phase 3 — Run pipeline

    Required fields that the user explicitly cannot answer:
      - condition: defaults to "healthy" with a polite note
      - gender / activity_level / height_cm / weight_kg: skipped gracefully,
        pipeline runs without them after one patient reminder

    The transition Phase 1 → Phase 2 → Phase 3 is managed by process_turn()
    using enrichment_index to track which question is next.
    """

    def __init__(self):
        # ── Required ──────────────────────────────────────────────
        self.age            = None
        self.condition      = None
        self.food           = None
        self.gender         = None
        self.activity_level = None
        self.height_cm      = None
        self.weight_kg      = None

        # ── Optional patient fields ───────────────────────────────
        self.ckd_stage      = None
        self.dialysis_type  = None
        self.diabetes_type  = None
        self.comorbidity    = None
        self.medication     = None

        # ── Healthy fallback note ─────────────────────────────────
        # Set to True when condition is defaulted to healthy because
        # the user said they don't know — included in the pipeline response.
        self.assumed_healthy = False

        # ── Enrichment tracking ───────────────────────────────────
        self.enrichment_index = 0
        self.enrichment_asked = 0
        self.enrichment_phase = False

        # ── Required field skip tracking ──────────────────────────
        # Counts how many turns the user has been asked for optional-required
        # fields (gender, activity, height/weight) without providing them.
        # After 1 reminder, we proceed to enrichment/pipeline anyway.
        self.profile_reminder_count = 0

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

    def core_required_complete(self) -> bool:
        """food + age + condition — these three are non-negotiable."""
        return bool(self.food and self.age and self.condition)

    def profile_fields_missing(self) -> list:
        """
        Returns which profile-level required fields (gender, activity,
        height/weight) are still missing. These are asked after the core
        three are collected, but are skippable after one reminder.
        """
        missing = []
        if not self.gender:
            missing.append('gender')
        if not self.activity_level:
            missing.append('activity_level')
        if not self.height_cm and not self.weight_kg:
            missing.append('height_weight')
        return missing

    def required_missing(self) -> list:
        """Legacy helper — returns missing core required fields."""
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
            if q['condition_filter'] is not None:
                if self.condition not in q['condition_filter']:
                    continue
            if not q['check'](self):
                continue
            self.enrichment_index = i + 1
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
            'age':              self.age,
            'condition':        self.condition,
            'food':             self.food,
            'gender':           self.gender,
            'height_cm':        self.height_cm,
            'weight_kg':        self.weight_kg,
            'activity_level':   self.activity_level,
            'ckd_stage':        self.ckd_stage,
            'dialysis_type':    self.dialysis_type,
            'diabetes_type':    self.diabetes_type,
            'comorbidity':      self.comorbidity,
            'medication':       self.medication,
            'assumed_healthy':  self.assumed_healthy,
            'enrichment_asked': self.enrichment_asked,
            'step':             self.step,
            'turns':            self.turns,
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


# ──────────────────────────────────────────────────────────────────────────────
# REQUIRED FIELD PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

def _ask_for_core_required(session: SessionState, gibberish: bool = False) -> str:
    """
    Asks for whichever of the 3 core required fields (food, age, condition)
    are still missing. Called one field at a time — most natural for the
    user since these are the most important questions.
    """
    if gibberish:
        return (
            "I'm sorry, I couldn't understand that. 😊 To help you, I need:\n\n"
            "- The **food** you'd like to check\n"
            "- Your **age**\n"
            "- Your **medical condition** (diabetes, hypertension, kidney disease, or none)\n\n"
            "Example: *\"I am 45 years old with diabetes, can I eat rice?\"*"
        )

    missing = session.required_missing()

    if 'food' in missing:
        return "Of course! Which **food** would you like me to check for you?"

    if 'age' in missing:
        return (
            f"Got it — you'd like to know about **{session.food}**. "
            "Could you also share your **age**?"
        )

    if 'condition' in missing:
        return (
            "Almost there! Do you have any medical conditions such as "
            "**diabetes**, **hypertension**, or **kidney disease**? "
            "If you have none, just say **healthy** — "
            "or if you're not sure, I'll assume you're healthy. 😊"
        )

    return None


def _build_profile_reminder(session: SessionState) -> str:
    """
    After core required fields are collected, gently asks for whichever
    profile fields (gender, activity level, height/weight) are still missing.
    Shows only what's still needed — never repeats what's already provided.
    Called at most once; after that the pipeline runs regardless.
    """
    missing = session.profile_fields_missing()
    parts = []

    if 'gender' in missing:
        parts.append("- 🧍 **Gender** — male or female?")
    if 'activity_level' in missing:
        parts.append(
            "- 🏃 **Activity level** — sedentary, lightly active, "
            "moderately active, or very active?"
        )
    if 'height_weight' in missing:
        parts.append(
            "- ⚖️ **Height & weight** — e.g. *\"170 cm and 65 kg\"* "
            "or *\"5ft 7 and 143 lbs\"*"
        )

    if not parts:
        return None

    intro = (
        "Just a couple more details to make the analysis more accurate — "
        "feel free to skip anything you'd rather not share:\n\n"
    )
    outro = "\n\nOr just say **skip** to continue without these. 😊"
    return intro + "\n".join(parts) + outro


# ──────────────────────────────────────────────────────────────────────────────
# HEALTHY FALLBACK
#
# If the user explicitly says they don't know their condition after being asked,
# we default condition to "healthy" and set assumed_healthy = True.
# The pipeline response layer must check assumed_healthy and prepend the note.
# ──────────────────────────────────────────────────────────────────────────────

_CONDITION_SKIP_PHRASES = {
    'skip', 'idk', "don't know", "dont know", "not sure", "no idea",
    'unsure', 'unknown', "i don't know", "i dont know", "not sure about",
    "no clue", "no idea", "haven't checked", "havent checked",
}

def _user_skipped_condition(text: str) -> bool:
    tl = text.strip().lower()
    return any(p in tl for p in _CONDITION_SKIP_PHRASES) and len(tl) < 35


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def process_turn(session_id: str, user_text: str) -> dict:
    """
    Processes one conversation turn.

    Phase 0 — Greeting: show onboarding message listing all required fields.
    Phase 1 — Collect core required fields (food, age, condition).
               If condition is skipped/unknown → assume healthy, set note.
    Phase 1b— Collect profile fields (gender, activity, height/weight).
               Asked once as a gentle reminder; skippable.
    Phase 2 — Ask enrichment questions (condition-specific, max 2).
    Phase 3 — Run pipeline.
    """
    session = get_session(session_id)
    session.turns += 1
    t  = user_text.strip()
    tl = t.lower()

    # ── Empty input ───────────────────────────────────────────────
    if not t:
        return {
            'action':      'reject',
            'message':     "Your message seems empty. Please tell me which food you'd like to check. 😊",
            'session':     session,
            'input_class': 'empty',
        }

    # ── Greeting → show full onboarding message ───────────────────
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'good afternoon', 'hiya', 'howdy']
    if any(tl == g or tl.startswith(g + ' ') for g in greetings):
        return {
            'action':      'greet',
            'message':     ONBOARDING_MESSAGE,
            'session':     session,
            'input_class': 'greeting',
        }

    # ── Thanks ────────────────────────────────────────────────────
    thanks = ['thank you', 'thanks', 'thx']
    thanks_exact = ['ty', 'thank']
    is_thanks = (
        any(w in tl for w in thanks) or
        any(tl == w or tl.startswith(w + ' ') or tl.endswith(' ' + w)
            for w in thanks_exact)
    )
    if is_thanks and len(tl) < 30:
        return {
            'action':      'thanks',
            'message':     "You're welcome! 😊 Feel free to ask about another food anytime.",
            'session':     session,
            'input_class': 'thanks',
        }

    # ── Reset ─────────────────────────────────────────────────────
    resets = ['reset', 'start over', 'restart', 'new question', 'clear', 'begin again']
    if any(w in tl for w in resets):
        clear_session(session_id)
        session = get_session(session_id)
        return {
            'action':      'reset',
            'message':     "Sure, let's start fresh! 😊\n\n" + ONBOARDING_MESSAGE,
            'session':     session,
            'input_class': 'reset',
        }

    # ── LLM extraction ────────────────────────────────────────────
    extracted = extract_entities_llm(user_text)
    session.update_from_extraction(extracted)
    session.step = 'collecting'

    # ── Phase 1: Core required fields (food, age, condition) ──────
    if not session.required_complete():

        # Special case: user is being asked for condition and says they don't know
        if session.food and session.age and not session.condition:
            if _user_skipped_condition(t):
                session.condition    = 'healthy'
                session.assumed_healthy = True
                # Fall through to profile / enrichment below

        # Re-check after possible healthy fallback
        if not session.required_complete():
            msg = _ask_for_core_required(session, gibberish=extracted.get('gibberish', False))
            return {
                'action':      'ask_missing',
                'message':     msg,
                'session':     session,
                'input_class': 'partial',
            }

    # ── Phase 1b: Profile fields (gender, activity, height/weight) ─
    # Ask once as a polite reminder. After one reminder, proceed regardless.
    profile_missing = session.profile_fields_missing()
    if profile_missing and session.profile_reminder_count == 0:
        # Check if the user already provided some in this turn — if all gone, skip reminder
        user_skipped_profile = any(p in tl for p in SKIP_PHRASES) and len(tl) < 25
        if not user_skipped_profile:
            reminder_msg = _build_profile_reminder(session)
            if reminder_msg:
                session.profile_reminder_count += 1
                return {
                    'action':      'ask_profile',
                    'message':     reminder_msg,
                    'session':     session,
                    'input_class': 'profile_reminder',
                }

    # Profile reminder was shown — mark it done regardless of what user answered
    if session.profile_reminder_count == 0:
        session.profile_reminder_count = 1   # don't ask again

    # ── Phase 2: Enrichment questions ─────────────────────────────
    session.enrichment_phase = True

    user_skipped = any(p in tl for p in SKIP_PHRASES) and len(tl) < 25

    if not user_skipped:
        if 'none' in tl and session.comorbidity is None and len(tl) < 20:
            session.comorbidity = 'none'

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
        'action':        'run_pipeline',
        'message':       None,
        'assumed_healthy': session.assumed_healthy,   # pipeline layer uses this
        'session':       session,
        'input_class':   'complete',
    }