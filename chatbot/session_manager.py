import time
from chatbot.entity_extractor import extract_entities_llm

# ── Session store ─────────────────────────────────────────────
# Holds one SessionState per user session in memory.
# Key = session_id (string), Value = SessionState object
_sessions: dict = {}


class SessionState:
    """
    Stores everything collected across multiple turns for one user.

    Why this exists:
    - Phi-3 has no memory between calls
    - User may send age in turn 1, food in turn 2, condition in turn 3
    - This class accumulates all three and triggers the pipeline
      only when all fields are present
    """

    def __init__(self):
        self.age        = None   # int
        self.condition  = None   # str
        self.food       = None   # str
        self.step       = 'idle' # idle | collecting | ready
        self.turns      = 0      # how many messages this session has seen
        self.created_at = time.time()

    def update_from_extraction(self, extracted: dict):
        """
        Merges newly extracted fields into session.
        Only updates a field if it was found AND not already set.
        This prevents a later vague message from overwriting
        a correctly captured field.
        """
        if extracted.get('age') and not self.age:
            self.age = extracted['age']
        if extracted.get('condition') and not self.condition:
            self.condition = extracted['condition']
        if extracted.get('food') and not self.food:
            self.food = extracted['food']

    def missing_fields(self) -> list:
        """Returns list of field names still needed, in priority order."""
        missing = []
        if not self.food:      missing.append('food')
        if not self.age:       missing.append('age')
        if not self.condition: missing.append('condition')
        return missing

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0

    def to_dict(self) -> dict:
        return {
            'age':       self.age,
            'condition': self.condition,
            'food':      self.food,
            'step':      self.step,
            'turns':     self.turns,
        }

    def reset(self):
        self.__init__()


# ── Session management functions ──────────────────────────────

def get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    return _sessions[session_id]


def clear_session(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].reset()


# ── Response messages ─────────────────────────────────────────

def _ask_for_missing(session: SessionState, gibberish: bool = False) -> str:
    """
    Returns the next polite question based on what's missing.
    Asks for one field at a time, in order: food → age → condition.
    """
    if gibberish:
        return (
            "I'm sorry, I couldn't find any health-related information in your message. "
            "To help you, I need:\n\n"
            "- The **food** you'd like to check\n"
            "- Your **age**\n"
            "- Your **medical condition** (e.g. diabetes, hypertension, or none)\n\n"
            "For example: \"I am 45 years old with diabetes, can I eat rice?\""
        )

    missing = session.missing_fields()

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
            "If you have none, just say \"healthy\"."
        )

    return None


# ── Main entry point ──────────────────────────────────────────

def process_turn(session_id: str, user_text: str) -> dict:
    """
    Processes one conversation turn.

    Flow:
    1. Handle trivial cases (empty, greeting, thanks, reset) without LLM
    2. Call Phi-3 to extract entities from this turn's message
    3. Merge extracted fields into session (session = memory across turns)
    4. If all fields collected → trigger pipeline
    5. If fields still missing → ask for next missing field politely

    Returns:
    {
      'action':      str,   # what to do next
      'message':     str,   # reply to user (None if running pipeline)
      'session':     SessionState,
      'input_class': str,
    }
    """
    session = get_session(session_id)
    session.turns += 1
    t = user_text.strip()
    tl = t.lower()

    # ── 1. Empty input ────────────────────────────
    if not t:
        return {
            'action':      'reject',
            'message':     "Your message appears to be empty. Please tell me which food you'd like to check for your health condition.",
            'session':     session,
            'input_class': 'empty',
        }

    # ── 2. Greeting — no LLM call needed ─────────
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

    # ── 3. Thanks — no LLM call needed ───────────
    thanks = ['thank you', 'thanks', 'thank', 'thx', 'ty']
    if any(w in tl for w in thanks) and len(tl) < 30:
        return {
            'action':      'thanks',
            'message':     "You're welcome! Feel free to ask about another food anytime.",
            'session':     session,
            'input_class': 'thanks',
        }

    # ── 4. Reset request ──────────────────────────
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

    # ── 5. LLM entity extraction ──────────────────
    extracted = extract_entities_llm(user_text)

    # ── 6. Merge into session memory ──────────────
    # This is the key step — fields accumulate across turns
    session.update_from_extraction(extracted)
    session.step = 'collecting'

    # ── 7. Check if complete ──────────────────────
    if session.is_complete():
        session.step = 'ready'
        return {
            'action':      'run_pipeline',
            'message':     None,
            'session':     session,
            'input_class': 'complete',
        }

    # ── 8. Still missing fields — ask politely ────
    message = _ask_for_missing(session, gibberish=extracted.get('gibberish', False))
    return {
        'action':      'ask_missing',
        'message':     message,
        'session':     session,
        'input_class': 'partial',
    }