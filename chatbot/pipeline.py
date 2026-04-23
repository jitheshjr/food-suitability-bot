import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from chatbot.session_manager import process_turn, clear_session
from chatbot.food_lookup     import lookup_food
from chatbot.fusion          import build_prompt, build_final_response
from chatbot.llm_response    import generate_response
from rag.retriever           import retrieve_context
from ml_model.predict        import predict_suitability


# ──────────────────────────────────────────────────────────────────────────────
# CONDITION-SPECIFIC NOT-FOUND ADVICE
# When a food is not in the database, skip the ML model entirely and return
# a condition-appropriate dietary guidance message instead.
# ──────────────────────────────────────────────────────────────────────────────

_NOT_FOUND_ADVICE = {
    'kidney_disease': (
        "For kidney disease, focus on low-potassium foods (apples, berries, white rice, pasta), "
        "low-phosphorus options (avoiding dairy, nuts, and processed foods), "
        "and limit protein to the amount your doctor recommends based on your CKD stage. "
        "Avoid high-sodium foods and processed snacks."
    ),
    'diabetes': (
        "For diabetes, focus on low-glycemic index foods such as vegetables, legumes, whole grains, "
        "and lean proteins. Prioritize fiber-rich foods (oats, lentils, broccoli) which slow glucose absorption, "
        "and avoid high-sugar or refined carbohydrate foods."
    ),
    'hypertension': (
        "For hypertension, focus on potassium-rich foods (bananas, spinach, sweet potato) which help lower blood pressure, "
        "and strictly limit sodium intake to under 1500mg per day. "
        "The DASH diet — rich in vegetables, fruits, whole grains, and low-fat dairy — is strongly recommended."
    ),
    'healthy': (
        "For a balanced diet, focus on whole foods including vegetables, fruits, lean proteins, "
        "whole grains, and healthy fats. Limit processed foods, added sugars, and excessive sodium."
    ),
}

def _build_not_found_response(food_name: str, condition: str | None) -> str:
    """
    Returns a clear, condition-specific response when a food is not in the database.
    Does NOT run the ML model — default nutrition values would produce unreliable results.
    """
    cond_key    = (condition or 'healthy').lower()
    advice      = _NOT_FOUND_ADVICE.get(cond_key, _NOT_FOUND_ADVICE['healthy'])
    cond_label  = cond_key.replace('_', ' ').title()
    food_label  = food_name.strip().title() if food_name else "This food"

    return (
        f"**{food_label}** was not found in our nutrition database, "
        f"so we cannot give a specific analysis for it.\n\n"
        f"**General guidance for {cond_label}:**\n"
        f"{advice}\n\n"
        f"If you'd like, try asking about a similar food we might have data for, "
        f"or consult your doctor or dietitian for personalized advice about {food_label}."
    )



def run_conversation_turn(session_id: str, user_text: str, verbose: bool = False) -> dict:
    """
    Main entry point for all UI layers.

    Every message from the user goes through here.

    Return dict:
    {
      'type':          'message' | 'response' | 'error',
      'text':          str,
      'ml_label':      str | None,
      'ml_confidence': float | None,
      'proba_dict':    dict | None,    # NEW — {'safe': x, 'moderate': y, 'avoid': z}
      'entities':      dict | None,
      'shap_reasons':  list | None,
      'session':       dict,
      'action':        str,
    }
    """

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"TURN  : session={session_id}")
        print(f"INPUT : {user_text[:80]}")
        print('=' * 55)

    # ── Step 1: Session manager — state tracking and entity extraction ──
    result  = process_turn(session_id, user_text)
    action  = result['action']
    session = result['session']

    if verbose:
        print(f"ACTION: {action}")
        print(f"STATE : {session.to_dict()}")

    # ── Step 2: Still collecting fields — return next question to user ──
    # Covers both required-field questions and enrichment questions.
    if action != 'run_pipeline':
        return {
            'type':          'message',
            'text':          result['message'],
            'ml_label':      None,
            'ml_confidence': None,
            'proba_dict':    None,
            'entities':      None,
            'shap_reasons':  None,
            'session':       session.to_dict(),
            'action':        action,
        }

    # ── Step 3: All required fields collected — build full patient dict ──
    # session.to_patient_dict() returns ALL fields (required + optional).
    # predict.py handles None values gracefully for optional fields.
    patient = session.to_patient_dict()

    # Entities dict for fusion.py (includes food + all patient fields)
    entities = {**patient, 'food': session.food, 'raw_text': user_text}

    if verbose:
        print(f"\n[PIPELINE STARTING]")
        # Show which optional fields were collected — confirms enrichment worked
        optional_filled = {k: v for k, v in patient.items()
                           if v is not None and k not in ('age', 'condition')}
        optional_missing = [k for k, v in patient.items()
                            if v is None and k not in ('age', 'condition')]
        print(f"[1] Patient (required): age={patient['age']} | condition={patient['condition']}")
        print(f"    Optional filled  : {optional_filled}")
        print(f"    Optional missing : {optional_missing}")
        print(f"    Food             : {session.food}")
        print(f"    Enrichment turns : {session.to_dict().get('enrichment_asked', 0)}")

    try:
        # ── Food nutrition lookup ──────────────────────────────────
        food_nutrition = lookup_food(entities['food'])
        if verbose:
            print(f"[2] Food lookup: {food_nutrition.get('canonical_name', food_nutrition.get('food_name'))}")
            print(f"    Nutrients : K={food_nutrition.get('potassium_mg')}mg  "
                  f"P={food_nutrition.get('phosphorus_mg')}mg  "
                  f"Na={food_nutrition.get('sodium_mg')}mg  "
                  f"GI={food_nutrition.get('gi_value')}  "
                  f"found={food_nutrition.get('found')}")

        # ── Not-found early exit ───────────────────────────────────
        # If the food is not in the database, the ML model would run on
        # DEFAULT_NUTRITION values (generic placeholder) — producing an
        # unreliable verdict. Instead, skip the model entirely and return
        # a condition-specific dietary guidance message.
        if not food_nutrition.get('found', True):
            if verbose:
                print(f"[!] Food not found in DB — returning not-found response")
            not_found_text = _build_not_found_response(
                food_name=entities.get('food', ''),
                condition=entities.get('condition'),
            )
            clear_session(session_id)
            return {
                'type':          'not_found',
                'text':          not_found_text,
                'ml_label':      None,
                'ml_confidence': None,
                'proba_dict':    None,
                'entities':      entities,
                'shap_reasons':  None,
                'session':       session.to_dict(),
                'action':        'not_found',
            }

        # ── RAG retrieval ──────────────────────────────────────────
        # Build a richer query using patient condition + food + food category
        food_category = food_nutrition.get('food_category', '')
        rag_query = " ".join(filter(None, [
            entities.get('condition', '').replace('_', ' '),
            entities.get('food', ''),
            food_category if food_category != 'other' else '',
            'dietary guidelines nutrition',
        ]))
        rag_results = retrieve_context(rag_query, k=3)
        if verbose:
            top_score = rag_results[0]['score'] if rag_results else 0
            print(f"[3] RAG: {len(rag_results)} chunks  (top score: {top_score:.3f})")

        # ── ML prediction ──────────────────────────────────────────
        # predict_suitability now returns 4 values — unpack all of them
        ml_label, ml_confidence, shap_reasons, proba_dict = predict_suitability(
            patient, food_nutrition
        )
        if verbose:
            print(f"[4] ML verdict  : {ml_label} ({ml_confidence}%)")
            print(f"    Probabilities: {proba_dict}")
            print(f"    SHAP top-5   : {shap_reasons}")

        # ── Fusion prompt ──────────────────────────────────────────
        prompt = build_prompt(
            entities=entities,
            food_nutrition=food_nutrition,
            ml_label=ml_label,
            ml_confidence=ml_confidence,
            shap_reasons=shap_reasons,
            rag_results=rag_results,
        )
        if verbose:
            print(f"[5] Prompt ({len(prompt)} chars):\n    {prompt[:120]}...")

        # ── LLM explanation ────────────────────────────────────────
        if verbose:
            print(f"[6] Calling {os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')}...")
        llm_explanation = generate_response(prompt)

        # ── Assemble final response ────────────────────────────────
        food_name = (
            food_nutrition.get('canonical_name') or
            food_nutrition.get('food_name') or
            entities['food']
        )
        final_response = build_final_response(
            ml_label=ml_label,
            ml_confidence=ml_confidence,
            shap_reasons=shap_reasons,
            food_name=food_name,
            condition=entities.get('condition'),
            rag_results=rag_results,
            llm_explanation=llm_explanation,
            proba_dict=proba_dict,           # NEW — passed through
            patient_context=patient,         # NEW — for contextual footer
        )

        if verbose:
            print(f"\n{'─' * 55}")
            print("RESPONSE:")
            print(final_response)
            print('─' * 55)

        # ── Reset session after successful response ────────────────
        clear_session(session_id)

        return {
            'type':          'response',
            'text':          final_response,
            'ml_label':      ml_label,
            'ml_confidence': ml_confidence,
            'proba_dict':    proba_dict,      # NEW
            'entities':      entities,
            'shap_reasons':  shap_reasons,
            'session':       session.to_dict(),
            'action':        'run_pipeline',
        }

    except Exception as e:
        clear_session(session_id)
        error_msg = (
            "Something went wrong while analyzing your query. "
            f"Please try again. (Error: {str(e)})"
        )
        if verbose:
            import traceback
            print(f"[ERROR] {e}")
            traceback.print_exc()
        return {
            'type':          'error',
            'text':          error_msg,
            'ml_label':      None,
            'ml_confidence': None,
            'proba_dict':    None,
            'entities':      entities,
            'shap_reasons':  None,
            'session':       session.to_dict(),
            'action':        'error',
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    session_id = "test_session_001"

    print("Food Suitability Advisor — Conversational Mode")
    print("Type 'quit' to exit, 'reset' to start over")
    print("─" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() == 'quit':
            break

        result = run_conversation_turn(session_id, user_input, verbose=True)
        print(f"\nBot: {result['text']}")

        if result['type'] == 'response':
            print(f"\n  Verdict    : {result['ml_label'].upper()} ({result['ml_confidence']}%)")
            print(f"  Proba      : {result['proba_dict']}")
            e = result['entities']
            print(f"  Patient    : age={e.get('age')} | condition={e.get('condition')} | "
                  f"ckd_stage={e.get('ckd_stage')} | activity={e.get('activity_level')}")
            print(f"  Food       : {e.get('food')}")