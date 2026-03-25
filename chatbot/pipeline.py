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


def run_conversation_turn(session_id: str, user_text: str, verbose: bool = False) -> dict:
    """
    Main entry point for all UI layers.

    Every message from the user goes through here.
    Returns a dict that the UI uses to decide what to display.

    Return dict:
    {
      'type':          'message' | 'response' | 'error',
      'text':          str,
      'ml_label':      str | None,
      'ml_confidence': float | None,
      'entities':      dict | None,
      'shap_reasons':  list | None,
      'session':       dict,
      'action':        str,
    }
    """

    if verbose:
        print(f"\n{'='*50}")
        print(f"TURN  : session={session_id}")
        print(f"INPUT : {user_text[:80]}")
        print('='*50)

    # ── Step 1: Session manager handles state and validation ──
    # This calls Phi-3 for entity extraction internally
    # and merges findings into session memory
    result  = process_turn(session_id, user_text)
    action  = result['action']
    session = result['session']

    if verbose:
        print(f"ACTION: {action}")
        print(f"STATE : {session.to_dict()}")

    # ── Step 2: If not all fields collected yet, return message ──
    if action != 'run_pipeline':
        return {
            'type':          'message',
            'text':          result['message'],
            'ml_label':      None,
            'ml_confidence': None,
            'entities':      None,
            'shap_reasons':  None,
            'session':       session.to_dict(),
            'action':        action,
        }

    # ── Step 3: All three fields collected — run full pipeline ──
    entities = {
        'age':       session.age,
        'condition': session.condition,
        'food':      session.food,
        'raw_text':  user_text,
    }

    if verbose:
        print(f"\n[PIPELINE STARTING]")
        print(f"[1] Entities: {entities}")

    try:
        # ── Food nutrition lookup ──────────────────
        food_nutrition = lookup_food(entities['food'])
        if verbose:
            print(f"[2] Food: {food_nutrition.get('food_name')} "
                  f"(sugar={food_nutrition.get('sugar_g')}g, "
                  f"gi={food_nutrition.get('gi_value')})")

        # ── RAG retrieval ──────────────────────────
        rag_query = (
            f"{entities['condition']} "
            f"{entities['food']} "
            f"dietary guidelines nutrition"
        ).strip()
        rag_results = retrieve_context(rag_query, k=3)
        if verbose:
            print(f"[3] RAG: {len(rag_results)} chunks retrieved "
                  f"(top score: {rag_results[0]['score'] if rag_results else 0})")

        # ── ML prediction ──────────────────────────
        patient = {
            'age':       entities['age'],
            'condition': entities['condition'],
        }
        ml_label, ml_confidence, shap_reasons = predict_suitability(
            patient, food_nutrition
        )
        if verbose:
            print(f"[4] ML verdict: {ml_label} ({ml_confidence}%)")
            print(f"    SHAP: {shap_reasons}")

        # ── Build fusion prompt ────────────────────
        prompt = build_prompt(
            entities=entities,
            food_nutrition=food_nutrition,
            ml_label=ml_label,
            ml_confidence=ml_confidence,
            shap_reasons=shap_reasons,
            rag_results=rag_results,
        )
        if verbose:
            print(f"[5] Prompt built ({len(prompt)} chars)")

        # ── Generate LLM explanation ───────────────
        if verbose:
            print("[6] Sending to Phi-3...")
        llm_explanation = generate_response(prompt)

        # ── Assemble final response ────────────────
        food_name = food_nutrition.get('food_name', entities['food'])
        final_response = build_final_response(
            ml_label=ml_label,
            ml_confidence=ml_confidence,
            shap_reasons=shap_reasons,
            food_name=food_name,
            llm_explanation=llm_explanation,
        )

        if verbose:
            print(f"\n{'─'*50}")
            print("RESPONSE:")
            print(final_response)
            print(f"{'─'*50}")

        # ── Reset session after successful response ─
        # User's question is fully answered — clear state
        # so they can ask about a different food next
        clear_session(session_id)

        return {
            'type':          'response',
            'text':          final_response,
            'ml_label':      ml_label,
            'ml_confidence': ml_confidence,
            'entities':      entities,
            'shap_reasons':  shap_reasons,
            'session':       session.to_dict(),
            'action':        'run_pipeline',
        }

    except Exception as e:
        # Reset session on error too — don't leave a broken state
        clear_session(session_id)
        error_msg = (
            "Something went wrong while analyzing your query. "
            f"Please try again. (Error: {str(e)})"
        )
        if verbose:
            print(f"[ERROR] {e}")
        return {
            'type':          'error',
            'text':          error_msg,
            'ml_label':      None,
            'ml_confidence': None,
            'entities':      entities,
            'shap_reasons':  None,
            'session':       session.to_dict(),
            'action':        'error',
        }


if __name__ == "__main__":
    session_id = "test_session_001"

    print("Food Suitability Advisor — Conversational Mode")
    print("Type 'quit' to exit, 'reset' to start over")
    print("─" * 50)

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
            print(f"\n     Verdict    : {result['ml_label'].upper()} ({result['ml_confidence']}%)")
            print(f"     Detected   : age={result['entities']['age']} | "
                  f"condition={result['entities']['condition']} | "
                  f"food={result['entities']['food']}")