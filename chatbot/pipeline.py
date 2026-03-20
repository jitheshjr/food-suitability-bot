import sys
import os

# Make sure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from chatbot.entity_extractor import extract_entities
from chatbot.food_lookup       import lookup_food
from chatbot.fusion            import build_prompt
from chatbot.llm_response      import generate_response
from rag.retriever             import retrieve_context
from ml_model.predict          import predict_suitability

def run_pipeline(user_query: str, verbose: bool = False) -> dict:
    """
    Full pipeline: query → entities → RAG + ML → fusion → LLM → response

    Returns dict with response and all intermediate results.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"PIPELINE: {user_query[:60]}")
        print('='*50)

    # ── Step 1: Extract entities ───────────────────
    entities = extract_entities(user_query)
    if verbose:
        print(f"[1] Entities: {entities}")

    # ── Step 2: Food nutrition lookup ──────────────
    food_nutrition = lookup_food(entities.get('food') or '')
    if verbose:
        print(f"[2] Food: {food_nutrition.get('food_name')} "
              f"(sugar={food_nutrition.get('sugar_g')}g, "
              f"gi={food_nutrition.get('gi_value')})")

    # ── Step 3: RAG retrieval ──────────────────────
    rag_query = (
        f"{entities.get('condition', '')} "
        f"{entities.get('food', '')} "
        f"dietary guidelines nutrition"
    ).strip()
    rag_results = retrieve_context(rag_query, k=3)
    if verbose:
        print(f"[3] RAG: retrieved {len(rag_results)} chunks "
              f"(top score: {rag_results[0]['score'] if rag_results else 0})")

    # ── Step 4: ML prediction ──────────────────────
    patient = {
        'age':       entities.get('age') or 40,
        'condition': entities.get('condition') or 'healthy',
    }
    ml_label, ml_confidence, shap_reasons = predict_suitability(
        patient, food_nutrition
    )
    if verbose:
        print(f"[4] ML verdict: {ml_label} ({ml_confidence}%)")
        print(f"    SHAP: {shap_reasons}")

    # ── Step 5: Build fusion prompt ────────────────
    # ── Step 5: Build fusion prompt ────────────────
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

    # ── Step 6: Generate LLM explanation ──────────
    if verbose:
        print("[6] Sending to TinyLlama...")
    llm_explanation = generate_response(prompt)

    # ── Step 7: Assemble final response ───────────
    from chatbot.fusion import build_final_response
    food_name = food_nutrition.get('food_name', entities.get('food', 'unknown'))
    final_response = build_final_response(
        ml_label=ml_label,
        ml_confidence=ml_confidence,
        shap_reasons=shap_reasons,
        food_name=food_name,
        llm_explanation=llm_explanation,
    )

    return {
        'response':       final_response,
        'entities':       entities,
        'food_nutrition': food_nutrition,
        'ml_label':       ml_label,
        'ml_confidence':  ml_confidence,
        'shap_reasons':   shap_reasons,
        'rag_results':    rag_results,
    }

if __name__ == "__main__":
    # The moment of truth — test your full pipeline
    test_queries = [
        "I am a 64 years old diabetic patient. I am craving ice cream. Is that ok?",
        "I am 55 with hypertension. Can I eat salty chips?",
        "I am 40 years old and healthy. Can I eat white rice daily?",
    ]

    for query in test_queries:
        result = run_pipeline(query, verbose=True)
        print(f"\n{'─'*50}")
        print("RESPONSE:")
        print(result['response'])
        print(f"{'─'*50}\n")
        input("Press Enter for next query...")
