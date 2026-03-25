def build_prompt(entities: dict, food_nutrition: dict,
                 ml_label: str, ml_confidence: float,
                 shap_reasons: list, rag_results: list) -> str:

    food_name = food_nutrition.get('food_name', entities.get('food', 'unknown'))
    age       = entities.get('age', 'unknown')
    condition = (entities.get('condition') or 'unknown').replace('_', ' ')
    sugar     = food_nutrition.get('sugar_g', '?')
    sodium    = food_nutrition.get('sodium_mg', '?')
    gi        = food_nutrition.get('gi_value', '?')
    found     = food_nutrition.get('found', True)

    # Guideline hint from RAG
    guideline_hint = ""
    if rag_results:
        first_sentence = rag_results[0]['text'].split('.')[0].strip()
        if len(first_sentence) > 20:
            guideline_hint = first_sentence + "."

    # Top SHAP factor
    top_reason = ""
    if shap_reasons:
        top_feat, top_val = shap_reasons[0]
        if top_val > 0:
            top_reason = f"particularly its {top_feat}"

    # Nutrition note
    if found:
        nutrition_note = (
            f"It contains {sugar}g of sugar, {sodium}mg of sodium, "
            f"and has a glycemic index of {gi}."
        )
    else:
        nutrition_note = (
            f"Nutritional data for this food was not found in the database."
        )

    # Verdict phrase
    verdict_phrase = {
        'avoid':    f"should avoid {food_name}",
        'moderate': f"should eat {food_name} only in very small amounts",
        'safe':     f"can generally eat {food_name}",
    }.get(ml_label, f"should be cautious about {food_name}")

    prompt = (
        f"A {age}-year-old patient with {condition} {verdict_phrase}. "
        f"{nutrition_note} "
        f"{guideline_hint} "
        f"In 2 clear sentences, explain the nutritional reason why "
        f"this verdict applies{(' — ' + top_reason) if top_reason else ''}."
    )

    return prompt


def build_final_response(ml_label: str, ml_confidence: float,
                         shap_reasons: list, food_name: str,
                         llm_explanation: str) -> str:
    """
    Python controls the verdict sentence.
    Phi-3 provides the explanation.
    This separation prevents the LLM from contradicting the ML model.
    """
    verdict_sentences = {
        'avoid': (
            f"Based on nutritional analysis, we strongly recommend "
            f"avoiding {food_name} ({ml_confidence}% confidence)."
        ),
        'moderate': (
            f"Based on nutritional analysis, {food_name} should be "
            f"consumed in strict moderation ({ml_confidence}% confidence)."
        ),
        'safe': (
            f"Based on nutritional analysis, {food_name} appears "
            f"generally safe for your condition ({ml_confidence}% confidence)."
        ),
    }

    verdict = verdict_sentences.get(ml_label, f"Verdict: {ml_label}.")

    # SHAP key factors
    shap_text = ""
    if shap_reasons:
        factors = ", ".join([
            f"{feat.title()} ({'risk' if val > 0 else 'protective'})"
            for feat, val in shap_reasons[:2]
        ])
        shap_text = f"Key factors: {factors}"

    disclaimer = "Please consult your doctor before making any dietary changes."

    parts = [verdict, "", llm_explanation]
    if shap_text:
        parts.append(shap_text)
    parts.extend(["", disclaimer])

    return "\n".join(parts)