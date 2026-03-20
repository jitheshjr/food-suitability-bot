def build_prompt(entities: dict, food_nutrition: dict,
                 ml_label: str, ml_confidence: float,
                 shap_reasons: list, rag_results: list) -> str:

    food_name = food_nutrition.get('food_name', entities.get('food', 'unknown'))
    age       = entities.get('age', 'unknown')
    condition = (entities.get('condition') or 'unknown').replace('_', ' ')
    sugar     = food_nutrition.get('sugar_g', '?')
    sodium    = food_nutrition.get('sodium_mg', '?')
    gi        = food_nutrition.get('gi_value', '?')

    # Pick one guideline sentence
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
            top_reason = f"especially its {top_feat}"

    prompt = (
        f"A {age}-year-old patient with {condition} asked about eating {food_name}. "
        f"It contains {sugar}g sugar, {sodium}mg sodium, glycemic index {gi}. "
        f"{guideline_hint} "
        f"In 2 sentences, explain why {food_name} is a health concern "
        f"for someone with {condition}{(' ' + top_reason) if top_reason else ''}. "
        f"Do not say it is safe or okay."
    )

    return prompt


def build_final_response(ml_label: str, ml_confidence: float,
                          shap_reasons: list, food_name: str,
                          llm_explanation: str) -> str:
    """
    Assembles the final user-facing response.
    Python controls the verdict sentence — LLM only provides explanation.
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

    # Top 2 SHAP reasons as bullet points
    shap_text = ""
    if shap_reasons:
        risk_factors = [
            f"- {feat.title()}: {'risk factor' if val > 0 else 'protective factor'}"
            for feat, val in shap_reasons[:2]
        ]
        shap_text = "\nKey factors: " + ", ".join(
            [f"{feat.title()} ({'risk' if val > 0 else 'protective'})"
             for feat, val in shap_reasons[:2]]
        )

    disclaimer = "Please consult your doctor before making dietary changes."

    return f"{verdict}\n\n{llm_explanation}\n{shap_text}\n\n{disclaimer}"
