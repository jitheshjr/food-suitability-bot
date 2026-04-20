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


def _humanize_food_name(food_name: str) -> str:
    """Convert raw dataset labels into something friendlier for users."""
    if not food_name:
        return "This food"

    text = str(food_name).replace("_", " ").strip()
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) >= 2:
            text = parts[0]
            if len(text.split()) <= 2:
                text = parts[1]
        else:
            text = parts[0]

    words = [word for word in text.split() if word.lower() not in {"prepared", "entree"}]
    cleaned = " ".join(words).strip(" ,.-")
    return cleaned.title() if cleaned else "This food"


def _format_rag_sources(rag_results: list) -> str:
    """Build a short, deduplicated citations line from retrieved RAG sources."""
    if not rag_results:
        return ""

    seen = set()
    ordered_sources = []
    for item in rag_results:
        source = (item.get("source") or "").strip()
        if source and source not in seen:
            seen.add(source)
            ordered_sources.append(source)

    if not ordered_sources:
        return ""

    return "Sources: " + "; ".join(ordered_sources[:3])


def _friendly_factor_text(shap_reasons: list) -> list[str]:
    lines = []
    for feature, value in shap_reasons[:3]:
        direction = "increases concern" if value > 0 else "slightly lowers concern"
        lines.append(f"{feature.title()} {direction}")
    return lines


def _plain_recommendation(label: str, food_name: str, condition: str | None = None) -> str:
    condition_text = (condition or "your condition").replace("_", " ")
    if condition_text == "healthy":
        audience_text = "someone without a known medical condition"
    else:
        audience_text = f"someone with {condition_text}"
    if label == "avoid":
        return f"{food_name} is best avoided for {audience_text}."
    if label == "moderate":
        return f"{food_name} is better as an occasional choice for {audience_text}."
    if label == "safe":
        return f"{food_name} is generally a reasonable choice for {audience_text}."
    return f"Be cautious with {food_name} for {audience_text}."


def build_final_response(ml_label: str, ml_confidence: float,
                         shap_reasons: list, food_name: str,
                         condition: str | None,
                         rag_results: list,
                         llm_explanation: str) -> str:
    """
    Python controls the verdict sentence.
    Phi-3 provides the explanation.
    This separation prevents the LLM from contradicting the ML model.
    """
    pretty_food_name = _humanize_food_name(food_name)
    recommendation = _plain_recommendation(ml_label, pretty_food_name, condition)
    confidence_text = f"Confidence: {ml_confidence}%"
    factor_lines = _friendly_factor_text(shap_reasons)
    citations_text = _format_rag_sources(rag_results)
    disclaimer = "Please consult your doctor before making any dietary changes."

    parts = [
        f"Recommendation: {recommendation}",
        "",
        llm_explanation.strip(),
    ]

    if factor_lines:
        parts.extend(["", "Top factors:"])
        parts.extend([f"- {line}" for line in factor_lines])

    if citations_text:
        parts.extend(["", f"Evidence: {citations_text.replace('Sources: ', '')}"])

    parts.extend(["", confidence_text, disclaimer])

    return "\n".join(parts)
