from __future__ import annotations


# ──────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _humanize_food_name(food_name: str, user_query: str | None = None) -> str:
    """
    Convert a food name into something display-friendly.

    Prefers user_query (exactly what the user typed) over the database name.
    The USDA database has pluralized or verbose names like "ice creams" or
    "crackers, whole wheat" that look wrong in responses.
    If user_query is provided and is a clean short string, use it directly.
    """
    # Prefer the user's own words — they said "ice cream", show "ice cream"
    if user_query and len(user_query.strip()) >= 2:
        return user_query.strip().title()

    if not food_name:
        return "This food"
    text = str(food_name).replace("_", " ").strip()
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        # Always take first part — USDA names lead with the primary food item
        # e.g. "ice creams, chocolate" → "ice creams" (correct)
        #      "soup, chicken noodle, canned" → "soup" (too generic — handled below)
        text = parts[0] if parts else text
        # Only swap to second part if first part is a single very generic word
        GENERIC_FIRSTS = {
            'soup', 'sauce', 'drink', 'oil', 'fat', 'flour',
            'bread', 'cake', 'candy', 'nuts', 'cheese', 'milk',
            'juice', 'beer', 'wine', 'salt', 'sugar',
        }
        if len(parts) >= 2 and text.lower() in GENERIC_FIRSTS:
            text = parts[1]
    words   = [w for w in text.split() if w.lower() not in {"prepared", "entree"}]
    cleaned = " ".join(words).strip(" ,.-")
    return cleaned.title() if cleaned else "This food"


def _format_rag_sources(rag_results: list) -> str:
    if not rag_results:
        return ""
    seen, ordered = set(), []
    for item in rag_results:
        src = (item.get("source") or "").strip()
        if src and src not in seen:
            seen.add(src)
            ordered.append(src)
    return "Sources: " + "; ".join(ordered[:3]) if ordered else ""


def _friendly_factor_text(shap_reasons: list) -> list[str]:
    lines = []
    for feature, value in shap_reasons[:3]:
        direction = "increases concern" if value > 0 else "slightly lowers concern"
        lines.append(f"{feature.title()} {direction}")
    return lines


def _plain_recommendation(label: str, food_name: str, condition: str | None = None) -> str:
    condition_text = (condition or "your condition").replace("_", " ")
    audience_text  = (
        "someone without a known medical condition"
        if condition_text == "healthy"
        else f"someone with {condition_text}"
    )
    if label == "avoid":
        return f"{food_name} is best avoided for {audience_text}."
    if label == "moderate":
        return f"{food_name} is better as an occasional, small-portion choice for {audience_text}."
    if label == "safe":
        return f"{food_name} is generally a reasonable choice for {audience_text}."
    return f"Be cautious with {food_name} for {audience_text}."


# ──────────────────────────────────────────────────────────────────────────────
# BUILD PROMPT
# Constructs the prompt sent to the LLM for the explanation paragraph.
# The ML model verdict is already decided — the LLM only explains WHY.
# Including patient context (BMI, CKD stage, activity) enables condition-specific
# explanations instead of generic diet advice.
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(
    entities:       dict,
    food_nutrition: dict,
    ml_label:       str,
    ml_confidence:  float,
    shap_reasons:   list,
    rag_results:    list,
) -> str:

    food_name = _humanize_food_name(
        food_nutrition.get('canonical_name') or
        food_nutrition.get('food_name') or
        entities.get('food', 'this food'),
        user_query=entities.get('food')   # prefer what the user typed
    )

    age       = entities.get('age', 'unknown')
    condition = (entities.get('condition') or 'unknown').replace('_', ' ')
    found     = food_nutrition.get('found', True)

    # ── Nutrient values ───────────────────────────────────────────
    sugar      = food_nutrition.get('sugar_g',      '?')
    sodium     = food_nutrition.get('sodium_mg',    '?')
    gi         = food_nutrition.get('gi_value',     '?')
    potassium  = food_nutrition.get('potassium_mg', '?')
    phosphorus = food_nutrition.get('phosphorus_mg','?')
    protein    = food_nutrition.get('protein_g',    '?')
    food_cat   = food_nutrition.get('food_category') or ''
    prep_state = food_nutrition.get('preparation_state') or ''

    # ── Patient context ───────────────────────────────────────────
    # Use (val or '') pattern everywhere — handles both missing keys AND
    # keys that exist but are explicitly None (new optional fields default to None)
    gender         = entities.get('gender') or None
    ckd_stage      = entities.get('ckd_stage') or None
    activity       = (entities.get('activity_level') or '').replace('_', ' ')
    comorbidity    = (entities.get('comorbidity') or '').replace('_', ' ')
    medication     = (entities.get('medication') or '').replace('_', ' ')
    bmi            = entities.get('bmi') or None
    bmi_category   = entities.get('bmi_category') or None
    dialysis_type  = entities.get('dialysis_type') or None
    diabetes_type  = entities.get('diabetes_type') or None

    # ── Build patient context sentence ────────────────────────────
    patient_parts = [f"A {age}-year-old"]
    if gender:
        patient_parts.append(gender)
    patient_parts.append(f"patient with {condition}")
    if ckd_stage:
        patient_parts.append(f"(stage {ckd_stage})")
    if dialysis_type and dialysis_type != 'none':
        patient_parts.append(f"on {dialysis_type.replace('_', ' ')} dialysis")
    if comorbidity and comorbidity not in ('none', condition):
        patient_parts.append(f"and {comorbidity}")
    if medication and medication != 'none':
        patient_parts.append(f"taking {medication.replace('_', ' ')}")
    if bmi_category and bmi_category != 'normal':
        patient_parts.append(f"with {bmi_category} BMI")
    if activity:
        patient_parts.append(f"({activity})")
    patient_context = " ".join(patient_parts)

    # ── Nutrition summary ─────────────────────────────────────────
    if found:
        nutrition_parts = []
        if condition == 'kidney disease':
            # CKD: lead with potassium and phosphorus — the most critical nutrients
            nutrition_parts.append(f"{potassium}mg potassium")
            nutrition_parts.append(f"{phosphorus}mg phosphorus")
            nutrition_parts.append(f"{sodium}mg sodium")
            nutrition_parts.append(f"{protein}g protein")
        elif condition == 'diabetes':
            nutrition_parts.append(f"glycemic index of {gi}")
            nutrition_parts.append(f"{sugar}g sugar")
            nutrition_parts.append(f"{sodium}mg sodium")
        elif condition == 'hypertension':
            nutrition_parts.append(f"{sodium}mg sodium")
            nutrition_parts.append(f"{potassium}mg potassium")
        else:
            nutrition_parts.append(f"{sugar}g sugar")
            nutrition_parts.append(f"{sodium}mg sodium")
            nutrition_parts.append(f"GI of {gi}")
        nutrition_note = f"It contains {', '.join(nutrition_parts)} per 100g."
        if prep_state and isinstance(prep_state, str):
            nutrition_note += f" ({prep_state.capitalize()} form.)"
    else:
        nutrition_note = "Exact nutritional data was not found for this food."

    # ── RAG guideline hint ────────────────────────────────────────
    guideline_hint = ""
    if rag_results:
        first = rag_results[0]['text'].split('.')[0].strip()
        if len(first) > 20:
            guideline_hint = first + "."

    # ── Top SHAP factor ───────────────────────────────────────────
    top_reason = ""
    if shap_reasons:
        top_feat, top_val = shap_reasons[0]
        if top_val > 0:
            top_reason = f"particularly its {top_feat}"

    # ── Verdict phrase ────────────────────────────────────────────
    verdict_phrase = {
        'avoid':    f"should avoid {food_name}",
        'moderate': f"should eat {food_name} only in small portions occasionally",
        'safe':     f"can generally include {food_name} in their diet",
    }.get(ml_label, f"should be cautious about {food_name}")

    prompt = (
        f"{patient_context} {verdict_phrase}. "
        f"{nutrition_note} "
        f"{guideline_hint} "
        f"In 2 clear sentences, explain the specific nutritional reason why "
        f"this verdict applies to this patient's condition"
        f"{(' — ' + top_reason) if top_reason else ''}. "
        f"Be specific to {condition}, not generic."
    )

    return prompt


# ──────────────────────────────────────────────────────────────────────────────
# BUILD FINAL RESPONSE
# Python controls the verdict sentence entirely.
# The LLM only provides the explanation paragraph.
# This separation prevents the LLM from contradicting the ML model verdict.
#
# Now accepts proba_dict (4th return value from predict_suitability) to show
# all class probabilities, not just the top prediction confidence.
# ──────────────────────────────────────────────────────────────────────────────

def build_final_response(
    ml_label:       str,
    ml_confidence:  float,
    shap_reasons:   list,
    food_name:      str,
    condition:      str | None,
    rag_results:    list,
    llm_explanation:str,
    proba_dict:     dict | None = None,   # NEW — {'safe':x, 'moderate':y, 'avoid':z}
    patient_context:dict | None = None,   # NEW — for contextual footer
) -> str:

    # food_name here is the canonical/DB name; user_query is passed separately
    # If caller passed a user_query via patient_context, prefer it
    user_query = (patient_context or {}).get('food') if patient_context else None
    pretty_food   = _humanize_food_name(food_name, user_query=user_query)
    recommendation= _plain_recommendation(ml_label, pretty_food, condition)
    factor_lines  = _friendly_factor_text(shap_reasons)
    citations     = _format_rag_sources(rag_results)
    disclaimer    = "Please consult your doctor before making any dietary changes."

    # ── Confidence display ────────────────────────────────────────
    # Show all class probabilities if available — more informative than one number
    if proba_dict:
        proba_parts = []
        for cls in ['safe', 'moderate', 'avoid']:
            if cls in proba_dict:
                emoji = {'safe': '🟢', 'moderate': '🟡', 'avoid': '🔴'}.get(cls, '')
                proba_parts.append(f"{emoji} {cls.capitalize()}: {proba_dict[cls]}%")
        confidence_text = "Confidence breakdown: " + "  |  ".join(proba_parts)
    else:
        confidence_text = f"Confidence: {ml_confidence}%"

    # ── Patient context note ──────────────────────────────────────
    # If optional fields were captured, acknowledge them subtly
    context_note = ""
    if patient_context:
        noted = []
        if patient_context.get('ckd_stage'):
            noted.append(f"CKD Stage {patient_context['ckd_stage']}")
        if patient_context.get('comorbidity') and patient_context['comorbidity'] != 'none':
            noted.append(patient_context['comorbidity'].replace('_', ' '))
        if patient_context.get('medication') and patient_context['medication'] != 'none':
            noted.append(f"{patient_context['medication'].replace('_', ' ')} medication")
        if patient_context.get('activity_level'):
            noted.append(f"{patient_context['activity_level'].replace('_', ' ')} lifestyle")
        if noted:
            context_note = f"(Based on your profile: {', '.join(noted)})"

    # ── Assemble response parts ───────────────────────────────────
    parts = [
        f"Recommendation: {recommendation}",
        "",
        llm_explanation.strip(),
    ]

    if context_note:
        parts.extend(["", context_note])

    if factor_lines:
        parts.extend(["", "Top factors:"])
        parts.extend([f"  {'🔴' if 'increases' in line else '🟢'} {line}"
                      for line in factor_lines])

    if citations:
        parts.extend(["", f"Evidence: {citations.replace('Sources: ', '')}"])

    parts.extend(["", confidence_text, disclaimer])

    return "\n".join(parts)