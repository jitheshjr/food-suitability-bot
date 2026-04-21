import os
import joblib
import pandas as pd
import numpy as np
import shap

# ──────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + ALL ENCODERS  (once at import time)
# Every encoder must match exactly what was fit and saved in train.py
# ──────────────────────────────────────────────────────────────────────────────

model                 = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
label_encoder         = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
condition_encoder     = joblib.load(os.path.join(MODEL_DIR, "condition_encoder.pkl"))
gender_encoder        = joblib.load(os.path.join(MODEL_DIR, "gender_encoder.pkl"))
bmi_encoder           = joblib.load(os.path.join(MODEL_DIR, "bmi_encoder.pkl"))
activity_encoder      = joblib.load(os.path.join(MODEL_DIR, "activity_encoder.pkl"))
comorbidity_encoder   = joblib.load(os.path.join(MODEL_DIR, "comorbidity_encoder.pkl"))
medication_encoder    = joblib.load(os.path.join(MODEL_DIR, "medication_encoder.pkl"))
food_category_encoder = joblib.load(os.path.join(MODEL_DIR, "food_category_encoder.pkl"))
prep_encoder          = joblib.load(os.path.join(MODEL_DIR, "prep_encoder.pkl"))
gi_cat_encoder        = joblib.load(os.path.join(MODEL_DIR, "gi_cat_encoder.pkl"))
dialysis_encoder      = joblib.load(os.path.join(MODEL_DIR, "dialysis_encoder.pkl"))
diabetes_encoder      = joblib.load(os.path.join(MODEL_DIR, "diabetes_encoder.pkl"))
source_encoder        = joblib.load(os.path.join(MODEL_DIR, "source_encoder.pkl"))
feature_cols          = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

explainer = shap.TreeExplainer(model)

# ──────────────────────────────────────────────────────────────────────────────
# HUMAN-READABLE FEATURE NAMES  (shown in SHAP explanations to the user)
# ──────────────────────────────────────────────────────────────────────────────

READABLE_NAMES = {
    # Patient numeric
    'age':                  'Patient Age',
    'bmi':                  'BMI',
    'height_cm':            'Height',
    'weight_kg':            'Weight',
    # Patient clinical
    'condition_enc':        'Medical Condition',
    'gender_enc':           'Gender',
    'bmi_category_enc':     'BMI Category',
    'activity_enc':         'Activity Level',
    'ckd_stage_enc':        'CKD Stage',
    'comorbidity_enc':      'Comorbidity',
    'medication_enc':       'Current Medication',
    'dialysis_type_enc':    'Dialysis Type',
    'diabetes_type_enc':    'Diabetes Type',
    # Nutrients
    'calories':             'Calories',
    'protein_g':            'Protein',
    'fat_g':                'Fat Content',
    'carbs_g':              'Carbohydrates',
    'sugar_g':              'Sugar Content',
    'fiber_g':              'Fiber Content',
    'sodium_mg':            'Sodium',
    'potassium_mg':         'Potassium',
    'phosphorus_mg':        'Phosphorus',
    # GI
    'gi_value':             'Glycemic Index',
    'gi_category_enc':      'GI Category',
    # Food metadata
    'food_category_enc':    'Food Category',
    'preparation_state_enc':'Preparation Method',
    'source_enc':           'Food Source',
    'is_composite_enc':     'Composite Dish',
}

# ──────────────────────────────────────────────────────────────────────────────
# SAFE ENCODER HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe_label_encode(encoder, value, default=0):
    """Encode a single value with a LabelEncoder; return default if unseen."""
    try:
        return int(encoder.transform([str(value)])[0])
    except (ValueError, TypeError):
        return default

def _safe_ordinal_encode(encoder, value, default=-1):
    """Encode a single value with an OrdinalEncoder; return default if unseen."""
    try:
        feature_name = getattr(encoder, "feature_names_in_", ["value"])[0]
        value_frame = pd.DataFrame([{feature_name: str(value)}])
        return float(encoder.transform(value_frame)[0][0])
    except (ValueError, TypeError):
        return default


# ──────────────────────────────────────────────────────────────────────────────
# BMI HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _compute_bmi(height_cm: float, weight_kg: float) -> tuple:
    """Returns (bmi, bmi_category) from height/weight."""
    if height_cm and weight_kg and height_cm > 0:
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    else:
        bmi = 22.0   # sensible default
    if bmi < 18.5:    cat = 'underweight'
    elif bmi < 25.0:  cat = 'normal'
    elif bmi < 30.0:  cat = 'overweight'
    else:             cat = 'obese'
    return bmi, cat


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def predict_suitability(patient: dict, food: dict) -> tuple:
    """
    Predict food suitability for a patient and return a structured result.

    Parameters
    ----------
    patient : dict
        Required keys  : 'age', 'condition'
        Optional keys  : 'gender', 'height_cm', 'weight_kg', 'bmi',
                         'bmi_category', 'activity_level', 'ckd_stage',
                         'dialysis_type', 'diabetes_type',
                         'comorbidity', 'medication'

    food : dict
        Required keys  : 'calories', 'protein_g', 'fat_g', 'carbs_g',
                         'sugar_g', 'fiber_g', 'sodium_mg'
        Optional keys  : 'potassium_mg', 'phosphorus_mg', 'gi_value',
                         'gi_category', 'food_category',
                         'preparation_state', 'source', 'is_composite_dish'

    Returns
    -------
    label       : str   — 'safe' | 'moderate' | 'avoid'
    confidence  : float — 0–100
    top_reasons : list  — [(readable_name, shap_value), ...]  top 5 features
    proba_dict  : dict  — {'safe': x, 'moderate': y, 'avoid': z}  all probabilities
    """

    # ── Patient: BMI ──────────────────────────────────────────────
    height_cm = float(patient.get('height_cm', 0) or 0)
    weight_kg = float(patient.get('weight_kg', 0) or 0)

    if patient.get('bmi'):
        bmi = float(patient['bmi'])
        bmi_category = patient.get('bmi_category', _compute_bmi(height_cm, weight_kg)[1])
    else:
        bmi, bmi_category = _compute_bmi(height_cm, weight_kg)

    # ── Patient: CKD stage ────────────────────────────────────────
    ckd_stage_raw = patient.get('ckd_stage', None)
    ckd_stage_enc = int(ckd_stage_raw) if ckd_stage_raw is not None else 0

    # ── Build feature row ─────────────────────────────────────────
    row = {
        # Patient numeric
        'age':          float(patient.get('age', 40)),
        'bmi':          bmi,
        'height_cm':    height_cm if height_cm > 0 else 165.0,
        'weight_kg':    weight_kg if weight_kg > 0 else 70.0,

        # Patient clinical
        'condition_enc':    _safe_label_encode(condition_encoder, patient.get('condition', 'healthy')),
        'gender_enc':       _safe_label_encode(gender_encoder,    patient.get('gender', 'male')),
        'bmi_category_enc': _safe_ordinal_encode(bmi_encoder,     bmi_category),
        'activity_enc':     _safe_ordinal_encode(activity_encoder, patient.get('activity_level', 'lightly_active')),
        'ckd_stage_enc':    ckd_stage_enc,
        'comorbidity_enc':  _safe_label_encode(comorbidity_encoder, patient.get('comorbidity', 'none')),
        'medication_enc':   _safe_label_encode(medication_encoder,  patient.get('medication', 'none')),
        'dialysis_type_enc':_safe_label_encode(dialysis_encoder,    patient.get('dialysis_type', 'none')),
        'diabetes_type_enc':_safe_label_encode(diabetes_encoder,    patient.get('diabetes_type', 'none')),

        # Nutrients
        'calories':     float(food.get('calories', 0) or 0),
        'protein_g':    float(food.get('protein_g', 0) or 0),
        'fat_g':        float(food.get('fat_g', 0) or 0),
        'carbs_g':      float(food.get('carbs_g', 0) or 0),
        'sugar_g':      float(food.get('sugar_g', 0) or 0),
        'fiber_g':      float(food.get('fiber_g', 0) or 0),
        'sodium_mg':    float(food.get('sodium_mg', 0) or 0),
        'potassium_mg': float(food.get('potassium_mg', 0) or 0),
        'phosphorus_mg':float(food.get('phosphorus_mg', 0) or 0),

        # GI
        'gi_value':         float(food.get('gi_value', 50) or 50),
        'gi_category_enc':  _safe_ordinal_encode(gi_cat_encoder, food.get('gi_category', 'medium')),

        # Food metadata
        'food_category_enc':    _safe_label_encode(food_category_encoder, food.get('food_category', 'other')),
        'preparation_state_enc':_safe_label_encode(prep_encoder,          food.get('preparation_state', 'unknown')),
        'source_enc':            _safe_label_encode(source_encoder,        food.get('source', 'none')),
        'is_composite_enc':      int(bool(food.get('is_composite_dish', False))),
    }

    X = pd.DataFrame([row])[feature_cols]

    # ── Predict ───────────────────────────────────────────────────
    pred_enc   = int(model.predict(X)[0])
    pred_proba = model.predict_proba(X)[0]
    label      = label_encoder.inverse_transform([pred_enc])[0]
    confidence = round(float(pred_proba.max()) * 100, 1)

    # All class probabilities
    proba_dict = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(label_encoder.classes_, pred_proba)
    }

    # ── SHAP explanation ──────────────────────────────────────────
    shap_vals = np.array(explainer.shap_values(X))  # (1, n_features, n_classes)
    sv = shap_vals[0, :, pred_enc]                  # features for predicted class
    contributions = list(zip(feature_cols, sv))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_reasons = [
        (READABLE_NAMES.get(f, f), round(float(v), 4))
        for f, v in contributions[:5]    # top 5 (was 3)
    ]

    return label, confidence, top_reasons, proba_dict


# ──────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Test 1: CKD patient — chicken pasta
    print("\n" + "=" * 55)
    print("TEST 1: CKD patient, chicken pasta")
    print("=" * 55)
    patient1 = {
        'age': 55,
        'condition': 'kidney_disease',
        'gender': 'male',
        'height_cm': 172,
        'weight_kg': 82,
        'activity_level': 'sedentary',
        'ckd_stage': 3,
        'comorbidity': 'hypertension',
        'medication': 'ace_inhibitor',
        'dialysis_type': None,
        'diabetes_type': None,
    }
    food1 = {
        'calories': 310, 'protein_g': 22, 'fat_g': 8,
        'carbs_g': 38, 'sugar_g': 3, 'fiber_g': 2,
        'sodium_mg': 420, 'potassium_mg': 380, 'phosphorus_mg': 210,
        'gi_value': 52, 'gi_category': 'low',
        'food_category': 'grain',
        'preparation_state': 'cooked',
        'source': None,
        'is_composite_dish': True,
    }
    label, conf, reasons, probas = predict_suitability(patient1, food1)
    print(f"Patient : age {patient1['age']}, {patient1['condition']} stage {patient1['ckd_stage']}, {patient1['comorbidity']}")
    print(f"Food    : Chicken Pasta")
    print(f"Verdict : {label.upper()} ({conf}% confidence)")
    print(f"Probabilities: {probas}")
    print(f"Top contributing factors:")
    for feat, val in reasons:
        direction = "increases concern" if val > 0 else "reduces concern"
        print(f"  {'🔴' if val > 0 else '🟢'} {feat}: {direction} (SHAP={val:+.4f})")

    # Test 2: Diabetic patient — white rice
    print("\n" + "=" * 55)
    print("TEST 2: Type 2 diabetic, white rice")
    print("=" * 55)
    patient2 = {
        'age': 62,
        'condition': 'diabetes',
        'gender': 'female',
        'height_cm': 158,
        'weight_kg': 78,
        'activity_level': 'lightly_active',
        'ckd_stage': None,
        'comorbidity': 'hypertension',
        'medication': 'metformin',
        'diabetes_type': 'type2',
    }
    food2 = {
        'calories': 130, 'protein_g': 2.7, 'fat_g': 0.3,
        'carbs_g': 28, 'sugar_g': 0, 'fiber_g': 0.4,
        'sodium_mg': 1, 'potassium_mg': 35, 'phosphorus_mg': 43,
        'gi_value': 73, 'gi_category': 'high',
        'food_category': 'grain',
        'preparation_state': 'cooked',
        'source': None,
        'is_composite_dish': False,
    }
    label, conf, reasons, probas = predict_suitability(patient2, food2)
    print(f"Patient : age {patient2['age']}, {patient2['condition']} ({patient2['diabetes_type']}), {patient2['comorbidity']}")
    print(f"Food    : White Rice")
    print(f"Verdict : {label.upper()} ({conf}% confidence)")
    print(f"Probabilities: {probas}")
    print(f"Top contributing factors:")
    for feat, val in reasons:
        direction = "increases concern" if val > 0 else "reduces concern"
        print(f"  {'🔴' if val > 0 else '🟢'} {feat}: {direction} (SHAP={val:+.4f})")
