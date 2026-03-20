import os
import joblib
import pandas as pd
import numpy as np
import shap

# Always resolve paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")


# Load everything once at import time
model             = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
label_encoder     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
condition_encoder = joblib.load(os.path.join(MODEL_DIR, "condition_encoder.pkl"))
feature_cols      = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
explainer         = shap.TreeExplainer(model)

READABLE_NAMES = {
    'age':        'patient age',
    'condition_enc': 'medical condition',
    'calories':   'calories',
    'protein_g':  'protein',
    'fat_g':      'fat content',
    'carbs_g':    'carbohydrates',
    'sugar_g':    'sugar content',
    'fiber_g':    'fiber content',
    'sodium_mg':  'sodium content',
    'gi_value':   'glycemic index',
}

def predict_suitability(patient: dict, food: dict):
    """
    patient: {'age': 64, 'condition': 'diabetes'}
    food:    {'calories':207, 'protein_g':3.5, 'fat_g':11,
               'carbs_g':24, 'sugar_g':21, 'fiber_g':0.7,
               'sodium_mg':80, 'gi_value':51}

    Returns: (label_string, top_reasons_list)
    """
    # Encode condition
    try:
        condition_enc = condition_encoder.transform([patient['condition']])[0]
    except ValueError:
        condition_enc = 0  # default to first class if unknown

    # Build feature row in correct order
    row = {
        'age':            patient.get('age', 40),
        'condition_enc':  condition_enc,
        'calories':       food.get('calories', 0),
        'protein_g':      food.get('protein_g', 0),
        'fat_g':          food.get('fat_g', 0),
        'carbs_g':        food.get('carbs_g', 0),
        'sugar_g':        food.get('sugar_g', 0),
        'fiber_g':        food.get('fiber_g', 0),
        'sodium_mg':      food.get('sodium_mg', 0),
        'gi_value':       food.get('gi_value', 50),
    }

    X = pd.DataFrame([row])[feature_cols]

    # Predict
    pred_enc   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    label      = label_encoder.inverse_transform([pred_enc])[0]
    confidence = round(float(pred_proba.max()) * 100, 1)

    # SHAP explanation — top 3 contributing features
    shap_vals = np.array(explainer.shap_values(X))  # shape (1, 10, 3)
    # For multiclass, shap_vals is a list — pick the predicted class
    sv = shap_vals[0, :, pred_enc]  # features for predicted class
    contributions = list(zip(feature_cols, sv))
    # Sort by absolute SHAP value descending
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_reasons = [
        (READABLE_NAMES.get(f, f), round(float(v), 3))
        for f, v in contributions[:3]
    ]

    return label, confidence, top_reasons


if __name__ == "__main__":
    # Quick test
    patient = {'age': 64, 'condition': 'diabetes'}
    food = {
        'calories': 207, 'protein_g': 3.5, 'fat_g': 11,
        'carbs_g': 24, 'sugar_g': 21, 'fiber_g': 0.7,
        'sodium_mg': 80, 'gi_value': 51
    }

    label, confidence, reasons = predict_suitability(patient, food)

    print(f"\nFood suitability prediction")
    print(f"Patient : age {patient['age']}, {patient['condition']}")
    print(f"Verdict : {label.upper()} ({confidence}% confidence)")
    print(f"Top reasons:")
    for feature, value in reasons:
        direction = "increases risk" if value > 0 else "reduces risk"
        print(f"  - {feature}: {direction} (SHAP={value:+.3f})")
