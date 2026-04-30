import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

foods = pd.read_csv("processed/foods_with_gi.csv")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

CONDITIONS      = ['diabetes', 'hypertension', 'kidney_disease', 'healthy']
ACTIVITY_LEVELS = ['sedentary', 'lightly_active', 'moderately_active', 'very_active']
CKD_STAGES      = [1, 2, 3, 4, 5]   # 5 = dialysis
GENDERS         = ['male', 'female']

MEDICATIONS = {
    'kidney_disease': ['ace_inhibitor', 'arb', 'phosphate_binder', 'diuretic', 'none'],
    'hypertension':   ['ace_inhibitor', 'diuretic', 'beta_blocker', 'arb', 'none'],
    'diabetes':       ['insulin', 'metformin', 'sglt2_inhibitor', 'none'],
    'healthy':        ['none'],
}

COMORBIDITY_PROFILES = {
    'kidney_disease': {
        'options': ['diabetes', 'hypertension', 'anemia', 'none'],
        'weights': [35, 60, 25, 20],
    },
    'diabetes': {
        'options': ['hypertension', 'obesity', 'kidney_disease', 'none'],
        'weights': [50, 40, 20, 30],
    },
    'hypertension': {
        'options': ['diabetes', 'obesity', 'none'],
        'weights': [30, 35, 50],
    },
    'healthy': {
        'options': ['none'],
        'weights': [100],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# PATIENT PROFILE GENERATOR
# ──────────────────────────────────────────────────────────────────────────────

def generate_patient(condition: str) -> dict:
    gender = random.choice(GENDERS)
    age    = random.randint(18, 80)

    if gender == 'male':
        height_cm = random.randint(158, 195)
    else:
        height_cm = random.randint(148, 180)

    if condition in ['diabetes', 'hypertension']:
        weight_kg = round(random.uniform(65, 135), 1)
    elif condition == 'kidney_disease':
        weight_kg = round(random.uniform(50, 120), 1)
    else:
        weight_kg = round(random.uniform(48, 100), 1)

    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)

    if bmi < 18.5:    bmi_category = 'underweight'
    elif bmi < 25.0:  bmi_category = 'normal'
    elif bmi < 30.0:  bmi_category = 'overweight'
    else:             bmi_category = 'obese'

    if condition in ['kidney_disease', 'diabetes']:
        activity = random.choices(ACTIVITY_LEVELS, weights=[40, 30, 20, 10], k=1)[0]
    else:
        activity = random.choices(ACTIVITY_LEVELS, weights=[20, 30, 30, 20], k=1)[0]

    if condition == 'kidney_disease':
        ckd_stage = random.choices(CKD_STAGES, weights=[5, 20, 35, 25, 15], k=1)[0]
        dialysis_type = random.choice(['hemodialysis', 'peritoneal', 'none']) \
                        if ckd_stage == 5 else None
    else:
        ckd_stage     = None
        dialysis_type = None

    if condition == 'diabetes':
        diabetes_type = random.choices(['type1', 'type2'], weights=[10, 90], k=1)[0]
    else:
        diabetes_type = None

    profile     = COMORBIDITY_PROFILES[condition]
    total       = sum(profile['weights'])
    norm_w      = [w / total for w in profile['weights']]
    comorbidity = random.choices(profile['options'], weights=norm_w, k=1)[0]
    medication  = random.choice(MEDICATIONS[condition])

    return {
        'age':            age,
        'gender':         gender,
        'height_cm':      height_cm,
        'weight_kg':      weight_kg,
        'bmi':            bmi,
        'bmi_category':   bmi_category,
        'condition':      condition,
        'ckd_stage':      ckd_stage,
        'dialysis_type':  dialysis_type,
        'diabetes_type':  diabetes_type,
        'comorbidity':    comorbidity,
        'medication':     medication,
        'activity_level': activity,
    }


# ──────────────────────────────────────────────────────────────────────────────
# SCORING FUNCTION
# All nutrients per 100g (USDA standard).
# ──────────────────────────────────────────────────────────────────────────────

def compute_label(
    age, gender, bmi, bmi_category, condition, activity,
    comorbidity, medication, ckd_stage, dialysis_type, diabetes_type,
    cal, protein, fat, carbs, sugar, fiber,
    sodium, potassium, phosphorus, gi,
    preparation_state, food_category,
) -> str:

    score = 0.0

    # ── Preparation state modifier ────────────────────────────────
    # Canned/processed foods: higher sodium concern
    if preparation_state in ['canned', 'pickled', 'smoked']:
        if condition in ['hypertension', 'kidney_disease']:
            score += 0.5
    # Raw foods: flag for vulnerable patients (food safety)
    if preparation_state == 'raw' and food_category in ['meat', 'poultry', 'seafood']:
        if condition == 'kidney_disease' or (condition == 'healthy' and age > 65):
            score += 0.5

    # ── Primary condition scoring ─────────────────────────────────

    if condition == 'diabetes':
        if gi > 70:           score += 3.0
        elif gi > 55:         score += 1.5
        if sugar > 15:        score += 2.0
        elif sugar > 8:       score += 1.0
        if fiber > 5:         score -= 1.0
        elif fiber > 3:       score -= 0.5
        if carbs > 60:        score += 1.0
        if diabetes_type == 'type1' and carbs > 40:
            score += 0.5

    elif condition == 'hypertension':
        if sodium > 600:      score += 3.0
        elif sodium > 300:    score += 1.5
        if fat > 30:          score += 2.0
        elif fat > 20:        score += 1.0
        if potassium > 1500:  score -= 1.0
        elif potassium > 800: score -= 0.5
        if sugar > 25:        score += 1.0

    elif condition == 'kidney_disease':
        stage = ckd_stage or 3

        if potassium > 2000:    score += 3.0
        elif potassium > 1000:  score += 1.5

        if phosphorus > 700:    score += 2.5
        elif phosphorus > 400:  score += 1.0

        if sodium > 400:        score += 2.0
        elif sodium > 200:      score += 0.5

        if stage < 5:
            if protein > 20:    score += 2.0
            elif protein > 10:  score += 1.0
        else:
            if protein < 6:     score += 1.5
            elif protein < 10:  score += 0.5

        if dialysis_type == 'peritoneal' and carbs > 40:
            score += 1.0
        if carbs > 60:          score += 0.5

    elif condition == 'healthy':
        if cal > 600:         score += 1.0
        if sodium > 800:      score += 1.0
        if sugar > 30:        score += 1.0
        if fat > 35:          score += 1.0
        if fiber > 5:         score -= 0.5

    # ── Comorbidity modifiers ─────────────────────────────────────

    if comorbidity == 'diabetes' and condition == 'kidney_disease':
        if gi > 55:       score += 1.5
        if sugar > 10:    score += 1.0
    if comorbidity == 'hypertension' and condition == 'kidney_disease':
        if sodium > 200:  score += 1.0
    if comorbidity == 'hypertension' and condition == 'diabetes':
        if sodium > 300:  score += 0.5
        if fat > 20:      score += 0.5
    if comorbidity == 'obesity' or bmi_category == 'obese':
        if cal > 400:     score += 0.5
        if fat > 20:      score += 0.5

    # ── Medication modifiers ──────────────────────────────────────

    if medication in ['ace_inhibitor', 'arb']:
        if potassium > 700:   score += 1.0
    if medication == 'diuretic':
        if potassium > 700:   score -= 0.5
    if medication == 'phosphate_binder':
        if phosphorus > 400:  score -= 0.5
    if medication == 'sglt2_inhibitor' and condition == 'diabetes':
        if gi > 55:           score -= 0.5

    # ── BMI modifier ──────────────────────────────────────────────

    if bmi_category == 'obese':
        if condition in ['diabetes', 'hypertension']:
            score *= 1.15
        if condition == 'kidney_disease':
            score += 0.5
    elif bmi_category == 'underweight':
        if condition == 'kidney_disease' and ckd_stage == 5:
            score -= 0.5
        if condition == 'healthy':
            score -= 0.5
    elif bmi_category == 'overweight':
        if condition in ['diabetes', 'hypertension']:
            score *= 1.05

    # ── Gender modifier ───────────────────────────────────────────

    if gender == 'female' and age >= 50:
        if condition == 'hypertension':
            score *= 1.05
        if condition == 'kidney_disease':
            score += 0.25
    if gender == 'male' and condition == 'kidney_disease':
        if ckd_stage and ckd_stage < 5 and protein > 25:
            score += 0.5

    # ── Age + condition modifier ──────────────────────────────────
    if age >= 65:
        if condition == 'diabetes' and gi > 55:
            score += 0.5
        if condition == 'kidney_disease' and potassium > 800:
            score += 0.5
        if condition == 'hypertension' and sodium > 300:
            score += 0.5

    # ── Age modifier ──────────────────────────────────────────────

    if age >= 65:
        score *= 1.20
    elif age >= 50:
        score *= 1.10
    elif age <= 25:
        score *= 0.95

    # ── Activity modifier ─────────────────────────────────────────

    if activity == 'very_active':
        score -= 0.5
    elif activity == 'moderately_active':
        score -= 0.25
    elif activity == 'sedentary':
        score += 0.5

    # ── Final label ───────────────────────────────────────────────

    if score >= 3.0:    return 'avoid'
    elif score >= 1.0:  return 'moderate'
    else:               return 'safe'


# ──────────────────────────────────────────────────────────────────────────────
# STRATIFIED SAMPLING
# ──────────────────────────────────────────────────────────────────────────────

def stratified_condition_sample(n_total: int) -> list:
    base = {
        'kidney_disease': 0.30,
        'diabetes':        0.25,
        'hypertension':    0.25,
        'healthy':         0.20,
    }
    result = []
    for condition, ratio in base.items():
        result.extend([condition] * int(n_total * ratio))
    while len(result) < n_total:
        result.append(random.choice(CONDITIONS))
    random.shuffle(result)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# MAIN GENERATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

N_ROWS = 10000
print(f"Generating {N_ROWS} synthetic patient-food records...")

condition_list = stratified_condition_sample(N_ROWS)
rows = []

def safe_get(col, default=0.0):
    val = food_row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)

def safe_str(col, default=None):
    val = food_row.get(col, default)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return str(val)

foods = foods[foods['gi_value'].notna()].reset_index(drop=True)

for condition in condition_list:
    patient  = generate_patient(condition)
    food_row = foods.sample(1).iloc[0]

    cal        = safe_get('calories')
    protein    = safe_get('protein_g')
    fat        = safe_get('fat_g')
    carbs      = safe_get('carbs_g')
    sugar      = safe_get('sugar_g')
    fiber      = safe_get('fiber_g')
    sodium     = safe_get('sodium_mg')
    potassium  = safe_get('potassium_mg')
    phosphorus = safe_get('phosphorus_mg')

    gi_raw = food_row.get('gi_value', None)
  
    gi = float(gi_raw)

    # NEW: food metadata from preprocess_foods.py
    canonical_name     = safe_str('canonical_name', food_row.get('food_name', ''))
    food_category      = safe_str('food_category', 'other')
    preparation_state  = safe_str('preparation_state')
    cooking_method     = safe_str('cooking_method')
    fat_descriptor     = safe_str('fat_descriptor')
    bone_status        = safe_str('bone_status')
    enrichment         = safe_str('enrichment')
    special_population = safe_str('special_population')
    source             = safe_str('source')
    brand              = safe_str('brand')
    cuisine            = safe_str('cuisine')
    is_composite       = bool(food_row.get('is_composite_dish', False))

    ckd_stage_val = patient['ckd_stage'] if patient['ckd_stage'] else 3

    label = compute_label(
        age=patient['age'], gender=patient['gender'],
        bmi=patient['bmi'], bmi_category=patient['bmi_category'],
        condition=condition, activity=patient['activity_level'],
        comorbidity=patient['comorbidity'], medication=patient['medication'],
        ckd_stage=ckd_stage_val,
        dialysis_type=patient['dialysis_type'],
        diabetes_type=patient['diabetes_type'],
        cal=cal, protein=protein, fat=fat, carbs=carbs,
        sugar=sugar, fiber=fiber, sodium=sodium,
        potassium=potassium, phosphorus=phosphorus, gi=gi,
        preparation_state=preparation_state,
        food_category=food_category,
    )

    rows.append({
        # ── Patient demographics ───────────────────────────────────
        'age':              patient['age'],
        'gender':           patient['gender'],
        'height_cm':        patient['height_cm'],
        'weight_kg':        patient['weight_kg'],
        'bmi':              patient['bmi'],
        'bmi_category':     patient['bmi_category'],

        # ── Clinical profile ───────────────────────────────────────
        'condition':        condition,
        'ckd_stage':        ckd_stage_val,
        'dialysis_type':    patient['dialysis_type'],
        'diabetes_type':    patient['diabetes_type'],
        'comorbidity':      patient['comorbidity'],
        'medication':       patient['medication'],
        'activity_level':   patient['activity_level'],

        # ── Food identity ──────────────────────────────────────────
        'food_name':            food_row['food_name'],   # original USDA name
        'canonical_name':       canonical_name,          # NEW: clean short name
        'food_category':        food_category,           # NEW: meat/grain/vegetable/…
        'preparation_state':    preparation_state,       # NEW: raw/cooked/canned/…
        'cooking_method':       cooking_method,          # NEW: roasted/boiled/…
        'fat_descriptor':       fat_descriptor,          # NEW: lean/lean and fat/…
        'bone_status':          bone_status,             # NEW: bone_in/boneless
        'enrichment':           enrichment,              # NEW: enriched/unenriched
        'special_population':   special_population,      # NEW: infant/toddler/None
        'source':               source,                  # NEW: restaurant/home/None
        'brand':                brand,                   # NEW: brand name or None
        'cuisine':              cuisine,                 # NEW: chinese/mexican/…
        'is_composite_dish':    is_composite,            # NEW: True for multi-ingredient
        'serving_size_g':       100,

        # ── Nutrients ──────────────────────────────────────────────
        'calories':         cal,
        'protein_g':        protein,
        'fat_g':            fat,
        'carbs_g':          carbs,
        'sugar_g':          sugar,
        'fiber_g':          fiber,
        'sodium_mg':        sodium,
        'potassium_mg':     potassium,
        'phosphorus_mg':    phosphorus,

        # ── GI ─────────────────────────────────────────────────────
        'gi_value':         gi,
        'gi_category':      food_row.get('gi_category', 'unknown'),
        'gi_match_type':    food_row.get('gi_match_type', 'unknown'),

        # ── Label ──────────────────────────────────────────────────
        'label':            label,
    })


# ──────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────────────────

df = pd.DataFrame(rows)

print("\n-- Label distribution -----------------------------------------")
counts = df['label'].value_counts()
pcts   = df['label'].value_counts(normalize=True).mul(100).round(1)
print(pd.DataFrame({'count': counts, 'pct': pcts.astype(str) + '%'}))

print("\n-- Label breakdown per condition ------------------------------")
print(pd.crosstab(df['condition'], df['label'], normalize='index')
      .mul(100).round(1).astype(str) + '%')

print("\n-- Condition distribution -------------------------------------")
print(df['condition'].value_counts())

print("\n-- Activity level distribution --------------------------------")
print(df['activity_level'].value_counts())

print("\n-- BMI category distribution ----------------------------------")
print(df['bmi_category'].value_counts())

print("\n-- Gender distribution ----------------------------------------")
print(df['gender'].value_counts())

print("\n-- Food category distribution ---------------------------------")
print(df['food_category'].value_counts())

print("\n-- Preparation state distribution -----------------------------")
print(df['preparation_state'].value_counts().head(10))

print("\n-- CKD stage distribution (kidney_disease rows only) ----------")
ckd_df = df[df['condition'] == 'kidney_disease']
print(ckd_df['ckd_stage'].value_counts().sort_index())

print("\n-- Comorbidity distribution -----------------------------------")
print(df['comorbidity'].value_counts())

print("\n-- Medication distribution ------------------------------------")
print(df['medication'].value_counts())

print(f"\nTotal rows : {len(df)}")
print(f"Columns    : {list(df.columns)}")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────

df.to_csv("processed/labeled_dataset.csv", index=False)
print("\nSaved -> processed/labeled_dataset.csv")