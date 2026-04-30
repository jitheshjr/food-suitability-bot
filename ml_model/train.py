import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("FOOD SUITABILITY ML MODEL TRAINING")
print("=" * 60)

os.makedirs("model", exist_ok=True)
os.makedirs("model/diagrams", exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv("../data/processed/labeled_dataset.csv")
print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. ENCODE CATEGORICAL FEATURES
# All label encoders are saved to disk so predict.py can use the exact same
# mapping at inference time. NEVER re-fit encoders at inference — load them.
# ──────────────────────────────────────────────────────────────────────────────

# -- 2a. Primary condition -----------------------------------------------------
condition_encoder = LabelEncoder()
df['condition_enc'] = condition_encoder.fit_transform(df['condition'])
print(f"\nConditions: {dict(zip(condition_encoder.classes_, condition_encoder.transform(condition_encoder.classes_)))}")

# -- 2b. Gender ----------------------------------------------------------------
gender_encoder = LabelEncoder()
df['gender_enc'] = gender_encoder.fit_transform(df['gender'].fillna('male'))

# -- 2c. BMI category (ordinal — meaningful order) -----------------------------
BMI_ORDER = ['underweight', 'normal', 'overweight', 'obese']
bmi_encoder = OrdinalEncoder(categories=[BMI_ORDER], handle_unknown='use_encoded_value', unknown_value=-1)
df['bmi_category_enc'] = bmi_encoder.fit_transform(df[['bmi_category']].fillna('normal'))

# -- 2d. Activity level (ordinal — meaningful order) ---------------------------
ACTIVITY_ORDER = ['sedentary', 'lightly_active', 'moderately_active', 'very_active']
activity_encoder = OrdinalEncoder(categories=[ACTIVITY_ORDER], handle_unknown='use_encoded_value', unknown_value=-1)
df['activity_enc'] = activity_encoder.fit_transform(df[['activity_level']].fillna('lightly_active'))

# -- 2e. Comorbidity -----------------------------------------------------------
comorbidity_encoder = LabelEncoder()
df['comorbidity_enc'] = comorbidity_encoder.fit_transform(df['comorbidity'].fillna('none'))

# -- 2f. Medication ------------------------------------------------------------
medication_encoder = LabelEncoder()
df['medication_enc'] = medication_encoder.fit_transform(df['medication'].fillna('none'))

# -- 2g. Food category ---------------------------------------------------------
food_category_encoder = LabelEncoder()
df['food_category_enc'] = food_category_encoder.fit_transform(df['food_category'].fillna('other'))

# -- 2h. Preparation state -----------------------------------------------------
prep_encoder = LabelEncoder()
df['preparation_state_enc'] = prep_encoder.fit_transform(df['preparation_state'].fillna('unknown'))

# -- 2i. GI category (ordinal) -------------------------------------------------
GI_ORDER = ['low', 'medium', 'high', 'unknown']
gi_cat_encoder = OrdinalEncoder(categories=[GI_ORDER], handle_unknown='use_encoded_value', unknown_value=-1)
df['gi_category_enc'] = gi_cat_encoder.fit_transform(df[['gi_category']].fillna('unknown'))

# -- 2j. CKD stage (numeric, but NaN for non-CKD patients → fill with 0) ------
# 0 = not applicable, 1-5 = CKD stage
df['ckd_stage_enc'] = df['ckd_stage'].fillna(0).astype(int)

# -- 2k. Dialysis type ---------------------------------------------------------
dialysis_encoder = LabelEncoder()
df['dialysis_type_enc'] = dialysis_encoder.fit_transform(
    df['dialysis_type'].fillna('none')
)

# -- 2l. Diabetes type ---------------------------------------------------------
diabetes_encoder = LabelEncoder()
df['diabetes_type_enc'] = diabetes_encoder.fit_transform(
    df['diabetes_type'].fillna('none')
)

# -- 2m. Source (restaurant / home / None) -------------------------------------
source_encoder = LabelEncoder()
df['source_enc'] = source_encoder.fit_transform(df['source'].fillna('none'))

# -- 2n. is_composite_dish (bool → int) ----------------------------------------
df['is_composite_enc'] = df['is_composite_dish'].fillna(False).astype(int)

print("\nAll categorical features encoded.")

# ──────────────────────────────────────────────────────────────────────────────
# 3. DEFINE FEATURE COLUMNS
#
# Split into groups so it's easy to see exactly what the model sees.
# These must exactly match what predict.py sends at inference time.
# ──────────────────────────────────────────────────────────────────────────────

# Patient demographics
PATIENT_NUMERIC = [
    'age',
    'bmi',
    'height_cm',
    'weight_kg',
]

# Patient clinical (encoded)
PATIENT_CLINICAL = [
    'condition_enc',
    'gender_enc',
    'bmi_category_enc',
    'activity_enc',
    'ckd_stage_enc',
    'comorbidity_enc',
    'medication_enc',
    'dialysis_type_enc',
    'diabetes_type_enc',
]

# Nutrients (continuous)
NUTRIENT_COLS = [
    'calories',
    'protein_g',
    'fat_g',
    'carbs_g',
    'sugar_g',
    'fiber_g',
    'sodium_mg',
    'potassium_mg',    # NEW — critical for CKD
    'phosphorus_mg',   # NEW — critical for CKD
]

# GI
GI_COLS = [
    'gi_value',
    'gi_category_enc',
]

# Food metadata (encoded)
FOOD_META_COLS = [
    'food_category_enc',
    'preparation_state_enc',
    'source_enc',
    'is_composite_enc',
]

FEATURE_COLS = (
    PATIENT_NUMERIC +
    PATIENT_CLINICAL +
    NUTRIENT_COLS +
    GI_COLS +
    FOOD_META_COLS
)

print(f"\nTotal features: {len(FEATURE_COLS)}")
print(f"Feature groups:")
print(f"  Patient numeric  : {len(PATIENT_NUMERIC)}  {PATIENT_NUMERIC}")
print(f"  Patient clinical : {len(PATIENT_CLINICAL)}  {PATIENT_CLINICAL}")
print(f"  Nutrients        : {len(NUTRIENT_COLS)}  {NUTRIENT_COLS}")
print(f"  GI               : {len(GI_COLS)}  {GI_COLS}")
print(f"  Food metadata    : {len(FOOD_META_COLS)}  {FOOD_META_COLS}")

# Guard: check all feature cols exist in dataframe
missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"\nMissing columns in dataset: {missing_cols}\n"
        "Re-run generate_dataset.py to regenerate the dataset with all fields."
    )

# Fill any remaining NaN in numeric features with column median
for col in FEATURE_COLS:
    if df[col].isna().any():
        median = df[col].median()
        df[col] = df[col].fillna(median)
        print(f"  Filled NaN in '{col}' with median={median:.2f}")

X = df[FEATURE_COLS]
y = df['label']

# ──────────────────────────────────────────────────────────────────────────────
# 4. ENCODE TARGET LABEL
# ──────────────────────────────────────────────────────────────────────────────

label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
print(f"\nLabel classes: {list(label_encoder.classes_)}")
print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print(f"\nLabel distribution:\n{y.value_counts()}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc   # preserve class proportions in both splits
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ──────────────────────────────────────────────────────────────────────────────
# 6. CLASS WEIGHTS
# Handles imbalance — avoids model always predicting 'moderate'
# ──────────────────────────────────────────────────────────────────────────────

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# ──────────────────────────────────────────────────────────────────────────────
# 7. TRAIN XGBOOST
# Hyperparameters tuned for a ~30-feature medical classification task.
# min_child_weight=3 prevents overfitting on rare patient profiles.
# ──────────────────────────────────────────────────────────────────────────────

print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,       # lowered from 0.1 — more robust with more features
    subsample=0.8,
    colsample_bytree=0.7,     # sample 70% of features per tree — handles wider feature set
    min_child_weight=3,       # NEW — prevents overfitting on rare comorbidity combos
    gamma=0.1,                # NEW — minimum loss reduction to make a split
    reg_alpha=0.1,            # NEW — L1 regularisation
    reg_lambda=1.0,           # L2 regularisation (default, kept explicit)
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0,
    use_label_encoder=False,
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("Training complete.")

# ──────────────────────────────────────────────────────────────────────────────
# 8. EVALUATE
# ──────────────────────────────────────────────────────────────────────────────

y_pred        = model.predict(X_test)
y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test_labels, y_pred_labels))

# ──────────────────────────────────────────────────────────────────────────────
# 9. CONFUSION MATRIX PLOT
# ──────────────────────────────────────────────────────────────────────────────

cm = confusion_matrix(
    y_test_labels, y_pred_labels,
    labels=['safe', 'moderate', 'avoid']
)
plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['safe', 'moderate', 'avoid'],
    yticklabels=['safe', 'moderate', 'avoid']
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/diagrams/confusion_matrix.png")
print("Confusion matrix saved → model/diagrams/confusion_matrix.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 10. FEATURE IMPORTANCE PLOT
# ──────────────────────────────────────────────────────────────────────────────

importances = model.feature_importances_
feat_df = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': importances
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))   # taller — more features now
plt.barh(feat_df['feature'], feat_df['importance'], color='steelblue')
plt.xlabel("Importance score")
plt.title("XGBoost Feature Importances")
plt.tight_layout()
plt.savefig("model/diagrams/feature_importance.png")
print("Feature importance plot saved → model/diagrams/feature_importance.png")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# 11. PER-CONDITION PERFORMANCE BREAKDOWN
# Critical for a medical app — we need to know if the model is weak on CKD
# specifically, not just overall accuracy.
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PER-CONDITION CLASSIFICATION REPORT")
print("=" * 60)

test_df = X_test.copy()
test_df['true_label'] = y_test_labels
test_df['pred_label'] = y_pred_labels
test_df['condition']  = condition_encoder.inverse_transform(test_df['condition_enc'])

for cond in sorted(test_df['condition'].unique()):
    mask = test_df['condition'] == cond
    subset_true = test_df.loc[mask, 'true_label']
    subset_pred = test_df.loc[mask, 'pred_label']
    print(f"\n--- {cond.upper()} (n={mask.sum()}) ---")
    print(classification_report(subset_true, subset_pred, zero_division=0))

# ──────────────────────────────────────────────────────────────────────────────
# 12. CROSS-VALIDATION
# Using StratifiedKFold to preserve class balance in every fold
# ──────────────────────────────────────────────────────────────────────────────

print("\nRunning 5-fold stratified cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_enc, cv=skf, scoring='f1_macro')
print(f"CV F1 scores : {cv_scores.round(3)}")
print(f"Mean CV F1   : {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ──────────────────────────────────────────────────────────────────────────────
# 13. SAVE MODEL + ALL ENCODERS
# Every encoder that was fit here must be saved.
# predict.py loads and uses all of them — never re-fits at inference.
# ──────────────────────────────────────────────────────────────────────────────

joblib.dump(model,                "model/xgb_model.pkl")
joblib.dump(label_encoder,        "model/label_encoder.pkl")
joblib.dump(condition_encoder,    "model/condition_encoder.pkl")
joblib.dump(gender_encoder,       "model/gender_encoder.pkl")
joblib.dump(bmi_encoder,          "model/bmi_encoder.pkl")
joblib.dump(activity_encoder,     "model/activity_encoder.pkl")
joblib.dump(comorbidity_encoder,  "model/comorbidity_encoder.pkl")
joblib.dump(medication_encoder,   "model/medication_encoder.pkl")
joblib.dump(food_category_encoder,"model/food_category_encoder.pkl")
joblib.dump(prep_encoder,         "model/prep_encoder.pkl")
joblib.dump(gi_cat_encoder,       "model/gi_cat_encoder.pkl")
joblib.dump(dialysis_encoder,     "model/dialysis_encoder.pkl")
joblib.dump(diabetes_encoder,     "model/diabetes_encoder.pkl")
joblib.dump(source_encoder,       "model/source_encoder.pkl")
joblib.dump(FEATURE_COLS,         "model/feature_cols.pkl")

print("\nAll model files saved → model/")
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)