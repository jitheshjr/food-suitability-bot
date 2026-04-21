import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ──────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "model")
DIAGRAM_DIR = os.path.join(MODEL_DIR, "diagrams")
DATA_DIR    = os.path.join(BASE_DIR, "..", "data", "processed")

os.makedirs(DIAGRAM_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + ENCODERS
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

# ──────────────────────────────────────────────────────────────────────────────
# LOAD + ENCODE DATASET
# Must apply exactly the same encoding as train.py — use saved encoders
# ──────────────────────────────────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(os.path.join(DATA_DIR, "labeled_dataset.csv"))
print(f"Dataset shape: {df.shape}")

# Apply all encoders (same order as train.py)
df['condition_enc']         = condition_encoder.transform(df['condition'])
df['gender_enc']            = gender_encoder.transform(df['gender'].fillna('male'))
df['bmi_category_enc']      = bmi_encoder.transform(df[['bmi_category']].fillna('normal'))
df['activity_enc']          = activity_encoder.transform(df[['activity_level']].fillna('lightly_active'))
df['ckd_stage_enc']         = df['ckd_stage'].fillna(0).astype(int)
df['comorbidity_enc']       = comorbidity_encoder.transform(df['comorbidity'].fillna('none'))
df['medication_enc']        = medication_encoder.transform(df['medication'].fillna('none'))
df['food_category_enc']     = food_category_encoder.transform(df['food_category'].fillna('other'))
df['preparation_state_enc'] = prep_encoder.transform(df['preparation_state'].fillna('unknown'))
df['gi_category_enc']       = gi_cat_encoder.transform(df[['gi_category']].fillna('unknown'))
df['dialysis_type_enc']     = dialysis_encoder.transform(df['dialysis_type'].fillna('none'))
df['diabetes_type_enc']     = diabetes_encoder.transform(df['diabetes_type'].fillna('none'))
df['source_enc']            = source_encoder.transform(df['source'].fillna('none'))
df['is_composite_enc']      = df['is_composite_dish'].fillna(False).astype(int)

# Fill numeric NaN with medians
for col in feature_cols:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Sample for SHAP — 500 rows is sufficient and keeps it fast
X = df[feature_cols].sample(500, random_state=42)
y_sample = df.loc[X.index, 'label']

print(f"\nSampled {len(X)} rows for SHAP analysis")
print(f"Label distribution in sample:\n{y_sample.value_counts()}")

# ──────────────────────────────────────────────────────────────────────────────
# COMPUTE SHAP VALUES
# ──────────────────────────────────────────────────────────────────────────────

print("\nComputing SHAP values (may take 30-90 seconds with more features)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# shap_values: list of 3 arrays (one per class) each shape (500, n_features)
# or a single 3D array (500, n_features, 3) depending on shap version
# Normalise to 3D array (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    shap_array = np.stack(shap_values, axis=2)   # (500, n_features, 3)
else:
    shap_array = np.array(shap_values)

print(f"SHAP array shape: {shap_array.shape}")
n_samples, n_features, n_classes = shap_array.shape

# Human-readable feature names for plot axes
READABLE = {
    'age': 'Age', 'bmi': 'BMI', 'height_cm': 'Height', 'weight_kg': 'Weight',
    'condition_enc': 'Condition', 'gender_enc': 'Gender',
    'bmi_category_enc': 'BMI Category', 'activity_enc': 'Activity Level',
    'ckd_stage_enc': 'CKD Stage', 'comorbidity_enc': 'Comorbidity',
    'medication_enc': 'Medication', 'dialysis_type_enc': 'Dialysis Type',
    'diabetes_type_enc': 'Diabetes Type',
    'calories': 'Calories', 'protein_g': 'Protein', 'fat_g': 'Fat',
    'carbs_g': 'Carbs', 'sugar_g': 'Sugar', 'fiber_g': 'Fiber',
    'sodium_mg': 'Sodium', 'potassium_mg': 'Potassium',
    'phosphorus_mg': 'Phosphorus',
    'gi_value': 'GI Value', 'gi_category_enc': 'GI Category',
    'food_category_enc': 'Food Category',
    'preparation_state_enc': 'Preparation',
    'source_enc': 'Food Source', 'is_composite_enc': 'Composite Dish',
}
readable_labels = [READABLE.get(f, f) for f in feature_cols]

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1: OVERALL SHAP SUMMARY  (mean |SHAP| across all classes)
# ──────────────────────────────────────────────────────────────────────────────

shap_mean = np.abs(shap_array).mean(axis=2)      # (500, n_features)
mean_importance = shap_mean.mean(axis=0)          # (n_features,)

feat_importance = sorted(
    zip(readable_labels, mean_importance),
    key=lambda x: x[1]
)
labels_sorted, values_sorted = zip(*feat_importance)

fig, ax = plt.subplots(figsize=(9, max(6, n_features * 0.35)))
bars = ax.barh(labels_sorted, values_sorted, color='steelblue')
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Overall SHAP Feature Importance (averaged across all classes)")

# Annotate bars with values
for bar, val in zip(bars, values_sorted):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=7)

plt.tight_layout()
summary_path = os.path.join(DIAGRAM_DIR, "shap_summary.png")
plt.savefig(summary_path, dpi=150)
print(f"\nSaved: {summary_path}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2: PER-CLASS SHAP IMPORTANCE
# ──────────────────────────────────────────────────────────────────────────────

class_names  = label_encoder.classes_   # ['avoid', 'moderate', 'safe']
class_colors = {
    'avoid':    '#e74c3c',
    'moderate': '#f39c12',
    'safe':     '#2ecc71',
}

for i, class_name in enumerate(class_names):
    class_shap  = shap_array[:, :, i]              # (500, n_features)
    mean_vals   = np.abs(class_shap).mean(axis=0)  # (n_features,)

    feat_imp = sorted(
        zip(readable_labels, mean_vals),
        key=lambda x: x[1]
    )
    feats, vals = zip(*feat_imp)

    fig, ax = plt.subplots(figsize=(9, max(6, n_features * 0.35)))
    bars = ax.barh(feats, vals, color=class_colors.get(class_name, 'steelblue'))
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"SHAP Feature Importance — Class: '{class_name}'")

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=7)

    plt.tight_layout()
    path = os.path.join(DIAGRAM_DIR, f"shap_{class_name}.png")
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3: CONDITION-STRATIFIED SHAP SUMMARY
# Shows which features matter most specifically for each medical condition.
# This is the most clinically useful plot — validates CKD vs diabetes features.
# ──────────────────────────────────────────────────────────────────────────────

print("\nGenerating per-condition SHAP plots...")

df_sample = df.loc[X.index].copy()
conditions = condition_encoder.classes_

fig, axes = plt.subplots(
    1, len(conditions),
    figsize=(5 * len(conditions), max(6, n_features * 0.35)),
    sharey=True
)
if len(conditions) == 1:
    axes = [axes]

for ax, cond in zip(axes, conditions):
    cond_enc = condition_encoder.transform([cond])[0]
    cond_mask = df_sample['condition_enc'] == cond_enc
    n_cond = cond_mask.sum()

    if n_cond == 0:
        ax.set_title(f"{cond}\n(no samples)")
        continue

    # Mean |SHAP| across all classes for this condition's rows
    cond_shap = np.abs(shap_array[cond_mask]).mean(axis=2).mean(axis=0)

    feat_imp = sorted(zip(readable_labels, cond_shap), key=lambda x: x[1])
    feats, vals = zip(*feat_imp)

    ax.barh(feats, vals, color='steelblue')
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"{cond}\n(n={n_cond})")

plt.suptitle("SHAP Feature Importance by Medical Condition", fontsize=13, y=1.01)
plt.tight_layout()
cond_path = os.path.join(DIAGRAM_DIR, "shap_by_condition.png")
plt.savefig(cond_path, dpi=150, bbox_inches='tight')
print(f"Saved: {cond_path}")
plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES (overall)")
print("=" * 60)
ranked = sorted(zip(readable_labels, mean_importance), key=lambda x: x[1], reverse=True)
for rank, (feat, val) in enumerate(ranked[:10], 1):
    print(f"  {rank:2d}. {feat:<28} {val:.5f}")

print("\nAll SHAP diagrams saved to:", DIAGRAM_DIR)
print("SHAP analysis complete.")