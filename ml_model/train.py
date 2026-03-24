import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

print("="*50)
print("FOOD SUITABILITY ML MODEL TRAINING")
print("="*50)

# ── 1. Load data ──────────────────────────────────
df = pd.read_csv("../data/processed/labeled_dataset.csv")
print(f"\nLoaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Encode condition (text → number) ───────────
condition_encoder = LabelEncoder()
df['condition_enc'] = condition_encoder.fit_transform(df['condition'])
print(f"\nConditions encoded: {dict(zip(condition_encoder.classes_, condition_encoder.transform(condition_encoder.classes_)))}")

# ── 3. Define features and target ─────────────────
FEATURE_COLS = [
    'age', 'condition_enc',
    'calories', 'protein_g', 'fat_g', 'carbs_g',
    'sugar_g', 'fiber_g', 'sodium_mg', 'gi_value'
]

X = df[FEATURE_COLS]
y = df['label']

# Encode target label
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
print(f"\nLabel classes: {list(label_encoder.classes_)}")
print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# ── 4. Train/test split ───────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── 5. Compute class weights ──────────────────────
# Handles class imbalance — gives more weight to minority classes
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# ── 6. Train XGBoost ──────────────────────────────
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    verbosity=0
)

model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("Training complete.")

# ── 7. Evaluate ───────────────────────────────────
y_pred = model.predict(X_test)

y_test_labels = label_encoder.inverse_transform(y_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test_labels, y_pred_labels))

# ── 8. Confusion matrix plot ──────────────────────
cm = confusion_matrix(y_test_labels, y_pred_labels,
                       labels=['safe','moderate','avoid'])
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['safe','moderate','avoid'],
            yticklabels=['safe','moderate','avoid'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/diagrams/confusion_matrix.png")
print("\nConfusion matrix saved to model/confusion_matrix.png")

# ── 9. Feature importance plot ────────────────────
importances = model.feature_importances_
feat_df = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance': importances
}).sort_values('importance', ascending=True)

feat_df.plot(kind='barh', x='feature', y='importance',
             legend=False, color='steelblue')
plt.title("Feature importances")
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("model/diagrams/feature_importance.png")
print("Feature importance plot saved to model/feature_importance.png")

# ── 10. Cross-validation ──────────────────────────
print("\nRunning 5-fold cross-validation...")
cv_scores = cross_val_score(model, X, y_enc, cv=5, scoring='f1_macro')
print(f"CV F1 scores: {cv_scores.round(3)}")
print(f"Mean CV F1:   {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ── 11. Save everything ───────────────────────────
os.makedirs("model", exist_ok=True)

joblib.dump(model,             "model/xgb_model.pkl")
joblib.dump(label_encoder,     "model/label_encoder.pkl")
joblib.dump(condition_encoder, "model/condition_encoder.pkl")
joblib.dump(FEATURE_COLS,      "model/feature_cols.pkl")

print("\nModel and encoders saved to ml_model/model/")
print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
