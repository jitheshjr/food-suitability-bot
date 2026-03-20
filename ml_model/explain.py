import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")

model             = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
label_encoder     = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
condition_encoder = joblib.load(os.path.join(MODEL_DIR, "condition_encoder.pkl"))
feature_cols      = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))

df = pd.read_csv(os.path.join(DATA_DIR, "labeled_dataset.csv"))

df['condition_enc'] = condition_encoder.transform(df['condition'])
X = df[feature_cols].sample(500, random_state=42)

print("Computing SHAP values (may take 30-60 seconds)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# shap_values is shape (500, 10, 3) for multiclass — one matrix per class
# Convert to numpy array if not already
shap_array = np.array(shap_values)
print(f"SHAP values shape: {shap_array.shape}")

# Plot mean absolute SHAP value across all classes
# Average across the class dimension (axis=2) → shape (500, 10)
shap_mean = np.abs(shap_array).mean(axis=2)

plt.figure(figsize=(8, 5))
mean_importance = shap_mean.mean(axis=0)
feat_importance = sorted(zip(feature_cols, mean_importance),
                          key=lambda x: x[1])
features, values = zip(*feat_importance)
plt.barh(features, values, color='steelblue')
plt.xlabel("Mean |SHAP value|")
plt.title("SHAP feature importance (averaged across classes)")
plt.tight_layout()
plt.savefig("model/shap_summary.png")
print("SHAP summary saved to model/shap_summary.png")
plt.close()

# Also save per-class SHAP plots
class_names = label_encoder.classes_  # ['avoid', 'moderate', 'safe']
for i, class_name in enumerate(class_names):
    plt.figure(figsize=(8, 5))
    class_shap = shap_array[:, :, i]  # shape (500, 10)
    mean_vals = np.abs(class_shap).mean(axis=0)
    feat_imp = sorted(zip(feature_cols, mean_vals), key=lambda x: x[1])
    feats, vals = zip(*feat_imp)
    plt.barh(feats, vals, color=['#e74c3c' if class_name=='avoid'
                                  else '#f39c12' if class_name=='moderate'
                                  else '#2ecc71'][0])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"SHAP importance for class: {class_name}")
    plt.tight_layout()
    plt.savefig(f"model/shap_{class_name}.png")
    print(f"Saved model/shap_{class_name}.png")
    plt.close()

print("\nAll SHAP plots saved successfully.")
print("SHAP setup complete.")
