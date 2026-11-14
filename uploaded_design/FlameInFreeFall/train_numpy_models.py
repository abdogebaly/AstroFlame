import json
import os
import pandas as pd
import numpy as np

os.makedirs("models", exist_ok=True)

print("="*60)
print("TRAINING PURE-NUMPY SPACE COMBUSTION MODELS")
print("="*60)

# ==========================================
# 1. BASS-II SPACECRAFT FIRE SAFETY MODEL
# ==========================================
df_b = pd.read_csv("cleaned_data/bass_ii_clean.csv")

# Feature preparation
# Numerical features
num_feats = ['thickness_mm', 'width_cm', 'fan_display', 'o2_initial']
X_num = df_b[num_feats].copy()
for col in num_feats:
    X_num[col] = X_num[col].fillna(X_num[col].median())

# One-hot encoding for material_type
mat_dummies = pd.get_dummies(df_b['material_type'], prefix='mat', drop_first=False, dtype=float)
geom_dummies = pd.get_dummies(df_b['geometry'], prefix='geom', drop_first=False, dtype=float)

X = pd.concat([X_num, mat_dummies, geom_dummies], axis=1)
feature_names = X.columns.tolist()

# Standardize numerical features
means = X[num_feats].mean()
stds = X[num_feats].std().replace(0, 1)
X_norm = X.copy()
X_norm[num_feats] = (X[num_feats] - means) / stds

# Add bias term
X_mat = np.hstack([np.ones((len(X), 1)), X_norm.values])
y_ext = df_b['extinguished'].values.astype(float)
y_co = df_b['delta_co'].fillna(0).values.astype(float)

# Logistic Regression for Extinction (via Gradient Descent with L2 regularization)
np.random.seed(42)
weights_ext = np.zeros(X_mat.shape[1])
lr = 0.05
l2_lambda = 0.01

for epoch in range(1500):
    logits = np.dot(X_mat, weights_ext)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25, 25)))
    grad = np.dot(X_mat.T, (probs - y_ext)) / len(y_ext) + l2_lambda * weights_ext
    weights_ext -= lr * grad

# Evaluate Extinction Model
final_probs = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X_mat, weights_ext), -25, 25)))
preds_ext = (final_probs >= 0.5).astype(int)
acc = np.mean(preds_ext == y_ext)
tp = np.sum((preds_ext == 1) & (y_ext == 1))
fp = np.sum((preds_ext == 1) & (y_ext == 0))
fn = np.sum((preds_ext == 0) & (y_ext == 1))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n[BASS-II] Extinction Classification Model:")
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1:.3f}")

# Linear Regression for Toxic CO (Ridge Regression via Normal Equation)
# w = (X^T X + lambda I)^-1 X^T y
I = np.eye(X_mat.shape[1])
I[0, 0] = 0 # do not regularize bias
ridge_lambda = 1.0
weights_co = np.linalg.inv(X_mat.T @ X_mat + ridge_lambda * I) @ X_mat.T @ y_co
preds_co = np.maximum(0, X_mat @ weights_co)
r2_co = 1.0 - (np.sum((y_co - preds_co)**2) / np.sum((y_co - np.mean(y_co))**2))
mae_co = np.mean(np.abs(y_co - preds_co))

print(f"\n[BASS-II] Toxic CO Regressor Model:")
print(f"R2 Score: {r2_co:.3f}")
print(f"MAE:      {mae_co:.2f} ppm")

# ==========================================
# 2. FLEX DROPLET BURNING RATE MODEL
# ==========================================
df_f = pd.read_csv("cleaned_data/flex_clean.csv")
flex_num = ['pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he', 'd0_mm']
X_f_num = df_f[flex_num].copy()
for col in flex_num:
    X_f_num[col] = X_f_num[col].fillna(X_f_num[col].median())

fuel_dummies = pd.get_dummies(df_f['fuel'], prefix='fuel', drop_first=False, dtype=float)
X_f = pd.concat([X_f_num, fuel_dummies], axis=1)
flex_features = X_f.columns.tolist()

means_f = X_f[flex_num].mean()
stds_f = X_f[flex_num].std().replace(0, 1)
X_f_norm = X_f.copy()
X_f_norm[flex_num] = (X_f[flex_num] - means_f) / stds_f

X_f_mat = np.hstack([np.ones((len(X_f), 1)), X_f_norm.values])
y_k = df_f['burning_rate'].values

I_f = np.eye(X_f_mat.shape[1])
I_f[0, 0] = 0
weights_k = np.linalg.inv(X_f_mat.T @ X_f_mat + 0.5 * I_f) @ X_f_mat.T @ y_k
preds_k = X_f_mat @ weights_k
r2_k = 1.0 - (np.sum((y_k - preds_k)**2) / np.sum((y_k - np.mean(y_k))**2))
mae_k = np.mean(np.abs(y_k - preds_k))

print(f"\n[FLEX] Droplet Burning Rate (K) Model:")
print(f"R2 Score: {r2_k:.3f}")
print(f"MAE:      {mae_k:.4f} mm^2/s")

# Save model parameters to JSON for universal deployment
model_export = {
    'bass_model': {
        'features': feature_names,
        'means': means.to_dict(),
        'stds': stds.to_dict(),
        'weights_extinction': weights_ext.tolist(),
        'weights_co': weights_co.tolist(),
        'metrics': {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'r2_co': float(r2_co),
            'mae_co': float(mae_co)
        }
    },
    'flex_model': {
        'features': flex_features,
        'means': means_f.to_dict(),
        'stds': stds_f.to_dict(),
        'weights_k': weights_k.tolist(),
        'metrics': {
            'r2_k': float(r2_k),
            'mae_k': float(mae_k)
        }
    }
}

with open("models/model_weights.json", "w") as f:
    json.dump(model_export, f, indent=2)

print("\nModel weights and metadata exported to models/model_weights.json successfully!")
