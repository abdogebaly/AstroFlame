import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("="*70)
print("TRAINING & OPTIMIZING THE TOP 3 CHAMPION CLASSIFIERS")
print("="*70)

# ==============================================================================
# 1. CHAMPION MODEL 1: BASS-II SOLID FLAME EXTINCTION CLASSIFIER
# ==============================================================================
print("\n[1/3] Training Champion 1: BASS-II Solid Extinction Classifier...")
df_b = pd.read_csv("cleaned_data/bass_ii_clean.csv")

# Filter out corrupted rows where total_frames was 0 due to 'none' missing data
df_b_valid = df_b[df_b['total_frames'] > 0].copy()

features_b = ['material_type', 'geometry', 'color', 'thickness_mm', 'width_cm', 
              'flow_restrictor', 'fan_display', 'air_display', 'o2_initial']
categorical_b = ['material_type', 'geometry', 'color']
numerical_b = ['thickness_mm', 'width_cm', 'flow_restrictor', 'fan_display', 'air_display', 'o2_initial']

X_b = df_b_valid[features_b].copy()
y_b = df_b_valid['extinguished'].copy()

pipe_b = Pipeline([
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_b),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_b)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=75, max_depth=7, min_samples_split=2, random_state=42))
])

cv_b = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_b_results = cross_validate(pipe_b, X_b, y_b, cv=cv_b, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'])

pipe_b.fit(X_b, y_b)
joblib.dump(pipe_b, "models/champion_bass_extinction.joblib")

print(f"  -> 5-Fold Cross-Val Accuracy: {cv_b_results['test_accuracy'].mean()*100:.2f}% (+/- {cv_b_results['test_accuracy'].std()*100:.2f}%)")
print(f"  -> Precision:                {cv_b_results['test_precision'].mean()*100:.2f}%")
print(f"  -> ROC-AUC:                  {cv_b_results['test_roc_auc'].mean():.4f}")

# ==============================================================================
# 2. CHAMPION MODEL 2: FLEX LIQUID DROPLET TERMINATION MODE CLASSIFIER
# ==============================================================================
print("\n[2/3] Training Champion 2: FLEX Droplet Termination Mode Classifier...")
df_f = pd.read_csv("cleaned_data/flex_clean.csv")

features_f = ['fuel', 'pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he', 'd0_mm']
categorical_f = ['fuel']
numerical_f = ['pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he', 'd0_mm']

X_f = df_f[features_f].copy()
y_f = df_f['test_end'].copy()

pipe_f = Pipeline([
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_f),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_f)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
])

cv_f = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f_results = cross_validate(pipe_f, X_f, y_f, cv=cv_f, scoring=['accuracy', 'f1_weighted'])

pipe_f.fit(X_f, y_f)
joblib.dump(pipe_f, "models/champion_flex_endmode.joblib")

print(f"  -> 5-Fold Cross-Val Accuracy: {cv_f_results['test_accuracy'].mean()*100:.2f}% (+/- {cv_f_results['test_accuracy'].std()*100:.2f}%)")
print(f"  -> Weighted F1-Score:         {cv_f_results['test_f1_weighted'].mean():.4f}")

# ==============================================================================
# 3. CHAMPION MODEL 3: CFI-G GASEOUS FLAME EXTINCTION CLASSIFIER
# ==============================================================================
print("\n[3/3] Training Champion 3: CFI-G Gaseous Flame Extinction Classifier...")
df_c = pd.read_csv("cleaned_data/cfig_normal_clean.csv").dropna(subset=['burn_time_s', 'peak_burner_t_c']).copy()

features_c = ['fuel', 'pressure_bar', 'xf', 'xo2_initial', 'mhc_mg_s', 'hot_ignition']
categorical_c = ['fuel', 'hot_ignition']
numerical_c = ['pressure_bar', 'xf', 'xo2_initial', 'mhc_mg_s']

X_c = df_c[features_c].copy()
y_c = (df_c['termination_mode'] == 'SE').astype(int) # 1 if Self-Extinguished, 0 if Sustained/FlowCut

pipe_c = Pipeline([
    ('preprocessor', ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_c),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_c)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=90, max_depth=5, random_state=42))
])

cv_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_c_results = cross_validate(pipe_c, X_c, y_c, cv=cv_c, scoring=['accuracy', 'precision', 'recall', 'f1'])

pipe_c.fit(X_c, y_c)
joblib.dump(pipe_c, "models/champion_cfig_extinction.joblib")

print(f"  -> 5-Fold Cross-Val Accuracy: {cv_c_results['test_accuracy'].mean()*100:.2f}% (+/- {cv_c_results['test_accuracy'].std()*100:.2f}%)")
print(f"  -> Precision:                {cv_c_results['test_precision'].mean()*100:.2f}%")
print(f"  -> F1-Score:                 {cv_c_results['test_f1'].mean():.4f}")

print("\n" + "="*70)
print("ALL 3 CHAMPION MODELS SUCCESSFULLY TRAINED & PERSISTED!")
print("="*70)
