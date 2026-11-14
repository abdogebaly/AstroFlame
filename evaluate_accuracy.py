import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, r2_score, mean_absolute_error,
    mean_squared_error, median_absolute_error
)

os.makedirs("results", exist_ok=True)

benchmark_results = {}

print("="*75)
print("COMPREHENSIVE ASTROFLAME ACCURACY & BENCHMARK TESTING SUITE")
print("="*75)

# ==============================================================================
# 1. TEST BASS-II SPACECRAFT SOLID COMBUSTION MODELS
# ==============================================================================
print("\n" + "-"*75)
print("TEST 1: BASS-II SPACECRAFT CABIN FIRE SAFETY (SOLID COMBUSTION)")
print("-" * 75)

df_bass = pd.read_csv("cleaned_data/bass_ii_clean.csv")
features_bass = ['material_type', 'geometry', 'color', 'thickness_mm', 'width_cm', 
                 'flow_restrictor', 'fan_display', 'air_display', 'o2_initial']

X_b = df_bass[features_bass].copy()
y_b_ext = df_bass['extinguished'].copy()
y_b_co = df_bass['delta_co'].fillna(0).copy()

bass_clf = joblib.load("models/bass_extinction_model.joblib")
bass_reg = joblib.load("models/bass_co_regressor.joblib")

# Stratified 5-Fold & 10-Fold Cross-Validation for Extinction
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv5_results = cross_validate(bass_clf, X_b, y_b_ext, cv=cv5, scoring=scoring_metrics)
cv10_results = cross_validate(bass_clf, X_b, y_b_ext, cv=cv10, scoring=scoring_metrics)

# Train/Test Split (80/20) for Detailed Matrix
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_b, y_b_ext, test_size=0.25, random_state=42, stratify=y_b_ext)
bass_clf.fit(X_train_b, y_train_b)
y_pred_b = bass_clf.predict(X_test_b)
y_prob_b = bass_clf.predict_proba(X_test_b)[:, 1]

cm_b = confusion_matrix(y_test_b, y_pred_b)
tn, fp, fn, tp = cm_b.ravel()

test_acc_b = accuracy_score(y_test_b, y_pred_b)
test_prec_b = precision_score(y_test_b, y_pred_b)
test_rec_b = recall_score(y_test_b, y_pred_b)
test_f1_b = f1_score(y_test_b, y_pred_b)
test_auc_b = roc_auc_score(y_test_b, y_prob_b)

print(f"[Extinction Classification]")
print(f"  * Hold-out Test Accuracy (25% split):  {test_acc_b*100:.2f}%")
print(f"  * Precision (True Safety Confidence): {test_prec_b*100:.2f}%")
print(f"  * Recall (Extinction Detection Rate): {test_rec_b*100:.2f}%")
print(f"  * F1-Score:                           {test_f1_b:.4f}")
print(f"  * ROC-AUC Score:                      {test_auc_b:.4f}")
print(f"  * 5-Fold Cross-Validation Accuracy:   {cv5_results['test_accuracy'].mean()*100:.2f}% (+/- {cv5_results['test_accuracy'].std()*100:.2f}%)")
print(f"  * 10-Fold Cross-Validation Accuracy:  {cv10_results['test_accuracy'].mean()*100:.2f}% (+/- {cv10_results['test_accuracy'].std()*100:.2f}%)")
print(f"  * Confusion Matrix Breakdown: TN={tn} (Sustained Fire Caught), FP={fp}, FN={fn}, TP={tp} (Safe Extinction Caught)")

# Evaluate CO Regressor
X_tr_co, X_te_co, y_tr_co, y_te_co = train_test_split(X_b, y_b_co, test_size=0.25, random_state=42)
bass_reg.fit(X_tr_co, y_tr_co)
y_pred_co = bass_reg.predict(X_te_co)
r2_co = r2_score(y_te_co, y_pred_co)
mae_co = mean_absolute_error(y_te_co, y_pred_co)
rmse_co = np.sqrt(mean_squared_error(y_te_co, y_pred_co))
medae_co = median_absolute_error(y_te_co, y_pred_co)

print(f"\n[Toxic CO Output Regression]")
print(f"  * R2 Score:                           {r2_co:.3f}")
print(f"  * Mean Absolute Error (MAE):          {mae_co:.2f} ppm")
print(f"  * Root Mean Squared Error (RMSE):     {rmse_co:.2f} ppm")
print(f"  * Median Absolute Error:              {medae_co:.2f} ppm")

benchmark_results['BASS_II'] = {
    'classification': {
        'test_accuracy': float(test_acc_b),
        'precision': float(test_prec_b),
        'recall': float(test_rec_b),
        'f1_score': float(test_f1_b),
        'roc_auc': float(test_auc_b),
        'cv5_accuracy_mean': float(cv5_results['test_accuracy'].mean()),
        'cv5_accuracy_std': float(cv5_results['test_accuracy'].std()),
        'cv10_accuracy_mean': float(cv10_results['test_accuracy'].mean()),
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}
    },
    'co_regression': {
        'r2': float(r2_co),
        'mae_ppm': float(mae_co),
        'rmse_ppm': float(rmse_co),
        'medae_ppm': float(medae_co)
    }
}

# ==============================================================================
# 2. TEST FLEX DROPLET COMBUSTION PHYSICS MODELS
# ==============================================================================
print("\n" + "-"*75)
print("TEST 2: FLEX LIQUID DROPLET COMBUSTION PHYSICS & d²-LAW")
print("-" * 75)

df_flex = pd.read_csv("cleaned_data/flex_clean.csv")
features_flex = ['fuel', 'pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he', 'd0_mm']
X_f = df_flex[features_flex].copy()
y_f_k = df_flex['burning_rate'].copy()
y_f_end = df_flex['test_end'].copy()

flex_k_model = joblib.load("models/flex_burning_rate_model.joblib")
flex_end_model = joblib.load("models/flex_end_mode_model.joblib")

# Test Burning Rate K
X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(X_f, y_f_k, test_size=0.20, random_state=42)
flex_k_model.fit(X_tr_f, y_tr_f)
y_pred_k = flex_k_model.predict(X_te_f)

r2_k = r2_score(y_te_f, y_pred_k)
mae_k = mean_absolute_error(y_te_f, y_pred_k)
rmse_k = np.sqrt(mean_squared_error(y_te_f, y_pred_k))
cv5_k = cross_val_score(flex_k_model, X_f, y_f_k, cv=5, scoring='r2')
pearson_corr = np.corrcoef(y_te_f, y_pred_k)[0, 1]

print(f"[Droplet Burning Rate K Regression]")
print(f"  * Hold-out Test R2 Score:             {r2_k:.3f}")
print(f"  * Pearson Correlation (Actual vs AI): {pearson_corr:.4f}")
print(f"  * Mean Absolute Error (MAE):          {mae_k:.4f} mm^2/s")
print(f"  * Root Mean Squared Error (RMSE):     {rmse_k:.4f} mm^2/s")
print(f"  * 5-Fold Cross-Validation R2 Mean:    {cv5_k.mean():.3f} (+/- {cv5_k.std():.3f})")

# Test Droplet End Mode (Extinction vs Disruption vs Completion)
X_tr_fe, X_te_fe, y_tr_fe, y_te_fe = train_test_split(X_f, y_f_end, test_size=0.20, random_state=42, stratify=y_f_end)
flex_end_model.fit(X_tr_fe, y_tr_fe)
y_pred_end = flex_end_model.predict(X_te_fe)
acc_end = accuracy_score(y_te_fe, y_pred_end)
f1_end_macro = f1_score(y_te_fe, y_pred_end, average='macro')
f1_end_weighted = f1_score(y_te_fe, y_pred_end, average='weighted')
cv5_end = cross_val_score(flex_end_model, X_f, y_f_end, cv=5, scoring='accuracy')

print(f"\n[Droplet Termination Mode 3-Class Classification]")
print(f"  * Test Accuracy:                      {acc_end*100:.2f}%")
print(f"  * Macro F1-Score:                     {f1_end_macro:.4f}")
print(f"  * Weighted F1-Score:                  {f1_end_weighted:.4f}")
print(f"  * 5-Fold Cross-Validation Accuracy:   {cv5_end.mean()*100:.2f}% (+/- {cv5_end.std()*100:.2f}%)")

benchmark_results['FLEX'] = {
    'burning_rate_regression': {
        'test_r2': float(r2_k),
        'pearson_correlation': float(pearson_corr),
        'mae_mm2_s': float(mae_k),
        'rmse_mm2_s': float(rmse_k),
        'cv5_r2_mean': float(cv5_k.mean())
    },
    'end_mode_classification': {
        'test_accuracy': float(acc_end),
        'macro_f1': float(f1_end_macro),
        'weighted_f1': float(f1_end_weighted),
        'cv5_accuracy_mean': float(cv5_end.mean())
    }
}

# ==============================================================================
# 3. TEST CFI-G GASEOUS COMBUSTION & EXTINCTION MODELS
# ==============================================================================
print("\n" + "-"*75)
print("TEST 3: CFI-G GASEOUS DIFFUSION & COOL FLAME EXTINCTION")
print("-" * 75)

df_cfig = pd.read_csv("cleaned_data/cfig_normal_clean.csv").dropna(subset=['burn_time_s', 'peak_burner_t_c']).copy()
features_cfig = ['fuel', 'pressure_bar', 'xf', 'xo2_initial', 'mhc_mg_s', 'hot_ignition']
X_c = df_cfig[features_cfig].copy()
y_c_time = df_cfig['burn_time_s'].copy()
y_c_ext = (df_cfig['termination_mode'] == 'SE').astype(int)

cfig_ext_model = joblib.load("models/cfig_self_ext_model.joblib")
cfig_time_model = joblib.load("models/cfig_burn_time_model.joblib")

# Test Self-Extinction Classifier
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_c, y_c_ext, test_size=0.25, random_state=42, stratify=y_c_ext)
cfig_ext_model.fit(X_tr_c, y_tr_c)
y_pred_ce = cfig_ext_model.predict(X_te_c)
acc_cfig_ext = accuracy_score(y_te_c, y_pred_ce)
prec_cfig_ext = precision_score(y_te_c, y_pred_ce)
rec_cfig_ext = recall_score(y_te_c, y_pred_ce)
f1_cfig_ext = f1_score(y_te_c, y_pred_ce)
cv5_cfig = cross_val_score(cfig_ext_model, X_c, y_c_ext, cv=5, scoring='accuracy')

print(f"[CFI-G Flame Self-Extinction Classification]")
print(f"  * Test Accuracy:                      {acc_cfig_ext*100:.2f}%")
print(f"  * Precision:                          {prec_cfig_ext*100:.2f}%")
print(f"  * Recall:                             {rec_cfig_ext*100:.2f}%")
print(f"  * F1-Score:                           {f1_cfig_ext:.4f}")
print(f"  * 5-Fold Cross-Validation Accuracy:   {cv5_cfig.mean()*100:.2f}% (+/- {cv5_cfig.std()*100:.2f}%)")

# Test Burn Time Regressor
cfig_time_model.fit(X_tr_c, y_c_time.loc[X_tr_c.index])
y_pred_ct = cfig_time_model.predict(X_te_c)
r2_ct = r2_score(y_c_time.loc[X_te_c.index], y_pred_ct)
mae_ct = mean_absolute_error(y_c_time.loc[X_te_c.index], y_pred_ct)
rmse_ct = np.sqrt(mean_squared_error(y_c_time.loc[X_te_c.index], y_pred_ct))

print(f"\n[CFI-G Burn Duration Regression]")
print(f"  * Test R2 Score:                      {r2_ct:.3f}")
print(f"  * Mean Absolute Error:                {mae_ct:.2f} seconds")
print(f"  * Root Mean Squared Error (RMSE):     {rmse_ct:.2f} seconds")

benchmark_results['CFI_G'] = {
    'self_extinction_classification': {
        'test_accuracy': float(acc_cfig_ext),
        'precision': float(prec_cfig_ext),
        'recall': float(rec_cfig_ext),
        'f1_score': float(f1_cfig_ext),
        'cv5_accuracy_mean': float(cv5_cfig.mean())
    },
    'burn_time_regression': {
        'r2': float(r2_ct),
        'mae_sec': float(mae_ct),
        'rmse_sec': float(rmse_ct)
    }
}

# ==============================================================================
# 4. TEST SPICE SOOT & SMOKE POINT MODELS
# ==============================================================================
print("\n" + "-"*75)
print("TEST 4: SPICE SOOT FORMATION & SMOKE POINT DYNAMICS")
print("-" * 75)

df_spice = pd.read_csv("cleaned_data/spice_clean.csv").dropna(subset=['hc_fuel_rate_mg_s', 'air_rate_cm_s', 'nozzle_mm']).copy()
features_spice = ['fuel', 'nozzle_mm', 'air_rate_cm_s']
X_s = df_spice[features_spice].copy()
y_s = df_spice['hc_fuel_rate_mg_s'].copy()

spice_model = joblib.load("models/spice_model.joblib")
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(X_s, y_s, test_size=0.20, random_state=42)
spice_model.fit(X_tr_s, y_tr_s)
y_pred_s = spice_model.predict(X_te_s)

r2_s = r2_score(y_te_s, y_pred_s)
mae_s = mean_absolute_error(y_te_s, y_pred_s)
rmse_s = np.sqrt(mean_squared_error(y_te_s, y_pred_s))
pearson_s = np.corrcoef(y_te_s, y_pred_s)[0, 1]
cv5_s = cross_val_score(spice_model, X_s, y_s, cv=5, scoring='r2')

print(f"[Critical Fuel Rate at Soot Smoke Point Regression]")
print(f"  * Hold-out Test R2 Score:             {r2_s:.3f}")
print(f"  * Pearson Correlation (Actual vs AI): {pearson_s:.4f}")
print(f"  * Mean Absolute Error (MAE):          {mae_s:.4f} mg/s")
print(f"  * Root Mean Squared Error (RMSE):     {rmse_s:.4f} mg/s")
print(f"  * 5-Fold Cross-Validation R2 Mean:    {cv5_s.mean():.3f} (+/- {cv5_s.std():.3f})")

benchmark_results['SPICE'] = {
    'smoke_point_regression': {
        'test_r2': float(r2_s),
        'pearson_correlation': float(pearson_s),
        'mae_mg_s': float(mae_s),
        'rmse_mg_s': float(rmse_s),
        'cv5_r2_mean': float(cv5_s.mean())
    }
}

# ==============================================================================
# SAVE JSON BENCHMARK REPORT
# ==============================================================================
with open("results/accuracy_benchmark_report.json", "w") as f:
    json.dump(benchmark_results, f, indent=2)

print("\n" + "="*75)
print("BENCHMARK TEST COMPLETE: Saved to results/accuracy_benchmark_report.json")
print("="*75)
