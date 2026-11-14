import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Set matplotlib style
plt.style.use('dark_background')

# ==============================================================================
# 1. BASS-II HYPERPARAMETER TUNED MODEL (SPACECRAFT SOLID COMBUSTION)
# ==============================================================================
def train_tuned_bass():
    print("\n" + "="*60)
    print("1. OPTIMIZING BASS-II SPACECRAFT FIRE SAFETY MODEL")
    print("="*60)
    df = pd.read_csv("cleaned_data/bass_ii_clean.csv")
    
    features = ['material_type', 'geometry', 'color', 'thickness_mm', 'width_cm', 
                'flow_restrictor', 'fan_display', 'air_display', 'o2_initial']
    
    categorical_cols = ['material_type', 'geometry', 'color']
    numeric_cols = ['thickness_mm', 'width_cm', 'flow_restrictor', 'fan_display', 'air_display', 'o2_initial']
    
    X = df[features].copy()
    y_ext = df['extinguished'].copy()
    y_co = df['delta_co'].fillna(0).copy()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ]
    )
    
    # GridSearch for Extinction Classifier
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [3, 5, 8, None],
        'classifier__min_samples_split': [2, 4],
        'classifier__class_weight': ['balanced', None]
    }
    
    base_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_ext, test_size=0.25, random_state=42, stratify=y_ext)
    grid_search = GridSearchCV(base_pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(best_clf, X, y_ext, cv=5)
    
    print(f"Optimal Hyperparameters: {grid_search.best_params_}")
    print(f"BASS-II Test Accuracy: {test_acc * 100:.2f}%")
    print(f"BASS-II 5-Fold Cross-Val Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    # Save Model
    joblib.dump(best_clf, "models/bass_extinction_model.joblib")
    
    # Generate Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("BASS-II Fire Extinction Confusion Matrix", fontsize=12, pad=10)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(['Sustained Fire', 'Self-Extinguished'], fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(['Sustained Fire', 'Self-Extinguished'], fontsize=10)
    ax.set_ylabel('True Experimental Result', fontsize=11)
    ax.set_xlabel('AI Predicted Result', fontsize=11)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/bass_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Feature Importance Plot
    rf = best_clf.named_steps['classifier']
    ohe = best_clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names
    importances = rf.feature_importances_
    
    feat_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_df['feature'][-10:], feat_df['importance'][-10:], color='#ff4b4b')
    ax.set_title("Top 10 Most Critical Factors for Spacecraft Flame Extinction", fontsize=12)
    ax.set_xlabel("Relative Importance Index", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/bass_feature_importance.png", dpi=300)
    plt.close()
    print("Generated results/bass_confusion_matrix.png and results/bass_feature_importance.png")

# ==============================================================================
# 2. FLEX DROPLET PHYSICS MODEL WITH HIGH-PRECISION REGRESSION
# ==============================================================================
def train_tuned_flex():
    print("\n" + "="*60)
    print("2. TRAINING HIGH-PRECISION FLEX DROPLET COMBUSTION MODELS")
    print("="*60)
    df = pd.read_csv("cleaned_data/flex_clean.csv")
    
    features = ['fuel', 'pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he', 'd0_mm']
    categorical_cols = ['fuel']
    numeric_cols = ['pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he', 'd0_mm']
    
    X = df[features].copy()
    y_k = df['burning_rate'].copy()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    k_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=180, learning_rate=0.06, max_depth=4, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_k, test_size=0.20, random_state=42)
    k_pipe.fit(X_train, y_train)
    
    y_pred = k_pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"FLEX Burning Rate K (Test R2): {r2:.3f}")
    print(f"FLEX Burning Rate K (Test MAE): {mae:.4f} mm^2/s")
    
    joblib.dump(k_pipe, "models/flex_burning_rate_model.joblib")
    
    # Save Predicted vs Actual Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, color='#38bdf8', edgecolors='white', alpha=0.8, s=60)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Agreement (1:1)")
    ax.set_title(f"FLEX Droplet Burning Rate (R² = {r2:.2f})", fontsize=12)
    ax.set_xlabel("Experimental Burning Rate K (mm²/s)", fontsize=11)
    ax.set_ylabel("AI Predicted Burning Rate K (mm²/s)", fontsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/flex_pred_vs_actual.png", dpi=300)
    plt.close()
    print("Generated results/flex_pred_vs_actual.png")

# ==============================================================================
# 3. CFI-G GASEOUS COMBUSTION & COOL FLAME MODEL
# ==============================================================================
def train_cfig_models():
    print("\n" + "="*60)
    print("3. TRAINING CFI-G GASEOUS & COOL FLAME DISCOVERY MODELS")
    print("="*60)
    df_norm = pd.read_csv("cleaned_data/cfig_normal_clean.csv")
    
    # Filter valid records
    df_norm = df_norm.dropna(subset=['burn_time_s', 'peak_burner_t_c']).copy()
    
    features = ['fuel', 'pressure_bar', 'xf', 'xo2_initial', 'mhc_mg_s', 'hot_ignition']
    cat_cols = ['fuel', 'hot_ignition']
    num_cols = ['pressure_bar', 'xf', 'xo2_initial', 'mhc_mg_s']
    
    X = df_norm[features].copy()
    y_time = df_norm['burn_time_s'].copy()
    y_term = (df_norm['termination_mode'] == 'SE').astype(int) # 1 if Self-Extinguished, 0 if Flow Cut
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    # Model 1: Burn Time Regressor
    time_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_time, test_size=0.25, random_state=42)
    time_pipe.fit(X_train, y_train)
    r2_time = r2_score(y_test, time_pipe.predict(X_test))
    
    # Model 2: Self Extinction Classifier
    term_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    term_pipe.fit(X_train, y_term.loc[X_train.index])
    acc_term = accuracy_score(y_term.loc[X_test.index], term_pipe.predict(X_test))
    
    print(f"CFI-G Normal Burn Time R2: {r2_time:.3f}")
    print(f"CFI-G Self-Extinction Predictor Accuracy: {acc_term*100:.2f}%")
    
    joblib.dump(time_pipe, "models/cfig_burn_time_model.joblib")
    joblib.dump(term_pipe, "models/cfig_self_ext_model.joblib")
    print("Saved CFI-G models successfully.")

# ==============================================================================
# 4. SPICE SOOT & SMOKE POINT PREDICTION MODEL
# ==============================================================================
def train_spice_model():
    print("\n" + "="*60)
    print("4. TRAINING SPICE SOOT & SMOKE POINT MODEL")
    print("="*60)
    df = pd.read_csv("cleaned_data/spice_clean.csv")
    df = df.dropna(subset=['hc_fuel_rate_mg_s', 'air_rate_cm_s', 'nozzle_mm']).copy()
    
    features = ['fuel', 'nozzle_mm', 'air_rate_cm_s']
    cat_cols = ['fuel']
    num_cols = ['nozzle_mm', 'air_rate_cm_s']
    
    X = df[features].copy()
    y = df['hc_fuel_rate_mg_s'].copy()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    spice_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    spice_pipe.fit(X_train, y_train)
    
    r2 = r2_score(y_test, spice_pipe.predict(X_test))
    mae = mean_absolute_error(y_test, spice_pipe.predict(X_test))
    
    print(f"SPICE Critical Smoke-Point Fuel Flow Rate (R2): {r2:.3f}")
    print(f"SPICE Fuel Flow Rate (MAE): {mae:.4f} mg/s")
    
    joblib.dump(spice_pipe, "models/spice_model.joblib")
    print("Saved SPICE model successfully.")

if __name__ == "__main__":
    train_tuned_bass()
    train_tuned_flex()
    train_cfig_models()
    train_spice_model()
    print("\n" + "="*60)
    print("ALL ADVANCED MODELS TRAINED & RESULTS EXPORTED!")
    print("="*60)
