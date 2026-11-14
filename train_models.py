import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def train_bass_models():
    print("\n==========================================")
    print("TRAINING BASS-II SPACECRAFT FIRE SAFETY MODELS")
    print("==========================================")
    df = pd.read_csv("cleaned_data/bass_ii_clean.csv")
    
    # Filter features
    features = ['material_type', 'geometry', 'color', 'thickness_mm', 'width_cm', 
                'flow_restrictor', 'fan_display', 'air_display', 'o2_initial']
    
    categorical_cols = ['material_type', 'geometry', 'color']
    numeric_cols = ['thickness_mm', 'width_cm', 'flow_restrictor', 'fan_display', 'air_display', 'o2_initial']
    
    X = df[features].copy()
    y_ext = df['extinguished'].copy()
    y_co = df['delta_co'].fillna(0).copy()
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ]), numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_cols)
        ]
    )
    
    # 1. Extinction Classifier
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_ext, test_size=0.25, random_state=42, stratify=y_ext)
    clf_pipeline.fit(X_train, y_train)
    
    y_pred = clf_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf_pipeline, X, y_ext, cv=5)
    
    print(f"Extinction Classifier Test Accuracy: {acc * 100:.2f}%")
    print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Sustained Fire', 'Self-Extinguished']))
    
    # 2. Toxic CO Emission Regressor
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6))
    ])
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_co, test_size=0.25, random_state=42)
    reg_pipeline.fit(X_train_r, y_train_r)
    
    y_pred_r = reg_pipeline.predict(X_test_r)
    r2 = r2_score(y_test_r, y_pred_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    print(f"Toxic CO Regressor R2 Score: {r2:.3f}")
    print(f"Toxic CO Regressor MAE: {mae:.2f} ppm")
    
    # Save models
    joblib.dump(clf_pipeline, "models/bass_extinction_model.joblib")
    joblib.dump(reg_pipeline, "models/bass_co_regressor.joblib")
    print("Saved BASS-II models successfully.")
    
    # Feature Importance for Extinction
    ohe = clf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names
    importances = clf_pipeline.named_steps['classifier'].feature_importances_
    
    feat_imp = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feat_imp = feat_imp.sort_values(by='importance', ascending=False)
    feat_imp.to_csv("results/bass_feature_importance.csv", index=False)
    print("\nTop 5 Most Important Features for Fire Extinction:")
    print(feat_imp.head(5).to_string(index=False))

def train_flex_models():
    print("\n==========================================")
    print("TRAINING FLEX DROPLET COMBUSTION MODELS")
    print("==========================================")
    df = pd.read_csv("cleaned_data/flex_clean.csv")
    
    features = ['fuel', 'pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he', 'd0_mm']
    categorical_cols = ['fuel']
    numeric_cols = ['pressure_bar', 'x_o2', 'x_n2', 'x_co2', 'x_he', 'd0_mm']
    
    X = df[features].copy()
    y_k = df['burning_rate'].copy()
    y_end = df['test_end'].copy()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # 1. Burning Rate (K) Regressor
    k_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=120, random_state=42, learning_rate=0.08, max_depth=4))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_k, test_size=0.20, random_state=42)
    k_pipeline.fit(X_train, y_train)
    
    y_pred = k_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Droplet Burning Rate (K) R2 Score: {r2:.3f}")
    print(f"Droplet Burning Rate (K) MAE: {mae:.4f} mm/s")
    
    # 2. Test End Mode Classifier (Extinction vs Disruption vs Completion)
    end_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5))
    ])
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_end, test_size=0.20, random_state=42, stratify=y_end)
    end_pipeline.fit(X_train_c, y_train_c)
    
    acc = accuracy_score(y_test_c, end_pipeline.predict(X_test_c))
    print(f"Droplet Extinction Mode Test Accuracy: {acc * 100:.2f}%")
    
    joblib.dump(k_pipeline, "models/flex_burning_rate_model.joblib")
    joblib.dump(end_pipeline, "models/flex_end_mode_model.joblib")
    print("Saved FLEX models successfully.")

if __name__ == "__main__":
    train_bass_models()
    train_flex_models()
    print("\nAll models trained and persisted successfully!")

