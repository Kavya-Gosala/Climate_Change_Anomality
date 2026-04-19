# =============================================================================
# COM747 - Data Science and Machine Learning
# Global Warming Dataset: 195 Countries (1900-2023)
#  Machine Learning Modelling & Evaluation
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Evaluation
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, KFold

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'Times New Roman'

# =============================================================================
#  RELOAD PREPARED DATA
# =============================================================================

df_model = pd.read_csv(r'C:\Users\uzwal\OneDrive\Desktop\DS\Climate changes\global_warming_features.csv')

REGRESSION_FEATURES = [
    'year', 'years_since_1900', 'post_1950', 'post_paris',
    'co2_emissions', 'co2_roll10', 'co2_yoy_change', 'co2_decade_avg',
    'temp_anomaly_roll10', 'temp_anomaly_lag1', 'country_encoded', 'decade',
]
optional_cols = ['sea_level_rise', 'arctic_ice_extent',
                 'extreme_weather_events', 'renewable_energy', 'co2_x_renewable']
for col in optional_cols:
    if col in df_model.columns:
        REGRESSION_FEATURES.append(col)

TARGET_REG = 'temperature_anomaly'
TARGET_CLF = 'warming_class'

train_mask = df_model['year'] < 2010
test_mask  = df_model['year'] >= 2010

X_train = df_model.loc[train_mask, REGRESSION_FEATURES]
X_test  = df_model.loc[test_mask,  REGRESSION_FEATURES]
y_train = df_model.loc[train_mask, TARGET_REG]
y_test  = df_model.loc[test_mask,  TARGET_REG]

X_train_clf = X_train.copy()
X_test_clf  = X_test.copy()
y_train_clf = df_model.loc[train_mask, TARGET_CLF]
y_test_clf  = df_model.loc[test_mask,  TARGET_CLF]

# =============================================================================
# 4.1 REGRESSION MODELS
# =============================================================================

print("=" * 60)
print("REGRESSION TASK: Predict Temperature Anomaly (°C)")
print("=" * 60)

# --- Helper function to evaluate a regression model ---
def evaluate_regression(name, model, X_tr, y_tr, X_te, y_te, cv=5):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)

    # Cross-validation R²
    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring='r2')

    print(f"\n--- {name} ---")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        'Model': name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'CV_R2_mean': round(cv_scores.mean(), 4),
        'CV_R2_std': round(cv_scores.std(), 4),
        'predictions': y_pred
    }

# --- Model 1: Linear Regression (Baseline) ---
lr = LinearRegression()
res_lr = evaluate_regression("Linear Regression (Baseline)", lr,
                              X_train, y_train, X_test, y_test)

# --- Model 2: Ridge Regression ---
ridge = Ridge(alpha=1.0)
res_ridge = evaluate_regression("Ridge Regression", ridge,
                                 X_train, y_train, X_test, y_test)

# --- Model 3: Random Forest Regressor ---
rf = RandomForestRegressor(n_estimators=50, max_depth=10,
                            min_samples_leaf=5, random_state=42, n_jobs=-1)
res_rf = evaluate_regression("Random Forest Regressor", rf,
                              X_train, y_train, X_test, y_test)

# --- Model 4: Gradient Boosting Regressor ---
gb = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1,
                                 max_depth=4, random_state=42)
res_gb = evaluate_regression("Gradient Boosting Regressor", gb,
                              X_train, y_train, X_test, y_test)

# =============================================================================
# 4.2 REGRESSION COMPARISON TABLE
# =============================================================================

results_reg = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'predictions'}
    for r in [res_lr, res_ridge, res_rf, res_gb]
])

print("\n" + "=" * 60)
print("REGRESSION MODEL COMPARISON")
print("=" * 60)
print(results_reg.to_string(index=False))
results_reg.to_csv('regression_results.csv', index=False)

# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics = ['MAE', 'RMSE', 'R2']
colors = ['steelblue', 'tomato', 'seagreen']

for ax, metric, color in zip(axes, metrics, colors):
    ax.bar(results_reg['Model'], results_reg[metric], color=color, alpha=0.8)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    ax.set_xticklabels(results_reg['Model'], rotation=20, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.5)

plt.suptitle('Regression Model Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig8_regression_comparison.png', bbox_inches='tight')
plt.show()

# =============================================================================
# 4.3 ACTUAL VS PREDICTED —  REGRESSION MODEL
# =============================================================================

best_reg = res_rf  # Random Forest is typically best
y_pred_best = best_reg['predictions']

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.4, color='steelblue', s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', linewidth=1.5, label='Perfect Fit')
plt.title('Actual vs Predicted: Random Forest Regressor', fontsize=13, fontweight='bold')
plt.xlabel('Actual Temperature Anomaly (°C)', fontsize=12)
plt.ylabel('Predicted Temperature Anomaly (°C)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('fig9_actual_vs_predicted.png', bbox_inches='tight')
plt.show()

# Residual plot
residuals = y_test.values - y_pred_best
plt.figure(figsize=(10, 4))
plt.scatter(y_pred_best, residuals, alpha=0.3, color='purple', s=10)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Residual Plot: Random Forest Regressor', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.tight_layout()
plt.savefig('fig10_residual_plot.png', bbox_inches='tight')
plt.show()

# =============================================================================
# 4.4 FEATURE IMPORTANCE (Random Forest)
# =============================================================================

rf.fit(X_train, y_train)  # ensure fitted
importances = pd.Series(rf.feature_importances_, index=REGRESSION_FEATURES)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 7))
importances.plot(kind='barh', color='steelblue', alpha=0.85)
plt.title('Feature Importance: Random Forest Regressor', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig('fig11_feature_importance.png', bbox_inches='tight')
plt.show()

print("\nTop 5 Most Important Features:")
print(importances.sort_values(ascending=False).head(5))

# =============================================================================
# 4.5 CLASSIFICATION MODELS
# =============================================================================

print("\n" + "=" * 60)
print("CLASSIFICATION TASK: Predict Warming Level (Low/Medium/High)")
print("=" * 60)

class_names = ['Low (<0.5°C)', 'Medium (0.5–1°C)', 'High (>1°C)']

def evaluate_classification(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv_acc = cross_val_score(model, X_tr, y_tr, cv=5, scoring='accuracy')

    print(f"\n--- {name} ---")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  CV Accuracy (mean ± std): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=class_names))

    return {'Model': name, 'Accuracy': round(acc, 4),
            'CV_Acc_mean': round(cv_acc.mean(), 4),
            'predictions': y_pred}

# --- Model 1: Logistic Regression (Baseline) ---
log_reg = LogisticRegression(max_iter=1000, random_state=42)
res_log = evaluate_classification("Logistic Regression (Baseline)",
                                   log_reg, X_train_clf, y_train_clf,
                                   X_test_clf, y_test_clf)

# --- Model 2: Random Forest Classifier ---
rf_clf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1)
res_rf_clf = evaluate_classification("Random Forest Classifier",
                                      rf_clf, X_train_clf, y_train_clf,
                                      X_test_clf, y_test_clf)

# --- Model 3: Gradient Boosting Classifier ---
gb_clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1,
                                      max_depth=4, random_state=42)
res_gb_clf = evaluate_classification("Gradient Boosting Classifier",
                                      gb_clf, X_train_clf, y_train_clf,
                                      X_test_clf, y_test_clf)

# =============================================================================
# 4.6 CONFUSION MATRIX — BEST CLASSIFIER
# =============================================================================

best_clf_preds = res_rf_clf['predictions']
cm = confusion_matrix(y_test_clf, best_clf_preds)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix: Random Forest Classifier', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('fig12_confusion_matrix.png', bbox_inches='tight')
plt.show()

# =============================================================================
# 4.7 CLASSIFICATION COMPARISON TABLE
# =============================================================================

results_clf = pd.DataFrame([
    {k: v for k, v in r.items() if k != 'predictions'}
    for r in [res_log, res_rf_clf, res_gb_clf]
])
print("\n" + "=" * 60)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 60)
print(results_clf.to_string(index=False))
results_clf.to_csv('classification_results.csv', index=False)

# =============================================================================
# 4.8 FUTURE TEMPERATURE PROJECTION 
# =============================================================================

print("\n" + "=" * 60)
print("FUTURE PROJECTION (2024–2050) — Illustrative")
print("=" * 60)

# Use global mean values to create a simple projection
# (this is for illustrative purposes in the paper, not a true forecast)
last_year_data = df_model[df_model['year'] == df_model['year'].max()].copy()
global_last = df_model.groupby('year')[REGRESSION_FEATURES].mean().reset_index(drop=True)
last_row = df_model[df_model['year'] == df_model['year'].max()][REGRESSION_FEATURES].mean()

future_years = range(2024, 2051)
future_rows = []
prev_temp = df_model[df_model['year'] == df_model['year'].max()]['temperature_anomaly'].mean()
prev_co2 = last_row['co2_emissions']

for yr in future_years:
    row = last_row.copy()
    row['year'] = yr
    row['years_since_1900'] = yr - 1900
    row['post_1950'] = 1
    row['post_paris'] = 1
    row['decade'] = (yr // 10) * 10
    row['temp_anomaly_lag1'] = prev_temp
    row['co2_emissions'] = prev_co2 * 1.01  # assume 1% annual CO2 growth
    row['co2_roll10'] = row['co2_emissions']
    row['co2_decade_avg'] = row['co2_emissions']
    row['co2_yoy_change'] = row['co2_emissions'] * 0.01
    future_rows.append(row)
    prev_co2 = row['co2_emissions']

future_df = pd.DataFrame(future_rows)
future_preds = rf.predict(future_df[REGRESSION_FEATURES])

# Plot projection
historical_global = df_model.groupby('year')['temperature_anomaly'].mean().reset_index()
plt.figure(figsize=(13, 5))
plt.plot(historical_global['year'], historical_global['temperature_anomaly'],
         color='tomato', linewidth=1.5, label='Historical (Observed)')
plt.plot(list(future_years), future_preds, color='darkorange', linewidth=1.5,
         linestyle='--', label='Projected (Random Forest, +1% CO₂/yr)')
plt.axvline(2023, color='gray', linestyle=':', linewidth=1.2, label='Projection Start')
plt.title('Historical & Projected Global Mean Temperature Anomaly',
          fontsize=13, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig13_future_projection.png', bbox_inches='tight')
plt.show()

print("\nAll models trained, evaluated, and figures saved.")
print("Phase 4 complete. Proceed to Phase 5 (IEEE Paper).")
