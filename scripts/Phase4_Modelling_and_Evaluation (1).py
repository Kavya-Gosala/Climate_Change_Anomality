# =============================================================================
# COM747 - Data Science and Machine Learning
# Global Warming Dataset: 195 Countries (1900-2023)
# PHASE 4: Machine Learning Modelling & Evaluation
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')   # For showing figure windows on Windows

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)

# Preprocessing / CV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =============================================================================
# 0.0 DISPLAY SETTINGS
# =============================================================================

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'Times New Roman'

# =============================================================================
# 1.0 LOAD PREPARED DATA
# =============================================================================

df_model = pd.read_csv(r'data\global_warming_features.csv')

# =============================================================================
# 2.0 DEFINE FEATURES AND TARGETS
# =============================================================================

REGRESSION_FEATURES = [
    'year',
    'years_since_1900',
    'post_1950',
    'post_paris',
    'co2_emissions',
    'co2_roll10',
    'co2_yoy_change',
    'co2_decade_avg',
    'temp_anomaly_roll10',
    'temp_anomaly_lag1',
    'country_encoded',
    'decade',
]

optional_cols = [
    'sea_level_rise',
    'arctic_ice_extent',
    'extreme_weather_events',
    'renewable_energy',
    'co2_x_renewable',
    'sea_level_rise_roll5',
    'arctic_ice_lag1',
    'gdp_per_capita_est'
]

for col in optional_cols:
    if col in df_model.columns:
        REGRESSION_FEATURES.append(col)

TARGET_REG = 'temperature_anomaly'
TARGET_CLF = 'warming_class'

# =============================================================================
# 3.0 TIME-BASED TRAIN / TEST SPLIT
# =============================================================================

train_mask = df_model['year'] < 2010
test_mask = df_model['year'] >= 2010

X_train = df_model.loc[train_mask, REGRESSION_FEATURES].copy()
X_test = df_model.loc[test_mask, REGRESSION_FEATURES].copy()
y_train = df_model.loc[train_mask, TARGET_REG].copy()
y_test = df_model.loc[test_mask, TARGET_REG].copy()

X_train_clf = X_train.copy()
X_test_clf = X_test.copy()
y_train_clf = df_model.loc[train_mask, TARGET_CLF].copy()
y_test_clf = df_model.loc[test_mask, TARGET_CLF].copy()

print("=" * 70)
print("DATA SPLIT SUMMARY")
print("=" * 70)
print(f"Training rows (before 2010): {len(X_train)}")
print(f"Testing rows  (2010 onwards): {len(X_test)}")
print(f"Number of regression features: {len(REGRESSION_FEATURES)}")

# Sort training data by year for time-series CV
train_reg_df = X_train.copy()
train_reg_df['target_reg'] = y_train.values
train_reg_df = train_reg_df.sort_values('year').reset_index(drop=True)

X_train_cv_reg = train_reg_df[REGRESSION_FEATURES]
y_train_cv_reg = train_reg_df['target_reg']

train_clf_df = X_train_clf.copy()
train_clf_df['target_clf'] = y_train_clf.values
train_clf_df = train_clf_df.sort_values('year').reset_index(drop=True)

X_train_cv_clf = train_clf_df[REGRESSION_FEATURES]
y_train_cv_clf = train_clf_df['target_clf']

# =============================================================================
# 4.0 HELPER FUNCTIONS
# =============================================================================

def evaluate_regression(name, model, X_tr, y_tr, X_te, y_te,
                        X_cv, y_cv, use_scaling=False, n_splits=5):
    if use_scaling:
        final_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        final_model = model

    final_model.fit(X_tr, y_tr)
    y_pred = final_model.predict(X_te)

    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = cross_val_score(
        final_model,
        X_cv,
        y_cv,
        cv=tscv,
        scoring='r2',
        n_jobs=1
    )

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
        'predictions': y_pred,
        'fitted_model': final_model
    }


def evaluate_classification(name, model, X_tr, y_tr, X_te, y_te,
                            X_cv, y_cv, class_names,
                            use_scaling=False, n_splits=3):
    if use_scaling:
        final_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        final_model = model

    final_model.fit(X_tr, y_tr)
    y_pred = final_model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = cross_val_score(
        final_model,
        X_cv,
        y_cv,
        cv=tscv,
        scoring='accuracy',
        n_jobs=1
    )

    print(f"\n--- {name} ---")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  CV Accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=class_names, zero_division=0))

    return {
        'Model': name,
        'Accuracy': round(acc, 4),
        'CV_Acc_mean': round(cv_scores.mean(), 4),
        'CV_Acc_std': round(cv_scores.std(), 4),
        'predictions': y_pred,
        'fitted_model': final_model
    }


def evaluate_classification_no_cv(name, model, X_tr, y_tr, X_te, y_te,
                                  class_names, use_scaling=False):
    if use_scaling:
        final_model = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    else:
        final_model = model

    final_model.fit(X_tr, y_tr)
    y_pred = final_model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)

    print(f"\n--- {name} ---")
    print(f"  Accuracy: {acc:.4f}")
    print("  CV Accuracy (mean ± std): skipped for faster execution")
    print("\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=class_names, zero_division=0))

    return {
        'Model': name,
        'Accuracy': round(acc, 4),
        'CV_Acc_mean': np.nan,
        'CV_Acc_std': np.nan,
        'predictions': y_pred,
        'fitted_model': final_model
    }

# =============================================================================
# 5.0 REGRESSION MODELS
# =============================================================================

print("\n" + "=" * 70)
print("REGRESSION TASK: Predict Temperature Anomaly (°C)")
print("=" * 70)

# Linear Regression
lr = LinearRegression()
res_lr = evaluate_regression(
    name="Linear Regression (Baseline)",
    model=lr,
    X_tr=X_train,
    y_tr=y_train,
    X_te=X_test,
    y_te=y_test,
    X_cv=X_train_cv_reg,
    y_cv=y_train_cv_reg,
    use_scaling=True
)

# Ridge Regression (Tuned)
ridge_cv = RidgeCV(
    alphas=np.logspace(-3, 4, 20),
    cv=5
)
res_ridge = evaluate_regression(
    name="Ridge Regression (Tuned)",
    model=ridge_cv,
    X_tr=X_train,
    y_tr=y_train,
    X_te=X_test,
    y_te=y_test,
    X_cv=X_train_cv_reg,
    y_cv=y_train_cv_reg,
    use_scaling=True
)

if hasattr(res_ridge['fitted_model'], 'named_steps'):
    best_alpha = res_ridge['fitted_model'].named_steps['model'].alpha_
    print(f"Best alpha selected by RidgeCV: {best_alpha}")

# Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=8,
    random_state=42,
    n_jobs=1
)
res_rf = evaluate_regression(
    name="Random Forest Regressor",
    model=rf,
    X_tr=X_train,
    y_tr=y_train,
    X_te=X_test,
    y_te=y_test,
    X_cv=X_train_cv_reg,
    y_cv=y_train_cv_reg,
    use_scaling=False
)

# Gradient Boosting Regressor
gb = GradientBoostingRegressor(
    n_estimators=120,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
res_gb = evaluate_regression(
    name="Gradient Boosting Regressor",
    model=gb,
    X_tr=X_train,
    y_tr=y_train,
    X_te=X_test,
    y_te=y_test,
    X_cv=X_train_cv_reg,
    y_cv=y_train_cv_reg,
    use_scaling=False
)

# =============================================================================
# 6.0 REGRESSION RESULTS TABLE
# =============================================================================

results_reg = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['predictions', 'fitted_model']}
    for r in [res_lr, res_ridge, res_rf, res_gb]
])

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)
print(results_reg.to_string(index=False))

results_reg.to_csv('data\regression_results.csv', index=False)

# =============================================================================
# 7.0 REGRESSION PERFORMANCE PLOT
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['MAE', 'RMSE', 'R2']
colors = ['steelblue', 'tomato', 'seagreen']

for ax, metric, color in zip(axes, metrics, colors):
    ax.bar(results_reg['Model'], results_reg[metric], color=color, alpha=0.85)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.4)

plt.suptitle('Regression Model Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'outputs\fig_regression_model_comparison.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)

# =============================================================================
# 8.0 BEST REGRESSOR SELECTION
# =============================================================================

best_reg_result = max([res_lr, res_ridge, res_rf, res_gb], key=lambda x: x['R2'])
best_reg_name = best_reg_result['Model']
best_reg_model = best_reg_result['fitted_model']
y_pred_best = best_reg_result['predictions']

print("\n" + "=" * 70)
print("BEST REGRESSION MODEL")
print("=" * 70)
print(f"Selected based on highest test R²: {best_reg_name}")

# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.4, color='steelblue', s=10)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=1.5,
    label='Perfect Fit'
)
plt.title(f'Actual vs Predicted: {best_reg_name}', fontsize=13, fontweight='bold')
plt.xlabel('Actual Temperature Anomaly (°C)', fontsize=12)
plt.ylabel('Predicted Temperature Anomaly (°C)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(r'outputs\fig_actual_vs_predicted.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)

# Residual plot
residuals = y_test.values - y_pred_best
plt.figure(figsize=(10, 4))
plt.scatter(y_pred_best, residuals, alpha=0.3, color='purple', s=10)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title(f'Residual Plot: {best_reg_name}', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.tight_layout()
plt.savefig(r'outputs\fig_residual_plot.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)

# =============================================================================
# 9.0 FEATURE IMPORTANCE (ALWAYS FROM RANDOM FOREST)
# =============================================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE (RANDOM FOREST)")
print("=" * 70)

rf.fit(X_train, y_train)

feature_importances = pd.Series(
    rf.feature_importances_,
    index=REGRESSION_FEATURES
).sort_values(ascending=True)

plt.figure(figsize=(10, 7))
feature_importances.plot(kind='barh', color='steelblue', alpha=0.85)
plt.title('Feature Importance: Random Forest Regressor', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.tight_layout()
plt.savefig(r'outputs\fig_feature_importance.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)

print("\nTop 5 Most Important Features:")
print(feature_importances.sort_values(ascending=False).head(5))

# =============================================================================
# 10.0 CLASSIFICATION MODELS
# =============================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION TASK: Predict Warming Level (Low/Medium/High)")
print("=" * 70)

class_names = ['Low (<0.5°C)', 'Medium (0.5–1°C)', 'High (>1°C)']

# Logistic Regression
log_reg = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight='balanced'
)
res_log = evaluate_classification(
    name="Logistic Regression (Baseline)",
    model=log_reg,
    X_tr=X_train_clf,
    y_tr=y_train_clf,
    X_te=X_test_clf,
    y_te=y_test_clf,
    X_cv=X_train_cv_clf,
    y_cv=y_train_cv_clf,
    class_names=class_names,
    use_scaling=True
)

# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=80,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=1,
    class_weight='balanced'
)
res_rf_clf = evaluate_classification(
    name="Random Forest Classifier",
    model=rf_clf,
    X_tr=X_train_clf,
    y_tr=y_train_clf,
    X_te=X_test_clf,
    y_te=y_test_clf,
    X_cv=X_train_cv_clf,
    y_cv=y_train_cv_clf,
    class_names=class_names,
    use_scaling=False
)

# Gradient Boosting Classifier (no CV to avoid interruption)
gb_clf = GradientBoostingClassifier(
    n_estimators=30,
    learning_rate=0.1,
    max_depth=2,
    random_state=42
)
res_gb_clf = evaluate_classification_no_cv(
    name="Gradient Boosting Classifier",
    model=gb_clf,
    X_tr=X_train_clf,
    y_tr=y_train_clf,
    X_te=X_test_clf,
    y_te=y_test_clf,
    class_names=class_names,
    use_scaling=False
)

# =============================================================================
# 11.0 CLASSIFICATION RESULTS TABLE
# =============================================================================

results_clf = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ['predictions', 'fitted_model']}
    for r in [res_log, res_rf_clf, res_gb_clf]
])

print("\n" + "=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)
print(results_clf.to_string(index=False))

results_clf.to_csv('data\classification_results.csv', index=False)

# =============================================================================
# 12.0 BEST CLASSIFIER SELECTION
# =============================================================================

best_clf_result = max([res_log, res_rf_clf, res_gb_clf], key=lambda x: x['Accuracy'])
best_clf_name = best_clf_result['Model']
best_clf_preds = best_clf_result['predictions']

print("\n" + "=" * 70)
print("BEST CLASSIFICATION MODEL")
print("=" * 70)
print(f"Selected based on highest test accuracy: {best_clf_name}")

cm = confusion_matrix(y_test_clf, best_clf_preds)

fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title(f'Confusion Matrix: {best_clf_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(r'outputs\fig_confusion_matrix.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)

# =============================================================================
# 13.0 FUTURE TEMPERATURE PROJECTION (ILLUSTRATIVE ONLY)
# =============================================================================

print("\n" + "=" * 70)
print("FUTURE PROJECTION (2024–2050) — ILLUSTRATIVE ONLY")
print("=" * 70)

projection_model = res_rf['fitted_model']

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
    row['co2_emissions'] = prev_co2 * 1.01
    row['co2_roll10'] = row['co2_emissions']
    row['co2_decade_avg'] = row['co2_emissions']
    row['co2_yoy_change'] = row['co2_emissions'] * 0.01

    if 'co2_x_renewable' in REGRESSION_FEATURES and 'renewable_energy' in row.index:
        row['co2_x_renewable'] = row['co2_emissions'] * (1 - row['renewable_energy'] / 100)

    future_rows.append(row)
    prev_co2 = row['co2_emissions']

future_df = pd.DataFrame(future_rows)
future_preds = projection_model.predict(future_df[REGRESSION_FEATURES])

historical_global = df_model.groupby('year')['temperature_anomaly'].mean().reset_index()

plt.figure(figsize=(13, 5))
plt.plot(
    historical_global['year'],
    historical_global['temperature_anomaly'],
    color='tomato',
    linewidth=1.5,
    label='Historical (Observed)'
)
plt.plot(
    list(future_years),
    future_preds,
    color='darkorange',
    linewidth=1.5,
    linestyle='--',
    label='Projected (Illustrative)'
)
plt.axvline(2023, color='gray', linestyle=':', linewidth=1.2, label='Projection Start')
plt.title('Historical & Projected Global Mean Temperature Anomaly', fontsize=13, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(r'outputs\fig_future_projection.png', bbox_inches='tight')
plt.show(block=False)
plt.pause(0.1)

# =============================================================================
# 14.0 FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 70)
print("TOP REGRESSION RESULTS")
print("=" * 70)
print(results_reg)

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)
print(results_clf)

print("\nAll models trained and evaluated successfully.")
print("Figures saved in the same folder as this script.")

plt.show(block=True)