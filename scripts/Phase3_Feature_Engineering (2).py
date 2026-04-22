# =============================================================================
# PHASE 3: Feature Engineering & Data Preparation
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# --- Load cleaned dataset ---
df = pd.read_csv(r'data\global_warming_clean.csv')

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# =============================================================================
# 3.1 TEMPORAL FEATURES
# =============================================================================

# Decade as a categorical bin
df['decade'] = (df['year'] // 10) * 10

# Years since 1900 (normalised time index)
df['years_since_1900'] = df['year'] - 1900

# Post-industrial era flag (after 1950 when industrialisation accelerated)
df['post_1950'] = (df['year'] >= 1950).astype(int)

# Post-Paris Agreement flag (after 2015)
df['post_paris'] = (df['year'] >= 2015).astype(int)

print("Temporal features created: decade, years_since_1900, post_1950, post_paris")

# =============================================================================
# 3.2 ROLLING STATISTICS (per country)
# =============================================================================
# Sort data by country and year before computing rolling features
df = df.sort_values(['country', 'year']).reset_index(drop=True)

# 10-year rolling mean temperature anomaly per country
df['temp_anomaly_roll10'] = (
    df.groupby('country')['temperature_anomaly']
    .transform(lambda x: x.rolling(window=10, min_periods=3).mean())
)

# 10-year rolling mean CO2 per country
df['co2_roll10'] = (
    df.groupby('country')['co2_emissions']
    .transform(lambda x: x.rolling(window=10, min_periods=3).mean())
)

# Year-on-year change in temperature anomaly (lag feature)
df['temp_anomaly_lag1'] = (
    df.groupby('country')['temperature_anomaly'].shift(1)
)

# Year-on-year change in CO2
df['co2_yoy_change'] = (
    df.groupby('country')['co2_emissions']
    .transform(lambda x: x.diff())
)

print("Rolling & lag features created: roll10, lag1, yoy_change")

# =============================================================================
# 3.3 INTERACTION FEATURES
# =============================================================================

# CO2 × Renewable Energy interaction 
if 'renewable_energy' in df.columns:
    df['co2_x_renewable'] = df['co2_emissions'] * (1 - df['renewable_energy'] / 100)
    print("Interaction feature created: co2_x_renewable")

# CO2 per decade average (captures long-term trend)
df['co2_decade_avg'] = df.groupby(['country', 'decade'])['co2_emissions'].transform('mean')

# Temperature anomaly relative to country historical mean
df['country_baseline_temp'] = df.groupby('country')['temperature_anomaly'].transform('mean')
df['temp_anomaly_relative'] = df['temperature_anomaly'] - df['country_baseline_temp']

print("Interaction & relative features created.")

# =============================================================================
# 3.4 ENCODE CATEGORICAL VARIABLES
# =============================================================================

# Encode 'country' using Label Encoding (for tree-based models)
le = LabelEncoder()
df['country_encoded'] = le.fit_transform(df['country'])
print(f"Country encoded: {df['country'].nunique()} unique countries → integer codes")

country_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
mapping_df = pd.DataFrame(list(country_mapping.items()), columns=['Country', 'Code'])
mapping_df.to_csv('data\country_encoding_map.csv', index=False)
print("Country encoding map saved.")

# =============================================================================
# 3.5 DEFINE TARGET & FEATURES — TASK 1: REGRESSION
# =============================================================================
# Target: temperature_anomaly (continuous — predict warming level)

# Drop rows where rolling/lag features are NaN (first few years per country)
df_model = df.dropna().copy()
print(f"\nRows after dropping NaN from lag/roll features: {len(df_model)}")

# Feature set for regression
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

# optional columns 
optional_cols = ['sea_level_rise', 'arctic_ice_extent',
                 'extreme_weather_events', 'renewable_energy',
                 'co2_x_renewable']
for col in optional_cols:
    if col in df_model.columns:
        REGRESSION_FEATURES.append(col)

TARGET_REGRESSION = 'temperature_anomaly'

X = df_model[REGRESSION_FEATURES]
y = df_model[TARGET_REGRESSION]

print(f"\nRegression Features ({len(REGRESSION_FEATURES)}): {REGRESSION_FEATURES}")
print(f"Target: {TARGET_REGRESSION}")

# =============================================================================
# 3.6 DEFINE TARGET & FEATURES — TASK 2: CLASSIFICATION
# =============================================================================
# Target: classify warming level into 3 categories
# Low: anomaly < 0.5°C | Medium: 0.5–1.0°C | High: > 1.0°C

def classify_warming(anomaly):
    if anomaly < 0.5:
        return 0  # Low
    elif anomaly < 1.0:
        return 1  # Medium
    else:
        return 2  # High

df_model['warming_class'] = df_model['temperature_anomaly'].apply(classify_warming)
TARGET_CLASSIFICATION = 'warming_class'

print(f"\nClassification target distribution:")
print(df_model['warming_class'].value_counts().rename({0: 'Low', 1: 'Medium', 2: 'High'}))

X_clf = df_model[REGRESSION_FEATURES]  # same features
y_clf = df_model[TARGET_CLASSIFICATION]

# =============================================================================
# 3.7 TRAIN / TEST SPLIT
# =============================================================================
# Use time-aware split: train on pre-2010, test on 2010 onwards

train_mask = df_model['year'] < 2010
test_mask = df_model['year'] >= 2010

X_train_reg = df_model.loc[train_mask, REGRESSION_FEATURES]
X_test_reg  = df_model.loc[test_mask, REGRESSION_FEATURES]
y_train_reg = df_model.loc[train_mask, TARGET_REGRESSION]
y_test_reg  = df_model.loc[test_mask, TARGET_REGRESSION]

X_train_clf = df_model.loc[train_mask, REGRESSION_FEATURES]
X_test_clf  = df_model.loc[test_mask, REGRESSION_FEATURES]
y_train_clf = df_model.loc[train_mask, TARGET_CLASSIFICATION]
y_test_clf  = df_model.loc[test_mask, TARGET_CLASSIFICATION]

print(f"\nTime-based Train/Test Split:")
print(f"  Train (pre-2010): {len(X_train_reg)} rows")
print(f"  Test  (2010+):    {len(X_test_reg)} rows")

# =============================================================================
# 3.8 FEATURE SCALING (for linear models)
# =============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reg)
X_test_scaled  = scaler.transform(X_test_reg)

print("\nFeature scaling applied (StandardScaler).")

# =============================================================================
# 3.9 SAVE PREPARED DATA
# =============================================================================

df_model.to_csv(r'data\global_warming_features.csv', index=False)
print("\nFeature-engineered dataset saved as 'global_warming_features.csv'")
print("\nPhase 3 complete. Proceed to Phase 4 (Modelling).")

# Summary of all features created
print("\n" + "=" * 60)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 60)
feature_info = {
    'years_since_1900':      'Linear time index from 1900',
    'post_1950':             'Binary flag: post-industrial era',
    'post_paris':            'Binary flag: post-Paris Agreement 2015',
    'decade':                'Decade bin (1900, 1910, ...)',
    'temp_anomaly_roll10':   '10-year rolling mean temp anomaly per country',
    'temp_anomaly_lag1':     'Previous year temperature anomaly (lag)',
    'co2_roll10':            '10-year rolling mean CO2 per country',
    'co2_yoy_change':        'Year-on-year CO2 change',
    'co2_decade_avg':        'Decade-average CO2 per country',
    'temp_anomaly_relative': 'Anomaly relative to country historical mean',
    'warming_class':         'Classification target (Low/Medium/High)',
    'country_encoded':       'Integer-encoded country label',
}
for feat, desc in feature_info.items():
    print(f"  {feat:<30} {desc}")
