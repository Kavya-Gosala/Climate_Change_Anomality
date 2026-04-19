# =============================================================================
# PHASE 2: Data Cleaning, EDA & Visualisation
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'Times New Roman'

# --- Load data (run after Phase 1) ---
df = pd.read_csv(r'C:\Users\uzwal\OneDrive\Desktop\DS\Climate changes\global_warming_dataset.csv')

# =============================================================================
# 2.1 DATA CLEANING
# =============================================================================

print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Standardise column names: lowercase, replace spaces with underscores
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"Cleaned column names: {list(df.columns)}")

# Drop duplicate rows
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows.")

# Handle missing values:
# - Numerical columns: fill with column median (robust to outliers)
# - Categorical columns: fill with mode
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  Filled missing values in '{col}' with median ({median_val:.3f})")

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Filled missing values in '{col}' with mode ({mode_val})")

# Confirm no missing values remain
assert df.isnull().sum().sum() == 0, "Warning: Missing values still present!"
print("\nAll missing values handled. Dataset is clean.")
print(f"Final shape: {df.shape}")

# Save cleaned dataset for subsequent phases
df.to_csv(r'C:\Users\uzwal\OneDrive\Desktop\DS\Climate changes\global_warming_clean.csv', index=False)
print("Cleaned dataset saved as 'global_warming_clean.csv'")

# =============================================================================
# 2.2 EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# --- 2.2.1 Descriptive Statistics ---
print("\nDescriptive Statistics (numerical features):")
print(df[num_cols].describe().round(3))

# --- 2.2.2 Global Temperature Anomaly Over Time ---
# Group by year, compute mean temperature anomaly across all countries
yearly_avg = df.groupby('year')['temperature_anomaly'].mean().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(yearly_avg['year'], yearly_avg['temperature_anomaly'],
         color='tomato', linewidth=1.5, label='Mean Temperature Anomaly')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Baseline (0°C)')
plt.fill_between(yearly_avg['year'], 0, yearly_avg['temperature_anomaly'],
                 where=(yearly_avg['temperature_anomaly'] > 0),
                 alpha=0.3, color='red', label='Above Baseline')
plt.fill_between(yearly_avg['year'], 0, yearly_avg['temperature_anomaly'],
                 where=(yearly_avg['temperature_anomaly'] < 0),
                 alpha=0.3, color='blue', label='Below Baseline')
plt.title('Global Mean Temperature Anomaly (1900–2023)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig1_temperature_anomaly_trend.png', bbox_inches='tight')
plt.show()

# --- 2.2.3 CO2 Emissions Trend Over Time ---
yearly_co2 = df.groupby('year')['co2_emissions'].mean().reset_index()

plt.figure(figsize=(12, 5))
plt.plot(yearly_co2['year'], yearly_co2['co2_emissions'],
         color='steelblue', linewidth=1.5)
plt.title('Global Mean CO₂ Emissions Over Time (1900–2023)', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO₂ Emissions (metric tonnes per capita)', fontsize=12)
plt.tight_layout()
plt.savefig('fig2_co2_trend.png', bbox_inches='tight')
plt.show()

# --- 2.2.4 Distribution of Temperature Anomaly ---
plt.figure(figsize=(10, 4))
sns.histplot(df['temperature_anomaly'], kde=True, bins=50, color='tomato')
plt.title('Distribution of Temperature Anomaly Across All Countries & Years',
          fontsize=13, fontweight='bold')
plt.xlabel('Temperature Anomaly (°C)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('fig3_temperature_distribution.png', bbox_inches='tight')
plt.show()

# --- 2.2.5 Correlation Heatmap ---
plt.figure(figsize=(10, 8))
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', vmin=-1, vmax=1,
            linewidths=0.5, square=True)
plt.title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_correlation_heatmap.png', bbox_inches='tight')
plt.show()

# --- 2.2.6 Top 10 Countries by Average Temperature Anomaly (recent 30 years) ---
recent = df[df['year'] >= 1993]
top_countries = (recent.groupby('country')['temperature_anomaly']
                 .mean()
                 .sort_values(ascending=False)
                 .head(10)
                 .reset_index())

plt.figure(figsize=(10, 5))
sns.barplot(data=top_countries, x='temperature_anomaly', y='country',
            palette='Reds_r')
plt.title('Top 10 Countries by Mean Temperature Anomaly (1993–2023)',
          fontsize=13, fontweight='bold')
plt.xlabel('Mean Temperature Anomaly (°C)', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.tight_layout()
plt.savefig('fig5_top_countries_anomaly.png', bbox_inches='tight')
plt.show()

# --- 2.2.7 Boxplot: Temperature Anomaly by Decade ---
df['decade'] = (df['year'] // 10) * 10
plt.figure(figsize=(14, 6))
sns.boxplot(data=df, x='decade', y='temperature_anomaly', palette='coolwarm')
plt.title('Temperature Anomaly Distribution by Decade', fontsize=14, fontweight='bold')
plt.xlabel('Decade', fontsize=12)
plt.ylabel('Temperature Anomaly (°C)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('fig6_anomaly_by_decade.png', bbox_inches='tight')
plt.show()

# --- 2.2.8 Sea Level Rise vs Temperature Anomaly Scatter ---
if 'sea_level_rise' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='temperature_anomaly', y='sea_level_rise',
                    alpha=0.3, color='steelblue')
    plt.title('Sea Level Rise vs Temperature Anomaly', fontsize=13, fontweight='bold')
    plt.xlabel('Temperature Anomaly (°C)', fontsize=12)
    plt.ylabel('Sea Level Rise (mm)', fontsize=12)
    plt.tight_layout()
    plt.savefig('fig7_sealevel_vs_temp.png', bbox_inches='tight')
    plt.show()

print("\nAll EDA visualisations saved.")
print("EDA complete. Proceed to Phase 3 (Feature Engineering).")
