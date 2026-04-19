# =============================================================================
# COM747 - Data Science and Machine Learning
# PHASE 1: Setup, Data Loading & Initial Inspection
# =============================================================================

# 1.1 Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set consistent plot style for the whole project
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.family'] = 'Times New Roman'  # Matches IEEE paper font

# --- 1.2 Load Dataset ---
# Download from: https://www.kaggle.com/datasets/ankushpanday1/global-warming-dataset-195-countries-1900-2023
# Place the CSV in the same directory as this script

df = pd.read_csv(r'C:\Users\uzwal\OneDrive\Desktop\DS\Climate changes\global_warming_dataset.csv')

# --- 1.3 Initial Inspection ---
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumn Names:\n{list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 Rows:\n{df.head()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# --- 1.4 Check for Missing Values ---
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_report = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct.round(2)
})
print(missing_report[missing_report['Missing Count'] > 0])

# --- 1.5 Check for Duplicate Rows ---
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# --- 1.6 Visualise Missing Data ---
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('missing_data_heatmap.png', bbox_inches='tight')
plt.show()
print("\nMissing data heatmap saved.")

# --- 1.7 Save cleaned column names for reference ---
print("\n" + "=" * 60)
print("COLUMN REFERENCE (for team use)")
print("=" * 60)
for i, col in enumerate(df.columns):
    print(f"  [{i}] {col}")
