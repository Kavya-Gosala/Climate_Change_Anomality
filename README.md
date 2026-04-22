# Climate Change Data Pipeline

A sequential data science pipeline for analyzing global warming data across 195 countries (1900-2023).

## Project Structure

```
Climate changes/
├── main.py                      # Entry point - runs all phases
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── scripts/
│   ├── Phase1_Setup_and_Loading (1).py
│   ├── Phase2_Cleaning_and_EDA.py
│   ├── Phase3_Feature_Engineering (2).py
│   └── Phase4_Modelling_and_Evaluation (1).py
├── data/
│   ├── global_warming_dataset.csv      # Raw input data
│   ├── global_warming_clean.csv        # Cleaned data (Phase 2)
│   ├── global_warming_features.csv     # Feature-engineered (Phase 3)
│   ├── country_encoding_map.csv        # Country label encoding
│   └── regression_results.csv          # Model results (Phase 4)
└── outputs/
    ├── missing_data_heatmap.png
    ├── fig1_temperature_anomaly_trend.png
    ├── fig2_co2_trend.png
    ├── fig3_temperature_distribution.png
    ├── fig4_correlation_heatmap.png
    ├── fig5_top_countries_anomaly.png
    ├── fig6_anomaly_by_decade.png
    ├── fig7_sealevel_vs_temp.png
    ├── fig8_regression_comparison.png
    ├── fig9_actual_vs_predicted.png
    ├── fig10_residual_plot.png
    ├── fig11_feature_importance.png
    ├── fig_confusion_matrix.png
    └── fig_future_projection.png
```

## Requirements

- Python 3.11+
- See `requirements.txt` for dependencies

## Installation

```powershell
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```powershell
python main.py
```

The pipeline executes in sequence:
1. **Phase 1**: Setup & Data Loading
2. **Phase 2**: Data Cleaning & EDA
3. **Phase 3**: Feature Engineering
4. **Phase 4**: Modelling & Evaluation (optional - run separately)

To run individual phases:

```powershell
python scripts/Phase1_Setup_and_Loading\ \(1\).py
python scripts/Phase2_Cleaning_and_EDA.py
python scripts/Phase3_Feature_Engineering\ \(2\).py
python scripts/Phase4_Modelling_and_Evaluation\ \(1\).py
```

## Data Source

- **Source**: [Kaggle - Global Warming Dataset](https://www.kaggle.com/datasets/ankushpanday1/global-warming-dataset-195-countries-1900-2023)
- **Records**: 100,000 rows × 26 columns
- **Countries**: 195
- **Time Range**: 1900-2023

## Output Files

| Phase | Output | Description |
|-------|--------|-------------|
| 1 | `missing_data_heatmap.png` | Visualizes missing data patterns |
| 2 | `global_warming_clean.csv` | Cleaned dataset |
| 2 | `fig1-7_*.png` | EDA visualizations |
| 3 | `global_warming_features.csv` | Feature-engineered dataset |
| 3 | `country_encoding_map.csv` | Country label encoding |
| 4 | `regression_results.csv` | Regression model metrics |
| 4 | `classification_results.csv` | Classification model metrics |
| 4 | `fig*.png` | Model evaluation plots |

## Features Engineered (Phase 3)

- **Temporal**: `decade`, `years_since_1900`, `post_1950`, `post_paris`
- **Rolling**: `temp_anomaly_roll10`, `co2_roll10`
- **Lag**: `temp_anomaly_lag1`, `co2_yoy_change`
- **Interaction**: `co2_x_renewable`, `co2_decade_avg`, `temp_anomaly_relative`
- **Encoded**: `country_encoded`

## Models (Phase 4)

### Regression (predict temperature anomaly)
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Classification (warming level: Low/Medium/High)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier