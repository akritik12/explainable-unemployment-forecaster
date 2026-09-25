# Explainable Unemployment Rate Forecaster
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://unemployment-forecaster-akriti.streamlit.app/)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Forecasting-green)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-orange)
![FRED](https://img.shields.io/badge/Data-FRED-red)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-ff4b4b)

An end-to-end economic forecasting project that combines machine learning with Explainable AI (SHAP) to forecast next month's U.S. unemployment rate while interpreting the economic drivers behind every prediction.

A gradient-boosted model that forecasts next month's **US unemployment rate**
from lagged macroeconomic indicators — and explains *why* it made each
prediction using SHAP, not just what it predicted.

The project automatically downloads live macroeconomic data from the Federal Reserve Economic Data (FRED) database, engineers time-series features, trains an XGBoost forecasting model, and explains every prediction using SHAP. A deployed Streamlit application allows users to interactively explore forecasts and the economic drivers behind them., trains the model, and generates every forecast and visualization from scratch—nothing is precomputed.
Built at the intersection of economics and applied machine learning, the forecasting pipeline follows time-series practices that an economist would expect before trusting a predictive model: chronological train/test splits, lagged features, rolling averages, and zero future-data leakage.


## Why explainability, not just accuracy

## Live Demo

Explore the deployed dashboard here:

**https://unemployment-forecaster-akriti.streamlit.app/**

The dashboard includes:

- Real-time unemployment trend visualization
- Interactive month selection
- Model forecast vs. actual comparison
- SHAP waterfall explanations
- Global feature importance

A forecast without a reason isn't useful to a policymaker or analyst — it's a
number to either blindly trust or ignore. 
This project pairs every prediction with a **SHAP waterfall plot** that quantifies exactly which indicators (rising Fed
funds rate, slowing industrial production, etc.) pushed the forecast up or
down, and by how much. 

## Data

Five monthly series pulled directly from [FRED](https://fred.stlouisfed.org)
(no API key needed — FRED serves public CSVs):

| Series | Description |
|---|---|
| `UNRATE` | Unemployment rate (%) — the forecast target |
| `CPIAUCSL` | Consumer Price Index (inflation proxy) |
| `FEDFUNDS` | Effective Federal Funds Rate (%) |
| `DGS10` | 10-Year Treasury yield (%) |
| `INDPRO` | Industrial Production Index |

Together, these indicators capture labor market conditions, inflation dynamics, monetary policy, financial markets, and real economic activity, providing a compact but economically meaningful forecasting feature set.
## Method

1. **Feature engineering** — 1/2/3/6/12-month lags, 3- and 6-month rolling
   means, and 1/3-month momentum for every series (`src/features.py`).
   
3. **Model** — `XGBRegressor`,configured with shallow trees and subsampling to improve generalization on a relatively small monthly time-series dataset.
4. **Evaluation** — a **chronological** train/test split (last 36 months held
   out). Shuffling time series data would leak the future into training and
   overstate accuracy — this project deliberately avoids that mistake.
5. **Explainability** — `shap.TreeExplainer` generates both a global feature
   importance summary and a per-prediction waterfall explanation.

## Quickstart

The project was developed in **Google Colab**, and the easiest way to reproduce the analysis is by running `notebooks/Explainable_Unemployment_forecast.ipynb`.

To run the modular pipeline locally:

```bash
pip install -r requirements.txt
python src/fetch_data.py
python src/train.py
python src/explain.py
```

## Results

_After running `train.py`, paste your own `outputs/metrics.json` numbers
here, e.g.:_

| Metric | Value |
|---|---|
| MAE (test) | ~ 0.188 pts |
| RMSE (test) | ~0.3 pts |
| R² (test) | ~0.85 |

_(Performance is evaluated on the most recent 36 months, held out chronologically to preserve the integrity of time-series forecasting)_

## Project structure

```
notebooks/
  colab_walkthrough.ipynb  Self-contained, narrated notebook -- run in Colab, no setup
src/
  fetch_data.py    Pulls FRED series, builds data/macro_monthly.csv
  features.py       Lag / rolling / momentum feature engineering
  train.py          Chronological split, XGBoost training, metrics
  explain.py         SHAP summary + waterfall plots -> outputs/

This dual structure makes the project approachable for readers while keeping the code reusable for future forecasting workflows.

app/
  streamlit_app.py  Interactive dashboard
data/                Downloaded macro series (generated, gitignored)
outputs/             Trained model, metrics, SHAP plots (generated, gitignored)
```

The notebook and the `src/` scripts implement the same pipeline two ways:
the notebook is for reading top-to-bottom and running in one click; the
scripts are for anyone who wants to import `features.py` or automate the
pipeline outside a notebook.

## Dashboard Preview

### Live Dashboard

The deployed Streamlit dashboard allows users to explore historical unemployment trends, compare forecasts with actual values, and understand model decisions through SHAP explainability.

![Dashboard](outputs/screenshots/dashboard.png)

### Forecast vs. Actual

Interactive comparison of predicted and observed unemployment rates.

![Forecast Chart](outputs/screenshots/forecast_chart.png)

### SHAP Waterfall Explanation

Each forecast is accompanied by a local SHAP explanation showing how individual macroeconomic indicators influenced the prediction.

![Waterfall](outputs/screenshots/waterfall.png)

### Global Feature Importance

SHAP summary plot highlighting the variables that consistently influence unemployment forecasts across the dataset.

![SHAP Summary](outputs/screenshots/shap_summary.png)
## Ideas for extending this

- Compare SHAP explanations across different economic regimes (pre- and post-pandemic).
- Add a naive/ARIMA baseline to `train.py` so the ML lift is quantified, not assumed.
- Backtest with a rolling-origin (walk-forward) evaluation instead of one fixed split.
- Deploy the Streamlit app on Streamlit Community Cloud and link it from this README.

## Disclaimer

Educational/portfolio project.
Not investment or policy advice — macro
forecasting is genuinely hard, and this model has not been validated for
real-world decision-making.
