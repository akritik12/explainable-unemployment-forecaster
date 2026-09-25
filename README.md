# explainable-unemployment-forecaster
a model that predicts an indicator (inflation, unemployment, GDP growth) and use SHAP to explain why it predicts what it does.
# Explainable Unemployment Rate Forecaster

A gradient-boosted model that forecasts next month's **US unemployment rate**
from lagged macroeconomic indicators — and explains *why* it made each
prediction using SHAP, not just what it predicted.

**▶ Run it instantly, no setup:** click the badge above to open the full
walkthrough in Google Colab. It fetches live data, trains the model, and
generates every plot from scratch — nothing is precomputed.

Built to sit at the intersection of economics and applied ML: the modeling
choices (chronological train/test split, lag/rolling features, no shuffling)
are the ones an economist would insist on before trusting a black box.

## Why explainability, not just accuracy

A forecast without a reason isn't useful to a policymaker or analyst — it's a
number to either blindly trust or ignore. 
This project pairs every prediction with a **SHAP waterfall plot** showing exactly which indicators (rising Fed
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

## Method

1. **Feature engineering** — 1/2/3/6/12-month lags, 3- and 6-month rolling
   means, and 1/3-month momentum for every series (`src/features.py`).
2. **Model** — `XGBRegressor`, tuned lightly for a small monthly dataset
   (shallow trees, subsampling to reduce overfitting).
3. **Evaluation** — a **chronological** train/test split (last 36 months held
   out). Shuffling time series data would leak the future into training and
   overstate accuracy — this project deliberately avoids that mistake.
4. **Explainability** — `shap.TreeExplainer` generates both a global feature
   importance summary and a per-prediction waterfall explanation.

## Quickstart

**Option A — Colab (recommended, zero setup):** click the badge at the top
of this README.

**Option B — run locally:**

```bash
git clone <your-repo-url>
cd econ-forecast-explainable-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/fetch_data.py    # downloads the 5 FRED series to data/
python src/train.py         # builds features, trains the model, saves metrics
python src/explain.py       # saves SHAP plots to outputs/

streamlit run app/streamlit_app.py   # interactive dashboard
```

The Streamlit app lets you pick any historical month, see the model's
forecast next to the actual outcome, and inspect the SHAP explanation behind
that specific prediction.

## Results

_After running `train.py`, paste your own `outputs/metrics.json` numbers
here, e.g.:_

| Metric | Value |
|---|---|
| MAE (test) | ~0.2 pts |
| RMSE (test) | ~0.3 pts |
| R² (test) | ~0.85 |

_(Evaluated on the most recent 36 months, held out chronologically.)_

## Project structure

```
notebooks/
  colab_walkthrough.ipynb  Self-contained, narrated notebook -- run in Colab, no setup
src/
  fetch_data.py    Pulls FRED series, builds data/macro_monthly.csv
  features.py       Lag / rolling / momentum feature engineering
  train.py          Chronological split, XGBoost training, metrics
  explain.py         SHAP summary + waterfall plots -> outputs/
app/
  streamlit_app.py  Interactive dashboard
data/                Downloaded macro series (generated, gitignored)
outputs/             Trained model, metrics, SHAP plots (generated, gitignored)
```

The notebook and the `src/` scripts implement the same pipeline two ways:
the notebook is for reading top-to-bottom and running in one click; the
scripts are for anyone who wants to import `features.py` or automate the
pipeline outside a notebook.

## Ideas for extending this

- Swap the target for CPI inflation or GDP growth to compare forecastability.
- Add a naive/ARIMA baseline to `train.py` so the ML lift is quantified, not assumed.
- Backtest with a rolling-origin (walk-forward) evaluation instead of one fixed split.
- Deploy the Streamlit app on Streamlit Community Cloud and link it from this README.

## Disclaimer

Educational/portfolio project.
Not investment or policy advice — macro
forecasting is genuinely hard, and this model has not been validated for
real-world decision-making.
