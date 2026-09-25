import json
import subprocess
import sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "macro_monthly.csv"
FEATURES_PATH = ROOT / "outputs" / "feature_frame.csv"
MODEL_PATH = ROOT / "outputs" / "model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics.json"

st.set_page_config(page_title="Explainable Unemployment Forecaster", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    raw = pd.read_csv(DATA_PATH, index_col="date", parse_dates=["date"])
    feature_frame = pd.read_csv(FEATURES_PATH, index_col="date", parse_dates=["date"])
    metrics = json.loads(METRICS_PATH.read_text())
    explainer = shap.TreeExplainer(model)
    return model, raw, feature_frame, metrics, explainer


def train_if_needed():
    """Train the model automatically if artifacts are missing."""
    if MODEL_PATH.exists() and FEATURES_PATH.exists() and METRICS_PATH.exists():
        return

    with st.spinner("Preparing the forecasting model for first launch..."):
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "fetch_data.py")],
            check=True,
            cwd=ROOT,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "train.py")],
            check=True,
            cwd=ROOT,
        )


def main():

    st.title("📈 Explainable US Unemployment Rate Forecaster")

    st.caption(
        "Gradient-boosted model forecasting next month's unemployment rate "
        "from lagged macro indicators (CPI, Fed funds rate, 10Y Treasury yield, "
        "industrial production). Every prediction is explained with SHAP."
    )

    try:
        train_if_needed()
        model, raw, feature_frame, metrics, explainer = load_artifacts()
    except Exception as e:
        st.error(f"Error loading project files: {e}")
        st.stop()

    target_col = "unemployment_rate_target_next"
    X = feature_frame.drop(columns=[target_col])
    y = feature_frame[target_col]

    col1, col2, col3 = st.columns(3)
    col1.metric("Test MAE", f"{metrics['mae']:.2f} pts")
    col2.metric("Test RMSE", f"{metrics['rmse']:.2f} pts")
    col3.metric("Test R²", f"{metrics['r2']:.2f}")

    st.subheader("Unemployment Rate Over Time")
    st.line_chart(raw["unemployment_rate"])

    st.subheader("Explore a Forecast")

    dates = X.index.strftime("%Y-%m").tolist()
    selected = st.select_slider("Pick a month", options=dates, value=dates[-1])
    selected_ts = pd.Timestamp(selected + "-01")

    row = X.loc[[selected_ts]]
    prediction = model.predict(row)[0]
    actual = y.loc[selected_ts]

    p1, p2 = st.columns(2)

    p1.metric("Forecast (next month)", f"{prediction:.2f}%")
    p2.metric(
        "Actual (next month)",
        f"{actual:.2f}%",
        delta=f"{prediction-actual:+.2f}",
    )

    st.subheader("Why did the model predict this?")

    shap_values = explainer(row)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False, max_display=12)
    st.pyplot(fig, clear_figure=True)

    with st.expander("Global Feature Importance"):
        fig2 = plt.figure()
        shap.summary_plot(explainer(X), X, show=False, max_display=15)
        st.pyplot(fig2, clear_figure=True)


if __name__ == "__main__":
    main()
