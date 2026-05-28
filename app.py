import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="Soyabean Forecast Dashboard",
    layout="wide"
)


# ---------------------------------
# Load Data
# ---------------------------------

results_df = pd.read_csv(
    'forecast_results.csv'
)

results_df['Date'] = pd.to_datetime(
    results_df['Date']
)


# ---------------------------------
# Title
# ---------------------------------

st.title(
    "Soyabean Price Forecast Dashboard"
)

st.write("""
AI-Based Short-Term Soyabean Mandi Price Forecasting
using AGMARKNET Data.
""")


# ---------------------------------
# Model Performance
# ---------------------------------

st.subheader("Model Performance")

metrics_df = pd.DataFrame({
    'Model': [
        'Linear Regression',
        'XGBoost',
        'Prophet'
    ],

    'MAPE': [
        1.55,
        2.08,
        10.15
    ]
})

st.dataframe(metrics_df)


# ---------------------------------
# Mandi Selector
# ---------------------------------

st.subheader("Select Mandi")

mandis = sorted(
    results_df['Market_Name'].unique()
)

selected_mandi = st.selectbox(
    "Choose a mandi",
    mandis
)


# ---------------------------------
# Filter Data
# ---------------------------------

filtered_df = results_df[
    results_df['Market_Name'] == selected_mandi
]


# ---------------------------------
# Forecast Plot
# ---------------------------------

st.subheader("Actual vs Predicted Prices")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    filtered_df['Date'],
    filtered_df['Actual'],
    label='Actual'
)

ax.plot(
    filtered_df['Date'],
    filtered_df['Predicted'],
    label='Predicted'
)

ax.set_title(
    f"{selected_mandi} Price Forecast"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.legend()

st.pyplot(fig)


# ---------------------------------
# Forecast Data
# ---------------------------------

st.subheader("Forecast Data")

st.dataframe(
    filtered_df.head(20)
)


# ---------------------------------
# Key Insights
# ---------------------------------

st.subheader("Key Insights")

st.write("""
- Soyabean prices showed strong autoregressive behavior.
- Lag features were the strongest predictors.
- Linear Regression outperformed XGBoost and Prophet.
- Forecasting using AGMARKNET-derived data was feasible.
""")


# ---------------------------------
# Footer
# ---------------------------------

st.markdown("---")

st.write(
    "Developed as part of AI-Based Agricultural Market Forecasting Project"
)