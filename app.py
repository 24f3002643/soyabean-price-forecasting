import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="Soyabean Forecast Dashboard",
    layout="wide"
)


# ---------------------------------
# Load Trained Model
# ---------------------------------

lr_model = joblib.load(
    'linear_model.pkl'
)


# ---------------------------------
# Load Engineered Dataset
# ---------------------------------

df = pd.read_csv(
    'final_engineered_data.csv'
)

df['Reported_Date'] = pd.to_datetime(
    df['Reported_Date']
)


# ---------------------------------
# Feature Columns
# ---------------------------------

feature_cols = [

    'Arrivals_Tonnes',

    'lag_1',
    'lag_3',
    'lag_7',
    'lag_14',

    'rolling_mean_7',
    'rolling_mean_30',
    'rolling_std_7',

    'arrival_lag_7',
    'arrival_roll_7',

    'year',
    'month',
    'day',
    'day_of_week',
    'quarter'
]


# ---------------------------------
# Forecast Function
# ---------------------------------

def forecast_next_7_days(
    mandi_name,
    model,
    df,
    feature_cols
):

    mandi_df = df[
        df['Market_Name'] == mandi_name
    ].copy()

    mandi_df = mandi_df.sort_values(
        'Reported_Date'
    )

    latest_row = mandi_df.iloc[-1].copy()

    future_predictions = []

    current_row = latest_row.copy()


    # ---------------------------------
    # Recursive Forecasting
    # ---------------------------------

    for i in range(7):

        features = pd.DataFrame([[
            
            current_row['Arrivals_Tonnes'],

            current_row['lag_1'],
            current_row['lag_3'],
            current_row['lag_7'],
            current_row['lag_14'],

            current_row['rolling_mean_7'],
            current_row['rolling_mean_30'],
            current_row['rolling_std_7'],

            current_row['arrival_lag_7'],
            current_row['arrival_roll_7'],

            current_row['year'],
            current_row['month'],
            current_row['day'],
            current_row['day_of_week'],
            current_row['quarter']

        ]], columns=feature_cols)


        pred_price = model.predict(features)[0]

        future_predictions.append(pred_price)


        # ---------------------------------
        # Update lag features
        # ---------------------------------

        current_row['lag_14'] = current_row['lag_7']
        current_row['lag_7'] = current_row['lag_3']
        current_row['lag_3'] = current_row['lag_1']
        current_row['lag_1'] = pred_price


        # ---------------------------------
        # Update rolling mean
        # ---------------------------------

        current_row['rolling_mean_7'] = (
            (
                current_row['rolling_mean_7'] * 6
            ) + pred_price
        ) / 7


        # ---------------------------------
        # Update date
        # ---------------------------------

        next_date = (
            current_row['Reported_Date']
            + pd.Timedelta(days=1)
        )

        current_row['Reported_Date'] = next_date

        current_row['year'] = next_date.year
        current_row['month'] = next_date.month
        current_row['day'] = next_date.day
        current_row['day_of_week'] = next_date.dayofweek
        current_row['quarter'] = next_date.quarter


    # ---------------------------------
    # Create Forecast DataFrame
    # ---------------------------------

    future_dates = pd.date_range(

        start=(
            mandi_df['Reported_Date'].max()
            + pd.Timedelta(days=1)
        ),

        periods=7
    )

    forecast_df = pd.DataFrame({

        'Date': future_dates,

        'Predicted_Price': future_predictions
    })

    return forecast_df


# ---------------------------------
# Load Historical Prediction Data
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
# Future Forecast
# ---------------------------------

forecast_df = forecast_next_7_days(
    selected_mandi,
    lr_model,
    df,
    feature_cols
)


# ---------------------------------
# Future Forecast Plot
# ---------------------------------

st.subheader("Next 7-Day Forecast")

fig2, ax2 = plt.subplots(figsize=(12,5))

ax2.plot(
    forecast_df['Date'],
    forecast_df['Predicted_Price'],
    marker='o'
)

ax2.set_title(
    f"7-Day Forecast: {selected_mandi}"
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Predicted Price")

st.pyplot(fig2)


# ---------------------------------
# Future Forecast Table
# ---------------------------------

st.subheader("Forecasted Prices")

st.dataframe(forecast_df)


# ---------------------------------
# Historical Prediction Comparison
# ---------------------------------

filtered_df = results_df[
    results_df['Market_Name'] == selected_mandi
]


# ---------------------------------
# Historical Plot
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
    f"{selected_mandi} Historical Forecast Performance"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.legend()

st.pyplot(fig)


# ---------------------------------
# Historical Data Preview
# ---------------------------------

st.subheader("Historical Forecast Data")

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