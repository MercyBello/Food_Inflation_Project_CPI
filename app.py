import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import numpy as np

# ==============================
# UK Food Inflation Dashboard
# ==============================
st.set_page_config(page_title="UK Food Inflation Dashboard", page_icon="📈", layout="wide")

# -------------------------------
# 1 Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "cleaned_cpi.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return df

# Load the dataset
df = load_data()


st.title("UK Food Inflation Dashboard")
st.markdown(
    """
This dashboard explores the **UK Consumer Price Index (CPI) – Food & Non-Alcoholic Beverages** data.
It provides interactive visualisations to understand historical trends, category-level inflation, and cost changes over time.

**Dataset:** cleaned_cpi.csv  
**Columns:** `date`, `category`, `value`  
**Skills Demonstrated:** Data cleaning, EDA, dashboard development (Streamlit, Seaborn, Matplotlib)
"""
)

# -------------------------------
# 2 Show Raw Data Preview
# -------------------------------
st.subheader("📁 Data Preview")
st.dataframe(df.head(10))

# -------------------------------
# 3 Overall Inflation Trend
# -------------------------------
st.subheader("Overall Food Inflation Trend (All Categories Combined)")

plt.figure(figsize=(12, 5))
sns.lineplot(data=df.groupby("date")["value"].mean().reset_index(), x="date", y="value", color="blue")
plt.title("Average Monthly Food Inflation Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Inflation Value (%)")
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()

# -------------------------------
# 4 Category-Level Trend
# -------------------------------
st.subheader("Category-Level Inflation Trend")

categories = sorted(df["category"].unique())
selected_category = st.selectbox("Select a Food Category:", categories)
filtered_df = df[df["category"] == selected_category]

plt.figure(figsize=(12, 5))
sns.lineplot(data=filtered_df, x="date", y="value", color="green")
plt.title(f"Inflation Trend: {selected_category}", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Inflation Value (%)")
plt.grid(True)
st.pyplot(plt.gcf())
plt.clf()

# -------------------------------
# 5️⃣ Insights & Observations
# -------------------------------
st.subheader("Insights & Observations")
st.markdown(
    """
- The overall food inflation trend helps identify long-term price changes.  
- Category-specific views allow businesses to track key product cost drivers.  

This type of dashboard is useful for **consulting, retail analytics, fintech, and public policy** applications.
"""
)

st.success("Dashboard loaded successfully! Use the dropdown to explore category trends.")

# -------------------------------
# 6️⃣ Forecasting Models – Baseline vs Advanced
# -------------------------------
st.header("Forecasting Models: Baseline vs Advanced")

st.markdown(
    """
We now compare baseline forecasting approaches with more advanced time-series models.  
The goal: **predict the next 6 months of food inflation** and see if advanced models outperform simple baselines.
"""
)

# Prepare data
series = df.groupby("date")["value"].mean()
train = series.iloc[:-6]
test = series.iloc[-6:]

# -------------------------------
# Evaluation function
# -------------------------------
def forecast_evaluation(y_true, y_pred, name):
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    st.write(f"**{name}** → MAPE: {mape:.3f}, RMSE: {rmse:.3f}")
    return mape, rmse

# ---------------------------------------
# 6.1 Baseline Models
# ---------------------------------------
st.subheader("Baseline Models")

# Naïve Forecast
naive_forecast = pd.Series(
    np.repeat(train.iloc[-1], len(test)), 
    index=test.index
)
forecast_evaluation(test, naive_forecast, "Naive Forecast")

# Seasonal Naïve (6-month lag)
seasonal_naive_forecast = pd.Series(
    series.shift(6).iloc[-6:].values, 
    index=test.index
)
forecast_evaluation(test, seasonal_naive_forecast, "Seasonal Naive Forecast")

# Filter only 2025 months for plotting
zoom_test = test[test.index.year == 2025]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(zoom_test.index, zoom_test, label="Actual", color="black", linewidth=2)
ax.plot(zoom_test.index, naive_forecast.loc[zoom_test.index], label="Naïve Forecast", linestyle="--", color="blue")
ax.plot(zoom_test.index, seasonal_naive_forecast.loc[zoom_test.index], label="Seasonal Naive", linestyle="--", color="orange")

ax.set_title("Baseline Forecast Comparison: 2025 Only", fontsize=14)
ax.set_ylabel("Monthly Inflation (%)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# ---------------------------------------
# 6.2 SARIMA (Advanced Statistical Model)
# ---------------------------------------
st.subheader("SARIMA Forecast")

sarima_model = SARIMAX(train, order=(2,1,2), seasonal_order=(2,1,2,12))
sarima_result = sarima_model.fit(disp=False)
sarima_forecast = sarima_result.forecast(steps=6)

forecast_evaluation(test, sarima_forecast, "SARIMA")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(series.index, series, label="Actual")
ax.plot(test.index, sarima_forecast, label="SARIMA Forecast", color="red")
ax.legend()
ax.set_title("SARIMA 6-Month Forecast")
st.pyplot(fig)

# ---------------------------------------
# 6.3 Prophet (Advanced ML Model)
# ---------------------------------------
st.subheader("Prophet Forecast")

prophet_df = series.reset_index().rename(columns={"date": "ds", "value": "y"})
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(prophet_df.iloc[:-6])

future = prophet_model.make_future_dataframe(periods=6, freq="M")
forecast = prophet_model.predict(future)
prophet_forecast = forecast.tail(6)["yhat"].values

forecast_evaluation(test, prophet_forecast, "Prophet")

fig2 = prophet_model.plot(forecast)
plt.title("Prophet Forecast: Next 6 Months", fontsize=14)
st.pyplot(fig2)


st.header("📈 Executive Summary & Insights")
st.markdown(
    """
**Key Findings:**

- **Baseline models performed poorly**, with both Naïve and Seasonal Naïve forecasts showing extremely high errors and failing to capture the underlying inflation dynamics.  
- **Advanced models significantly improved forecast accuracy**, with SARIMA reducing errors substantially and **Prophet delivering the most consistent performance** overall.  
- Prophet slightly outperformed SARIMA, demonstrating stronger capability in capturing **seasonal patterns and long-term trends** in UK food inflation.  
- The forecast suggests inflation will remain **volatile but shows early signs of moderation** by late **2025**, which may indicate a gradual easing of price pressures.

This dashboard demonstrates how combining exploratory analysis with time-series forecasting can support **business planning, cost-monitoring, and public-policy decision-making**.
"""
)


#fig2 = prophet_model.plot(forecast)
plt.title("Prophet Forecast: Next 6 Months", fontsize=14)
st.pyplot(fig2)

# Download Button under the plot
forecast_export = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(6)
forecast_export["ds"] = pd.to_datetime(forecast_export["ds"]).dt.strftime("%Y-%m-%d")

csv = forecast_export.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Forecast Data (CSV)",
    data=csv,
    file_name="food_inflation_forecast_2025.csv",
    mime="text/csv"
)



