"""
app.py
--------
Module 6: Streamlit Dashboard
Interactive dashboard that ties together all 5 analysis modules:
Sales Data Analysis, Trend Analysis, ARIMA Forecasting, ML Prediction,
and Inventory Recommendation.

Run:
    streamlit run app.py
"""

import sys
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), "modules"))

from data_analysis import load_data, get_kpis, revenue_by_dimension, daily_sales
from trend_analysis import (
    add_moving_averages, month_over_month_growth,
    day_of_week_pattern, monthly_seasonality,
)
from ml_prediction import train_model, predict_future, feature_importance
from inventory_recommendation import recommend_inventory, slow_vs_fast_movers

try:
    from forecasting_arima import forecast as arima_forecast, STATSMODELS_AVAILABLE
except ImportError:
    STATSMODELS_AVAILABLE = False

st.set_page_config(page_title="Retail Sales Forecasting Dashboard", layout="wide", page_icon="📊")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sales_data.csv")


# ---------- Data loading (cached) ----------
@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_data
def get_daily(df, store, category):
    return daily_sales(df, store=store, category=category)


@st.cache_resource
def get_ml_model(daily_df, value_col, model_type):
    return train_model(daily_df, value_col=value_col, model_type=model_type)


st.title("📊 Retail Sales Forecasting & Business Dashboard")

if not os.path.exists(DATA_PATH):
    st.error(f"No data found at {DATA_PATH}. Run `python data/generate_data.py` first, "
              f"or drop your own sales_data.csv in the data/ folder (columns: "
              f"date, store, category, product, units_sold, price, revenue).")
    st.stop()

df = get_data()

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
stores = ["All"] + sorted(df["store"].unique().tolist())
categories = ["All"] + sorted(df["category"].unique().tolist())
store_filter = st.sidebar.selectbox("Store", stores)
category_filter = st.sidebar.selectbox("Category", categories)

st.sidebar.header("Forecast Settings")
forecast_periods = st.sidebar.slider("Forecast horizon (days)", 7, 90, 30)
value_col = st.sidebar.radio("Forecast target", ["revenue", "units_sold"], horizontal=True)

st.sidebar.header("Inventory Settings")
lead_time = st.sidebar.slider("Supplier lead time (days)", 1, 30, 7)
service_level = st.sidebar.select_slider("Service level", options=[0.90, 0.95, 0.97, 0.99], value=0.95)

filtered_df = df.copy()
if store_filter != "All":
    filtered_df = filtered_df[filtered_df["store"] == store_filter]
if category_filter != "All":
    filtered_df = filtered_df[filtered_df["category"] == category_filter]

daily = get_daily(df, store_filter, category_filter)

tabs = st.tabs([
    "📈 Overview", "📉 Trends", "🔮 ARIMA Forecast",
    "🤖 ML Prediction", "📦 Inventory", "🗂️ Raw Data",
])

# ============================================================
# TAB 1: OVERVIEW (Module 1 - Sales Data Analysis)
# ============================================================
with tabs[0]:
    st.subheader("Business KPIs")
    kpis = get_kpis(filtered_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
    c2.metric("Total Units Sold", f"{kpis['total_units_sold']:,}")
    c3.metric("Avg Unit Price", f"${kpis['avg_unit_price']:.2f}")
    c4.metric("Avg Daily Revenue", f"${kpis['avg_daily_revenue']:,.0f}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Top Category", kpis["top_category"])
    c6.metric("Top Product", kpis["top_product"])
    c7.metric("Top Store", kpis["top_store"])

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Revenue by Category**")
        rev_cat = revenue_by_dimension(filtered_df, "category")
        fig = px.bar(rev_cat, x="category", y="total_revenue", color="category")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Revenue by Store**")
        rev_store = revenue_by_dimension(filtered_df, "store")
        fig = px.pie(rev_store, names="store", values="total_revenue", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Daily Revenue Over Time**")
    fig = px.line(daily, x="date", y="revenue")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: TRENDS (Module 2 - Trend Analysis)
# ============================================================
with tabs[1]:
    st.subheader("Trend Analysis")

    ma_df = add_moving_averages(daily, windows=(7, 30))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ma_df["date"], y=ma_df["revenue"], name="Daily Revenue", opacity=0.4))
    fig.add_trace(go.Scatter(x=ma_df["date"], y=ma_df["ma_7"], name="7-day MA"))
    fig.add_trace(go.Scatter(x=ma_df["date"], y=ma_df["ma_30"], name="30-day MA"))
    fig.update_layout(title="Revenue with Moving Averages", height=450)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Month-over-Month Growth**")
        mom = month_over_month_growth(daily)
        fig = px.bar(mom, x="month", y="mom_growth_pct", color="mom_growth_pct",
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Day-of-Week Pattern**")
        dow = day_of_week_pattern(daily)
        fig = px.bar(dow, x="day_of_week", y="avg_revenue")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Monthly Seasonality (avg across all years)**")
    seasonality = monthly_seasonality(daily)
    fig = px.line(seasonality, x="month_name", y="avg_revenue", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3: ARIMA FORECAST (Module 3)
# ============================================================
with tabs[2]:
    st.subheader("ARIMA Sales Forecasting")

    if not STATSMODELS_AVAILABLE:
        st.warning("`statsmodels` is not installed. Run `pip install statsmodels` to enable ARIMA forecasting.")
    else:
        with st.spinner("Fitting ARIMA model..."):
            try:
                fc_df, order = arima_forecast(daily, periods=forecast_periods, value_col=value_col)
                st.caption(f"Model order used: ARIMA{order}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily["date"], y=daily[value_col], name="Historical"))
                fig.add_trace(go.Scatter(x=fc_df["date"], y=fc_df["forecast"], name="Forecast", line=dict(dash="dash")))
                fig.add_trace(go.Scatter(
                    x=pd.concat([fc_df["date"], fc_df["date"][::-1]]),
                    y=pd.concat([fc_df["upper_ci"], fc_df["lower_ci"][::-1]]),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(color="rgba(255,255,255,0)"), name="95% Confidence Interval",
                ))
                fig.update_layout(title=f"ARIMA {forecast_periods}-Day Forecast", height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Forecast Table**")
                st.dataframe(fc_df.round(2), use_container_width=True)
            except Exception as e:
                st.error(f"ARIMA fitting failed: {e}")

# ============================================================
# TAB 4: ML PREDICTION (Module 4)
# ============================================================
with tabs[3]:
    st.subheader("Machine Learning Prediction")

    model_type = st.selectbox("Model", ["random_forest", "gradient_boosting"])

    with st.spinner("Training model..."):
        model, feats, metrics = get_ml_model(daily, value_col, model_type)
        future_preds = predict_future(model, feats, daily, periods=forecast_periods, value_col=value_col)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:,.0f}")
    c2.metric("RMSE", f"{metrics['rmse']:,.0f}")
    c3.metric("R²", f"{metrics['r2']:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily[value_col], name="Historical"))
    fig.add_trace(go.Scatter(x=future_preds["date"], y=future_preds["prediction"],
                              name="ML Prediction", line=dict(dash="dash")))
    fig.update_layout(title=f"ML {forecast_periods}-Day Prediction ({model_type})", height=500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Feature Importance**")
        fi = feature_importance(model, feats).head(10)
        fig = px.bar(fi, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Prediction Table**")
        st.dataframe(future_preds, use_container_width=True, height=380)

# ============================================================
# TAB 5: INVENTORY (Module 5)
# ============================================================
with tabs[4]:
    st.subheader("Inventory Recommendations")
    st.caption(f"Based on lead time = {lead_time} days, service level = {service_level*100:.0f}%")

    rec = recommend_inventory(filtered_df, lead_time_days=lead_time, service_level=service_level)

    reorder_count = (rec["status"] == "Reorder Now").sum()
    st.metric("Products Needing Reorder", f"{reorder_count} / {len(rec)}")

    def highlight_status(row):
        color = "background-color: #ffcccc" if row["status"] == "Reorder Now" else "background-color: #ccffcc"
        return [color] * len(row)

    st.dataframe(rec.style.apply(highlight_status, axis=1), use_container_width=True, height=400)

    st.divider()
    movers = slow_vs_fast_movers(filtered_df)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔥 Fast Movers**")
        st.dataframe(movers["fast_movers"], use_container_width=True)
    with col2:
        st.markdown("**🐢 Slow Movers**")
        st.dataframe(movers["slow_movers"], use_container_width=True)

# ============================================================
# TAB 6: RAW DATA
# ============================================================
with tabs[5]:
    st.subheader("Raw Sales Data")
    st.dataframe(filtered_df, use_container_width=True, height=500)
    st.download_button(
        "Download filtered data as CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        "filtered_sales_data.csv",
        "text/csv",
    )
