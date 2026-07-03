"""
data_analysis.py
------------------
Module 1: Sales Data Analysis
Handles loading, cleaning, and computing core business KPIs from
the raw sales data.
"""

import pandas as pd
import numpy as np


def load_data(csv_path: str) -> pd.DataFrame:
    """Load sales data from CSV and enforce correct dtypes."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = clean_data(df)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop nulls/duplicates, fix negative values, sort."""
    df = df.drop_duplicates()
    df = df.dropna(subset=["date", "units_sold", "revenue"])

    # Guard against negative/nonsensical values
    df = df[df["units_sold"] >= 0]
    df = df[df["price"] > 0]

    # Recompute revenue defensively in case source data is inconsistent
    df["revenue"] = (df["units_sold"] * df["price"]).round(2)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_kpis(df: pd.DataFrame) -> dict:
    """Return headline KPIs for dashboard cards."""
    total_revenue = df["revenue"].sum()
    total_units = df["units_sold"].sum()
    avg_order_value = total_revenue / max(total_units, 1)
    n_days = df["date"].nunique()
    avg_daily_revenue = total_revenue / max(n_days, 1)
    top_category = (
        df.groupby("category")["revenue"].sum().sort_values(ascending=False).index[0]
    )
    top_product = (
        df.groupby("product")["revenue"].sum().sort_values(ascending=False).index[0]
    )
    top_store = (
        df.groupby("store")["revenue"].sum().sort_values(ascending=False).index[0]
    )

    return {
        "total_revenue": round(total_revenue, 2),
        "total_units_sold": int(total_units),
        "avg_unit_price": round(avg_order_value, 2),
        "avg_daily_revenue": round(avg_daily_revenue, 2),
        "top_category": top_category,
        "top_product": top_product,
        "top_store": top_store,
        "date_range": (df["date"].min().date(), df["date"].max().date()),
    }


def revenue_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate revenue & units by a given column (category/store/product)."""
    agg = (
        df.groupby(dimension)
        .agg(total_revenue=("revenue", "sum"), total_units=("units_sold", "sum"))
        .sort_values("total_revenue", ascending=False)
        .reset_index()
    )
    return agg


def daily_sales(df: pd.DataFrame, store: str = None, category: str = None) -> pd.DataFrame:
    """Return a daily aggregated revenue/units time series, optionally filtered."""
    filtered = df.copy()
    if store and store != "All":
        filtered = filtered[filtered["store"] == store]
    if category and category != "All":
        filtered = filtered[filtered["category"] == category]

    daily = (
        filtered.groupby("date")
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        .reset_index()
        .sort_values("date")
    )
    return daily


if __name__ == "__main__":
    df = load_data("data/sales_data.csv")
    print("Loaded:", df.shape)
    print(get_kpis(df))
    print(revenue_by_dimension(df, "category"))
