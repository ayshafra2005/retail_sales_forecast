"""
trend_analysis.py
-------------------
Module 2: Trend Analysis
Computes moving averages, period-over-period growth, and seasonality
patterns (day-of-week, monthly) from the daily sales series.
"""

import pandas as pd
import numpy as np


def add_moving_averages(daily_df: pd.DataFrame, windows=(7, 30)) -> pd.DataFrame:
    """Add rolling mean columns (e.g. 7-day, 30-day) to a daily series."""
    out = daily_df.copy().sort_values("date")
    for w in windows:
        out[f"ma_{w}"] = out["revenue"].rolling(window=w, min_periods=1).mean()
    return out


def month_over_month_growth(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly revenue and compute % growth vs previous month."""
    monthly = (
        daily_df.set_index("date")["revenue"]
        .resample("MS")
        .sum()
        .to_frame("revenue")
    )
    monthly["mom_growth_pct"] = monthly["revenue"].pct_change().mul(100).round(2)
    monthly = monthly.reset_index().rename(columns={"date": "month"})
    return monthly


def year_over_year_growth(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly revenue and compute % growth vs same month last year."""
    monthly = (
        daily_df.set_index("date")["revenue"]
        .resample("MS")
        .sum()
        .to_frame("revenue")
    )
    monthly["yoy_growth_pct"] = monthly["revenue"].pct_change(periods=12).mul(100).round(2)
    monthly = monthly.reset_index().rename(columns={"date": "month"})
    return monthly


def day_of_week_pattern(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Average revenue by day of week — reveals weekly seasonality."""
    tmp = daily_df.copy()
    tmp["day_of_week"] = tmp["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pattern = (
        tmp.groupby("day_of_week")["revenue"]
        .mean()
        .reindex(order)
        .round(2)
        .reset_index()
        .rename(columns={"revenue": "avg_revenue"})
    )
    return pattern


def monthly_seasonality(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Average revenue by calendar month across all years — reveals yearly seasonality."""
    tmp = daily_df.copy()
    tmp["month_name"] = tmp["date"].dt.month_name()
    tmp["month_num"] = tmp["date"].dt.month
    pattern = (
        tmp.groupby(["month_num", "month_name"])["revenue"]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("month_num")
        .drop(columns="month_num")
        .rename(columns={"revenue": "avg_revenue"})
    )
    return pattern


if __name__ == "__main__":
    from data_analysis import load_data, daily_sales

    df = load_data("data/sales_data.csv")
    daily = daily_sales(df)
    print(add_moving_averages(daily).tail())
    print(month_over_month_growth(daily).tail())
    print(day_of_week_pattern(daily))
