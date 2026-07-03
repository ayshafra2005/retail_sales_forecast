"""
inventory_recommendation.py
------------------------------
Module 5: Inventory Recommendation
Translates demand forecasts into actionable reorder recommendations
using a safety-stock / reorder-point model.

Reorder Point (ROP) = (Avg Daily Demand x Lead Time) + Safety Stock
Safety Stock = Z x StdDev(Daily Demand) x sqrt(Lead Time)

Z is the service-level factor (e.g. Z=1.65 for ~95% service level).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

SERVICE_LEVEL_Z = {
    0.90: 1.28,
    0.95: 1.65,
    0.97: 1.88,
    0.99: 2.33,
}


def compute_demand_stats(df: pd.DataFrame, group_cols=("product",)) -> pd.DataFrame:
    """Compute average and std-dev of daily units sold per product (or product+store)."""
    daily_by_group = (
        df.groupby(list(group_cols) + ["date"])["units_sold"]
        .sum()
        .reset_index()
    )
    stats = (
        daily_by_group.groupby(list(group_cols))["units_sold"]
        .agg(avg_daily_demand="mean", std_daily_demand="std")
        .reset_index()
    )
    stats["std_daily_demand"] = stats["std_daily_demand"].fillna(0)
    return stats


def recommend_inventory(
    df: pd.DataFrame,
    group_cols=("product",),
    lead_time_days: int = 7,
    service_level: float = 0.95,
    current_stock: dict = None,
) -> pd.DataFrame:
    """
    Compute reorder point, safety stock, and recommended order quantity
    for each product (or product/store combination).

    current_stock: optional dict mapping the group key (e.g. product name,
    or a tuple if group_cols has multiple columns) -> units currently on hand.
    If not provided, assumes 0 current stock (i.e. shows full reorder point).
    """
    z = SERVICE_LEVEL_Z.get(service_level, 1.65)
    stats = compute_demand_stats(df, group_cols=group_cols)

    stats["safety_stock"] = (z * stats["std_daily_demand"] * np.sqrt(lead_time_days)).round(0)
    stats["reorder_point"] = (
        stats["avg_daily_demand"] * lead_time_days + stats["safety_stock"]
    ).round(0)
    # Suggest ~30 days of coverage as the target order-up-to level
    stats["target_stock_level"] = (
        stats["avg_daily_demand"] * 30 + stats["safety_stock"]
    ).round(0)

    if current_stock:
        if len(group_cols) == 1:
            stats["current_stock"] = stats[group_cols[0]].map(current_stock).fillna(0)
        else:
            stats["current_stock"] = stats.apply(
                lambda r: current_stock.get(tuple(r[c] for c in group_cols), 0), axis=1
            )
    else:
        stats["current_stock"] = 0

    stats["recommended_order_qty"] = (
        stats["target_stock_level"] - stats["current_stock"]
    ).clip(lower=0).round(0)

    stats["status"] = np.where(
        stats["current_stock"] <= stats["reorder_point"],
        "Reorder Now",
        "Sufficient Stock",
    )

    return stats.sort_values("recommended_order_qty", ascending=False).reset_index(drop=True)


def slow_vs_fast_movers(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Identify fastest and slowest moving products by total units sold."""
    totals = df.groupby("product")["units_sold"].sum().sort_values(ascending=False)
    return {
        "fast_movers": totals.head(top_n).reset_index().rename(columns={"units_sold": "total_units"}),
        "slow_movers": totals.tail(top_n).sort_values().reset_index().rename(columns={"units_sold": "total_units"}),
    }


if __name__ == "__main__":
    from data_analysis import load_data

    df = load_data("data/sales_data.csv")
    rec = recommend_inventory(df, lead_time_days=7, service_level=0.95)
    print(rec.head(10))

    movers = slow_vs_fast_movers(df)
    print("\nFast movers:\n", movers["fast_movers"])
    print("\nSlow movers:\n", movers["slow_movers"])
