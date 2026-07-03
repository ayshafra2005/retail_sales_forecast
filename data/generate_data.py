"""
generate_data.py
-----------------
Creates a synthetic but realistic daily retail sales dataset covering
3 years, multiple stores, and multiple product categories.

If you already have your own sales data, skip this script entirely —
just make sure your CSV has these columns (rename if needed):
    date, store, category, product, units_sold, price, revenue

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 12, 31)

STORES = ["Store_A", "Store_B", "Store_C"]

CATEGORIES = {
    "Electronics": {"products": ["Headphones", "Smartphone", "Laptop", "Smartwatch"], "base_price": (50, 900)},
    "Apparel": {"products": ["T-Shirt", "Jeans", "Jacket", "Sneakers"], "base_price": (15, 120)},
    "Groceries": {"products": ["Rice_5kg", "Cooking_Oil", "Snacks_Pack", "Beverages"], "base_price": (2, 25)},
    "Home_Decor": {"products": ["Lamp", "Cushion", "Wall_Art", "Curtains"], "base_price": (10, 150)},
    "Beauty": {"products": ["Shampoo", "Moisturizer", "Perfume", "Lipstick"], "base_price": (5, 80)},
}

def seasonal_multiplier(date, category):
    """Adds yearly seasonality (e.g. holiday spikes) + weekly pattern."""
    day_of_year = date.timetuple().tm_yday
    # Yearly seasonal wave (peaks around November-December for most categories)
    yearly = 1 + 0.35 * np.sin(2 * np.pi * (day_of_year - 300) / 365)

    # Category-specific boosts
    month = date.month
    if category == "Electronics" and month in (11, 12):
        yearly *= 1.6  # holiday shopping
    if category == "Apparel" and month in (10, 11):
        yearly *= 1.3
    if category == "Groceries":
        yearly = 1 + 0.08 * np.sin(2 * np.pi * day_of_year / 365)  # groceries are steadier

    # Weekly pattern: weekends higher
    weekday = date.weekday()  # 0=Mon
    weekly = 1.25 if weekday >= 5 else 1.0

    return yearly * weekly

def generate():
    rows = []
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    # slow upward trend over the 3 years (business growth)
    total_days = (END_DATE - START_DATE).days

    for date in dates:
        day_index = (date - START_DATE).days
        growth = 1 + 0.00025 * day_index  # gradual growth trend

        for store in STORES:
            store_factor = {"Store_A": 1.2, "Store_B": 1.0, "Store_C": 0.8}[store]

            for category, info in CATEGORIES.items():
                season = seasonal_multiplier(date, category)
                for product in info["products"]:
                    base_demand = np.random.poisson(lam=8) + 1
                    noise = np.random.normal(1, 0.15)
                    units_sold = max(
                        0,
                        int(base_demand * season * growth * store_factor * noise)
                    )

                    low, high = info["base_price"]
                    price = round(np.random.uniform(low, high), 2)
                    revenue = round(units_sold * price, 2)

                    rows.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "store": store,
                        "category": category,
                        "product": product,
                        "units_sold": units_sold,
                        "price": price,
                        "revenue": revenue,
                    })

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate()
    out_path = "data/sales_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} rows -> {out_path}")
    print(df.head())
