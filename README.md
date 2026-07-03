# Retail Sales Forecasting & Business Dashboard

An end-to-end system that analyzes historical retail sales data, forecasts
future sales with two different methods (ARIMA and Machine Learning),
generates inventory reorder recommendations, and presents everything in
an interactive Streamlit dashboard.

## Project Structure

```
retail_sales_forecasting/
├── app.py                          # Streamlit dashboard (Module 6)
├── requirements.txt
├── data/
│   ├── generate_data.py            # Synthetic data generator
│   └── sales_data.csv              # Generated sample dataset (3 yrs, 3 stores, 5 categories)
└── modules/
    ├── data_analysis.py            # Module 1: Sales Data Analysis
    ├── trend_analysis.py           # Module 2: Trend Analysis
    ├── forecasting_arima.py        # Module 3: ARIMA Forecasting
    ├── ml_prediction.py            # Module 4: ML Prediction
    └── inventory_recommendation.py # Module 5: Inventory Recommendation
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Regenerate the sample dataset
python data/generate_data.py

# 4. Launch the dashboard
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Using Your Own Data

Replace `data/sales_data.csv` with your own file using these columns:

| column      | type   | description                    |
|-------------|--------|---------------------------------|
| date        | date   | YYYY-MM-DD                     |
| store       | string | store name/ID                  |
| category    | string | product category               |
| product     | string | product name/SKU               |
| units_sold  | int    | units sold that day             |
| price       | float  | unit price                     |
| revenue     | float  | units_sold * price (recomputed automatically) |

No code changes needed — the dashboard picks up any store/category/product
values automatically.

## Module Overview

**1. Sales Data Analysis** (`data_analysis.py`)
Loads and cleans raw data, computes headline KPIs (total revenue, units
sold, average price, top performers), and aggregates revenue by store,
category, or product.

**2. Trend Analysis** (`trend_analysis.py`)
Computes 7-day / 30-day moving averages, month-over-month and
year-over-year growth rates, and seasonality patterns (day-of-week,
month-of-year).

**3. Sales Forecasting — ARIMA** (`forecasting_arima.py`)
Fits a statistical ARIMA(p,d,q) time-series model on the daily
revenue/units series and produces a forecast with 95% confidence
intervals. Includes an Augmented Dickey-Fuller stationarity check and
a lightweight AIC-based grid search (`auto_order`) if you want the
model to pick its own order.

**4. Machine Learning Prediction** (`ml_prediction.py`)
Trains a Random Forest or Gradient Boosting regressor on engineered
features (day-of-week, month, lag values, rolling averages) and
recursively predicts forward. Reports MAE, RMSE, and R² on a
chronological (not random) train/test split, plus feature importances.
This is a good cross-check against the ARIMA forecast since it can
capture non-linear patterns ARIMA misses.

**5. Inventory Recommendation** (`inventory_recommendation.py`)
Applies a classic reorder-point / safety-stock model:

```
Safety Stock   = Z x StdDev(daily demand) x sqrt(lead time)
Reorder Point  = (Avg daily demand x lead time) + Safety Stock
```

Z is chosen from your selected service level (90/95/97/99%). Flags each
product as "Reorder Now" or "Sufficient Stock" and suggests an order
quantity to reach a ~30-day target coverage level.

**6. Streamlit Dashboard** (`app.py`)
Ties everything together with sidebar filters (store, category, forecast
horizon, lead time, service level) across six tabs: Overview, Trends,
ARIMA Forecast, ML Prediction, Inventory, and Raw Data (with CSV export).

## Notes

- The bundled dataset is synthetically generated (see `generate_data.py`)
  with realistic seasonality (holiday spikes, weekend lift, gradual
  growth trend) so every module has meaningful patterns to find. Swap in
  your real sales history any time.
- ARIMA fitting with `auto_select=True` runs a grid search and is slower;
  the default fixed order (2,1,2) is fine for most daily retail series
  and is what the dashboard uses.
- The ML recursive forecast feeds each prediction back in as a lag
  feature for the next step, so accuracy naturally degrades the further
  out you forecast — this is normal and worth showing users honestly
  rather than hiding.
