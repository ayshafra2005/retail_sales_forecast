"""
forecasting_arima.py
----------------------
Module 3: Sales Forecasting using ARIMA (statsmodels).

Fits an ARIMA model on a daily revenue (or units) time series and
produces a forward-looking forecast with confidence intervals.

Requires: statsmodels  (pip install statsmodels)
"""

import warnings
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")


def check_stationarity(series: pd.Series) -> dict:
    """Run an Augmented Dickey-Fuller test. p-value < 0.05 => stationary."""
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required: pip install statsmodels")
    result = adfuller(series.dropna())
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
    }


def fit_arima(series: pd.Series, order=(2, 1, 2)):
    """
    Fit an ARIMA(p,d,q) model on a series indexed by date.
    Default order (2,1,2) works reasonably well for noisy daily retail
    data without needing full grid search, but auto_order() below can
    be used to search for a better fit.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required: pip install statsmodels")

    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def auto_order(series: pd.Series, p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3)):
    """
    Lightweight grid search over (p,d,q) using AIC as the selection
    criterion. Good enough for a dashboard without pulling in pmdarima.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required: pip install statsmodels")

    best_aic = np.inf
    best_order = (1, 1, 1)
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order, best_aic


def forecast(daily_df: pd.DataFrame, periods: int = 30, value_col: str = "revenue",
             order=(2, 1, 2), auto_select: bool = False) -> pd.DataFrame:
    """
    Forecast `periods` days ahead.

    daily_df: DataFrame with a 'date' column and the value_col to forecast.
    Returns a DataFrame with columns: date, forecast, lower_ci, upper_ci
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required: pip install statsmodels")

    series = daily_df.set_index("date")[value_col].asfreq("D").interpolate()

    if auto_select:
        order, _ = auto_order(series)

    fitted = fit_arima(series, order=order)
    result = fitted.get_forecast(steps=periods)
    conf_int = result.conf_int(alpha=0.05)

    future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=periods, freq="D")

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": result.predicted_mean.values,
        "lower_ci": conf_int.iloc[:, 0].values,
        "upper_ci": conf_int.iloc[:, 1].values,
    })
    # Forecasted sales can't be negative
    forecast_df["forecast"] = forecast_df["forecast"].clip(lower=0)
    forecast_df["lower_ci"] = forecast_df["lower_ci"].clip(lower=0)

    return forecast_df, order


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from data_analysis import load_data, daily_sales

    df = load_data("data/sales_data.csv")
    daily = daily_sales(df)
    fc, order = forecast(daily, periods=14)
    print(f"Order used: {order}")
    print(fc)
