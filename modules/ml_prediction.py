"""
ml_prediction.py
-------------------
Module 4: Machine Learning Prediction
Uses a Random Forest / Gradient Boosting regressor with engineered
date + lag features to predict future daily sales. This complements
ARIMA by capturing non-linear patterns (day-of-week, month, lags)
and gives a second forecast to cross-check against.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_features(daily_df: pd.DataFrame, value_col: str = "revenue", lags=(1, 7, 14, 30)) -> pd.DataFrame:
    """Engineer date-based and lag features for ML forecasting."""
    df = daily_df.copy().sort_values("date").reset_index(drop=True)

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_index"] = (df["date"] - df["date"].min()).dt.days  # trend proxy

    for lag in lags:
        df[f"lag_{lag}"] = df[value_col].shift(lag)

    df["rolling_mean_7"] = df[value_col].shift(1).rolling(window=7, min_periods=1).mean()
    df["rolling_mean_30"] = df[value_col].shift(1).rolling(window=30, min_periods=1).mean()

    return df


def train_model(daily_df: pd.DataFrame, value_col: str = "revenue",
                 model_type: str = "random_forest", test_size: float = 0.15):
    """
    Train an ML regressor on engineered features. Returns the fitted
    model, feature list, and evaluation metrics on a held-out tail split
    (chronological split, not random, since this is time series data).
    """
    feat_df = build_features(daily_df[["date", value_col]], value_col=value_col).dropna().reset_index(drop=True)

    feature_cols = [c for c in feat_df.columns if c not in ("date", value_col)]
    X = feat_df[feature_cols]
    y = feat_df[value_col]

    split_idx = int(len(feat_df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if model_type == "gradient_boosting":
        model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        "r2": round(r2_score(y_test, preds), 3),
        "test_size": len(y_test),
    }

    return model, feature_cols, metrics


def predict_future(model, feature_cols, daily_df: pd.DataFrame, periods: int = 30, value_col: str = "revenue") -> pd.DataFrame:
    """
    Recursively predict `periods` days ahead, feeding each new prediction
    back in as a lag feature for the next step (since future lags are
    unknown).
    """
    history = daily_df[["date", value_col]].copy().sort_values("date").reset_index(drop=True)
    predictions = []

    for step in range(periods):
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        temp = pd.concat([history, pd.DataFrame({"date": [next_date], value_col: [np.nan]})], ignore_index=True)
        feat_df = build_features(temp, value_col=value_col)
        row = feat_df.iloc[[-1]][feature_cols]

        pred = max(0, model.predict(row)[0])
        predictions.append({"date": next_date, "prediction": round(pred, 2)})

        history = pd.concat([history, pd.DataFrame({"date": [next_date], value_col: [pred]})], ignore_index=True)

    return pd.DataFrame(predictions)


def feature_importance(model, feature_cols) -> pd.DataFrame:
    """Return sorted feature importances for interpretability."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    fi = pd.DataFrame({"feature": feature_cols, "importance": importances})
    return fi.sort_values("importance", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from data_analysis import load_data, daily_sales

    df = load_data("data/sales_data.csv")
    daily = daily_sales(df)

    model, feats, metrics = train_model(daily)
    print("Metrics:", metrics)
    print(feature_importance(model, feats).head(8))

    future = predict_future(model, feats, daily, periods=14)
    print(future)
