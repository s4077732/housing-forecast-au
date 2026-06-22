"""
Phase 3c: Prophet model — trend/seasonality baseline, fit per state.

Prophet decomposes the series into trend + seasonality + changepoints.
Key advantage over ARIMA: automatically detects structural breaks
(like the 2022 COVID boom) as trend changepoints rather than treating
them as noise.

Run: python src/models/prophet_model.py
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

DATA_PATH   = Path("data/processed/dwelling_prices_clean.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

TRAIN_CUTOFF     = "2024-12-31"
FORECAST_HORIZON = 4
STATES           = ["NSW", "VIC", "QLD", "SA", "WA"]


def compute_metrics(actual, predicted) -> dict:
    """RMSE, MAE, MAPE — identical formula to ARIMA/XGBoost for fair comparison."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((np.array(actual) - np.array(predicted))
                          / (np.array(actual) + 1e-8))) * 100
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 4)}


def fit_and_evaluate_prophet(df_state: pd.DataFrame, state: str) -> dict:
    """Fit Prophet per state and evaluate on test set.

    Prophet requires columns named exactly 'ds' (datestamp) and 'y' (value).
    changepoint_prior_scale=0.1: slightly more flexible than default (0.05)
    to detect the 2022 boom/bust structural break.
    seasonality_mode='multiplicative': seasonality scales with price level
    (appropriate since a 5% seasonal swing on $1.3M NSW differs from $973k SA).
    """
    # Train/test split
    train_df = df_state[df_state["date"] <= TRAIN_CUTOFF].copy()
    test_df  = df_state[df_state["date"] >  TRAIN_CUTOFF].copy()

    # Rename to Prophet's required format
    train_prophet = train_df[["date", "mean_price_aud"]].rename(
        columns={"date": "ds", "mean_price_aud": "y"}
    )
    test_prophet = test_df[["date", "mean_price_aud"]].rename(
        columns={"date": "ds", "mean_price_aud": "y"}
    )

    print(f"\n{state}: train={len(train_prophet)} quarters, "
          f"test={len(test_prophet)} quarters")

    # Fit Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # not relevant for quarterly data
        daily_seasonality=False,    # not relevant for quarterly data
        changepoint_prior_scale=0.1,
        seasonality_mode="multiplicative",
        interval_width=0.95,        # 95% uncertainty interval
    )
    model.fit(train_prophet)

    # Build future dataframe (covers test period + extra forecast horizon)
    n_future = len(test_df) + FORECAST_HORIZON
    future   = model.make_future_dataframe(periods=n_future, freq="QS")
    forecast = model.predict(future)

    # Extract predictions for test period dates only
    test_forecast = forecast[forecast["ds"].isin(test_prophet["ds"].values)]

    # Evaluate
    if len(test_prophet) > 0 and len(test_forecast) > 0:
        metrics = compute_metrics(
            test_prophet["y"].values,
            test_forecast["yhat"].values
        )
        print(f"  RMSE: ${metrics['rmse']:,.1f}k | "
              f"MAE: ${metrics['mae']:,.1f}k | "
              f"MAPE: {metrics['mape']:.2f}%")
    else:
        metrics = {"rmse": None, "mae": None, "mape": None}

    return {
        "state": state, "model": model,
        "train": train_prophet, "test": test_prophet,
        "forecast": forecast, "test_forecast": test_forecast,
        "metrics": metrics,
    }


def plot_prophet_results(results: list):
    """Plot actual vs Prophet forecast for all states."""
    fig, axes = plt.subplots(nrows=len(results), ncols=1,
                             figsize=(12, 4 * len(results)))

    for ax, res in zip(axes, results):
        state    = res["state"]
        train    = res["train"]
        test     = res["test"]
        forecast = res["forecast"]

        # Training actuals
        ax.plot(train["ds"], train["y"],
                color="steelblue", label="Actual (train)", linewidth=2)

        # Test actuals
        if len(test) > 0:
            ax.plot(test["ds"], test["y"],
                    color="green", label="Actual (test)", linewidth=2)

        # Prophet forecast line (yhat = point estimate)
        ax.plot(forecast["ds"], forecast["yhat"],
                color="orange", linestyle="--",
                label="Prophet Forecast", linewidth=2)

        # Uncertainty interval (yhat_lower to yhat_upper)
        ax.fill_between(forecast["ds"],
                        forecast["yhat_lower"],
                        forecast["yhat_upper"],
                        alpha=0.2, color="orange", label="95% Uncertainty")

        # Train/test split line
        ax.axvline(pd.Timestamp(TRAIN_CUTOFF),
                   color="red", linestyle=":", linewidth=1.5,
                   label="Train/test split")

        m = res["metrics"]
        if m["rmse"] is not None:
            title = (f"{state} — Prophet | "
                     f"RMSE: ${m['rmse']:,.0f}k | "
                     f"MAE: ${m['mae']:,.0f}k | "
                     f"MAPE: {m['mape']:.1f}%")
        else:
            title = f"{state} — Prophet Forecast"

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Price (AUD $000s)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "prophet_forecast.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved Prophet forecast plot to {path}")
    plt.close()


def run_prophet_pipeline():
    """Main: fit Prophet per state, log to MLflow, plot, save metrics."""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    all_results = []
    all_metrics = []

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("housing_forecast_prophet")

    for state in STATES:
        df_state = df[df["state"] == state].copy()

        with mlflow.start_run(run_name=f"Prophet_{state}"):
            res = fit_and_evaluate_prophet(df_state, state)
            all_results.append(res)

            mlflow.log_param("model_type", "Prophet")
            mlflow.log_param("state", state)
            mlflow.log_param("changepoint_prior_scale", 0.1)
            mlflow.log_param("seasonality_mode", "multiplicative")
            mlflow.log_param("train_cutoff", TRAIN_CUTOFF)

            if res["metrics"]["rmse"] is not None:
                mlflow.log_metric("rmse", res["metrics"]["rmse"])
                mlflow.log_metric("mae",  res["metrics"]["mae"])
                mlflow.log_metric("mape", res["metrics"]["mape"])

            all_metrics.append({
                "state": state, "model": "Prophet",
                **res["metrics"]
            })

    plot_prophet_results(all_results)

    print("\n=== Prophet Results Summary ===")
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df[["state", "rmse", "mae", "mape"]].to_string(index=False))

    metrics_path = REPORTS_DIR / "prophet_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    return all_results


if __name__ == "__main__":
    run_prophet_pipeline()