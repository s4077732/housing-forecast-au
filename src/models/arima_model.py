"""
Phase 3b: ARIMA model — statistical baseline, fit per state.

ARIMA = AutoRegressive Integrated Moving Average.
  - AR (AutoRegressive): uses past values to predict future values
  - I (Integrated): differences the series to make it stationary
  - MA (Moving Average): uses past forecast errors to improve predictions

We use auto_arima (from pmdarima) which automatically selects the best
(p, d, q) order per state using AIC (Akaike Information Criterion),
so we don't have to hand-tune parameters.

Why ARIMA first: it's the standard statistical baseline in time series
literature. If ARIMA beats our ML models, that's a meaningful finding.
If ML wins, we can cite the gap as justification for the complexity.

Run: python src/models/arima_model.py
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow                          # experiment tracking
import mlflow.sklearn
from pathlib import Path
from pmdarima import auto_arima        # auto-selects best ARIMA order
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress harmless convergence warnings from ARIMA fitting
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/processed/dwelling_prices_clean.csv")
REPORTS_DIR  = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
# We train on data up to end of 2024 and test on 2025 onwards.
# Why time-based split (not random): you MUST NOT randomly shuffle time series
# data. Randomly splitting would let the model 'see the future' during training,
# making evaluation results falsely optimistic (data leakage).
TRAIN_CUTOFF = "2024-12-31"

# How many quarters ahead to forecast (4 = 1 year into the future)
FORECAST_HORIZON = 4

# The 5 states we care about for international students
STATES = ["NSW", "VIC", "QLD", "SA", "WA"]


def compute_metrics(actual: pd.Series, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, and MAPE evaluation metrics.

    RMSE (Root Mean Squared Error): penalises large errors more than small
          ones. Good for catching cases where the model is badly wrong.
    MAE  (Mean Absolute Error): average dollar error. Easy to interpret:
          'on average, our forecast is off by $X thousand'.
    MAPE (Mean Absolute Percentage Error): scale-independent error.
          Lets us compare error across states with different price levels
          (NSW at $1.3M vs SA at $973k — same MAPE means same relative accuracy).
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    # MAPE: avoid division by zero with a small epsilon
    mape = np.mean(np.abs((actual.values - predicted) / (actual.values + 1e-8))) * 100
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 4)}


def fit_and_evaluate_arima(df_state: pd.DataFrame, state: str) -> dict:
    """Fit ARIMA on training data, evaluate on test data, return metrics + forecast.

    Args:
        df_state: rows for one state only, sorted by date
        state:    state abbreviation (e.g. 'NSW')

    Returns:
        dict with metrics, fitted model, and forecast values
    """
    # ── Train/test split ─────────────────────────────────────────────────────
    # Everything up to and including 2024-Q4 is training data
    train = df_state[df_state["date"] <= TRAIN_CUTOFF]["mean_price_aud"]
    # Everything after 2024-Q4 is test data (what we evaluate against)
    test  = df_state[df_state["date"] >  TRAIN_CUTOFF]["mean_price_aud"]

    print(f"\n{state}: train={len(train)} quarters, test={len(test)} quarters")

    # ── Fit ARIMA ────────────────────────────────────────────────────────────
    # auto_arima tries many (p,d,q) combinations and picks the best by AIC.
    # seasonal=True with m=4 tells it to also look for quarterly seasonality
    # (SARIMA — Seasonal ARIMA). stepwise=True makes the search faster.
    model = auto_arima(
        train,
        seasonal=True,
        m=4,               # m=4 = quarterly seasonality (4 periods per year)
        stepwise=True,     # faster than exhaustive grid search
        suppress_warnings=True,
        error_action="ignore",
        information_criterion="aic",  # AIC balances fit quality vs model complexity
    )
    print(f"  Best ARIMA order: {model.order}, seasonal: {model.seasonal_order}")

    # ── In-sample fitted values (on training data) ───────────────────────────
    # These are the model's predictions on data it already saw - used to
    # visually check if the model captured the training trend correctly
    fitted_values = model.predict_in_sample()

    # ── Out-of-sample forecast (on test data) ────────────────────────────────
    # n_periods tells ARIMA how many future steps to predict.
    # We predict as many steps as we have test data, or FORECAST_HORIZON,
    # whichever is larger, so we always get a full 4-quarter future forecast.
    n_predict = max(len(test), FORECAST_HORIZON)
    forecast, conf_int = model.predict(
        n_periods=n_predict,
        return_conf_int=True,  # also return 95% confidence intervals
        alpha=0.05,            # 95% confidence level
    )

    # ── Evaluate on test data ────────────────────────────────────────────────
    # Only evaluate on quarters that actually have test data
    # (may be fewer than FORECAST_HORIZON if we're forecasting beyond available data)
    if len(test) > 0:
        metrics = compute_metrics(test, forecast[:len(test)])
        print(f"  RMSE: ${metrics['rmse']:,.1f}k  |  MAE: ${metrics['mae']:,.1f}k  |  MAPE: {metrics['mape']:.2f}%")
    else:
        metrics = {"rmse": None, "mae": None, "mape": None}
        print(f"  No test data available — forecast only mode")

    return {
        "state": state,
        "model": model,
        "train": train,
        "test": test,
        "fitted": fitted_values,
        "forecast": forecast,
        "conf_int": conf_int,
        "metrics": metrics,
        "train_dates": df_state[df_state["date"] <= TRAIN_CUTOFF]["date"],
        "test_dates":  df_state[df_state["date"] >  TRAIN_CUTOFF]["date"],
    }


def plot_arima_results(results: list):
    """Plot actual vs fitted + forecast for all states in one figure."""
    fig, axes = plt.subplots(
        nrows=len(results), ncols=1,
        figsize=(12, 4 * len(results)),
        sharex=False
    )

    for ax, res in zip(axes, results):
        state    = res["state"]
        train    = res["train"]
        test     = res["test"]
        forecast = res["forecast"]
        conf_int = res["conf_int"]

        # Build a continuous date index for the forecast period
        # Starting from the last training date, step forward by 1 quarter
        last_train_date = res["train_dates"].iloc[-1]
        forecast_dates  = pd.date_range(
            start=last_train_date + pd.DateOffset(months=3),
            periods=len(forecast),
            freq="QS"  # QS = Quarter Start frequency
        )

        # Plot training actuals (solid blue line)
        ax.plot(res["train_dates"], train,
                color="steelblue", label="Actual (train)", linewidth=2)

        # Plot test actuals if they exist (solid green line)
        if len(test) > 0:
            ax.plot(res["test_dates"], test,
                    color="green", label="Actual (test)", linewidth=2)

        # Plot ARIMA forecast (dashed orange line)
        ax.plot(forecast_dates, forecast,
                color="orange", linestyle="--", label="ARIMA Forecast", linewidth=2)

        # Shade the 95% confidence interval around the forecast
        # conf_int columns: [lower_bound, upper_bound]
        ax.fill_between(
            forecast_dates,
            conf_int[:, 0],  # lower bound
            conf_int[:, 1],  # upper bound
            alpha=0.2, color="orange", label="95% CI"
        )

        # Add vertical line showing where training ended / forecast begins
        ax.axvline(
            pd.Timestamp(TRAIN_CUTOFF),
            color="red", linestyle=":", linewidth=1.5, label="Train/test split"
        )

        # Add metrics to the plot title if available
        m = res["metrics"]
        if m["rmse"] is not None:
            title = f"{state} — ARIMA | RMSE: ${m['rmse']:,.0f}k | MAE: ${m['mae']:,.0f}k | MAPE: {m['mape']:.1f}%"
        else:
            title = f"{state} — ARIMA Forecast"
        ax.set_title(title, fontsize=11, fontweight="bold")

        ax.set_ylabel("Mean Price (AUD $000s)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "arima_forecast.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved forecast plot to {path}")
    plt.close()


def run_arima_pipeline():
    """Main function: load data, fit ARIMA per state, log to MLflow, plot results."""

    # Load the clean price data (not the features.csv — ARIMA only needs the
    # raw price series, it computes its own internal lags automatically)
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    all_results = []
    all_metrics = []

    # ── MLflow experiment setup ───────────────────────────────────────────────
    # MLflow tracks every model run so we can compare them later.
    # Explicitly set tracking URI to avoid SQLite conflicts on re-runs

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("housing_forecast_arima")

    for state in STATES:
        # Filter to this state's data only
        df_state = df[df["state"] == state].copy()

        # Each state gets its own MLflow run so metrics are tracked separately
        with mlflow.start_run(run_name=f"ARIMA_{state}"):

            # Fit and evaluate
            res = fit_and_evaluate_arima(df_state, state)
            all_results.append(res)

            # ── Log to MLflow ─────────────────────────────────────────────────
            # Log the ARIMA order parameters chosen by auto_arima
            mlflow.log_param("model_type", "ARIMA")
            mlflow.log_param("state", state)
            mlflow.log_param("arima_order", str(res["model"].order))
            mlflow.log_param("seasonal_order", str(res["model"].seasonal_order))
            mlflow.log_param("train_cutoff", TRAIN_CUTOFF)
            mlflow.log_param("forecast_horizon", FORECAST_HORIZON)

            # Log evaluation metrics (only if test data existed)
            if res["metrics"]["rmse"] is not None:
                mlflow.log_metric("rmse", res["metrics"]["rmse"])
                mlflow.log_metric("mae",  res["metrics"]["mae"])
                mlflow.log_metric("mape", res["metrics"]["mape"])

            # Collect metrics for summary table
            all_metrics.append({
                "state":   state,
                "model":   "ARIMA",
                "order":   res["model"].order,
                **res["metrics"]
            })

    # ── Plot all states ───────────────────────────────────────────────────────
    plot_arima_results(all_results)

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n=== ARIMA Results Summary ===")
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df[["state", "order", "rmse", "mae", "mape"]].to_string(index=False))

    # Save metrics to CSV so we can compare with Prophet/XGBoost/TFT later
    metrics_path = REPORTS_DIR / "arima_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    return all_results


if __name__ == "__main__":
    run_arima_pipeline()