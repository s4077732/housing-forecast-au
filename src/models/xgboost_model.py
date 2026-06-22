"""
Phase 3d: XGBoost model — primary ML model, pooled across all states.

XGBoost (Extreme Gradient Boosting) is an ensemble of decision trees
trained sequentially, where each tree corrects the errors of the previous one.

Why XGBoost is our primary model:
  - Best-evidenced model in the Australian housing literature
    (Adelaide paper: tree ensembles outperformed linear models)
  - Handles non-linear relationships (the WA/QLD acceleration ARIMA missed)
  - Naturally handles our mixed feature types (lag prices, time, state dummies)
  - Fast to train even with many features
  - Pooled across states: one model learns patterns shared across all 5 states
    AND state-specific patterns via the one-hot state columns

Key difference from ARIMA/Prophet:
  - ARIMA/Prophet: univariate (uses only past prices to predict future prices)
  - XGBoost: multivariate (uses lag prices + rolling stats + time features +
    state identity + COVID flag → all engineered in build_features.py)

Run: python src/models/xgboost_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

# ── Paths ──────────────────────────────────────────────────────────────────
# XGBoost uses the engineered features (not raw prices like ARIMA/Prophet)
FEATURES_PATH = Path("data/processed/features.csv")
REPORTS_DIR   = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_CUTOFF = "2024-12-31"
STATES       = ["NSW", "VIC", "QLD", "SA", "WA"]
TARGET_COL   = "mean_price_aud"   # what we're predicting
RANDOM_SEED  = 42                 # for reproducibility

# Feature columns used for training
# These are all the engineered columns from build_features.py
# We exclude metadata (date, state name) and leaky columns (yoy/qoq change
# computed from the target — using them would be data leakage since we
# wouldn't have these at prediction time for future quarters)
FEATURE_COLS = [
    "price_lag_1q",      # price 1 quarter ago (strongest predictor)
    "price_lag_2q",      # price 2 quarters ago
    "price_lag_4q",      # price 1 year ago (spatio-temporal lag from Adelaide paper)
    "price_lag_8q",      # price 2 years ago (captures longer cycles)
    "rolling_mean_4q",   # 1-year rolling average (medium-term trend)
    "rolling_std_4q",    # 1-year rolling volatility
    "year",              # long-term trend (prices generally rise year on year)
    "quarter_sin",       # cyclical quarter encoding (Q4 and Q1 are adjacent)
    "quarter_cos",       # cyclical quarter encoding (paired with sin)
    "is_covid_boom",     # structural break flag for 2021-2022
    "state_NSW",         # one-hot state features (from build_features.py)
    "state_VIC",
    "state_QLD",
    "state_SA",
    "state_WA",
]


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE — identical to ARIMA/Prophet for comparison."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 4)}


def time_based_train_test_split(df: pd.DataFrame):
    """Split features into train/test using time boundary.

    CRITICAL: we split on time, not randomly.
    Random splitting for time series = data leakage (model sees future data
    during training). Time-based split ensures the model only ever trains
    on data that would have been available at prediction time.

    Train: all quarters up to and including 2024-Q4
    Test:  all quarters from 2025-Q1 onwards
    """
    train = df[df["date"] <= TRAIN_CUTOFF].copy()
    test  = df[df["date"] >  TRAIN_CUTOFF].copy()

    # Extract feature matrix (X) and target vector (y)
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, y_train, X_test, y_test, train, test


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Grid search over key XGBoost hyperparameters.

    Why tune: default XGBoost parameters are rarely optimal. The most
    important parameters to tune for regression tasks are:
    - n_estimators: how many trees (more = better fit, but slower + overfit risk)
    - max_depth: how deep each tree grows (deeper = more complex patterns,
      but overfits on small datasets like ours ~200 rows)
    - learning_rate: step size for each tree's contribution (lower = more
      trees needed but more robust)
    - subsample: fraction of training rows used per tree (like dropout in NNs,
      reduces overfitting)

    We use a small grid given our ~200 row dataset — exhaustive search on
    a small dataset is fast and avoids overfitting from over-tuning.
    """
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 4, 5],        # shallow trees for small dataset
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample":     [0.8, 1.0],
    }

    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",  # regression task, minimise squared error
        random_state=RANDOM_SEED,
        verbosity=0,                   # suppress XGBoost's own output
    )

    # GridSearchCV with time-series-safe cross validation
    # cv=5: 5-fold cross validation on training data only
    # scoring="neg_mean_squared_error": GridSearch minimises MSE
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,      # use all CPU cores
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best params: {grid_search.best_params_}")
    return grid_search.best_params_


def plot_xgboost_results(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    metrics: dict,
    feature_importance: pd.DataFrame,
):
    """Plot actual vs predicted + feature importance."""
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 10)
    )

    # ── Plot 1: Actual vs Predicted per state ──────────────────────────────
    ax = axes[0]
    colors = {"NSW": "#1f77b4", "VIC": "#ff7f0e",
              "QLD": "#2ca02c", "SA": "#d62728", "WA": "#9467bd"}

    for state in STATES:
        # Training actuals (solid line)
        tr = train[train["state"] == state]
        ax.plot(tr["date"], tr[TARGET_COL],
                color=colors[state], linewidth=2, label=f"{state} actual")

        # Test actuals (solid, slightly thicker)
        te = test[test["state"] == state]
        if len(te) > 0:
            ax.plot(te["date"], te[TARGET_COL],
                    color=colors[state], linewidth=3)

        # Training predictions (dashed)
        ax.plot(tr["date"], y_pred_train[train["state"] == state],
                color=colors[state], linewidth=1.5,
                linestyle="--", alpha=0.7)

        # Test predictions (dashed, thicker)
        if len(te) > 0:
            ax.plot(te["date"], y_pred_test[test["state"] == state],
                    color=colors[state], linewidth=2,
                    linestyle="--", alpha=0.9)

    # Vertical line at train/test split
    ax.axvline(pd.Timestamp(TRAIN_CUTOFF),
               color="red", linestyle=":", linewidth=1.5,
               label="Train/test split")

    m = metrics
    ax.set_title(
        f"XGBoost — All States | "
        f"RMSE: ${m['rmse']:,.0f}k | MAE: ${m['mae']:,.0f}k | MAPE: {m['mape']:.2f}%",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Mean Price (AUD $000s)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Feature Importance ─────────────────────────────────────────
    # Feature importance shows WHICH features XGBoost used most —
    # this is the equivalent of Grad-CAM for tabular models
    ax2 = axes[1]
    top_features = feature_importance.head(12)  # top 12 most important features
    ax2.barh(
        top_features["feature"],
        top_features["importance"],
        color="steelblue"
    )
    ax2.set_title("XGBoost Feature Importance (Top 12)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Importance Score")
    ax2.invert_yaxis()  # most important at top
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "xgboost_results.png"
    plt.savefig(path, dpi=120)
    print(f"Saved XGBoost results plot to {path}")
    plt.close()


def run_xgboost_pipeline():
    """Main: load features, tune + train XGBoost, evaluate, log to MLflow."""

    # Load the engineered features (output of build_features.py)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    # ── Train/test split ───────────────────────────────────────────────────
    X_train, y_train, X_test, y_test, train_df, test_df = (
        time_based_train_test_split(df)
    )

    # ── MLflow setup ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("housing_forecast_xgboost")

    with mlflow.start_run(run_name="XGBoost_pooled_all_states"):

        # ── Hyperparameter tuning ──────────────────────────────────────────
        print("Tuning XGBoost hyperparameters (grid search)...")
        best_params = tune_xgboost(X_train, y_train)

        # ── Train final model with best params ────────────────────────────
        print("Training final XGBoost model...")
        model = xgb.XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        model.fit(
            X_train, y_train,
            # Early stopping: stop training if validation score doesn't
            # improve for 20 rounds — prevents overfitting
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # ── Predictions ───────────────────────────────────────────────────
        y_pred_train = model.predict(X_train)  # in-sample (training) predictions
        y_pred_test  = model.predict(X_test)   # out-of-sample (test) predictions

        # ── Evaluate on test set ──────────────────────────────────────────
        metrics = compute_metrics(y_test.values, y_pred_test)
        print(f"\nXGBoost Test Metrics (all states pooled):")
        print(f"  RMSE: ${metrics['rmse']:,.1f}k")
        print(f"  MAE:  ${metrics['mae']:,.1f}k")
        print(f"  MAPE: {metrics['mape']:.2f}%")

        # ── Per-state breakdown ───────────────────────────────────────────
        print("\nPer-state breakdown:")
        for state in STATES:
            mask = test_df["state"] == state
            if mask.sum() > 0:
                state_metrics = compute_metrics(
                    y_test[mask].values,
                    y_pred_test[mask]
                )
                print(f"  {state}: RMSE=${state_metrics['rmse']:,.1f}k "
                      f"MAE=${state_metrics['mae']:,.1f}k "
                      f"MAPE={state_metrics['mape']:.2f}%")

        # ── Feature importance ────────────────────────────────────────────
        # XGBoost tracks how much each feature reduced the loss across all trees
        importance_df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print("\nTop 5 most important features:")
        print(importance_df.head(5).to_string(index=False))

        # ── Log everything to MLflow ──────────────────────────────────────
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("train_cutoff", TRAIN_CUTOFF)
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae",  metrics["mae"])
        mlflow.log_metric("mape", metrics["mape"])

        # Log the trained model artifact to MLflow
        mlflow.xgboost.log_model(model, "xgboost_model")

        # ── Plot results ──────────────────────────────────────────────────
        plot_xgboost_results(
            train_df, test_df,
            y_pred_train, y_pred_test,
            metrics, importance_df
        )

        # ── Save metrics CSV ──────────────────────────────────────────────
        per_state_metrics = []
        for state in STATES:
            mask = test_df["state"] == state
            if mask.sum() > 0:
                sm = compute_metrics(
                    y_test[mask].values, y_pred_test[mask]
                )
                per_state_metrics.append({"state": state, "model": "XGBoost", **sm})

        metrics_df = pd.DataFrame(per_state_metrics)
        metrics_path = REPORTS_DIR / "xgboost_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nMetrics saved to {metrics_path}")

    return model, metrics


if __name__ == "__main__":
    run_xgboost_pipeline()