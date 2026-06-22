"""
Phase 3e: Random Forest model — ensemble baseline, pooled across states.

Random Forest builds many decision trees independently (in parallel) on
random subsets of data/features, then averages their predictions.

Key difference from XGBoost:
  - XGBoost: trees built SEQUENTIALLY, each correcting the last (boosting)
  - Random Forest: trees built INDEPENDENTLY in PARALLEL (bagging)
  - Random Forest is generally more robust to overfitting on small datasets
    but typically less accurate than XGBoost on structured tabular data
  - Literature finding: XGBoost usually beats RF on housing price prediction
    (our hypothesis going in — we're testing this explicitly)

Same feature set as XGBoost for direct apples-to-apples comparison.

Run: python src/models/random_forest_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH   = Path("data/processed/features.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

TRAIN_CUTOFF = "2024-12-31"
STATES       = ["NSW", "VIC", "QLD", "SA", "WA"]
TARGET_COL   = "mean_price_aud"
RANDOM_SEED  = 42

# Exact same feature set as XGBoost — required for fair comparison
FEATURE_COLS = [
    "price_lag_1q", "price_lag_2q", "price_lag_4q", "price_lag_8q",
    "rolling_mean_4q", "rolling_std_4q",
    "year", "quarter_sin", "quarter_cos", "is_covid_boom",
    "state_NSW", "state_VIC", "state_QLD", "state_SA", "state_WA",
]


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """RMSE, MAE, MAPE — identical formula to all other models."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 4)}


def time_based_split(df: pd.DataFrame):
    """Time-based train/test split — same as XGBoost for consistency."""
    train = df[df["date"] <= TRAIN_CUTOFF].copy()
    test  = df[df["date"] >  TRAIN_CUTOFF].copy()
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, y_train, X_test, y_test, train, test


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Grid search for Random Forest hyperparameters.

    Key RF params:
    - n_estimators: number of trees. More = better but diminishing returns.
      RF is less sensitive to this than XGBoost (trees are independent).
    - max_depth: limits tree depth to prevent overfitting on our small dataset.
      None = grow until pure leaves (can overfit). 5-10 is safer here.
    - min_samples_leaf: minimum samples in a leaf node. Higher = more
      regularisation, less overfitting on small datasets like ours.
    - max_features: number of features considered at each split.
      'sqrt' = sqrt(n_features) is the standard RF default for regression.
    """
    param_grid = {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [5, 10, None],    # None = unlimited depth
        "min_samples_leaf": [1, 2, 4],        # higher = more regularisation
        "max_features":     ["sqrt", 0.8],    # features per split
    }

    base_model = RandomForestRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,    # use all CPU cores
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)
    print(f"  Best RF params: {grid_search.best_params_}")
    return grid_search.best_params_


def plot_rf_results(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    metrics: dict,
    importance_df: pd.DataFrame,
):
    """Plot actual vs RF predicted + feature importance."""
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    colors = {"NSW": "#1f77b4", "VIC": "#ff7f0e",
              "QLD": "#2ca02c", "SA": "#d62728", "WA": "#9467bd"}

    # ── Plot 1: Actual vs Predicted ────────────────────────────────────────
    ax = axes[0]
    for state in STATES:
        tr = train[train["state"] == state]
        te = test[test["state"] == state]

        # Solid = actual, dashed = predicted
        ax.plot(tr["date"], tr[TARGET_COL],
                color=colors[state], linewidth=2, label=f"{state} actual")
        ax.plot(tr["date"], y_pred_train[train["state"] == state],
                color=colors[state], linewidth=1.5, linestyle="--", alpha=0.6)

        if len(te) > 0:
            ax.plot(te["date"], te[TARGET_COL],
                    color=colors[state], linewidth=3)
            ax.plot(te["date"], y_pred_test[test["state"] == state],
                    color=colors[state], linewidth=2, linestyle="--", alpha=0.9)

    ax.axvline(pd.Timestamp(TRAIN_CUTOFF),
               color="red", linestyle=":", linewidth=1.5,
               label="Train/test split")

    m = metrics
    ax.set_title(
        f"Random Forest — All States | "
        f"RMSE: ${m['rmse']:,.0f}k | MAE: ${m['mae']:,.0f}k | MAPE: {m['mape']:.2f}%",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Mean Price (AUD $000s)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Feature Importance ─────────────────────────────────────────
    ax2 = axes[1]
    top = importance_df.head(12)
    ax2.barh(top["feature"], top["importance"], color="forestgreen")
    ax2.set_title("Random Forest Feature Importance (Top 12)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Importance Score (Mean Decrease Impurity)")
    ax2.invert_yaxis()
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "rf_results.png"
    plt.savefig(path, dpi=120)
    print(f"Saved Random Forest results plot to {path}")
    plt.close()


def run_rf_pipeline():
    """Main: load features, tune + train RF, evaluate, log to MLflow."""
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    X_train, y_train, X_test, y_test, train_df, test_df = time_based_split(df)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("housing_forecast_rf")

    with mlflow.start_run(run_name="RandomForest_pooled_all_states"):

        # Tune
        print("Tuning Random Forest (grid search)...")
        best_params = tune_random_forest(X_train, y_train)

        # Train final model
        print("Training final Random Forest model...")
        model = RandomForestRegressor(
            **best_params,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        # Overall metrics
        metrics = compute_metrics(y_test.values, y_pred_test)
        print(f"\nRandom Forest Test Metrics (all states pooled):")
        print(f"  RMSE: ${metrics['rmse']:,.1f}k")
        print(f"  MAE:  ${metrics['mae']:,.1f}k")
        print(f"  MAPE: {metrics['mape']:.2f}%")

        # Per-state breakdown
        print("\nPer-state breakdown:")
        per_state_metrics = []
        for state in STATES:
            mask = test_df["state"] == state
            if mask.sum() > 0:
                sm = compute_metrics(
                    y_test[mask].values,
                    y_pred_test[mask]
                )
                print(f"  {state}: RMSE=${sm['rmse']:,.1f}k "
                      f"MAE=${sm['mae']:,.1f}k "
                      f"MAPE={sm['mape']:.2f}%")
                per_state_metrics.append({
                    "state": state, "model": "RandomForest", **sm
                })

        # Feature importance
        importance_df = pd.DataFrame({
            "feature":    FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print("\nTop 5 most important features:")
        print(importance_df.head(5).to_string(index=False))

        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("train_cutoff", TRAIN_CUTOFF)
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae",  metrics["mae"])
        mlflow.log_metric("mape", metrics["mape"])

        # Plot
        plot_rf_results(
            train_df, test_df,
            y_pred_train, y_pred_test,
            metrics, importance_df
        )

        # Save metrics
        metrics_df = pd.DataFrame(per_state_metrics)
        metrics_path = REPORTS_DIR / "rf_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\nMetrics saved to {metrics_path}")

    return model, metrics


if __name__ == "__main__":
    run_rf_pipeline()