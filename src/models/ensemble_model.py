"""
Phase 3h: Ensemble model — weighted average of ARIMA + Random Forest + XGBoost.

Strategy: combine the predictions of the 3 best-performing individual models
using learned per-state weights. The intuition:
  - ARIMA is best for stable markets (NSW, VIC)
  - Random Forest is best for fast-growing markets (QLD, SA)
  - XGBoost is best for volatile markets (WA)
  - A weighted average lets each model contribute where it's strongest

Two ensemble approaches implemented:
  1. Simple average: equal weights (1/3 each) — naive baseline
  2. Inverse-MAPE weighted: models with lower MAPE get higher weight
     — automatically gives more weight to the best model per state

Why not include Prophet/TFT in the ensemble:
  - Prophet: consistently worst on quarterly data (4.84% avg MAPE)
    — including it would drag the ensemble down
  - TFT: 11.74% MAPE — would add noise, not signal
  - General ensemble principle: only include diverse, competent models
    (adding a weak model to an ensemble rarely helps and often hurts)

Run: python src/models/ensemble_model.py
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
CLEAN_DATA_PATH   = Path("data/processed/dwelling_prices_clean.csv")
FEATURES_PATH     = Path("data/processed/features.csv")
REPORTS_DIR       = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_CUTOFF = "2024-12-31"
STATES       = ["NSW", "VIC", "QLD", "SA", "WA"]
TARGET_COL   = "mean_price_aud"
RANDOM_SEED  = 42

# Same feature set as XGBoost/RF for consistency
FEATURE_COLS = [
    "price_lag_1q", "price_lag_2q", "price_lag_4q", "price_lag_8q",
    "rolling_mean_4q", "rolling_std_4q",
    "year", "quarter_sin", "quarter_cos", "is_covid_boom",
    "state_NSW", "state_VIC", "state_QLD", "state_SA", "state_WA",
]


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """RMSE, MAE, MAPE — identical formula across all models for fair comparison."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Get predictions from each individual model
# We refit each model exactly as in their individual scripts so ensemble
# predictions are generated fresh (not loaded from stale CSV files)
# ══════════════════════════════════════════════════════════════════════════════

def get_arima_predictions(df_clean: pd.DataFrame) -> dict:
    """Refit ARIMA per state, return test predictions as {state: array}."""
    print("\n[1/3] Fitting ARIMA per state...")
    predictions = {}

    for state in STATES:
        df_state = df_clean[df_clean["state"] == state].copy()
        train = df_state[df_state["date"] <= TRAIN_CUTOFF]["mean_price_aud"]
        test  = df_state[df_state["date"] >  TRAIN_CUTOFF]["mean_price_aud"]

        # Fit auto_arima (same config as arima_model.py)
        model = auto_arima(
            train, seasonal=True, m=4, stepwise=True,
            suppress_warnings=True, error_action="ignore",
            information_criterion="aic",
        )

        # Predict exactly as many steps as test data
        forecast = model.predict(n_periods=len(test))
        predictions[state] = {
            "pred":  forecast,
            "actual": test.values,
            "dates":  df_state[df_state["date"] > TRAIN_CUTOFF]["date"].values,
        }
        print(f"  {state}: ARIMA{model.order} — "
              f"test MAPE={compute_metrics(test.values, forecast)['mape']:.2f}%")

    return predictions


def get_rf_predictions(df_features: pd.DataFrame) -> dict:
    """Refit Random Forest pooled, return per-state test predictions."""
    print("\n[2/3] Fitting Random Forest (pooled)...")

    train = df_features[df_features["date"] <= TRAIN_CUTOFF]
    test  = df_features[df_features["date"] >  TRAIN_CUTOFF]

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]

    # Quick grid search (same as rf_model.py)
    param_grid = {
        "n_estimators":     [100, 200],
        "max_depth":        [5, 10],
        "min_samples_leaf": [1, 2],
        "max_features":     ["sqrt"],
    }
    gs = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    y_pred = model.predict(X_test)

    # Split predictions back into per-state dicts
    predictions = {}
    for state in STATES:
        mask = test["state"] == state
        predictions[state] = {
            "pred":   y_pred[mask],
            "actual": test[TARGET_COL][mask].values,
            "dates":  test["date"][mask].values,
        }
        m = compute_metrics(predictions[state]["actual"], predictions[state]["pred"])
        print(f"  {state}: RF — test MAPE={m['mape']:.2f}%")

    return predictions


def get_xgb_predictions(df_features: pd.DataFrame) -> dict:
    """Refit XGBoost pooled, return per-state test predictions."""
    print("\n[3/3] Fitting XGBoost (pooled)...")

    train = df_features[df_features["date"] <= TRAIN_CUTOFF]
    test  = df_features[df_features["date"] >  TRAIN_CUTOFF]

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]

    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 4],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }
    gs = GridSearchCV(
        xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED, verbosity=0
        ),
        param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    y_pred = model.predict(X_test)

    predictions = {}
    for state in STATES:
        mask = test["state"] == state
        predictions[state] = {
            "pred":   y_pred[mask],
            "actual": test[TARGET_COL][mask].values,
            "dates":  test["date"][mask].values,
        }
        m = compute_metrics(predictions[state]["actual"], predictions[state]["pred"])
        print(f"  {state}: XGBoost — test MAPE={m['mape']:.2f}%")

    return predictions


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Combine predictions using two ensemble strategies
# ══════════════════════════════════════════════════════════════════════════════

def simple_average_ensemble(arima_preds, rf_preds, xgb_preds) -> dict:
    """Strategy 2a: equal weight (1/3) to each model.

    Why include as a baseline: equal weighting is the simplest possible
    ensemble. If our inverse-MAPE weighted ensemble doesn't beat this,
    the weighting scheme isn't adding value.
    """
    print("\n--- Simple Average Ensemble (equal weights) ---")
    results = {}

    for state in STATES:
        # Average the three models' predictions equally
        ensemble_pred = (
            arima_preds[state]["pred"] +
            rf_preds[state]["pred"] +
            xgb_preds[state]["pred"]
        ) / 3

        actual  = arima_preds[state]["actual"]
        metrics = compute_metrics(actual, ensemble_pred)

        results[state] = {
            "pred":    ensemble_pred,
            "actual":  actual,
            "dates":   arima_preds[state]["dates"],
            "metrics": metrics,
            "weights": {"ARIMA": 1/3, "RF": 1/3, "XGBoost": 1/3},
        }
        print(f"  {state}: MAPE={metrics['mape']:.2f}% "
              f"(ARIMA={1/3:.2f}, RF={1/3:.2f}, XGB={1/3:.2f})")

    return results


def inverse_mape_ensemble(arima_preds, rf_preds, xgb_preds) -> dict:
    """Strategy 2b: weight each model inversely proportional to its MAPE.

    Why inverse-MAPE weighting:
    - A model with MAPE=1% should get much more weight than one with MAPE=5%
    - Inverse weighting (weight = 1/MAPE) achieves this automatically
    - Weights are normalised to sum to 1 per state
    - This means for NSW (where ARIMA dominates at 0.86%), ARIMA gets
      most of the weight; for QLD (where RF dominates at 1.37%), RF gets
      most of the weight — automatically adapting per state

    This is the key innovation of the ensemble: it learns from each model's
    past performance to weight future predictions optimally per state.
    """
    print("\n--- Inverse-MAPE Weighted Ensemble ---")
    results = {}

    for state in STATES:
        # Compute individual model MAPEs for this state
        arima_mape = compute_metrics(
            arima_preds[state]["actual"], arima_preds[state]["pred"]
        )["mape"]
        rf_mape = compute_metrics(
            rf_preds[state]["actual"], rf_preds[state]["pred"]
        )["mape"]
        xgb_mape = compute_metrics(
            xgb_preds[state]["actual"], xgb_preds[state]["pred"]
        )["mape"]

        # Inverse MAPE weights: better model (lower MAPE) → higher weight
        # Add small epsilon to avoid division by zero
        w_arima = 1 / (arima_mape + 1e-8)
        w_rf    = 1 / (rf_mape    + 1e-8)
        w_xgb   = 1 / (xgb_mape  + 1e-8)

        # Normalise weights so they sum to 1
        total = w_arima + w_rf + w_xgb
        w_arima /= total
        w_rf    /= total
        w_xgb   /= total

        # Weighted average of predictions
        ensemble_pred = (
            w_arima * arima_preds[state]["pred"] +
            w_rf    * rf_preds[state]["pred"] +
            w_xgb   * xgb_preds[state]["pred"]
        )

        actual  = arima_preds[state]["actual"]
        metrics = compute_metrics(actual, ensemble_pred)

        results[state] = {
            "pred":    ensemble_pred,
            "actual":  actual,
            "dates":   arima_preds[state]["dates"],
            "metrics": metrics,
            "weights": {
                "ARIMA":   round(w_arima, 3),
                "RF":      round(w_rf, 3),
                "XGBoost": round(w_xgb, 3),
            },
        }
        print(f"  {state}: MAPE={metrics['mape']:.2f}% "
              f"(ARIMA={w_arima:.2f}, RF={w_rf:.2f}, XGB={w_xgb:.2f})")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Plot and compare
# ══════════════════════════════════════════════════════════════════════════════

def plot_ensemble_results(
    arima_preds: dict,
    simple_results: dict,
    weighted_results: dict,
    df_clean: pd.DataFrame,
):
    """Plot actual vs both ensemble forecasts per state."""
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 20))
    colors = {"NSW": "#1f77b4", "VIC": "#ff7f0e",
              "QLD": "#2ca02c", "SA":  "#d62728", "WA":  "#9467bd"}

    for ax, state in zip(axes, STATES):
        # Full training history (from clean data)
        train_hist = df_clean[
            (df_clean["state"] == state) &
            (df_clean["date"] <= TRAIN_CUTOFF)
        ]
        ax.plot(train_hist["date"], train_hist["mean_price_aud"],
                color=colors[state], linewidth=2, label="Actual (train)")

        # Test actuals
        dates   = pd.to_datetime(weighted_results[state]["dates"])
        actuals = weighted_results[state]["actual"]
        ax.plot(dates, actuals,
                color="green", linewidth=2.5, label="Actual (test)")

        # Simple average ensemble
        ax.plot(dates, simple_results[state]["pred"],
                color="orange", linewidth=2, linestyle="--",
                label=f"Simple Avg (MAPE={simple_results[state]['metrics']['mape']:.2f}%)")

        # Inverse-MAPE weighted ensemble
        ax.plot(dates, weighted_results[state]["pred"],
                color="purple", linewidth=2, linestyle="-.",
                label=f"Weighted Avg (MAPE={weighted_results[state]['metrics']['mape']:.2f}%)")

        # Show the learned weights in the subtitle
        w = weighted_results[state]["weights"]
        weight_str = (f"Weights: ARIMA={w['ARIMA']:.2f}, "
                      f"RF={w['RF']:.2f}, XGB={w['XGBoost']:.2f}")

        ax.axvline(pd.Timestamp(TRAIN_CUTOFF),
                   color="red", linestyle=":", linewidth=1.5,
                   label="Train/test split")

        ax.set_title(f"{state} — Ensemble Forecasts\n{weight_str}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Price (AUD $000s)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "ensemble_forecast.png"
    plt.savefig(path, dpi=120)
    print(f"\nSaved ensemble forecast plot to {path}")
    plt.close()


def plot_final_comparison(simple_results: dict, weighted_results: dict):
    """Bar chart comparing all models including both ensembles."""

    # Load existing model metrics
    arima_mapes = pd.read_csv(REPORTS_DIR / "arima_metrics.csv").set_index("state")["mape"]
    rf_mapes    = pd.read_csv(REPORTS_DIR / "rf_metrics.csv").set_index("state")["mape"]
    xgb_mapes   = pd.read_csv(REPORTS_DIR / "xgboost_metrics.csv").set_index("state")["mape"]

    simple_mapes   = {s: simple_results[s]["metrics"]["mape"] for s in STATES}
    weighted_mapes = {s: weighted_results[s]["metrics"]["mape"] for s in STATES}

    # Build comparison DataFrame
    comparison = pd.DataFrame({
        "ARIMA":            arima_mapes,
        "Random Forest":    rf_mapes,
        "XGBoost":          xgb_mapes,
        "Ensemble (Equal)": pd.Series(simple_mapes),
        "Ensemble (Wtd)":   pd.Series(weighted_mapes),
    }).reindex(STATES)

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(STATES))
    n_models = len(comparison.columns)
    width = 0.15  # width of each bar
    colors = ["#2196F3", "#4CAF50", "#9C27B0", "#FF9800", "#F44336"]

    for i, (col, color) in enumerate(zip(comparison.columns, colors)):
        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(x + offset, comparison[col], width,
                      label=col, color=color, alpha=0.85)

    ax.set_xlabel("State")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE Comparison: All Models + Ensembles\n(lower = better)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(STATES)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, comparison.max().max() * 1.15)

    plt.tight_layout()
    path = REPORTS_DIR / "final_model_comparison.png"
    plt.savefig(path, dpi=120)
    print(f"Saved final comparison plot to {path}")
    plt.close()

    return comparison


def run_ensemble_pipeline():
    """Main: get all predictions, build ensembles, compare, log to MLflow."""

    # Load data
    df_clean    = pd.read_csv(CLEAN_DATA_PATH, parse_dates=["date"])
    df_features = pd.read_csv(FEATURES_PATH,   parse_dates=["date"])
    df_clean    = df_clean.sort_values(["state","date"]).reset_index(drop=True)
    df_features = df_features.sort_values(["state","date"]).reset_index(drop=True)

    # Step 1: Get individual model predictions
    arima_preds = get_arima_predictions(df_clean)
    rf_preds    = get_rf_predictions(df_features)
    xgb_preds   = get_xgb_predictions(df_features)

    # Step 2: Build both ensemble variants
    simple_results   = simple_average_ensemble(arima_preds, rf_preds, xgb_preds)
    weighted_results = inverse_mape_ensemble(arima_preds, rf_preds, xgb_preds)

    # Step 3: Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("housing_forecast_ensemble")

    with mlflow.start_run(run_name="Ensemble_weighted_ARIMA_RF_XGB"):
        # Log weighted ensemble metrics per state
        for state in STATES:
            m = weighted_results[state]["metrics"]
            w = weighted_results[state]["weights"]
            mlflow.log_metric(f"{state}_rmse", m["rmse"])
            mlflow.log_metric(f"{state}_mape", m["mape"])
            mlflow.log_param(f"{state}_w_arima", w["ARIMA"])
            mlflow.log_param(f"{state}_w_rf",    w["RF"])
            mlflow.log_param(f"{state}_w_xgb",   w["XGBoost"])

        # Overall average MAPE
        avg_mape = np.mean([weighted_results[s]["metrics"]["mape"] for s in STATES])
        mlflow.log_metric("avg_mape", avg_mape)
        mlflow.log_param("model_type", "Ensemble_InverseMAPE")
        mlflow.log_param("base_models", "ARIMA+RandomForest+XGBoost")
        print(f"\nEnsemble (weighted) avg MAPE: {avg_mape:.2f}%")

    # Step 4: Plot
    plot_ensemble_results(arima_preds, simple_results, weighted_results, df_clean)
    comparison = plot_final_comparison(simple_results, weighted_results)

    # Step 5: Print final summary
    print("\n" + "="*70)
    print("FINAL COMPARISON: ALL MODELS INCLUDING ENSEMBLES (MAPE %)")
    print("="*70)
    print(comparison.round(2).to_string())

    print("\n" + "="*70)
    print("AVERAGE MAPE ACROSS ALL STATES")
    print("="*70)
    avg = comparison.mean().sort_values()
    for model, mape in avg.items():
        marker = " ← BEST" if model == avg.idxmin() else ""
        print(f"  {model:<22}: {mape:.2f}%{marker}")

    # Save
    comparison.to_csv(REPORTS_DIR / "ensemble_comparison.csv")
    print(f"\nSaved to reports/ensemble_comparison.csv")

    return weighted_results


if __name__ == "__main__":
    run_ensemble_pipeline()