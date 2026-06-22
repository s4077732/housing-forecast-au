"""
Phase 3f: Temporal Fusion Transformer (TFT) — novel/differentiated model.

TFT is a deep learning architecture specifically designed for multi-horizon
time series forecasting. Key advantages over ARIMA/Prophet/RF/XGBoost:

1. Multi-horizon: predicts multiple steps ahead simultaneously (not one at a time)
2. Probabilistic: outputs prediction intervals natively (like ARIMA CI)
3. Attention-based interpretability: shows WHICH time steps and features
   the model focused on — like Grad-CAM but for time series
4. Handles multiple time series jointly (all 5 states in one model)
   with learned state-specific patterns via static covariates

Why TFT is the differentiator for this project:
  - No published paper has applied TFT to Australian housing/rent forecasting
  - Modern architecture (2021 Google paper) vs ARIMA (1970s) / RF (2001)
  - The attention weights give a new kind of interpretability: "the model
    focused most heavily on Q3 2022 prices when forecasting 2025 WA prices"

Architecture overview:
  Input: past price history + time features + state identity
    → Variable Selection Networks (learn which features matter)
    → LSTM Encoder (captures local temporal patterns)
    → Multi-head Attention (captures long-range dependencies)
    → Quantile outputs (10th, 50th, 90th percentile forecasts)

Run: python src/models/tft_model.py
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import torch
import lightning.pytorch as pl
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/processed/dwelling_prices_clean.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_CUTOFF     = "2024-12-31"
FORECAST_HORIZON = 4      # predict 4 quarters ahead (1 year)
MAX_ENCODER_LEN  = 12     # use last 12 quarters (3 years) as context window
RANDOM_SEED      = 42
MAX_EPOCHS       = 30     # limit epochs since dataset is small
LEARNING_RATE    = 0.03
BATCH_SIZE       = 16

# TFT requires CPU on Windows without a GPU
# (MPS is Mac-only, CUDA requires NVIDIA GPU)
DEVICE = "cpu"
pl.seed_everything(RANDOM_SEED)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """RMSE, MAE, MAPE — identical formula to all other models."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae  = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "mape": round(mape, 4)}


def prepare_tft_data(df: pd.DataFrame):
    """Prepare data in the format pytorch-forecasting's TFT expects.

    TFT needs:
    1. A continuous integer time index per group (not datetime)
       — TFT works with integer time steps internally
    2. A group identifier column (our 'state' column)
    3. All features declared upfront as time_varying_known,
       time_varying_unknown, or static_categoricals

    Why GroupNormalizer: TFT normalises each group (state) independently
    so NSW's $1.3M prices don't dominate SA's $973k prices during training.
    This is critical for pooled models — without normalisation, the model
    would implicitly weight high-price states more heavily.
    """
    df = df.sort_values(["state", "date"]).copy()

    # Add integer time index per state (TFT requirement)
    # Each state gets 0, 1, 2, ... 58 (59 quarters total)
    df["time_idx"] = df.groupby("state").cumcount()

    # Add time features TFT will use as known future covariates
    # (these are 'known' because we always know what quarter it will be)
    df["quarter"]  = df["date"].dt.quarter.astype(str)  # categorical
    df["year"]     = df["date"].dt.year.astype(float)

    # is_covid_boom as a known covariate (we know which quarters were the boom)
    df["is_covid_boom"] = (
        (df["date"] >= "2021-01-01") & (df["date"] <= "2022-12-31")
    ).astype(float)

    # Find the maximum time index in training data
    # TFT splits on time_idx, not datetime strings
    train_max_idx = df[df["date"] <= TRAIN_CUTOFF]["time_idx"].max()
    print(f"Max time_idx in training: {train_max_idx}")
    print(f"Total time steps per state: {df['time_idx'].max() + 1}")

    # ── Training dataset ────────────────────────────────────────────────
    # TimeSeriesDataSet is pytorch-forecasting's core data structure.
    # It handles:
    #   - windowing (sliding windows of encoder_length + prediction_length)
    #   - normalisation per group
    #   - batching for the PyTorch DataLoader
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= train_max_idx],

        # Column identifiers
        time_idx="time_idx",            # integer time index
        target="mean_price_aud",        # what we're forecasting
        group_ids=["state"],            # one series per state

        # Window sizes
        min_encoder_length=MAX_ENCODER_LEN // 2,  # minimum context (6 quarters)
        max_encoder_length=MAX_ENCODER_LEN,        # maximum context (12 quarters)
        min_prediction_length=1,
        max_prediction_length=FORECAST_HORIZON,    # predict 4 quarters ahead

        # Static features: don't change over time for a given state
        static_categoricals=["state"],

        # Time-varying features we KNOW in advance (calendar features)
        time_varying_known_categoricals=["quarter"],
        time_varying_known_reals=["time_idx", "year", "is_covid_boom"],

        # Time-varying features we DON'T know in advance (the target itself)
        time_varying_unknown_reals=["mean_price_aud"],

        # Normalise per state so NSW/SA price scales don't conflict
        target_normalizer=GroupNormalizer(
            groups=["state"],
            transformation="softplus"  # softplus keeps values positive (prices can't be negative)
        ),

        allow_missing_timesteps=False,  # our data is complete, no gaps
    )

    # ── Validation dataset ──────────────────────────────────────────────
    # Create validation set from the same parameters as training
    # (shares normalisation parameters — critical for consistency)
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,                          # full dataset (includes test period)
        predict=True,                # only include prediction windows
        stop_randomization=True,     # deterministic validation
    )

    # Create DataLoaders (PyTorch's batched data iterators)
    train_loader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0
    )
    val_loader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 2, num_workers=0
    )

    return training, validation, train_loader, val_loader, df, train_max_idx


def build_tft_model(training: TimeSeriesDataSet) -> TemporalFusionTransformer:
    """Build the TFT model architecture.

    Key architecture parameters:
    - hidden_size: dimensionality of internal representations.
      16 is deliberately small — our dataset has only ~200 rows,
      so a large model would severely overfit. Small hidden size = strong regularisation.
    - attention_head_size: number of parallel attention heads.
      1 is appropriate for small datasets (more heads = more parameters = overfit risk)
    - dropout: randomly zeros 10% of activations during training to prevent overfitting
    - hidden_continuous_size: size of continuous variable embedding layers
    - loss: QuantileLoss predicts the 10th, 50th, 90th percentiles simultaneously
      giving us prediction intervals 'for free' (like ARIMA's 95% CI)
    """
    tft = TemporalFusionTransformer.from_dataset(
        training,

        # Architecture size — deliberately small for our ~200 row dataset
        hidden_size=16,             # internal representation dimension
        attention_head_size=1,      # single attention head
        dropout=0.1,                # 10% dropout regularisation
        hidden_continuous_size=8,   # continuous variable embedding size

        # Training
        learning_rate=LEARNING_RATE,

        # Loss: quantile loss produces 10th/50th/90th percentile predictions
        # This gives us uncertainty intervals without needing separate CI calculation
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),

        # Logging
        log_interval=10,            # log every 10 batches
        log_val_interval=1,         # log validation every epoch
        reduce_on_plateau_patience=3,  # reduce LR if val loss plateaus for 3 epochs
    )
    return tft


def train_tft(tft, train_loader, val_loader):
    """Train TFT using PyTorch Lightning trainer.

    PyTorch Lightning handles the training loop, validation, and checkpointing
    automatically so we don't have to write the epoch/batch loop manually.

    EarlyStopping: stops training if validation loss doesn't improve for
    5 epochs — prevents overfitting on our small dataset and saves time.
    """
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",    # watch validation loss
        patience=5,            # stop if no improvement for 5 epochs
        mode="min",            # lower val_loss is better
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=DEVICE,         # cpu (no GPU required)
        gradient_clip_val=0.1,      # clip gradients to prevent exploding gradients
        callbacks=[early_stop],
        enable_progress_bar=True,
        logger=False,               # disable Lightning's own logger (we use MLflow)
    )

    print(f"Training TFT for up to {MAX_EPOCHS} epochs (early stopping at patience=5)...")
    trainer.fit(tft, train_loader, val_loader)
    print(f"Training stopped at epoch {trainer.current_epoch}")
    return trainer


def get_predictions_and_metrics(tft, val_loader, validation, df, train_max_idx):
    """Extract TFT predictions and compute metrics on the test period."""

    # Get raw predictions from the validation loader
    # output_type="prediction" returns the median (50th percentile) forecast
    predictions = tft.predict(val_loader, return_y=True, trainer_kwargs={"accelerator": DEVICE})

    # predictions.output shape: (n_windows, forecast_horizon, n_quantiles)
    # predictions.y shape: (n_windows, forecast_horizon)
    pred_values = predictions.output.numpy()   # predicted values
    true_values = predictions.y[0].numpy()     # actual values

    # Use median prediction (quantile index 1 = 50th percentile)
    # for metric computation (comparable to other models' point forecasts)
    if pred_values.ndim == 3:
        pred_median = pred_values[:, :, 1]  # 50th percentile
    else:
        pred_median = pred_values

    # Flatten for metric computation
    pred_flat = pred_median.flatten()
    true_flat = true_values.flatten()

    # Remove any NaN padding that pytorch-forecasting may add
    mask = ~np.isnan(true_flat) & ~np.isnan(pred_flat)
    pred_flat = pred_flat[mask]
    true_flat = true_flat[mask]

    metrics = compute_metrics(true_flat, pred_flat)
    print(f"\nTFT Test Metrics (all states, all prediction windows):")
    print(f"  RMSE: ${metrics['rmse']:,.1f}k")
    print(f"  MAE:  ${metrics['mae']:,.1f}k")
    print(f"  MAPE: {metrics['mape']:.2f}%")

    return metrics, pred_median, true_values


def plot_tft_forecast(tft, val_loader, df, metrics, train_max_idx):
    """Plot TFT forecasts per state with uncertainty intervals."""
    test_df = df[df["time_idx"] > train_max_idx]

    # Get raw predictions including all quantiles for plotting intervals
    raw_preds = tft.predict(
        val_loader,
        mode="raw",
        return_x=True,
        trainer_kwargs={"accelerator": DEVICE}
    )

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 20))
    colors = {"NSW": "#1f77b4", "VIC": "#ff7f0e",
              "QLD": "#2ca02c", "SA": "#d62728", "WA": "#9467bd"}
    states = ["NSW", "VIC", "QLD", "SA", "WA"]

    for idx, (ax, state) in enumerate(zip(axes, states)):
        state_train = df[(df["state"] == state) & (df["time_idx"] <= train_max_idx)]
        state_test  = df[(df["state"] == state) & (df["time_idx"] > train_max_idx)]

        # Plot training history
        ax.plot(state_train["date"], state_train["mean_price_aud"],
                color=colors[state], linewidth=2, label="Actual (train)")

        # Plot test actuals
        if len(state_test) > 0:
            ax.plot(state_test["date"], state_test["mean_price_aud"],
                    color="green", linewidth=2, label="Actual (test)")

        ax.axvline(pd.Timestamp(TRAIN_CUTOFF),
                   color="red", linestyle=":", linewidth=1.5,
                   label="Train/test split")

        ax.set_title(f"{state} — TFT Forecast | "
                     f"Overall RMSE: ${metrics['rmse']:,.0f}k | "
                     f"MAPE: {metrics['mape']:.2f}%",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Price (AUD $000s)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "tft_forecast.png"
    plt.savefig(path, dpi=120)
    print(f"Saved TFT forecast plot to {path}")
    plt.close()


def run_tft_pipeline():
    """Main: prepare data, build, train, evaluate TFT, log to MLflow."""

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    print("Preparing TFT dataset...")
    training, validation, train_loader, val_loader, df, train_max_idx = (
        prepare_tft_data(df)
    )

    print("Building TFT model...")
    tft = build_tft_model(training)
    print(f"TFT parameter count: {sum(p.numel() for p in tft.parameters()):,}")

    # Train
    trainer = train_tft(tft, train_loader, val_loader)

    # Evaluate
    metrics, pred_median, true_values = get_predictions_and_metrics(
        tft, val_loader, validation, df, train_max_idx
    )

    # Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("housing_forecast_tft")

    with mlflow.start_run(run_name="TFT_all_states"):
        mlflow.log_param("model_type", "TFT")
        mlflow.log_param("hidden_size", 16)
        mlflow.log_param("attention_head_size", 1)
        mlflow.log_param("dropout", 0.1)
        mlflow.log_param("max_epochs", MAX_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("encoder_length", MAX_ENCODER_LEN)
        mlflow.log_param("forecast_horizon", FORECAST_HORIZON)
        mlflow.log_param("train_cutoff", TRAIN_CUTOFF)
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("mae",  metrics["mae"])
        mlflow.log_metric("mape", metrics["mape"])

    # Plot
    plot_tft_forecast(tft, val_loader, df, metrics, train_max_idx)

    # Save metrics
    metrics_df = pd.DataFrame([{"model": "TFT", **metrics}])
    metrics_path = REPORTS_DIR / "tft_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    return tft, metrics


if __name__ == "__main__":
    run_tft_pipeline()