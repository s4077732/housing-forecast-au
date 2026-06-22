"""
Phase 3a: Feature Engineering for pooled models (RF, XGBoost, TFT).

Takes the clean dwelling prices data and engineers:
  1. Lag features (prices from 1, 2, 4, 8 quarters ago per state)
  2. Spatio-temporal lag (same quarter last year, same state)
  3. Rolling statistics (4-quarter rolling mean and std per state)
  4. Time features (quarter number, year, is_covid_boom flag)
  5. State as one-hot encoded categorical

Output: data/processed/features.csv
Run: python src/features/build_features.py
"""

# pandas: for loading and manipulating tabular data (DataFrames)
# numpy: for mathematical operations like sin/cos for cyclical encoding
# Path: cleaner way to handle file paths across Windows/Mac/Linux
import pandas as pd
import numpy as np
from pathlib import Path

# Where to read the cleaned data from (output of clean_abs_data.py)
INPUT_PATH = Path("data/processed/dwelling_prices_clean.csv")

# Where to save the engineered features (input for model training)
OUTPUT_PATH = Path("data/processed/features.csv")


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged price values per state.

    A lag feature is simply: what was the price N quarters ago?
    e.g. price_lag_1q for 2026-Q1 NSW = NSW price in 2025-Q4

    Why lags matter: housing prices are 'autoregressive' - meaning
    last quarter's price is the single best predictor of this quarter's
    price. The Adelaide paper found spatio-temporal lag was the most
    important feature for tree models on Australian housing data.
    """
    # Loop over 4 different lag amounts: 1 quarter, 2 quarters, 4 quarters (1 year), 8 quarters (2 years)
    for lag in [1, 2, 4, 8]:
        # groupby("state") ensures NSW's lag only looks back through NSW data,
        # not into VIC's prices. shift(lag) moves the price value 'lag' rows down.
        df[f"price_lag_{lag}q"] = (
            df.groupby("state")["mean_price_aud"].shift(lag)
        )
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and standard deviation over 4 quarters per state.

    Rolling mean = average price over the last 4 quarters (1 year window)
    Rolling std  = how much prices varied over the last 4 quarters

    Why rolling features matter:
    - Rolling mean captures the medium-term trend, smoothing out noisy
      quarter-to-quarter jumps. It tells the model: 'is the underlying
      trend up or down?' rather than reacting to single outlier quarters.
    - Rolling std captures volatility. High std = unstable market (like
      WA right now). Low std = stable market. Tree models can use this
      to give wider/narrower predictions.
    """
    # transform() applies the function per state group and returns a
    # result with the same index as the original DataFrame (needed for assignment)
    df["rolling_mean_4q"] = (
        df.groupby("state")["mean_price_aud"]
        # rolling(4) = 4-quarter window. min_periods=2 means we still
        # compute a value even if we only have 2 data points (avoids
        # losing too many rows at the start of the series)
        .transform(lambda x: x.rolling(4, min_periods=2).mean())
    )
    df["rolling_std_4q"] = (
        df.groupby("state")["mean_price_aud"]
        .transform(lambda x: x.rolling(4, min_periods=2).std())
    )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based and structural break features.

    Calendar features capture seasonality and time trends.
    Structural break flags tell the model about unusual historical periods
    so it doesn't treat the COVID boom as 'normal' recurring behaviour.
    """
    # Extract year (2011, 2012, ... 2026) from the date column
    df["year"] = df["date"].dt.year

    # Extract quarter number (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
    df["quarter"] = df["date"].dt.quarter

    # Cyclical encoding: encode quarter as sine and cosine waves
    # Why: if we used quarter=1,2,3,4 as a raw number, the model would think
    # Q4 (value=4) and Q1 (value=1) are far apart - but seasonally they're
    # adjacent (Dec -> Jan). Sin/cos encoding wraps the cycle so Q4 and Q1
    # are close together numerically, which tree models can learn correctly.
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

    # Structural break flag for the COVID housing boom
    # Our EDA (yoy_changes.png) clearly showed a massive spike in 2021-2022
    # across all states (+25-30% YoY). This was caused by record-low interest
    # rates and pandemic-driven demand - not a repeatable pattern.
    # Flagging it as 1 (vs 0 for normal periods) tells the model: 'prices
    # during this window were abnormally high - don't generalise from this'.
    df["is_covid_boom"] = (
        (df["date"] >= "2021-01-01") & (df["date"] <= "2022-12-31")
    ).astype(int)  # convert True/False to 1/0 (models need numbers, not booleans)

    return df


def add_state_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the state column for tree-based models.

    Tree models (Random Forest, XGBoost) require all inputs to be numeric.
    'State' is a categorical variable (NSW, VIC, QLD, SA, WA).

    One-hot encoding creates a separate 0/1 column for each state:
      state_NSW=1, state_VIC=0, state_QLD=0, state_SA=0, state_WA=0  <- a NSW row
      state_NSW=0, state_VIC=1, state_QLD=0, state_SA=0, state_WA=0  <- a VIC row

    Why not just use numbers (NSW=1, VIC=2...)? Because that would imply
    NSW > VIC > QLD in some numerical order, which has no meaning for states.
    One-hot treats each state as completely independent.
    """
    # pd.get_dummies creates the binary columns automatically
    # prefix="state" names them state_NSW, state_VIC etc (not just NSW, VIC)
    state_dummies = pd.get_dummies(df["state"], prefix="state")

    # Attach the new state columns to the right of the existing DataFrame
    df = pd.concat([df, state_dummies], axis=1)  # axis=1 = add as new columns
    return df


def build_features() -> pd.DataFrame:
    """Master function: loads clean data, runs all feature engineering steps,
    saves the result. This is the function called when the script runs."""

    # Load the cleaned CSV - parse_dates converts the 'date' column from
    # string ("2011-07-01") to a real datetime object automatically
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    # Sort by state then date so lag/rolling calculations work correctly
    # (shift() and rolling() depend on row order within each group)
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    print(f"Input rows: {len(df)}")  # should be 295 rows (59 quarters x 5 states)

    # Run each feature engineering step in sequence
    df = add_lag_features(df)      # adds price_lag_1q, _2q, _4q, _8q
    df = add_rolling_features(df)  # adds rolling_mean_4q, rolling_std_4q
    df = add_time_features(df)     # adds year, quarter, quarter_sin/cos, is_covid_boom
    df = add_state_encoding(df)    # adds state_NSW, state_VIC, state_QLD, state_SA, state_WA

    # The 8-quarter lag means the first 8 rows per state will have NaN
    # (there's no price from 8 quarters ago at the start of the series).
    # dropna removes these incomplete rows so models don't fail on missing values.
    df_model = df.dropna(subset=["price_lag_8q"]).reset_index(drop=True)
    print(f"Rows after dropping early NaN lags: {len(df_model)}")
    # Expected: 295 - (8 dropped per state x 5 states) = 255 rows

    # Create the output folder if it doesn't exist, then save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(OUTPUT_PATH, index=False)  # index=False avoids saving row numbers
    print(f"Saved features to {OUTPUT_PATH}")

    # Print the list of all feature columns (everything except metadata columns)
    feature_cols = [c for c in df_model.columns
                    if c not in ["date", "state", "region_name"]]
    print(f"\nFeatures built ({len(feature_cols)} total):")
    for col in feature_cols:
        print(f"  {col}")

    return df_model


if __name__ == "__main__":
    # __name__ == "__main__" means: only run this block when the script is
    # executed directly (python src/features/build_features.py), NOT when
    # it's imported by another script (like the model training scripts later).
    # This is a standard Python best practice pattern.

    df = build_features()

    # Print a sample showing the key engineered columns side by side
    # so we can visually verify the lag/rolling values look correct
    print("\nSample (key feature columns):")
    print(df[["date", "state", "mean_price_aud",
              "price_lag_1q",       # should be ~same as mean_price_aud from last row
              "price_lag_4q",       # should be ~1 year ago price
              "rolling_mean_4q",    # should be between recent prices
              "yoy_change_pct",     # year-on-year % change
              "quarter",            # 1, 2, 3, or 4
              "is_covid_boom"]      # 1 for 2021-2022, 0 otherwise
             ].head(10).to_string())