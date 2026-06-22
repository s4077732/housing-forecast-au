"""
Step 1d: Clean and filter the decoded ABS dwelling data for modelling.

From the decoded data we know:
  - MEASURE 5 = 'Mean price of residential dwellings' - this is our target
  - Regions: NSW, VIC, QLD, SA, WA are the 5 relevant states for
    international students (dropping TAS, NT, ACT - small populations)
  - TIME_PERIOD is quarterly e.g. '2011-Q3'
  - OBS_VALUE is in AUD (mean dwelling price)

Output: data/processed/dwelling_prices_clean.csv
  Columns: date (datetime), state, mean_price_aud, yoy_change_pct, qoq_change_pct

Run from project root:
    python src/data/clean_abs_data.py
"""

import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/abs_rppi_decoded.csv")
OUTPUT_PATH = Path("data/processed/dwelling_prices_clean.csv")

# States relevant to international students
TARGET_STATES = {
    "New South Wales": "NSW",
    "Victoria": "VIC",
    "Queensland": "QLD",
    "South Australia": "SA",
    "Western Australia": "WA",
}

# We want mean price of residential dwellings (measure 5)
TARGET_MEASURE = "Mean price of residential dwellings"


def parse_abs_quarter(period: str) -> pd.Timestamp:
    """Convert ABS quarter string '2011-Q3' to a proper datetime.

    We anchor each quarter to its first day (e.g. Q3 = July 1).
    All time-series models need actual datetime objects, not strings.
    """
    year, q = period.split("-")
    quarter_start_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
    month = quarter_start_month[q]
    return pd.Timestamp(year=int(year), month=month, day=1)


def compute_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Add quarter-on-quarter and year-on-year percentage change columns.

    Percentage changes are more stationary than raw price levels and
    directly answer the student-facing question: 'is it increasing?'
    Computed per state using groupby so states don't bleed into each other.
    """
    df = df.sort_values(["state", "date"]).copy()

    df["qoq_change_pct"] = (
        df.groupby("state")["mean_price_aud"]
        .pct_change() * 100
    )
    df["yoy_change_pct"] = (
        df.groupby("state")["mean_price_aud"]
        .pct_change(periods=4) * 100  # 4 quarters = 1 year
    )
    return df


def clean_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH, dtype={"REGION": str})

    # Step 1: Filter to mean price only
    df = df[df["measure_name"] == TARGET_MEASURE].copy()
    print(f"Rows after measure filter: {len(df)}")

    # Step 2: Filter to target states
    df = df[df["region_name"].isin(TARGET_STATES.keys())].copy()
    print(f"Rows after state filter: {len(df)}")

    # Step 3: Map full state names to abbreviations
    df["state"] = df["region_name"].map(TARGET_STATES)

    # Step 4: Parse quarter strings into real datetime objects
    df["date"] = df["TIME_PERIOD"].apply(parse_abs_quarter)

    # Step 5: Keep only columns we need
    df = df[["date", "state", "OBS_VALUE"]].rename(
        columns={"OBS_VALUE": "mean_price_aud"}
    )

    # Step 6: Add change features
    df = compute_changes(df)

    # Step 7: Sort and save
    df = df.sort_values(["state", "date"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved clean data to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    df = clean_data()

    print("\n=== Clean data sample (first 10 rows) ===")
    print(df.head(10).to_string())

    print("\n=== Summary per state ===")
    summary = df.groupby("state").agg(
        start=("date", "min"),
        end=("date", "max"),
        n_quarters=("date", "count"),
        latest_price_aud=("mean_price_aud", "last"),
        latest_yoy_pct=("yoy_change_pct", "last"),
    )
    print(summary.to_string())