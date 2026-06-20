"""
Step 1b: Rental market data acquisition.

Unlike the RPPI (Step 1a), the ABS rental indicators are not confirmed to be
available as a clean dataflow on the Data API as of this writing - they are
primarily published as part of the CPI release and as standalone Excel
spreadsheets ("New Insights into the Rental Market" series).

This script documents the manual download path. Re-run find_dataflow.py
style discovery (see fetch_abs_rppi.py) against keywords like "rent" if you
want to check whether a clean API dataflow exists by the time you run this.

Manual download steps:

1. Go to: https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia
   -> Download the "Rent" time series spreadsheet under Data downloads.

2. Alternative (more granular, city-level): SQM Research publishes free
   weekly asking rent data by city:
   https://sqmresearch.com.au/weekly-rents.php?region=nsw-Sydney&type=c&t=1
   (repeat for each capital city by changing the region parameter)

3. Save downloaded files into data/raw/rents/ with a clear naming
   convention, e.g.:
       data/raw/rents/abs_rent_index.xlsx
       data/raw/rents/sqm_sydney_weekly_rents.csv
       data/raw/rents/sqm_melbourne_weekly_rents.csv
       ...

Run this script after downloading to standardise the raw rent files into
one combined CSV for the pipeline.
"""

import pandas as pd
from pathlib import Path

RAW_RENTS_DIR = Path("data/raw/rents")
OUTPUT_PATH = Path("data/processed/rents_combined.csv")

CITY_FILE_MAP = {
    "sydney": "sqm_sydney_weekly_rents.csv",
    "melbourne": "sqm_melbourne_weekly_rents.csv",
    "brisbane": "sqm_brisbane_weekly_rents.csv",
    "perth": "sqm_perth_weekly_rents.csv",
    "adelaide": "sqm_adelaide_weekly_rents.csv",
}


def combine_city_rent_files() -> pd.DataFrame:
    """Combine per-city rent CSVs (downloaded manually) into a single
    long-format DataFrame with columns: date, city, median_rent.

    Why long format: this is the structure needed for the pooled models
    (Random Forest, XGBoost, TFT) where 'city' becomes a feature column,
    versus wide format which is easier for per-city models (ARIMA, Prophet).
    We will pivot as needed in the feature engineering step.
    """
    frames = []
    for city, filename in CITY_FILE_MAP.items():
        path = RAW_RENTS_DIR / filename
        if not path.exists():
            print(f"WARNING: {path} not found - skipping {city}. "
                  f"Download it first (see docstring instructions).")
            continue
        df = pd.read_csv(path)
        df["city"] = city
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No rent files found. Download at least one city's data into "
            f"{RAW_RENTS_DIR} before running this script."
        )

    combined = pd.concat(frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved combined rent data ({len(combined)} rows) to {OUTPUT_PATH}")
    return combined


if __name__ == "__main__":
    RAW_RENTS_DIR.mkdir(parents=True, exist_ok=True)
    combine_city_rent_files()
