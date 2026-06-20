"""
Step 1a: ABS Data API acquisition for Residential Property Price Indexes (RPPI).

The ABS Data API is SDMX 2.1 compliant and free, no API key required.
Docs: https://www.abs.gov.au/about/data-services/application-programming-interfaces-apis/data-api-user-guide

Because ABS dataflow IDs can change/version over time, this script first
*discovers* the correct dataflow for residential property prices rather than
hardcoding a guessed ID - this makes the script robust to ABS catalogue changes.

Run from the project root:
    python src/data/fetch_abs_rppi.py
"""

import requests
import pandas as pd
from pathlib import Path

ABS_BASE = "https://data.api.abs.gov.au/rest"
OUTPUT_DIR = Path("data/raw")


def find_rppi_dataflow() -> str:
    """Search the full ABS dataflow catalogue for the Residential Property
    Price Indexes dataflow and return its identifier.

    Why this approach: hardcoding a dataflow ID risks silent failure if ABS
    updates their catalogue (this has happened before, e.g. RPPI moved from
    cat. no. 6416.0 to being folded into broader housing collections).
    Searching the live catalogue is slightly slower but far more reliable.
    """
    url = f"{ABS_BASE}/dataflow/ABS?format=json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dataflows = resp.json().get("data", {}).get("dataflows", [])

    candidates = []
    for df in dataflows:
        name = df.get("name", "").lower()
        if "residential" in name and "price" in name:
            candidates.append(df)
        elif "rppi" in df.get("id", "").lower():
            candidates.append(df)

    if not candidates:
        raise RuntimeError(
            "Could not find an RPPI dataflow automatically. "
            "Visit https://data.api.abs.gov.au/rest/dataflow/ABS in a "
            "browser and search the response for 'Residential Property' "
            "to find the correct dataflow id manually."
        )

    # Prefer an exact "Residential Property Price Indexes" match if present
    for c in candidates:
        if "eight capital" in c.get("name", "").lower():
            return c["id"]

    return candidates[0]["id"]


def fetch_rppi(dataflow_id: str, start_period: str = "2010-Q1") -> pd.DataFrame:
    """Pull the full RPPI time series (all capital cities) as a DataFrame."""
    url = (
        f"{ABS_BASE}/data/ABS,{dataflow_id}/all"
        f"?startPeriod={start_period}&format=csv"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "abs_rppi_raw.csv"
    out_path.write_bytes(resp.content)

    df = pd.read_csv(out_path)
    return df


if __name__ == "__main__":
    print("Searching ABS dataflow catalogue for RPPI dataset...")
    dataflow_id = find_rppi_dataflow()
    print(f"Found dataflow: {dataflow_id}")

    print("Fetching data...")
    df = fetch_rppi(dataflow_id)
    print(f"Saved {len(df)} rows to data/raw/abs_rppi_raw.csv")
    print(df.head())
