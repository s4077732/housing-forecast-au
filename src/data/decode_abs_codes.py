"""
Step 1c: Decode ABS SDMX codes (REGION, MEASURE) into human-readable labels.

The raw RPPI data uses numeric/short codes (REGION=4, MEASURE=2, etc.) instead
of names. SDMX stores the code -> label mapping separately as a "codelist".
This script fetches the relevant codelists from the ABS API and uses them to
decode data/raw/abs_rppi_raw.csv into a clean, labeled CSV.

Run from the project root (after fetch_abs_rppi.py has been run):
    python src/data/decode_abs_codes.py
"""

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

ABS_BASE = "https://data.api.abs.gov.au/rest"
RAW_PATH = Path("data/raw/abs_rppi_raw.csv")
OUTPUT_PATH = Path("data/processed/abs_rppi_decoded.csv")

NS = {
    "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
}


def fetch_codelist(codelist_id: str) -> dict:
    url = f"{ABS_BASE}/codelist/ABS/{codelist_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    mapping = {}
    for code in root.iter("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}Code"):
        code_id = code.attrib.get("id")
        name_el = code.find("com:Name", NS)
        name = name_el.text if name_el is not None else code_id
        mapping[code_id] = name

    return mapping


def discover_codelist_ids(dataflow_id: str = "RES_DWELL_ST") -> dict:
    url = f"{ABS_BASE}/datastructure/ABS/{dataflow_id}?references=codelist"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    codelists = {}
    for cl in root.iter("{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure}Codelist"):
        cl_id = cl.attrib.get("id")
        codelists[cl_id] = cl_id

    return codelists


def decode_data(region_map: dict, measure_map: dict) -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, dtype={"REGION": str})
    df["region_name"] = df["REGION"].map(region_map).fillna(df["REGION"])
    df["measure_name"] = df["MEASURE"].astype(str).map(measure_map).fillna(df["MEASURE"].astype(str))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    print("Discovering codelist IDs used by RES_DWELL_ST dataflow...")
    codelists = discover_codelist_ids()
    print("Codelists found:", list(codelists.keys()))

    # ABS naming varies by dataflow: region/city codes may live under
    # CL_REGION, CL_STATE, or similar - check several known patterns.
    region_codelist_id = next(
        (c for c in codelists if any(k in c.upper() for k in ["REGION", "STATE", "GCCSA", "CITY"])),
        None,
    )
    # Avoid accidentally matching CL_UNIT_MEASURE when looking for the
    # dataflow's own measure codelist (e.g. CL_RES_DWELL_ST_MEASURE).
    measure_codelist_id = next(
        (c for c in codelists if "MEASURE" in c.upper() and "UNIT" not in c.upper()),
        None,
    )

    if not region_codelist_id or not measure_codelist_id:
        print("Could not auto-detect REGION/MEASURE codelist IDs.")
        print("Available codelists:", list(codelists.keys()))
        print("Inspect these manually at:")
        print(f"  {ABS_BASE}/datastructure/ABS/RES_DWELL_ST?references=codelist")
        raise SystemExit(1)

    print(f"Fetching region codelist: {region_codelist_id}")
    region_map = fetch_codelist(region_codelist_id)
    print("Region codes decoded:", region_map)

    print(f"Fetching measure codelist: {measure_codelist_id}")
    measure_map = fetch_codelist(measure_codelist_id)
    print("Measure codes decoded:", measure_map)

    print("Decoding raw data...")
    df = decode_data(region_map, measure_map)
    print(f"Saved decoded data to {OUTPUT_PATH}")
    print(df[["region_name", "measure_name", "TIME_PERIOD", "OBS_VALUE"]].head(10))