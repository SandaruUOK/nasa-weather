"""Rebuild the 11-column dengue dataset using NASA POWER IMERG daily rainfall.

Keep the source's exact reporting-week boundaries. Temperature, humidity,
case counts and reporting labels are unchanged; only weekly rainfall and the
four documented climatic-zone corrections are updated.

Run from the nasa-weather repository root:
    python build_imerg_weekly.py --input data/dengue_weather_base_11cols.csv

Dependencies: pandas, requests, urllib3. Reuses the existing districts.py.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from districts import DISTRICTS  # Existing repo: 26 RDHS points including Kalmunai

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
ZONE_FIXES = {
    "Ampara": "Dry",
    "Batticaloa": "Dry",
    "Kalmunai": "Dry",
    "Matale": "Intermediate",
}


def fetch_daily_imerg(session, district, start_date, end_date, cache_dir):
    """Return daily IMERG_PRECTOT for one district, with a local CSV cache."""
    cache_file = cache_dir / f"{district.replace(' ', '_')}_daily.csv"
    if cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["Date"])
        series = cached.set_index("Date")["IMERG_PRECTOT"].sort_index()
        if (series.index.min() <= start_date and
                series.index.max() >= end_date and
                series.index.is_unique):
            return series

    coords = DISTRICTS[district]
    parts = []
    # Same two API periods used in the archived research notebook.
    windows = [
        (start_date, min(end_date, pd.Timestamp("2020-12-31"))),
        (max(start_date, pd.Timestamp("2021-01-01")), end_date),
    ]
    for start, end in windows:
        if start > end:
            continue
        params = {
            "parameters": "IMERG_PRECTOT",
            "community": "AG",
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
            "time-standard": "UTC",
        }
        response = session.get(POWER_URL, params=params, timeout=(20, 180))
        response.raise_for_status()
        daily_values = response.json()["properties"]["parameter"]["IMERG_PRECTOT"]
        part = pd.Series(daily_values, dtype=float)
        part.index = pd.to_datetime(part.index, format="%Y%m%d")
        parts.append(part)
        time.sleep(0.5)

    daily = pd.concat(parts).sort_index()
    if not daily.index.is_unique:
        raise ValueError(f"Duplicate API dates for {district}")
    daily = daily.mask(daily <= -900)  # NASA missing-data sentinels
    daily.name = "IMERG_PRECTOT"
    daily.to_csv(cache_file, index_label="Date")
    return daily


def weekly_total(daily, start, end):
    """Sum exactly seven valid daily observations; never use a 7-day mean."""
    dates = pd.date_range(start, end, freq="D")
    values = daily.reindex(dates)
    if len(dates) != 7 or values.notna().sum() != 7:
        raise ValueError(f"Incomplete reporting week: {start} to {end}")
    return round(float(values.sum()), 2)


def build_dataset(source_csv, output_csv, cache_dir):
    base = pd.read_csv(source_csv, parse_dates=["Week_Start", "Week_End"])
    needed = {"District", "Zone", "Week_Start", "Week_End", "Total_Rainfall_mm"}
    if not needed.issubset(base.columns):
        raise ValueError(f"Missing input columns: {sorted(needed - set(base.columns))}")
    if set(base["District"]) != set(DISTRICTS):
        raise ValueError("RDHS areas do not match districts.py (must include Kalmunai)")

    cache_dir.mkdir(parents=True, exist_ok=True)
    start_date, end_date = base["Week_Start"].min(), base["Week_End"].max()
    output = base.copy()

    session = requests.Session()
    retry = Retry(total=4, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))

    for district, group in base.groupby("District", sort=False):
        daily = fetch_daily_imerg(session, district, start_date, end_date, cache_dir)
        for idx, row in group.iterrows():
            output.loc[idx, "Total_Rainfall_mm"] = weekly_total(
                daily, row["Week_Start"], row["Week_End"]
            )
        print(f"{district}: {len(group)} reporting weeks complete")

    for district, zone in ZONE_FIXES.items():
        output.loc[output["District"] == district, "Zone"] = zone

    # Preserve exactly the existing source record set, including 2026 week changes.
    assert output.shape == base.shape
    assert output["Total_Rainfall_mm"].notna().all()
    assert (output["Total_Rainfall_mm"] >= 0).all()
    assert not output.duplicated(["District", "Week_Start"]).any()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    print(f"Saved {len(output):,} district-week records: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Original 11-column dengue/weather CSV")
    parser.add_argument("--output", type=Path, default=Path(
        "outputs/imerg_experiment/dengue_weather_11cols_imerg_zones_corrected_reproduced.csv"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("outputs/imerg_experiment/daily_cache"))
    args = parser.parse_args()
    build_dataset(args.input, args.output, args.cache_dir)
