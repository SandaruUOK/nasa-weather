"""Sri Lanka 26-RDHS NASA POWER weather fetcher (direct GPM IMERG rainfall).

Output: daily and epidemiological-weekly weather only; dengue cases are joined
later using District + Week_Start + Week_End. Requires the existing districts.py
module with the 26 RDHS coordinates, including Kalmunai.

Install: pip install streamlit pandas requests openpyxl urllib3
Run:     streamlit run streamlit_weather_imerg.py
"""
from __future__ import annotations

import io
import time
from datetime import date

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from districts import DISTRICTS

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
FIRST_RESEARCH_DAY = pd.Timestamp("2015-12-26")
OLD_WEEK_END = pd.Timestamp("2025-12-26")
NEW_WEEK_START = pd.Timestamp("2025-12-29")
SKIPPED_TRANSITION_DATES = {
    pd.Timestamp("2025-12-27"),
    pd.Timestamp("2025-12-28"),
}
ZONE_FIXES = {
    "Ampara": "Dry",
    "Batticaloa": "Dry",
    "Kalmunai": "Dry",
    "Matale": "Intermediate",
}


def _request_windows(start: pd.Timestamp, end: pd.Timestamp):
    """Split long research extraction at 2020-12-31, like the archived IMERG flow."""
    windows = [
        (start, min(end, pd.Timestamp("2020-12-31"))),
        (max(start, pd.Timestamp("2021-01-01")), end),
    ]
    return [(a, b) for a, b in windows if a <= b]


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def _fetch_parameter_group(
    lat: float,
    lon: float,
    start_str: str,
    end_str: str,
    parameters: str,
) -> pd.DataFrame:
    """NASA POWER daily point data, all sourced in UTC for matching date labels.

    Calls are split by variable group to avoid dependence on multi-source
    parameter availability in a single mixed NASA POWER request.
    """
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    pieces = []

    with _session() as session:
        for begin, finish in _request_windows(start, end):
            params = {
                "parameters": parameters,
                "community": "AG",
                "latitude": lat,
                "longitude": lon,
                "start": begin.strftime("%Y%m%d"),
                "end": finish.strftime("%Y%m%d"),
                "format": "JSON",
                "time-standard": "UTC",  # IMERG_PRECTOT is daily UTC only.
            }
            response = session.get(POWER_URL, params=params, timeout=(20, 180))
            response.raise_for_status()
            payload = response.json().get("properties", {}).get("parameter", {})
            required = parameters.split(",")
            missing_params = set(required).difference(payload)
            if missing_params:
                raise ValueError(f"NASA response missing parameters: {sorted(missing_params)}")

            values = {
                name: pd.Series(payload[name], dtype=float)
                for name in required
            }
            frame = pd.DataFrame(values)
            frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
            pieces.append(frame)
            time.sleep(0.3)

    daily = pd.concat(pieces).sort_index()
    if not daily.index.is_unique:
        raise ValueError("NASA returned duplicate dates")
    # NASA POWER uses large negative values for missing source observations.
    daily = daily.mask(daily <= -900)
    daily.index.name = "Date"
    return daily


def fetch_district_data(
    district: str,
    lat: float,
    lon: float,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Fetch daily T2M, RH2M and IMERG_PRECTOT; no PRECTOTCORR request."""
    start_str, end_str = start_date.isoformat(), end_date.isoformat()
    meteorology = _fetch_parameter_group(lat, lon, start_str, end_str, "T2M,RH2M")
    imerg = _fetch_parameter_group(lat, lon, start_str, end_str, "IMERG_PRECTOT")

    daily = meteorology.join(imerg, how="outer", validate="one_to_one")
    daily = daily.rename(
        columns={"T2M": "Temp_C", "RH2M": "Humidity_%", "IMERG_PRECTOT": "Precip_mm"}
    )
    daily = daily.reset_index()
    daily["District"] = district
    daily["Zone"] = ZONE_FIXES.get(district, DISTRICTS[district]["zone"])
    return daily[["Date", "District", "Zone", "Temp_C", "Humidity_%", "Precip_mm"]]


def prepare_daily_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Keep original research calendar: omit the two 2025 transition dates."""
    data = daily_df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.loc[
        data["Date"].ge(FIRST_RESEARCH_DAY)
        & ~data["Date"].isin(SKIPPED_TRANSITION_DATES)
    ].copy()
    return data.sort_values(["Date", "District"]).reset_index(drop=True)


def aggregate_weekly(
    daily_df: pd.DataFrame,
    requested_start: date | None = None,
    requested_end: date | None = None,
) -> pd.DataFrame:
    """Seven valid daily readings for each epidemiological week.

    2015-12-26 through 2025-12-26: Saturday-Friday weeks.
    From 2025-12-29: Monday-Sunday weeks.
    A partial week due to user's chosen date boundaries is omitted; a full
    week with missing NASA data is rejected rather than silently summed.
    """
    df = prepare_daily_data(daily_df)
    df["Week_Start"] = pd.NaT
    old_mask = df["Date"].le(OLD_WEEK_END)
    offsets = (df.loc[old_mask, "Date"] - FIRST_RESEARCH_DAY).dt.days
    df.loc[old_mask, "Week_Start"] = FIRST_RESEARCH_DAY + pd.to_timedelta(
        (offsets // 7) * 7, unit="D"
    )
    new_mask = df["Date"].ge(NEW_WEEK_START)
    df.loc[new_mask, "Week_Start"] = df.loc[new_mask, "Date"] - pd.to_timedelta(
        df.loc[new_mask, "Date"].dt.weekday, unit="D"
    )
    df = df.dropna(subset=["Week_Start"]).copy()
    df["Week_End"] = df["Week_Start"] + pd.Timedelta(days=6)
    df["Year"] = df["Week_End"].dt.year

    # First, check if a 7-day reporting interval was entirely requested.
    group_cols = ["District", "Zone", "Year", "Week_Start", "Week_End"]
    groups = df.groupby(group_cols, as_index=False)
    weekly = groups.agg(
        Avg_Temp_C=("Temp_C", "mean"),
        Avg_Humidity_pct=("Humidity_%", "mean"),
        Total_Rainfall_mm=("Precip_mm", "sum"),
        Days_Count=("Date", "nunique"),
        Temp_Valid=("Temp_C", "count"),
        Humidity_Valid=("Humidity_%", "count"),
        Rainfall_Valid=("Precip_mm", "count"),
    )

    # A complete calendar week requested by the user must have all seven
    # dates AND seven valid values for all three meteorological variables.
    # A 6-day interior week means missing NASA data, not a partial boundary.
    bound_start = pd.Timestamp(requested_start) if requested_start else df["Date"].min()
    bound_end = pd.Timestamp(requested_end) if requested_end else df["Date"].max()
    fully_requested = weekly["Week_Start"].ge(bound_start) & weekly["Week_End"].le(bound_end)
    invalid_week = (weekly["Days_Count"] != 7) | (
        weekly[["Temp_Valid", "Humidity_Valid", "Rainfall_Valid"]] != 7
    ).any(axis=1)
    incomplete = weekly.loc[fully_requested & invalid_week]
    if not incomplete.empty:
        sample = incomplete.iloc[0]
        raise ValueError(
            f"NASA missing climate values for {sample['District']} "
            f"week {sample['Week_Start'].date()}; cannot create a complete week."
        )

    # Drop only boundary-partial weeks, never fill them with a shorter sum.
    weekly = weekly.loc[weekly["Days_Count"].eq(7)].copy()
    if weekly.empty:
        raise ValueError("No complete seven-day epidemiological weeks in the date range")
    if (weekly["Total_Rainfall_mm"] < 0).any():
        raise ValueError("Unexpected negative weekly IMERG rainfall")

    weekly["Month"] = weekly["Week_End"].dt.strftime("%B")
    weekly = weekly.sort_values(["District", "Year", "Week_Start"])
    # Stable epi week numbering even when a user selects only part of a year.
    # For years through 2025 the first reporting week ends on the first Friday;
    # from 2026 onwards it ends on the first Sunday.
    jan_first = pd.to_datetime(weekly["Year"].astype(str) + "-01-01")
    end_weekday = weekly["Year"].ge(2026).map({True: 6, False: 4})
    days_until_first_week_end = (end_weekday - jan_first.dt.weekday) % 7
    first_week_end = jan_first + pd.to_timedelta(days_until_first_week_end, unit="D")
    weekly["Week_No"] = ((weekly["Week_End"] - first_week_end).dt.days // 7 + 1).astype(int)
    for col in ["Avg_Temp_C", "Avg_Humidity_pct", "Total_Rainfall_mm"]:
        weekly[col] = weekly[col].round(2)
    cols = [
        "Year", "Month", "Week_No", "Week_Start", "Week_End", "District",
        "Zone", "Avg_Temp_C", "Avg_Humidity_pct", "Total_Rainfall_mm", "Days_Count",
    ]
    return weekly.sort_values(["Year", "Week_No", "District"])[cols].reset_index(drop=True)


def to_excel(daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> bytes:
    """Preserve the original two-sheet weather-only Excel output."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        prepare_daily_data(daily_df).to_excel(writer, sheet_name="Daily Weather", index=False)
        weekly_df.to_excel(writer, sheet_name="Weekly Weather", index=False)
    return output.getvalue()


def main():
    st.set_page_config(page_title="Sri Lanka Weather Fetcher – IMERG", layout="wide")
    st.title("Sri Lanka RDHS Weather — NASA POWER / GPM IMERG")
    st.caption(
        "Daily T2M + RH2M + IMERG_PRECTOT (UTC); epidemiological weekly "
        "temperature/humidity averages and rainfall totals. Weather only — no dengue cases."
    )

    with st.sidebar:
        st.header("Settings")
        start_date = st.date_input(
            "Start Date", value=date(2015, 12, 26),
            min_value=date(2015, 12, 26), max_value=date.today(),
        )
        end_date = st.date_input(
            "End Date", value=min(date(2026, 5, 31), date.today()),
            min_value=date(2015, 12, 26), max_value=date.today(),
        )
        choices = list(DISTRICTS.keys())
        selected = st.multiselect(
            "RDHS areas (leave empty = all available)", options=choices, default=[]
        )
        target_districts = selected or choices
        fetch_btn = st.button("Fetch Weather Data", type="primary", use_container_width=True)

    if not fetch_btn:
        return
    if start_date >= end_date:
        st.error("Start date must be before end date")
        return

    all_daily = []
    bar = st.progress(0, text="Starting NASA POWER extraction…")
    for i, district in enumerate(target_districts, start=1):
        coords = DISTRICTS[district]
        bar.progress((i - 1) / len(target_districts), text=f"Fetching {district} ({i}/{len(target_districts)})")
        try:
            all_daily.append(
                fetch_district_data(district, coords["lat"], coords["lon"], start_date, end_date)
            )
        except (requests.RequestException, ValueError, KeyError) as exc:
            bar.empty()
            st.error(f"Fetch failed for {district}: {exc}")
            st.warning("Export stopped to avoid producing a silently incomplete 26-RDHS dataset.")
            return
    bar.empty()
    daily = pd.concat(all_daily, ignore_index=True)
    try:
        weekly = aggregate_weekly(daily, start_date, end_date)
    except ValueError as exc:
        st.error(f"Weekly validation failed: {exc}")
        return

    st.success(f"Fetched {len(target_districts)} RDHS areas; {len(weekly):,} complete district-week rows")
    st.info("The workbook contains weather only. Dengue cases must be merged separately by RDHS area and exact week dates.")
    st.subheader("Weekly weather preview")
    st.dataframe(weekly, use_container_width=True)

    st.download_button(
        "Download Daily + Weekly Weather (Excel)",
        data=to_excel(daily, weekly),
        file_name=f"sl_weather_imerg_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Download Weekly Weather (CSV)",
        data=weekly.to_csv(index=False).encode("utf-8"),
        file_name=f"sl_weather_imerg_weekly_{start_date}_{end_date}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
