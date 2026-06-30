import io
import time
import requests
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from districts import DISTRICTS

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameters to fetch from NASA POWER API
PARAMETERS = "T2M,RH2M,PRECTOTCORR"

def fetch_district_data(district: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily weather data for a single district from NASA POWER API."""
    params = {
        "parameters": PARAMETERS,
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    try:
        resp = requests.get(NASA_POWER_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["properties"]["parameter"]

        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "Date"
        df.rename(columns={
            "T2M": "Temp_C",
            "RH2M": "Humidity_%",
            "PRECTOTCORR": "Precip_mm",
        }, inplace=True)
        df["District"] = district
        df["Zone"] = DISTRICTS[district]["zone"]
        return df

    except Exception as e:
        st.warning(f"⚠ Failed to fetch data for {district}: {e}")
        return None


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily data to weekly averages and build the final report columns."""
    numeric_cols = ["Temp_C", "Humidity_%", "Precip_mm"]
    weekly = df.groupby("District")[numeric_cols].resample("W-MON", label="left", closed="left").mean()
    weekly = weekly.reset_index()
    weekly.rename(columns={"Date": "Week_Start"}, inplace=True)

    # Attach zone back (lost during groupby/resample on numeric-only columns)
    weekly["Zone"] = weekly["District"].map(lambda d: DISTRICTS[d]["zone"])

    # Derive Year, Month, Week_End, and a per-(Year) running Week_No
    weekly["Year"] = weekly["Week_Start"].dt.year
    weekly["Month"] = weekly["Week_Start"].dt.strftime("%B")
    weekly["Week_End"] = weekly["Week_Start"] + pd.Timedelta(days=6)

    # Week_No = sequential week count within each district's data (1, 2, 3...)
    weekly["Week_No"] = weekly.groupby("District").cumcount() + 1

    # Format dates as plain YYYY-MM-DD strings for Excel
    weekly["Week_Start"] = weekly["Week_Start"].dt.strftime("%Y-%m-%d")
    weekly["Week_End"] = weekly["Week_End"].dt.strftime("%Y-%m-%d")

    # Reorder columns to match the target report layout
    weekly = weekly[[
        "Year", "Month", "Week_No", "Week_Start", "Week_End",
        "District", "Zone", "Temp_C", "Humidity_%", "Precip_mm",
    ]]

    # Sort chronologically: Year -> Month order -> Week_No -> District
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    weekly["Month"] = pd.Categorical(weekly["Month"], categories=month_order, ordered=True)
    weekly = weekly.sort_values(["Year", "Month", "Week_No", "District"]).reset_index(drop=True)
    weekly["Month"] = weekly["Month"].astype(str)

    return weekly


def to_excel(df: pd.DataFrame) -> bytes:
    """Write the dataframe to a single-sheet Excel file."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Weekly Weather", index=False)
    return buffer.getvalue()


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sri Lanka Weather Fetcher", layout="wide")
st.title("Sri Lanka District Weather — NASA POWER")
st.caption("Fetches daily data and aggregates to weekly averages for all 25 districts.")

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90), min_value=date(1981, 1, 1), max_value=date.today())
    end_date   = st.date_input("End Date",   value=date.today() - timedelta(days=1),  min_value=date(1981, 1, 1), max_value=date.today())

    selected_districts = st.multiselect(
        "Districts (leave empty = all 25)",
        options=list(DISTRICTS.keys()),
        default=[],
    )
    fetch_btn = st.button("Fetch Weather Data", type="primary", use_container_width=True)

target_districts = selected_districts if selected_districts else list(DISTRICTS.keys())

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if fetch_btn:
    start_str = start_date.strftime("%Y%m%d")
    end_str   = end_date.strftime("%Y%m%d")

    all_daily = []
    progress = st.progress(0, text="Starting fetch…")

    for i, district in enumerate(target_districts):
        coords = DISTRICTS[district]
        progress.progress((i + 1) / len(target_districts), text=f"Fetching {district}…")
        df = fetch_district_data(district, coords["lat"], coords["lon"], start_str, end_str)
        if df is not None:
            all_daily.append(df)
        time.sleep(0.3)

    progress.empty()

    if not all_daily:
        st.error("No data fetched. Check your date range or network.")
        st.stop()

    daily_combined = pd.concat(all_daily)
    weekly_df = aggregate_weekly(daily_combined)

    st.success(f"✅ Fetched {len(all_daily)}/{len(target_districts)} districts.")

    st.subheader("Weekly Data Preview")
    st.dataframe(weekly_df, width='stretch')

    excel_bytes = to_excel(weekly_df)
    st.download_button(
        label="📥 Download Excel",
        data=excel_bytes,
        file_name=f"sl_weather_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )