import io
import time
import requests
import pandas as pd
import streamlit as st
from datetime import date, timedelta
from districts import DISTRICTS

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "T2M,RH2M,PRECTOTCORR"

def fetch_district_data(district, lat, lon, start, end):
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
        df.rename(columns={"T2M": "Temp_C", "RH2M": "Humidity_%", "PRECTOTCORR": "Precip_mm"}, inplace=True)
        df["District"] = district
        return df
    except Exception as e:
        st.warning(f"Failed to fetch data for {district}: {e}")
        return None

def aggregate_weekly(df):
    numeric_cols = ["Temp_C", "Humidity_%", "Precip_mm"]
    weekly = df.groupby("District")[numeric_cols].resample("W-MON", label="left", closed="left").mean()
    weekly = weekly.reset_index()
    weekly.rename(columns={"Date": "Week_Start"}, inplace=True)
    weekly["Week_Start"] = weekly["Week_Start"].dt.strftime("%Y-%m-%d")
    return weekly

def to_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Districts", index=False)
        for district, group in df.groupby("District"):
            group.to_excel(writer, sheet_name=district[:31], index=False)
    return buffer.getvalue()

st.set_page_config(page_title="Sri Lanka Weather Fetcher", layout="wide")
st.title("Sri Lanka District Weather - NASA POWER")
st.caption("Fetches daily data and aggregates to weekly averages for all 25 districts.")

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=90))
    end_date   = st.date_input("End Date",   value=date.today() - timedelta(days=1))
    selected_districts = st.multiselect("Districts (leave empty = all 25)", options=list(DISTRICTS.keys()), default=[])
    fetch_btn = st.button("Fetch Weather Data", type="primary", use_container_width=True)

target_districts = selected_districts if selected_districts else list(DISTRICTS.keys())

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if fetch_btn:
    start_str = start_date.strftime("%Y%m%d")
    end_str   = end_date.strftime("%Y%m%d")
    all_daily = []
    progress = st.progress(0, text="Starting fetch...")

    for i, district in enumerate(target_districts):
        coords = DISTRICTS[district]
        progress.progress((i + 1) / len(target_districts), text=f"Fetching {district}...")
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

    st.success(f"Fetched {len(all_daily)}/{len(target_districts)} districts.")
    st.subheader("Weekly Data Preview")
    st.dataframe(weekly_df, use_container_width=True)

    excel_bytes = to_excel(weekly_df)
    st.download_button(
        label="Download Excel",
        data=excel_bytes,
        file_name=f"sl_weather_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
