#!/usr/bin/env python3
from __future__ import annotations

import os
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# INPUT PATHS (your exact paths)
# ------------------------------------------------------------
FLIGHTS_CSV = "/home/jon/Documents/grad_school/OR568/project/OR568_ML_Project/data_pipeline/bwi_delays.csv"
WEATHER_CSV = "/home/jon/Documents/grad_school/OR568/project/OR568_ML_Project/data_pipeline/BWI_Weather_from_Iowa Environmental Mesonet_IEM_ASOS_27 June 2022(in).csv"
REGISTRY_CSV = "/home/jon/Documents/grad_school/OR568/project/OR568_ML_Project/data_pipeline/raw_data/faa_aircraft_registry.csv"

# Output
OUT_CSV = "/home/jon/Documents/grad_school/OR568/project/OR568_ML_Project/data_pipeline/enriched_flights.csv"

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
# Nearest weather observation allowed distance from scheduled time
WEATHER_TOLERANCE = pd.Timedelta("2h")

# Regional features thresholds
STORM_SEVERITY_THRESHOLD = 1.5
SYSTEM_INDEX_THRESHOLD = 1.0
SYSTEM_STORM_COUNT_THRESHOLD = 2

# BTS scheduled times are local; for BWI this is America/New_York
# We convert scheduled times to UTC to align with IEM timestamps (typically UTC).
LOCAL_TZ_DEFAULT = "America/New_York"

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def hhmm_to_minutes(hhmm) -> float:
    """Convert HHMM (e.g., 1209) to minutes after midnight."""
    if pd.isna(hhmm):
        return np.nan
    # CRSDepTime can be int/float; preserve leading zeros
    s = str(int(float(hhmm))).zfill(4)
    hh = int(s[:2])
    mm = int(s[2:])
    return hh * 60 + mm


def build_local_ts(flight_date: pd.Series, hhmm: pd.Series, local_tz: str) -> pd.Series:
    """
    Build timezone-aware timestamps from FlightDate + HHMM in local time, then keep tz-aware.
    """
    d = pd.to_datetime(flight_date, errors="coerce")  # date only
    mins = hhmm.apply(hhmm_to_minutes)
    ts_naive = d + pd.to_timedelta(mins, unit="m")
    # Localize to local_tz, then convert to UTC
    ts_local = ts_naive.dt.tz_localize(local_tz, nonexistent="NaT", ambiguous="NaT")
    ts_utc = ts_local.dt.tz_convert("UTC")
    return ts_utc


def normalize_upper(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def compute_weather_severity(wind_knots, vis_miles, precip_in, ceiling_ft) -> pd.Series:
    """
    Excel-matching formula:
      (wind/30) + (1 - vis/10) + IF(precip>0,1,0) + IF(ceiling<1000,1,0)
    """
    wind = pd.to_numeric(wind_knots, errors="coerce").fillna(0.0)
    vis = pd.to_numeric(vis_miles, errors="coerce").fillna(10.0).clip(0.0, 10.0)
    p01i = pd.to_numeric(precip_in, errors="coerce").fillna(0.0)
    ceil = pd.to_numeric(ceiling_ft, errors="coerce").fillna(99999.0)

    precip_flag = (p01i > 0).astype(int)
    low_ceiling_flag = (ceil < 1000).astype(int)

    return (wind / 30.0) + (1.0 - (vis / 10.0)) + precip_flag + low_ceiling_flag


def prep_weather(weather_csv: str) -> pd.DataFrame:
    w = pd.read_csv(weather_csv)

    # Expected columns: station, valid, tmpf, sknt, vsby, p01i, skyl1, ...
    w["station"] = normalize_upper(w["station"])

    # IEM 'valid' is typically UTC; parse as UTC-aware.
    w["valid_ts"] = pd.to_datetime(w["valid"], errors="coerce", utc=True)

    # Keep only needed columns (safe if extras exist)
    keep = ["station", "valid_ts", "tmpf", "sknt", "vsby", "p01i", "skyl1", "gust", "wxcodes"]
    keep = [c for c in keep if c in w.columns]
    w = w[keep].copy()

    w = w.dropna(subset=["station", "valid_ts"]).sort_values(["station", "valid_ts"])
    return w


def attach_weather_nearest(
    flights: pd.DataFrame,
    weather: pd.DataFrame,
    airport_col: str,
    time_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Nearest-time weather join using merge_asof.

    Key requirement: left/right must be sorted by the 'on' key globally (ts).
    When using `by=station`, we still sort by ts first to satisfy pandas.
    """
    f = flights.copy()

    # Station key (airport code)
    f[f"{prefix}_station"] = normalize_upper(f[airport_col])

    # Stable row id so we can join back safely
    f["_row_id"] = np.arange(len(f), dtype=np.int64)

    # Left frame: one row per flight with join keys
    left = f[["_row_id", f"{prefix}_station", time_col]].rename(
        columns={f"{prefix}_station": "station", time_col: "ts"}
    )

    # Ensure datetime (UTC) and drop NaT after coercion
    left["station"] = normalize_upper(left["station"])
    left["ts"] = pd.to_datetime(left["ts"], utc=True, errors="coerce")
    left = left.dropna(subset=["station", "ts"])

    # Right frame: weather observations
    right = weather.copy()
    right = right.rename(columns={"valid_ts": "ts"})
    right["station"] = normalize_upper(right["station"])
    right["ts"] = pd.to_datetime(right["ts"], utc=True, errors="coerce")
    right = right.dropna(subset=["station", "ts"])

    # IMPORTANT: sort by ts FIRST (global monotonic requirement), then station
    left = left.sort_values(["ts", "station"]).reset_index(drop=True)
    right = right.sort_values(["ts", "station"]).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        on="ts",
        by="station",
        direction="nearest",
        tolerance=WEATHER_TOLERANCE,
    )

    # Join merged weather back to flights by row_id
    f = f.merge(
        merged.drop(columns=["station", "ts"], errors="ignore"),
        on="_row_id",
        how="left",
    )

    # Derived features
    f[f"{prefix}_tmpf"] = f.get("tmpf", np.nan)
    f[f"{prefix}_wind"] = f.get("sknt", np.nan)
    f[f"{prefix}_visibility"] = f.get("vsby", np.nan)

    p01i = pd.to_numeric(f.get("p01i", 0), errors="coerce").fillna(0.0)
    f[f"{prefix}_precip_flag"] = (p01i > 0).astype(int)

    ceil = pd.to_numeric(f.get("skyl1", np.nan), errors="coerce")
    f[f"{prefix}_ceiling_ft"] = ceil
    f[f"{prefix}_low_ceiling_flag"] = (ceil.fillna(99999.0) < 1000).astype(int)

    f[f"{prefix}_weather_severity"] = compute_weather_severity(
        f[f"{prefix}_wind"],
        f[f"{prefix}_visibility"],
        p01i,
        f[f"{prefix}_ceiling_ft"],
    )

    # Optional: drop raw weather columns
    for c in ["tmpf", "sknt", "vsby", "p01i", "skyl1", "gust", "wxcodes"]:
        if c in f.columns:
            f.drop(columns=[c], inplace=True, errors="ignore")

    # Cleanup helper
    f.drop(columns=["_row_id"], inplace=True, errors="ignore")

    return f


def add_regional_features(flights: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Computes hourly regional summaries across the stations present in `weather`.
    If your weather file includes only BWI, the regional index == BWI severity (per hour).
    """
    w = weather.copy()
    for col in ["sknt", "vsby", "p01i", "skyl1"]:
        if col not in w.columns:
            w[col] = np.nan

    w["severity"] = compute_weather_severity(w["sknt"], w["vsby"], w["p01i"], w["skyl1"])

    # FIX: use "h" not "H"
    w["hour"] = w["valid_ts"].dt.floor("h")

    regional = (
        w.groupby("hour")
         .agg(
             regional_weather_index=("severity", "mean"),
             regional_storm_count=("severity", lambda s: int((s >= STORM_SEVERITY_THRESHOLD).sum())),
         )
         .reset_index()
         .sort_values("hour")
    )

    f = flights.copy()

    # FIX: use "h" not "H"
    f["dep_hour"] = f["crs_dep_ts_utc"].dt.floor("h")

    f = f.merge(regional, left_on="dep_hour", right_on="hour", how="left").drop(columns=["hour"], errors="ignore")

    f["regional_storm_count"] = pd.to_numeric(f["regional_storm_count"], errors="coerce").fillna(0).astype(int)

    f["system_weather_flag"] = np.where(
        (pd.to_numeric(f["regional_weather_index"], errors="coerce") > SYSTEM_INDEX_THRESHOLD) |
        (f["regional_storm_count"] >= SYSTEM_STORM_COUNT_THRESHOLD),
        1,
        0
    )

    return f


def attach_registry(flights: pd.DataFrame, registry_csv: str) -> pd.DataFrame:
    """
    Join registry fields using Tail_Number (BTS) -> n_number (registry).
    """
    reg = pd.read_csv(registry_csv)

    # Normalize keys
    f = flights.copy()
    f["Tail_Number"] = normalize_upper(f["Tail_Number"])

    if "n_number" not in reg.columns:
        raise ValueError("Registry CSV missing required column: n_number")

    reg["n_number"] = normalize_upper(reg["n_number"])

    # Choose columns to bring in (add more if you want)
    registry_cols = [
        "n_number",
        "aircraft_type",
        "num_seats",
        "aircraft_manufacturer",
        "aircraft_model",
        "num_engines",
        "manufacturing_year",
        "registrant_type",
        "registrant_name",
        "registrant_city",
        "registrant_state",
        "registrant_country",
    ]
    registry_cols = [c for c in registry_cols if c in reg.columns]

    reg_small = reg[registry_cols].drop_duplicates(subset=["n_number"])

    f = f.merge(reg_small, left_on="Tail_Number", right_on="n_number", how="left")
    f = f.drop(columns=["n_number"], errors="ignore")
    return f


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main() -> None:
    # 1) Flights (BTS delays)
    flights = pd.read_csv(FLIGHTS_CSV)

    # Normalize airport codes
    flights["Origin"] = normalize_upper(flights["Origin"])
    flights["Dest"] = normalize_upper(flights["Dest"])

    # Build scheduled timestamps in UTC (assume BTS times are local to BWI timezone)
    flights["crs_dep_ts_utc"] = build_local_ts(flights["FlightDate"], flights["CRSDepTime"], LOCAL_TZ_DEFAULT)
    flights["crs_arr_ts_utc"] = build_local_ts(flights["FlightDate"], flights["CRSArrTime"], LOCAL_TZ_DEFAULT)

    # 2) Weather
    weather = prep_weather(WEATHER_CSV)

    # 3) Attach dep weather (Origin)
    flights = attach_weather_nearest(
        flights=flights,
        weather=weather,
        airport_col="Origin",
        time_col="crs_dep_ts_utc",
        prefix="dep"
    )

    # 4) Attach arr weather (Dest)
    flights = attach_weather_nearest(
        flights=flights,
        weather=weather,
        airport_col="Dest",
        time_col="crs_arr_ts_utc",
        prefix="arr"
    )

    # 5) Regional system features
    flights = add_regional_features(flights, weather)

    # 6) Registry join
    flights = attach_registry(flights, REGISTRY_CSV)

    # 7) Output columns (your key set + useful IDs)
    out_cols = [
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "CRSDepTime",
        "ArrDelay",
        "Tail_Number",
        # Registry
        "aircraft_type",
        "num_seats",
        # Dep wx
        "dep_tmpf",
        "dep_wind",
        "dep_visibility",
        "dep_precip_flag",
        "dep_ceiling_ft",
        "dep_weather_severity",
        # Arr wx
        "arr_tmpf",
        "arr_wind",
        "arr_visibility",
        "arr_precip_flag",
        "arr_ceiling_ft",
        "arr_weather_severity",
        # Regional system
        "regional_weather_index",
        "regional_storm_count",
        "system_weather_flag",
    ]
    out_cols = [c for c in out_cols if c in flights.columns]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    flights[out_cols].to_csv(OUT_CSV, index=False)

    print(f"✅ Enriched dataset saved: {OUT_CSV}")
    print(f"Rows: {len(flights):,} | Columns written: {len(out_cols)}")


if __name__ == "__main__":
    main()