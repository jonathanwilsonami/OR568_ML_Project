"""
data.py
-------
Load, engineer, and cache flight data for the demo app.

Strategy for 7M+ rows
──────────────────────
1. First run  → read raw parquet, engineer all features, write
   data/flights_canonical_2019_cache.parquet.  Takes ~60-120s once.

2. Every subsequent startup  → read cache directly (~5-10s).
   Cache is rebuilt automatically when source file is newer.

3. A FlightIndex is built from the loaded DataFrame — pre-built
   dicts mapping tail-number and flight-number to row positions.
   Every callback touches only the ~5-20 rows it needs.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Holidays 2019-2025
# ---------------------------------------------------------------------------
_HOLIDAYS = {pd.Timestamp(d) for d in [
    "2019-01-01","2019-01-21","2019-02-18","2019-05-27","2019-07-04",
    "2019-09-02","2019-10-14","2019-11-11","2019-11-28","2019-12-25",
    "2020-01-01","2020-01-20","2020-02-17","2020-05-25","2020-07-03",
    "2020-09-07","2020-10-12","2020-11-11","2020-11-26","2020-12-25",
    "2021-01-01","2021-01-18","2021-02-15","2021-05-31","2021-07-05",
    "2021-09-06","2021-10-11","2021-11-11","2021-11-25","2021-12-24",
    "2022-01-01","2022-01-17","2022-02-21","2022-05-30","2022-07-04",
    "2022-09-05","2022-10-10","2022-11-11","2022-11-24","2022-12-25",
    "2023-01-01","2023-01-16","2023-02-20","2023-05-29","2023-07-04",
    "2023-09-04","2023-10-09","2023-11-10","2023-11-23","2023-12-25",
    "2024-01-01","2024-01-15","2024-02-19","2024-05-27","2024-07-04",
    "2024-09-02","2024-10-14","2024-11-11","2024-11-28","2024-12-25",
    "2025-01-01","2025-01-20","2025-02-17","2025-05-26","2025-07-04",
    "2025-09-01","2025-10-13","2025-11-11","2025-11-27","2025-12-25",
]}
_HOLIDAY_SET   = frozenset(h.date() for h in _HOLIDAYS)
_HOLIDAY_DATES = sorted(_HOLIDAYS)

# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------
XGB_FULL_FEATURES = [
    "distance","dep_hour_local","dep_weekday_local","dep_month_local",
    "dep_time_bucket","is_weekend","is_holiday","days_to_nearest_holiday",
    "crs_elapsed_time",
    "dep_temp_c","dep_wind_speed_m_s","dep_wind_dir_deg","dep_ceiling_height_m",
    "arr_temp_c","arr_wind_speed_m_s","arr_wind_dir_deg","arr_ceiling_height_m",
    "route_frequency","origin_flight_volume","dest_flight_volume",
    "prev_arr_delay","prev_dep_delay","prev_arr_del15","prev_dep_del15",
    "prev_arr_delayed_flag","prev_total_delay",
    "turnaround_minutes","tight_turnaround_flag","rotation_continuity_flag",
    "aircraft_leg_number_day","relative_leg_position",
    "cum_dep_delay_aircraft_day","cum_arr_delay_aircraft_day",
    "prev1_arr_delay","prev1_dep_delay","prev1_arr_del15","prev1_dep_del15",
    "prev2_arr_delay","prev2_dep_delay","prev2_arr_del15","prev2_dep_del15",
    "prev1_turnaround_minutes","time_since_prev2_arrival_minutes",
]

TARGET_CLASS = "arr_del15"
TARGET_REG   = "arr_delay"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_col(col: str) -> str:
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    col = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", col)
    col = re.sub(r"([a-z0-9])([A-Z])",    r"\1_\2", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_").lower()


def _dep_time_bucket_vec(hours: pd.Series) -> pd.Series:
    b = pd.Series(6, index=hours.index, dtype="int8")
    b = b.where(hours >= 21, 5).where(hours >= 18, 4).where(
        hours >= 14, 3).where(hours >= 11, 2).where(hours >= 6, 1)
    return b


def _is_holiday_vec(dates: pd.Series) -> pd.Series:
    return dates.dt.date.map(lambda d: int(d in _HOLIDAY_SET)).astype("int8")


def _days_to_nearest_holiday_vec(dates: pd.Series) -> pd.Series:
    holiday_arr = np.array([h.value for h in _HOLIDAY_DATES], dtype="int64")
    ns_per_day  = 86_400_000_000_000
    ts_ns = dates.values.astype("int64")
    result = np.array([int(np.min(np.abs(holiday_arr - t)) // ns_per_day)
                       for t in ts_ns], dtype="int32")
    return pd.Series(result, index=dates.index)


def _parse_date(date_str: str):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(date_str, format=fmt).date()
        except Exception:
            pass
    try:
        return pd.to_datetime(date_str).date()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature engineering  (runs once, result cached to disk)
# ---------------------------------------------------------------------------

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  [eng] {len(df):,} rows — engineering features …")

    # Timestamps
    for col in ("dep_ts_actual_utc", "arr_ts_actual_utc", "flight_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    if "dep_ts_actual_utc" in df.columns:
        ts = df["dep_ts_actual_utc"]
        if "dep_hour_local"    not in df.columns: df["dep_hour_local"]    = ts.dt.hour.astype("int8")
        if "dep_weekday_local" not in df.columns: df["dep_weekday_local"] = (ts.dt.weekday+1).astype("int8")
        if "dep_month_local"   not in df.columns: df["dep_month_local"]   = ts.dt.month.astype("int8")
        if "flight_date"       not in df.columns: df["flight_date"]       = ts.dt.normalize()

    if "dep_hour_local" in df.columns:
        df["dep_time_bucket"] = _dep_time_bucket_vec(df["dep_hour_local"])
    if "dep_weekday_local" in df.columns:
        df["is_weekend"] = df["dep_weekday_local"].isin([6,7]).astype("int8")

    if "flight_date" in df.columns:
        print("  [eng] holiday features …")
        df["is_holiday"]              = _is_holiday_vec(df["flight_date"])
        df["days_to_nearest_holiday"] = _days_to_nearest_holiday_vec(df["flight_date"])

    # Traffic volumes
    print("  [eng] traffic volumes …")
    if "origin" in df.columns and "dest" in df.columns:
        if "route_key" not in df.columns:
            df["route_key"] = df["origin"].str.cat(df["dest"], sep="_")
        if "route_frequency"      not in df.columns:
            df["route_frequency"]      = df.groupby("route_key")["route_key"].transform("count")
        if "origin_flight_volume" not in df.columns:
            df["origin_flight_volume"] = df.groupby("origin")["origin"].transform("count")
        if "dest_flight_volume"   not in df.columns:
            df["dest_flight_volume"]   = df.groupby("dest")["dest"].transform("count")

    # Sort for time-based shifts
    sort_cols = [c for c in ("tail_number","dep_ts_actual_utc") if c in df.columns]
    if sort_cols:
        print("  [eng] sorting …")
        df = df.sort_values(sort_cols).reset_index(drop=True)

    has_tail = "tail_number" in df.columns

    def _gs(col: str, n: int) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype="float32")
        return (df.groupby("tail_number")[col].shift(n)
                if has_tail else df[col].shift(n))

    # Lag features
    print("  [eng] lag features …")
    for src, dst in [
        ("arr_delay","prev_arr_delay"),  ("dep_delay","prev_dep_delay"),
        ("arr_del15","prev_arr_del15"),  ("dep_del15","prev_dep_del15"),
        ("arr_delay","prev1_arr_delay"), ("dep_delay","prev1_dep_delay"),
        ("arr_del15","prev1_arr_del15"), ("dep_del15","prev1_dep_del15"),
    ]:
        if dst not in df.columns:
            df[dst] = _gs(src, 1)

    for src, dst in [
        ("arr_delay","prev2_arr_delay"), ("dep_delay","prev2_dep_delay"),
        ("arr_del15","prev2_arr_del15"), ("dep_del15","prev2_dep_del15"),
    ]:
        if dst not in df.columns:
            df[dst] = _gs(src, 2)

    if "prev_arr_delay" in df.columns:
        if "prev_arr_delayed_flag" not in df.columns:
            df["prev_arr_delayed_flag"] = (df["prev_arr_delay"].fillna(0)>15).astype("int8")
        if "prev_total_delay" not in df.columns:
            df["prev_total_delay"] = df["prev_arr_delay"].fillna(0)+df["prev_dep_delay"].fillna(0)

    # Turnaround
    if "dep_ts_actual_utc" in df.columns and "arr_ts_actual_utc" in df.columns:
        print("  [eng] turnaround times …")
        arr_s1 = _gs("arr_ts_actual_utc",1)
        arr_s2 = _gs("arr_ts_actual_utc",2)
        if "prev1_turnaround_minutes" not in df.columns:
            df["prev1_turnaround_minutes"] = (
                (df["dep_ts_actual_utc"]-arr_s1).dt.total_seconds().div(60).astype("float32"))
        if "turnaround_minutes" not in df.columns:
            df["turnaround_minutes"] = df["prev1_turnaround_minutes"]
        if "time_since_prev2_arrival_minutes" not in df.columns:
            df["time_since_prev2_arrival_minutes"] = (
                (df["dep_ts_actual_utc"]-arr_s2).dt.total_seconds().div(60).astype("float32"))

    # Per-aircraft-day stats
    if "tail_number" in df.columns and "flight_date" in df.columns:
        print("  [eng] per-aircraft-day stats …")
        grp = df.groupby(["tail_number","flight_date"])
        if "aircraft_leg_number_day" not in df.columns:
            df["aircraft_leg_number_day"] = (grp.cumcount()+1).astype("int8")
        if "cum_dep_delay_aircraft_day" not in df.columns:
            df["cum_dep_delay_aircraft_day"] = (
                grp["dep_delay"].cumsum().shift(1).fillna(0).astype("float32")
                if "dep_delay" in df.columns else 0.0)
        if "cum_arr_delay_aircraft_day" not in df.columns:
            df["cum_arr_delay_aircraft_day"] = (
                grp["arr_delay"].cumsum().shift(1).fillna(0).astype("float32")
                if "arr_delay" in df.columns else 0.0)

    # Derived flags
    if "turnaround_minutes" in df.columns and "tight_turnaround_flag" not in df.columns:
        df["tight_turnaround_flag"] = (df["turnaround_minutes"].fillna(999)<60).astype("int8")
    if "aircraft_leg_number_day" in df.columns and "relative_leg_position" not in df.columns:
        max_leg = (df.groupby("tail_number")["aircraft_leg_number_day"].transform("max")
                   if has_tail else pd.Series(1, index=df.index))
        df["relative_leg_position"] = (
            df["aircraft_leg_number_day"]/max_leg.replace(0,1)).astype("float32")
    if "rotation_continuity_flag" not in df.columns:
        df["rotation_continuity_flag"] = (
            (df["prev_arr_delay"].fillna(0)<60).astype("int8")
            if "prev_arr_delay" in df.columns else 1)

    # Fill missing features
    for col in XGB_FULL_FEATURES:
        if col not in df.columns:
            df[col] = np.float32(0)
        elif df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0).astype("float32")

    print("  [eng] done.")
    return df


# ---------------------------------------------------------------------------
# Public loader with disk cache
# ---------------------------------------------------------------------------

def load_and_engineer(
    parquet_path: str | Path,
    cache_suffix: str = "_cache",
) -> pd.DataFrame:
    """
    Load parquet + engineer features, with automatic disk caching.
    First run: slow (feature engineering on all rows).
    Subsequent runs: fast (reads cache parquet directly).
    """
    src   = Path(parquet_path)
    cache = src.with_name(src.stem + cache_suffix + ".parquet")

    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        print(f"[cache] Reading {cache.name} …")
        df = pd.read_parquet(cache)
        print(f"[cache] {len(df):,} rows ready.")
        return df

    print(f"[cache] No cache — processing {src.name} from scratch …")
    df = pd.read_parquet(src)
    df.columns = [_clean_col(c) for c in df.columns]
    print(f"[raw]   {len(df):,} rows × {df.shape[1]} cols")

    for flag in ("is_cancelled","is_diverted"):
        if flag in df.columns:
            df = df[df[flag]==0]
    if TARGET_CLASS in df.columns:
        df = df[df[TARGET_CLASS].notna()]
    print(f"[raw]   {len(df):,} rows after filter")

    df = _engineer(df)

    print(f"[cache] Writing {cache.name} …")
    df.to_parquet(cache, index=False, compression="snappy")
    print(f"[cache] Written ({cache.stat().st_size/1e6:.0f} MB).")
    return df


# ---------------------------------------------------------------------------
# FlightIndex — fast O(1) lookups over 7M rows
# ---------------------------------------------------------------------------

class FlightIndex:
    """
    Pre-built lookup dicts so callbacks never scan the full DataFrame.

    tail_index   : {"N823AW": [pos, pos, ...]}   sorted by dep_ts at build time
    flight_index : {"1294":   [pos, pos, ...]}
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

        print("[index] Building tail-number index …")
        self.tail_index: dict[str, list[int]] = {}
        if "tail_number" in df.columns:
            for pos, val in enumerate(df["tail_number"].astype(str).str.upper()):
                self.tail_index.setdefault(val, []).append(pos)

        print("[index] Building flight-number index …")
        self.flight_index: dict[str, list[int]] = {}
        # BTS canonical column is flight_number_reporting_airline (int).
        # After _clean_col it keeps the same name. Also handle common aliases.
        for col in (
            "flight_number_reporting_airline",
            "flight_number",
            "fl_num",
            "op_carrier_fl_num",
        ):
            if col in df.columns:
                print(f"[index] Flight number column: '{col}'")
                for pos, val in enumerate(df[col].astype(str).str.strip()):
                    self.flight_index.setdefault(val, []).append(pos)
                    stripped = val.lstrip("0")
                    if stripped and stripped != val:
                        self.flight_index.setdefault(stripped, []).append(pos)
                break

        print("[index] Building flight-id index …")
        self.flight_id_index: dict[str, int] = {}   # flight_id → single row pos
        if "flight_id" in df.columns:
            for pos, val in enumerate(df["flight_id"].astype(str)):
                self.flight_id_index[val] = pos   # flight_id is unique per row

        print(f"[index] {len(self.tail_index):,} tails | "
              f"{len(self.flight_index):,} flight numbers | "
              f"{len(self.flight_id_index):,} flight IDs  — ready.")

    def search(
        self,
        tail_number: str | None = None,
        date: str | None = None,
        flight_number: str | None = None,
        origin: str | None = None,
        dest: str | None = None,
    ) -> pd.DataFrame:
        """
        Return a small DataFrame of matching rows.
        Modes:
          tail_number [+ date] [+ origin/dest]
          flight_number [+ date] [+ origin/dest]
        """
        df = self._df

        if tail_number:
            key = tail_number.strip().upper()
            positions = self.tail_index.get(key, [])
            if not positions:
                return pd.DataFrame()
            subset = df.iloc[positions].copy()
        elif flight_number:
            # flight_number_reporting_airline is an integer — match as string
            # Accept "552", "0552", "G4 552" etc.
            raw = str(flight_number).strip().split()[-1]   # drop carrier prefix if any
            positions = (
                self.flight_index.get(raw)
                or self.flight_index.get(raw.lstrip("0"))
                or []
            )
            if not positions:
                return pd.DataFrame()
            subset = df.iloc[positions].copy()
        else:
            return pd.DataFrame()

        # Optional date filter
        if date:
            d = _parse_date(date)
            if d is not None and "flight_date" in subset.columns:
                subset = subset[subset["flight_date"].dt.date == d]

        # Optional route filters
        if origin:
            subset = subset[subset["origin"].str.upper() == origin.strip().upper()]
        if dest:
            subset = subset[subset["dest"].str.upper() == dest.strip().upper()]

        return subset.reset_index(drop=True)

    def get_chain(self, row: pd.Series) -> dict:
        """
        Return prev1 and prev2 airport codes for a matched flight row.

        The BTS canonical parquet already has prev_origin / prev_dest
        pre-computed on every row, so we read them directly — no scan needed.
        We fall back to a tail-index scan only if those columns are absent.
        """
        out = {"prev2_origin":"","prev2_dest":"","prev1_origin":"","prev1_dest":""}

        # ── Fast path: columns already on the row ─────────────────────
        p_orig = str(row.get("prev_origin","") or "")
        p_dest = str(row.get("prev_dest","")   or "")
        if p_orig and p_orig not in ("nan","None",""):
            out["prev1_origin"] = p_orig
            out["prev1_dest"]   = p_dest  # prev1 route: prev_origin → prev_dest
            # For prev2 we look up the flight whose id == prev_flight_id_same_tail
            prev_fid = str(row.get("prev_flight_id_same_tail","") or "")
            if prev_fid and prev_fid not in ("nan","None",""):
                fid_pos = self.flight_id_index.get(prev_fid)
                if fid_pos is not None:
                    prev1_row = self._df.iloc[fid_pos]
                    p2_orig = str(prev1_row.get("prev_origin","") or "")
                    p2_dest = str(prev1_row.get("prev_dest","")   or "")
                    if p2_orig not in ("nan","None",""):
                        out["prev2_origin"] = p2_orig
                        out["prev2_dest"]   = p2_dest
            return out

        # ── Fallback: scan tail index ─────────────────────────────────
        tail   = str(row.get("tail_number","")).upper()
        dep_ts = row.get("dep_ts_actual_utc")
        if not tail or pd.isna(dep_ts):
            return out
        positions = self.tail_index.get(tail, [])
        if len(positions) < 2:
            return out
        tail_df = self._df.iloc[positions]
        if "dep_ts_actual_utc" not in tail_df.columns:
            return out
        before = tail_df[tail_df["dep_ts_actual_utc"] < dep_ts]
        if len(before) >= 1:
            p1 = before.iloc[-1]
            out["prev1_origin"] = str(p1.get("origin",""))
            out["prev1_dest"]   = str(p1.get("dest",""))
        if len(before) >= 2:
            p2 = before.iloc[-2]
            out["prev2_origin"] = str(p2.get("origin",""))
            out["prev2_dest"]   = str(p2.get("dest",""))
        return out


# ---------------------------------------------------------------------------
# Legacy wrappers (so existing app.py imports don't break)
# ---------------------------------------------------------------------------

def find_flight(df, flight_id=None, tail_number=None, origin=None, dest=None):
    result = df
    if flight_id:
        fid = str(flight_id).strip().upper()
        for col in ("flight_id","fl_num","op_carrier_fl_num"):
            if col in result.columns:
                mask = result[col].astype(str).str.strip().str.upper() == fid
                if mask.any():
                    result = result[mask]; break
    if tail_number:
        tn = str(tail_number).strip().upper()
        for col in ("tail_number","tail_num"):
            if col in result.columns:
                result = result[result[col].astype(str).str.strip().str.upper()==tn]; break
    if origin:
        result = result[result["origin"].astype(str).str.upper()==origin.strip().upper()]
    if dest:
        result = result[result["dest"].astype(str).str.upper()==dest.strip().upper()]
    return result.reset_index(drop=True)


def get_propagation_chain(df, row):
    chain = {"prev2_origin":"","prev2_dest":"","prev1_origin":"","prev1_dest":""}
    if "tail_number" not in df.columns or "dep_ts_actual_utc" not in df.columns:
        return chain
    tail   = row.get("tail_number","")
    dep_ts = row.get("dep_ts_actual_utc")
    if not tail or pd.isna(dep_ts):
        return chain
    tail_df = df[df["tail_number"]==tail].sort_values("dep_ts_actual_utc")
    before  = tail_df[tail_df["dep_ts_actual_utc"]<dep_ts]
    if len(before)>=1:
        p1=before.iloc[-1]; chain["prev1_origin"]=str(p1.get("origin","")); chain["prev1_dest"]=str(p1.get("dest",""))
    if len(before)>=2:
        p2=before.iloc[-2]; chain["prev2_origin"]=str(p2.get("origin","")); chain["prev2_dest"]=str(p2.get("dest",""))
    return chain