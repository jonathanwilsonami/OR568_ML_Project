#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import polars as pl

# ============================================================
# Build Enriched Flight Delay Dataset (BTS + Hourly Weather + Registry)
# Fixes:
#  - Polars "streaming" deprecation -> use engine="streaming"
#  - Weather CSV parsing error: "could not parse `null` as dtype f64"
#    -> force weather numeric columns to Utf8 on read + convert later
# ============================================================

# ----------------------------
# Paths (your directory layout)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "raw_data"

bts_path = data_dir / "BTS_data_2019.csv"                      # BTS flights CSV
weather_path = data_dir / "weather.csv"                        # Weather CSV (IEM/ASOS export)
registry_path = data_dir / "faa_flight_registry_2025.parquet"  # Registry parquet

out_dir = BASE_DIR / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

OUT_CSV = out_dir / "enriched_flights.csv"
OUT_PARQUET = out_dir / "enriched_flights.parquet"

# ----------------------------
# Config
# ----------------------------
FORCE_REGISTRY_STRINGS = True  # fill ALL registry cols with "Registry Not Found"
AIRPORT_TO_STATION: dict[str, str] = {
    # "BWI": "KBWI",
}

# Weather columns that often contain "null" strings; read as strings first
WEATHER_NUMERIC_COLS = [
    "lon", "lat", "elevation", "tmpf", "dwpf", "relh", "drct", "sknt",
    "p01i", "alti", "mslp", "vsby", "gust",
    "skyl1", "skyl2", "skyl3", "skyl4",
    "ice_accretion_1hr", "ice_accretion_3hr", "ice_accretion_6hr",
    "peak_wind_gust", "peak_wind_drct", "feel", "snowdepth",
]

# ----------------------------
# Helpers
# ----------------------------
def lazy_read_bts(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(path, infer_schema_length=50000)

def lazy_read_registry(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)

def lazy_read_weather(path: Path) -> pl.LazyFrame:
    """
    Weather CSV may contain literal 'null' strings in numeric columns.
    We prevent parser errors by reading those columns as Utf8, then converting later.
    """
    return pl.scan_csv(
        path,
        infer_schema_length=50000,
        null_values=["null", "NULL", ""],
        schema_overrides={c: pl.Utf8 for c in WEATHER_NUMERIC_COLS},
        ignore_errors=True,  # tolerate any weird rows; you can set False once stable
    )

def normalize_upper(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_uppercase()

def parse_flight_date_expr(date_col: str) -> pl.Expr:
    s = pl.col(date_col).cast(pl.Utf8)
    return (
        pl.when(s.str.contains(r"/"))
        .then(s.str.strptime(pl.Date, "%m/%d/%Y", strict=False))
        .otherwise(s.str.strptime(pl.Date, "%Y-%m-%d", strict=False))
    )

def hhmm_to_hour_start_expr(time_col: str) -> pl.Expr:
    t = pl.col(time_col).cast(pl.Int64, strict=False)
    s = t.cast(pl.Utf8).str.zfill(4)
    hh = s.str.slice(0, 2)
    return hh + ":00:00"

def map_airport_to_station_expr(airport_col: str, out_col: str) -> pl.Expr:
    if not AIRPORT_TO_STATION:
        return normalize_upper(airport_col).alias(out_col)
    return (
        normalize_upper(airport_col)
        .map_dict(AIRPORT_TO_STATION, default=None)
        .fill_null(normalize_upper(airport_col))
        .alias(out_col)
    )

def compute_arrival_date_rollover(bts: pl.LazyFrame) -> pl.LazyFrame:
    dep = pl.col("CRSDepTime").cast(pl.Int64, strict=False)
    arr = pl.col("CRSArrTime").cast(pl.Int64, strict=False)
    return bts.with_columns(
        pl.when(arr.is_not_null() & dep.is_not_null() & (arr < dep))
          .then(pl.col("flight_date") + pl.duration(days=1))
          .otherwise(pl.col("flight_date"))
          .alias("arr_flight_date")
    )

def prefix_weather(lf: pl.LazyFrame, prefix: str) -> pl.LazyFrame:
    schema_cols = lf.collect_schema().names()
    key_cols = {"station", "wx_hour"}
    exprs = [pl.col("station"), pl.col("wx_hour")]
    for c in schema_cols:
        if c in key_cols:
            continue
        exprs.append(pl.col(c).alias(f"{prefix}{c}"))
    return lf.select(exprs)

def weather_severity(prefix: str) -> pl.Expr:
    wind = pl.col(f"{prefix}sknt").cast(pl.Float64, strict=False).fill_null(0.0)
    vis  = pl.col(f"{prefix}vsby").cast(pl.Float64, strict=False).fill_null(10.0).clip(0.0, 10.0)
    p01i = pl.col(f"{prefix}p01i").cast(pl.Float64, strict=False).fill_null(0.0)
    ceil = pl.col(f"{prefix}skyl1").cast(pl.Float64, strict=False).fill_null(99999.0)
    return (
        (wind / 30.0)
        + (1.0 - (vis / 10.0))
        + (p01i.gt(0).cast(pl.Int64))
        + (ceil.lt(1000).cast(pl.Int64))
    )

def to_float(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_lowercase().replace("null", None).cast(pl.Float64, strict=False)

# ----------------------------
# Load (Lazy)
# ----------------------------
bts = lazy_read_bts(bts_path)
wx = lazy_read_weather(weather_path)
reg = lazy_read_registry(registry_path)

# ----------------------------
# Prep BTS
# ----------------------------
bts = bts.with_columns([
    parse_flight_date_expr("FlightDate").alias("flight_date"),
    normalize_upper("Origin").alias("Origin"),
    normalize_upper("Dest").alias("Dest"),
    normalize_upper("Tail_Number").alias("Tail_Number"),
])

bts = compute_arrival_date_rollover(bts)

bts = bts.with_columns([
    (pl.col("flight_date").cast(pl.Utf8) + " " + hhmm_to_hour_start_expr("CRSDepTime"))
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        .alias("dep_hour"),

    (pl.col("arr_flight_date").cast(pl.Utf8) + " " + hhmm_to_hour_start_expr("CRSArrTime"))
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
        .alias("arr_hour"),

    map_airport_to_station_expr("Origin", "dep_station"),
    map_airport_to_station_expr("Dest", "arr_station"),
])

# ----------------------------
# Prep Weather
# - parse valid
# - truncate to hour
# - normalize station
# - convert key numeric fields to float AFTER read
# ----------------------------
wx = (
    wx.with_columns([
        normalize_upper("station").alias("station"),
        pl.col("valid").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).alias("valid_ts"),
    ])
    .with_columns([
        pl.col("valid_ts").dt.truncate("1h").alias("wx_hour"),
    ])
)

# Convert commonly used fields for severity, safely
for c in ["sknt", "vsby", "p01i", "skyl1", "tmpf"]:
    if c in wx.collect_schema().names():
        wx = wx.with_columns(to_float(c).alias(c))

# Prefix for dep/arr joins
wx_dep = prefix_weather(wx, "dep_")
wx_arr = prefix_weather(wx, "arr_")

# ----------------------------
# Join BTS ↔ Weather by hour (CRS-based)
# ----------------------------
joined = (
    bts.join(
        wx_dep,
        left_on=["dep_station", "dep_hour"],
        right_on=["station", "wx_hour"],
        how="left",
    )
    .drop(["dep_station", "station", "wx_hour"], strict=False)
)

joined = (
    joined.join(
        wx_arr,
        left_on=["arr_station", "arr_hour"],
        right_on=["station", "wx_hour"],
        how="left",
    )
    .drop(["arr_station", "station", "wx_hour"], strict=False)
)

# Add severity if inputs exist
cols_now = set(joined.collect_schema().names())
if {"dep_sknt", "dep_vsby", "dep_p01i", "dep_skyl1"}.issubset(cols_now):
    joined = joined.with_columns(weather_severity("dep_").alias("dep_weather_severity"))
if {"arr_sknt", "arr_vsby", "arr_p01i", "arr_skyl1"}.issubset(cols_now):
    joined = joined.with_columns(weather_severity("arr_").alias("arr_weather_severity"))

# ----------------------------
# Registry join (Tail_Number == n_number)
# ----------------------------
reg = reg.with_columns([normalize_upper("n_number").alias("n_number")])

joined = joined.join(
    reg,
    left_on="Tail_Number",
    right_on="n_number",
    how="left",
)

# Fill missing registry values
reg_cols = [c for c in reg.collect_schema().names() if c != "n_number"]
joined_cols = set(joined.collect_schema().names())

if FORCE_REGISTRY_STRINGS:
    joined = joined.with_columns([
        pl.when(pl.col(c).is_null())
          .then(pl.lit("Registry Not Found"))
          .otherwise(pl.col(c).cast(pl.Utf8, strict=False))
          .alias(c)
        for c in reg_cols
        if c in joined_cols
    ])

# ----------------------------
# Materialize + Write
# - Polars 1.25+: use engine="streaming"
# ----------------------------
df = joined.collect(engine="streaming")

df.write_csv(OUT_CSV)
df.write_parquet(OUT_PARQUET)

print(f"✅ Wrote CSV: {OUT_CSV}")
print(f"✅ Wrote Parquet: {OUT_PARQUET}")
print(df.head(5))