"""
Download + clean NOAA NCEI "global-hourly" data for one or more stations,
with configurable date range, and write Parquet (and optional raw Parquet).

- Uses Polars for parsing/cleaning
- Uses Parquet as the primary output
- Mirrors your R cleaning logic for TMP / WND / CIG

Example:
  python ncei_global_hourly_pull_clean.py \
    --stations 72406093721 \
    --start 2022-06-27T00:00:00Z \
    --end   2022-06-27T23:59:59Z \
    --outdir ncei_out \
    --prefix bwi_june27

Multiple stations:
  python ncei_global_hourly_pull_clean.py \
    --stations 72406093721 72295023174 \
    --start 2019-01-01T00:00:00Z \
    --end   2019-12-31T23:59:59Z \
    --outdir ncei_out \
    --prefix two_stations_2019 \
    --raw-parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import requests
import polars as pl

BASE_URL = "https://www.ncei.noaa.gov/access/services/data/v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull + clean NCEI global-hourly data to Parquet (Polars).")
    p.add_argument(
        "--stations",
        nargs="+",
        required=True,
        help="One or more NCEI station IDs (e.g., 72406093721). Space-separated.",
    )
    p.add_argument(
        "--start",
        required=True,
        help="Start datetime in ISO-8601 (e.g., 2022-06-27T00:00:00Z).",
    )
    p.add_argument(
        "--end",
        required=True,
        help="End datetime in ISO-8601 (e.g., 2022-06-27T23:59:59Z).",
    )
    p.add_argument(
        "--units",
        default="metric",
        choices=["metric", "standard"],
        help="NCEI units parameter. 'metric' recommended (matches your R).",
    )
    p.add_argument(
        "--include-attributes",
        action="store_true",
        help="Include data quality flags/attributes (includeAttributes=true).",
    )
    p.add_argument(
        "--include-station-name",
        action="store_true",
        help="Include station name (includeStationName=true).",
    )
    p.add_argument(
        "--outdir",
        default="ncei_downloads",
        help="Output directory.",
    )
    p.add_argument(
        "--prefix",
        default="ncei",
        help="Output filename prefix.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds.",
    )
    p.add_argument(
        "--raw-parquet",
        action="store_true",
        help="Also write the raw (un-cleaned) dataset to Parquet.",
    )
    return p.parse_args()


def fetch_ncei_csv_text(
    stations: list[str],
    start: str,
    end: str,
    units: str,
    include_attributes: bool,
    include_station_name: bool,
    timeout: int,
) -> str:
    params = {
        "dataset": "global-hourly",
        "stations": ",".join(stations),
        "startDate": start,
        "endDate": end,
        "format": "csv",
        "units": units,
        "includeAttributes": "true" if include_attributes else "false",
        "includeStationName": "true" if include_station_name else "false",
    }

    r = requests.get(BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.text


def clean_global_hourly(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create ML-ready features mirroring your R pipeline:
      - TMP -> temp_c (missing +9999)
      - WND -> wind_speed_m_s (missing 9999; /10), wind_dir_deg (missing 999)
      - CIG -> ceiling_height_m (missing 99999)

    Keeps DATE plus derived numeric features.
    """
    cols = set(df.columns)

    # The global-hourly CSV typically has "DATE" plus these code columns:
    needed = {"DATE", "TMP", "WND", "CIG"}
    missing = sorted(needed - cols)
    if missing:
        raise KeyError(f"Missing expected columns {missing}. Available columns: {sorted(cols)[:50]}...")

    # Split fields by comma. Some rows can have missing/short strings; use safe splits.
    # TMP: "temp_raw,temp_qual" e.g. "+0123,1"
    tmp_parts = pl.col("TMP").cast(pl.Utf8).str.split_exact(",", 1)
    # WND: "dir,dir_q,type,speed,speed_q" e.g. "180,1,N,0010,1"
    wnd_parts = pl.col("WND").cast(pl.Utf8).str.split_exact(",", 4)
    # CIG: "height,qual,det,cavok" e.g. "00500,1,9,N"
    cig_parts = pl.col("CIG").cast(pl.Utf8).str.split_exact(",", 3)

    df2 = (
        df.with_columns(
            # TMP
            tmp_parts.struct.field("field_0").alias("temp_raw"),
            tmp_parts.struct.field("field_1").alias("temp_qual"),
            # WND
            wnd_parts.struct.field("field_0").alias("wind_dir_raw"),
            wnd_parts.struct.field("field_1").alias("wind_dir_qual"),
            wnd_parts.struct.field("field_2").alias("wind_type"),
            wnd_parts.struct.field("field_3").alias("wind_speed_raw"),
            wnd_parts.struct.field("field_4").alias("wind_speed_qual"),
            # CIG
            cig_parts.struct.field("field_0").alias("ceiling_raw"),
            cig_parts.struct.field("field_1").alias("ceiling_qual"),
            cig_parts.struct.field("field_2").alias("ceiling_det"),
            cig_parts.struct.field("field_3").alias("ceiling_cavok"),
        )
        .with_columns(
            # temp_c: missing "+9999" else numeric/10
            pl.when(pl.col("temp_raw") == "+9999")
            .then(None)
            .otherwise(pl.col("temp_raw").cast(pl.Float64, strict=False) / 10.0)
            .alias("temp_c"),
            # wind_speed_m_s: missing "9999" else numeric/10
            pl.when(pl.col("wind_speed_raw") == "9999")
            .then(None)
            .otherwise(pl.col("wind_speed_raw").cast(pl.Float64, strict=False) / 10.0)
            .alias("wind_speed_m_s"),
            # wind_dir_deg: missing "999" else numeric
            pl.when(pl.col("wind_dir_raw") == "999")
            .then(None)
            .otherwise(pl.col("wind_dir_raw").cast(pl.Int64, strict=False))
            .alias("wind_dir_deg"),
            # ceiling_height_m: missing "99999" else numeric
            pl.when(pl.col("ceiling_raw") == "99999")
            .then(None)
            .otherwise(pl.col("ceiling_raw").cast(pl.Int64, strict=False))
            .alias("ceiling_height_m"),
        )
        .select(
            [
                pl.col("DATE"),
                pl.col("temp_c"),
                pl.col("wind_speed_m_s"),
                pl.col("wind_dir_deg"),
                pl.col("ceiling_height_m"),
            ]
        )
    )

    return df2


def main() -> int:
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Fetch CSV payload
    print(f"Fetching NCEI global-hourly for stations={args.stations} start={args.start} end={args.end}")
    csv_text = fetch_ncei_csv_text(
        stations=args.stations,
        start=args.start,
        end=args.end,
        units=args.units,
        include_attributes=args.include_attributes,
        include_station_name=args.include_station_name,
        timeout=args.timeout,
    )

    # 2) Read into Polars
    # Using read_csv from in-memory text
    df_raw = pl.read_csv(
        source=csv_text.encode("utf-8"),
        infer_schema_length=2000,
        ignore_errors=True,  # in case a few rows have odd quoting/fields
    )

    print(f"Raw rows: {df_raw.height:,}  | Raw cols: {len(df_raw.columns)}")

    # Optional: write raw parquet
    if args.raw_parquet:
        raw_path = outdir / f"{args.prefix}_raw.parquet"
        df_raw.write_parquet(raw_path)
        print(f"Wrote raw parquet -> {raw_path}")

    # 3) Clean features
    df_clean = clean_global_hourly(df_raw)
    print(f"Clean rows: {df_clean.height:,}  | Clean cols: {len(df_clean.columns)}")
    print(df_clean.head(5))

    # 4) Write cleaned parquet
    clean_path = outdir / f"{args.prefix}_clean.parquet"
    df_clean.write_parquet(clean_path)
    print(f"Wrote cleaned parquet -> {clean_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise