from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from data_pipeline.archive.version_2.config import BTSConfig, JoinConfig, RouteFilterConfig
from data_pipeline.archive.version_2.utils import (
    ensure_dir,
    make_retry_session,
    download_file_with_backoff,
    extract_first_csv,
)


def get_months_for_year(
    year: int,
    months_by_year: dict[int, list[int]] | None,
) -> list[int]:
    if months_by_year is None:
        return list(range(1, 13))
    return months_by_year.get(year, list(range(1, 13)))


def build_bts_zip_name(year: int, month: int, cfg: BTSConfig) -> str:
    return cfg.zip_name_template.format(year=year, month=month)


def build_bts_url(year: int, month: int, cfg: BTSConfig) -> str:
    zip_name = build_bts_zip_name(year, month, cfg)
    return f"{cfg.prezip_base}/{zip_name}"


def apply_route_filter(df: pl.DataFrame, route_cfg: RouteFilterConfig) -> pl.DataFrame:
    """
    For network analysis:
    - airports => keep flights where Origin OR Dest is in the airport list
    - airport_pairs => keep only explicit directed pairs
    - origin_filter / dest_filter => additional restrictions
    """
    if route_cfg.airports:
        airport_set = set(route_cfg.airports)
        df = df.filter(
            pl.col("Origin").is_in(airport_set) | pl.col("Dest").is_in(airport_set)
        )

    if route_cfg.airport_pairs:
        pair_expr = None
        for origin, dest in route_cfg.airport_pairs:
            expr = (pl.col("Origin") == origin) & (pl.col("Dest") == dest)
            pair_expr = expr if pair_expr is None else (pair_expr | expr)
        if pair_expr is not None:
            df = df.filter(pair_expr)

    if route_cfg.origin_filter:
        df = df.filter(pl.col("Origin").is_in(route_cfg.origin_filter))

    if route_cfg.dest_filter:
        df = df.filter(pl.col("Dest").is_in(route_cfg.dest_filter))

    return df


def _format_hhmm(col_name: str) -> pl.Expr:
    """
    Convert BTS HHMM-style columns into zero-padded 4-digit strings.
    Examples:
      5 -> '0005'
      45 -> '0045'
      930 -> '0930'
      1427 -> '1427'
    """
    return (
        pl.col(col_name)
        .cast(pl.Int64, strict=False)
        .cast(pl.Utf8)
        .str.zfill(4)
    )


def add_bts_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build scheduled and actual departure/arrival timestamps from BTS columns.

    Expected columns:
    - FlightDate
    - CRSDepTime
    - DepTime
    - CRSArrTime
    - ArrTime

    Handles:
    - numeric HHMM fields
    - zero-padding
    - null values
    - overnight arrival rollover
    """
    required = ["FlightDate", "CRSDepTime", "DepTime", "CRSArrTime", "ArrTime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required BTS timestamp columns: {missing}. "
            f"Available columns include: {df.columns[:80]}"
        )

    print("\nColumns available for timestamp build:")
    print(df.columns)

    # Parse date and create zero-padded HHMM strings
    df = df.with_columns([
        pl.col("FlightDate")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
        .alias("flight_date"),

        _format_hhmm("CRSDepTime").alias("crs_dep_hhmm"),
        _format_hhmm("DepTime").alias("dep_hhmm"),
        _format_hhmm("CRSArrTime").alias("crs_arr_hhmm"),
        _format_hhmm("ArrTime").alias("arr_hhmm"),
    ])

    # Build base timestamps
    df = df.with_columns([
        (
            pl.col("flight_date").cast(pl.Utf8) + " " + pl.col("crs_dep_hhmm")
        ).str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False)
        .alias("dep_ts_sched"),

        (
            pl.col("flight_date").cast(pl.Utf8) + " " + pl.col("dep_hhmm")
        ).str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False)
        .alias("dep_ts_actual"),

        (
            pl.col("flight_date").cast(pl.Utf8) + " " + pl.col("crs_arr_hhmm")
        ).str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False)
        .alias("arr_ts_sched_raw"),

        (
            pl.col("flight_date").cast(pl.Utf8) + " " + pl.col("arr_hhmm")
        ).str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False)
        .alias("arr_ts_actual_raw"),
    ])

    # If arrival clock time is earlier than departure clock time, assume next-day arrival
    df = df.with_columns([
        pl.when(
            pl.col("arr_ts_sched_raw").is_not_null()
            & pl.col("dep_ts_sched").is_not_null()
            & (pl.col("arr_ts_sched_raw") < pl.col("dep_ts_sched"))
        )
        .then(pl.col("arr_ts_sched_raw") + pl.duration(days=1))
        .otherwise(pl.col("arr_ts_sched_raw"))
        .alias("arr_ts_sched"),

        pl.when(
            pl.col("arr_ts_actual_raw").is_not_null()
            & pl.col("dep_ts_actual").is_not_null()
            & (pl.col("arr_ts_actual_raw") < pl.col("dep_ts_actual"))
        )
        .then(pl.col("arr_ts_actual_raw") + pl.duration(days=1))
        .otherwise(pl.col("arr_ts_actual_raw"))
        .alias("arr_ts_actual"),
    ])

    # Add convenience date columns used downstream
    df = df.with_columns([
        pl.col("arr_ts_sched").dt.date().alias("crs_arr_date"),
        pl.col("arr_ts_actual").dt.date().alias("act_arr_date"),
    ])

    # Validation prints
    for c in ["dep_ts_sched", "dep_ts_actual", "arr_ts_sched", "arr_ts_actual"]:
        non_null = df.select(pl.col(c).is_not_null().sum()).item()
        print(f"Non-null {c}: {non_null:,}")
        if non_null == 0:
            raise ValueError(
                f"{c} failed to build and has 0 non-null values. "
                "Check BTS timestamp construction."
            )

    # Drop temp columns
    df = df.drop([
        "flight_date",
        "crs_dep_hhmm",
        "dep_hhmm",
        "crs_arr_hhmm",
        "arr_hhmm",
        "arr_ts_sched_raw",
        "arr_ts_actual_raw",
    ])

    return df


def process_bts_month(
    year: int,
    month: int,
    bts_cfg: BTSConfig,
    route_cfg: RouteFilterConfig,
    joins: JoinConfig,
) -> tuple[pl.DataFrame, tuple[Path, Path]]:
    """
    Download one BTS month, extract CSV, apply route filter, build timestamps,
    and return the processed DataFrame plus download record:
      (zip_path, extract_dir)
    """
    ensure_dir(bts_cfg.out_dir)

    zip_name = build_bts_zip_name(year, month, bts_cfg)
    zip_path = bts_cfg.out_dir / zip_name
    extract_dir = bts_cfg.out_dir / f"extracted_{year}_{month:02d}"

    url = build_bts_url(year, month, bts_cfg)

    session = make_retry_session(max_retries=bts_cfg.max_retries)

    download_file_with_backoff(
        session=session,
        url=url,
        out_path=zip_path,
        timeout=bts_cfg.timeout,
        verify_ssl=bts_cfg.verify_ssl,
        max_retries=bts_cfg.max_retries,
        backoff_base_seconds=bts_cfg.backoff_base_seconds,
    )

    csv_path = extract_first_csv(zip_path, extract_dir)
    print(f"Using BTS CSV -> {csv_path}")

    # Read CSV
    df = pl.read_csv(
        csv_path,
        infer_schema_length=10000,
        ignore_errors=False,
        null_values=["", "NA", "NULL", "null"],
        try_parse_dates=False,
    )

    print(f"Raw BTS rows before route filter: {df.height:,}")

    # Apply route/network filter
    df = apply_route_filter(df, route_cfg)
    print(f"BTS rows after route filter: {df.height:,}")

    if df.height == 0:
        # Return early if nothing remains
        return df, (zip_path, extract_dir)

    # Build timestamps
    df = add_bts_timestamps(df)

    print("Timestamp preview:")
    preview_cols = [
        c for c in [
            "FlightDate",
            "Origin",
            "Dest",
            "CRSDepTime",
            "DepTime",
            "CRSArrTime",
            "ArrTime",
            "dep_ts_sched",
            "dep_ts_actual",
            "arr_ts_sched",
            "arr_ts_actual",
        ]
        if c in df.columns
    ]
    print(df.select(preview_cols).head(10))

    # Optional polite pause between monthly pulls
    if bts_cfg.chunk_pause_seconds > 0:
        time.sleep(bts_cfg.chunk_pause_seconds)

    return df, (zip_path, extract_dir)


def write_bts_parquet(df: pl.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.write_parquet(out_path)
    print(f"Wrote BTS -> {out_path}")