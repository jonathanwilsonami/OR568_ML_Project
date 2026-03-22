from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from config import BTSConfig, JoinConfig, RouteFilterConfig
from utils import (
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
    return (
        pl.col(col_name)
        .cast(pl.Int64, strict=False)
        .cast(pl.Utf8)
        .str.zfill(4)
    )


def add_bts_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    required = ["FlightDate", "CRSDepTime", "DepTime", "CRSArrTime", "ArrTime"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required BTS timestamp columns: {missing}. "
            f"Available columns include: {df.columns[:80]}"
        )

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

    df = df.with_columns([
        (
            pl.col("flight_date").cast(pl.Utf8) + " " + pl.col("crs_dep_hhmm")
        ).str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False).alias("dep_ts_sched"),

        (
            pl.col("flight_date").cast(pl.Utf8) + " " + pl.col("dep_hhmm")
        ).str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False).alias("dep_ts_actual"),

        (
            pl.col("flight_date").cast(pl.Utf8) + " " + pl.col("crs_arr_hhmm")
        ).str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False).alias("arr_ts_sched_raw"),

        (
            pl.col("flight_date").cast(pl.Utf8) + " " + pl.col("arr_hhmm")
        ).str.strptime(pl.Datetime, format="%Y-%m-%d %H%M", strict=False).alias("arr_ts_actual_raw"),
    ])

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

    df = df.with_columns([
        pl.col("arr_ts_sched").dt.date().alias("crs_arr_date"),
        pl.col("arr_ts_actual").dt.date().alias("act_arr_date"),
    ])

    for c in ["dep_ts_sched", "dep_ts_actual", "arr_ts_sched", "arr_ts_actual"]:
        non_null = df.select(pl.col(c).is_not_null().sum()).item()
        print(f"Non-null {c}: {non_null:,}")
        if non_null == 0:
            raise ValueError(
                f"{c} failed to build and has 0 non-null values. "
                "Check BTS timestamp construction."
            )

    return df.drop([
        "flight_date",
        "crs_dep_hhmm",
        "dep_hhmm",
        "crs_arr_hhmm",
        "arr_hhmm",
        "arr_ts_sched_raw",
        "arr_ts_actual_raw",
    ])


def process_bts_month(
    year: int,
    month: int,
    bts_cfg: BTSConfig,
    route_cfg: RouteFilterConfig,
    joins: JoinConfig,
) -> tuple[pl.DataFrame, tuple[Path, Path]]:
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

    df = pl.read_csv(
        csv_path,
        infer_schema_length=10000,
        ignore_errors=False,
        null_values=["", "NA", "NULL", "null"],
        try_parse_dates=False,
    )

    df = df.select([c for c in df.columns if c != ""])

    print(f"Raw BTS rows before route filter: {df.height:,}")

    df = apply_route_filter(df, route_cfg)
    print(f"BTS rows after route filter: {df.height:,}")

    if df.height == 0:
        return df, (zip_path, extract_dir)

    df = add_bts_timestamps(df)

    if bts_cfg.chunk_pause_seconds > 0:
        time.sleep(bts_cfg.chunk_pause_seconds)

    return df, (zip_path, extract_dir)


def write_bts_parquet(df: pl.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.write_parquet(out_path)
    print(f"Wrote BTS -> {out_path}")