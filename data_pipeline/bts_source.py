from __future__ import annotations

import time
from pathlib import Path
from zoneinfo import ZoneInfo

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
    """
    Safe version: works even if RouteFilterConfig is missing fields
    """

    core_airports = set(getattr(route_cfg, "core_airports", []) or [])
    two_hop = getattr(route_cfg, "two_hop_inbound_to_core", False)

    if core_airports and two_hop:
        core_set = core_airports

        hop1 = (
            df
            .filter(~pl.col("Origin").is_in(core_set) & pl.col("Dest").is_in(core_set))
            .select(pl.col("Origin").alias("airport"))
            .drop_nulls()
            .unique()
        )
        hop1_set = set(hop1["airport"].to_list())

        expr = (
            (pl.col("Origin").is_in(core_set) & pl.col("Dest").is_in(core_set))
            | (~pl.col("Origin").is_in(core_set) & pl.col("Dest").is_in(core_set))
            | (~pl.col("Origin").is_in(core_set) & pl.col("Dest").is_in(hop1_set))
        )

        df = df.filter(expr)

    # --- Safe access for optional fields ---
    if getattr(route_cfg, "airports", None):
        airport_set = set(route_cfg.airports)
        df = df.filter(
            pl.col("Origin").is_in(airport_set) | pl.col("Dest").is_in(airport_set)
        )

    if getattr(route_cfg, "airport_pairs", None):
        pair_expr = None
        for origin, dest in route_cfg.airport_pairs:
            expr = (pl.col("Origin") == origin) & (pl.col("Dest") == dest)
            pair_expr = expr if pair_expr is None else (pair_expr | expr)
        if pair_expr is not None:
            df = df.filter(pair_expr)

    if getattr(route_cfg, "origin_filter", None):
        df = df.filter(pl.col("Origin").is_in(route_cfg.origin_filter))

    if getattr(route_cfg, "dest_filter", None):
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

    return df.drop([
        "flight_date",
        "crs_dep_hhmm",
        "dep_hhmm",
        "crs_arr_hhmm",
        "arr_hhmm",
        "arr_ts_sched_raw",
        "arr_ts_actual_raw",
    ])


def add_timezone_columns(
    df: pl.DataFrame,
    airport_to_timezone: dict[str, str],
    joins: JoinConfig,
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(joins.origin_col)
        .replace(airport_to_timezone, default=None)
        .alias(joins.dep_timezone_col),

        pl.col(joins.dest_col)
        .replace(airport_to_timezone, default=None)
        .alias(joins.arr_timezone_col),
    )


def _convert_local_series_to_utc(
    ts_series: pl.Series,
    tz_series: pl.Series,
) -> pl.Series:
    out = []

    for ts_val, tz_val in zip(ts_series.to_list(), tz_series.to_list()):
        if ts_val is None or tz_val is None:
            out.append(None)
            continue

        try:
            aware = ts_val.replace(tzinfo=ZoneInfo(tz_val))
            utc_val = aware.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            out.append(utc_val)
        except Exception:
            out.append(None)

    return pl.Series(out)


def add_utc_timestamps(
    df: pl.DataFrame,
    joins: JoinConfig,
) -> pl.DataFrame:
    dep_sched_utc = _convert_local_series_to_utc(
        df["dep_ts_sched"], df[joins.dep_timezone_col]
    )
    dep_actual_utc = _convert_local_series_to_utc(
        df["dep_ts_actual"], df[joins.dep_timezone_col]
    )
    arr_sched_utc = _convert_local_series_to_utc(
        df["arr_ts_sched"], df[joins.arr_timezone_col]
    )
    arr_actual_utc = _convert_local_series_to_utc(
        df["arr_ts_actual"], df[joins.arr_timezone_col]
    )

    return df.with_columns([
        dep_sched_utc.alias("dep_ts_sched_utc"),
        dep_actual_utc.alias("dep_ts_actual_utc"),
        arr_sched_utc.alias("arr_ts_sched_utc"),
        arr_actual_utc.alias("arr_ts_actual_utc"),
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