from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from config import BTSConfig, JoinConfig, RouteFilterConfig
from utils import (
    download_file_with_backoff,
    extract_first_csv,
    ensure_dir,
    make_retry_session,
)


def get_months_for_year(year: int, months_by_year: dict[int, list[int]] | None) -> list[int]:
    if months_by_year and year in months_by_year:
        return months_by_year[year]
    return list(range(1, 13))


def get_date_expr(cols: set[str]) -> pl.Expr:
    if "FL_DATE" in cols:
        return pl.col("FL_DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    if "FlightDate" in cols:
        return pl.col("FlightDate").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
    raise KeyError("Could not find 'FL_DATE' or 'FlightDate'.")


def find_origin_dest_cols(cols: set[str]) -> tuple[str, str]:
    origin_col = "Origin" if "Origin" in cols else ("ORIGIN" if "ORIGIN" in cols else None)
    dest_col = "Dest" if "Dest" in cols else ("DEST" if "DEST" in cols else None)

    if origin_col is None or dest_col is None:
        raise KeyError("Could not find Origin/Dest columns.")
    return origin_col, dest_col


def build_route_filter(
    origin_col: str,
    dest_col: str,
    route_cfg: RouteFilterConfig,
) -> pl.Expr | None:
    exprs: list[pl.Expr] = []

    if route_cfg.airports:
        airports = route_cfg.airports
        exprs.append(
            pl.col(origin_col).is_in(airports) | pl.col(dest_col).is_in(airports)
        )

    if route_cfg.airport_pairs:
        pair_exprs: list[pl.Expr] = []
        for a, b in route_cfg.airport_pairs:
            pair_exprs.append(
                ((pl.col(origin_col) == a) & (pl.col(dest_col) == b))
                | ((pl.col(origin_col) == b) & (pl.col(dest_col) == a))
            )
        if pair_exprs:
            pair_filter = pair_exprs[0]
            for e in pair_exprs[1:]:
                pair_filter = pair_filter | e
            exprs.append(pair_filter)

    if route_cfg.origin_filter:
        exprs.append(pl.col(origin_col).is_in(route_cfg.origin_filter))

    if route_cfg.dest_filter:
        exprs.append(pl.col(dest_col).is_in(route_cfg.dest_filter))

    if not exprs:
        return None

    final_expr = exprs[0]
    for e in exprs[1:]:
        final_expr = final_expr & e
    return final_expr


def standardize_bts_columns(df: pl.DataFrame, joins: JoinConfig) -> pl.DataFrame:
    cols = set(df.columns)

    renames: dict[str, str] = {}
    if "ORIGIN" in cols and joins.origin_col not in cols:
        renames["ORIGIN"] = joins.origin_col
    if "DEST" in cols and joins.dest_col not in cols:
        renames["DEST"] = joins.dest_col

    if renames:
        df = df.rename(renames)

    return df


def _find_col(df: pl.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _clean_hhmm_expr(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.replace_all(r"\.0$", "")
        .str.zfill(4)
    )


def _hhmm_hour_expr(col_name: str) -> pl.Expr:
    hhmm = _clean_hhmm_expr(col_name)
    return (
        pl.when(pl.col(col_name).is_null())
        .then(None)
        .when(hhmm == "2400")
        .then(0)
        .otherwise(hhmm.str.slice(0, 2).cast(pl.Int64, strict=False))
    )


def _hhmm_minute_expr(col_name: str) -> pl.Expr:
    hhmm = _clean_hhmm_expr(col_name)
    return (
        pl.when(pl.col(col_name).is_null())
        .then(None)
        .when(hhmm == "2400")
        .then(0)
        .otherwise(hhmm.str.slice(2, 2).cast(pl.Int64, strict=False))
    )


def _hhmm_add_day_expr(col_name: str) -> pl.Expr:
    hhmm = _clean_hhmm_expr(col_name)
    return (
        pl.when(pl.col(col_name).is_null())
        .then(0)
        .when(hhmm == "2400")
        .then(1)
        .otherwise(0)
    )


def enrich_bts_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    flight_date_col = _find_col(df, ["FL_DATE", "FlightDate"])
    if flight_date_col is None:
        raise KeyError("Could not find FL_DATE or FlightDate in BTS data.")

    crs_dep_col = _find_col(df, ["CRS_DEP_TIME", "CRSDepTime", "CRSDepTime".lower(), "CRSDepTime".upper(), "CRSDepTime"])
    dep_col = _find_col(df, ["DEP_TIME", "DepTime"])
    crs_arr_col = _find_col(df, ["CRS_ARR_TIME", "CRSArrTime"])
    arr_col = _find_col(df, ["ARR_TIME", "ArrTime"])

    # also support camel-case names commonly present after some transforms
    if crs_dep_col is None and "CRSDepTime" in df.columns:
        crs_dep_col = "CRSDepTime"
    if dep_col is None and "DepTime" in df.columns:
        dep_col = "DepTime"
    if crs_arr_col is None and "CRSArrTime" in df.columns:
        crs_arr_col = "CRSArrTime"
    if arr_col is None and "ArrTime" in df.columns:
        arr_col = "ArrTime"

    missing = [
        name for name, value in {
            "CRS_DEP_TIME/CRSDepTime": crs_dep_col,
            "DEP_TIME/DepTime": dep_col,
            "CRS_ARR_TIME/CRSArrTime": crs_arr_col,
            "ARR_TIME/ArrTime": arr_col,
        }.items()
        if value is None
    ]
    if missing:
        raise KeyError(
            f"Missing one or more BTS time columns: {missing}. "
            f"Available columns include: {df.columns[:60]}"
        )

    if flight_date_col == "FL_DATE":
        date_expr = pl.col(flight_date_col).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    else:
        date_expr = pl.col(flight_date_col).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)

    df = df.with_columns(
        date_expr.alias("_flight_date"),

        _hhmm_hour_expr(crs_dep_col).alias("_crs_dep_hour"),
        _hhmm_minute_expr(crs_dep_col).alias("_crs_dep_min"),
        _hhmm_add_day_expr(crs_dep_col).alias("_crs_dep_add_day"),

        _hhmm_hour_expr(dep_col).alias("_dep_hour"),
        _hhmm_minute_expr(dep_col).alias("_dep_min"),
        _hhmm_add_day_expr(dep_col).alias("_dep_add_day"),

        _hhmm_hour_expr(crs_arr_col).alias("_crs_arr_hour"),
        _hhmm_minute_expr(crs_arr_col).alias("_crs_arr_min"),
        _hhmm_add_day_expr(crs_arr_col).alias("_crs_arr_add_day"),

        _hhmm_hour_expr(arr_col).alias("_arr_hour"),
        _hhmm_minute_expr(arr_col).alias("_arr_min"),
        _hhmm_add_day_expr(arr_col).alias("_arr_add_day"),
    )

    df = df.with_columns(
        pl.when(pl.col("_crs_dep_hour").is_null() | pl.col("_crs_dep_min").is_null())
        .then(None)
        .otherwise(
            pl.datetime(
                pl.col("_flight_date").dt.year(),
                pl.col("_flight_date").dt.month(),
                pl.col("_flight_date").dt.day(),
                pl.col("_crs_dep_hour"),
                pl.col("_crs_dep_min"),
            ) + pl.duration(days=pl.col("_crs_dep_add_day"))
        )
        .alias("dep_ts_sched"),

        pl.when(pl.col("_dep_hour").is_null() | pl.col("_dep_min").is_null())
        .then(None)
        .otherwise(
            pl.datetime(
                pl.col("_flight_date").dt.year(),
                pl.col("_flight_date").dt.month(),
                pl.col("_flight_date").dt.day(),
                pl.col("_dep_hour"),
                pl.col("_dep_min"),
            ) + pl.duration(days=pl.col("_dep_add_day"))
        )
        .alias("dep_ts_actual"),

        pl.when(pl.col("_crs_arr_hour").is_null() | pl.col("_crs_arr_min").is_null())
        .then(None)
        .otherwise(
            pl.datetime(
                pl.col("_flight_date").dt.year(),
                pl.col("_flight_date").dt.month(),
                pl.col("_flight_date").dt.day(),
                pl.col("_crs_arr_hour"),
                pl.col("_crs_arr_min"),
            ) + pl.duration(days=pl.col("_crs_arr_add_day"))
        )
        .alias("_arr_ts_sched_same_day"),

        pl.when(pl.col("_arr_hour").is_null() | pl.col("_arr_min").is_null())
        .then(None)
        .otherwise(
            pl.datetime(
                pl.col("_flight_date").dt.year(),
                pl.col("_flight_date").dt.month(),
                pl.col("_flight_date").dt.day(),
                pl.col("_arr_hour"),
                pl.col("_arr_min"),
            ) + pl.duration(days=pl.col("_arr_add_day"))
        )
        .alias("_arr_ts_actual_same_day"),
    )

    df = df.with_columns(
        pl.when(pl.col("_arr_ts_sched_same_day").is_null() | pl.col("dep_ts_sched").is_null())
        .then(pl.col("_arr_ts_sched_same_day"))
        .when(pl.col("_arr_ts_sched_same_day") < pl.col("dep_ts_sched"))
        .then(pl.col("_arr_ts_sched_same_day") + pl.duration(days=1))
        .otherwise(pl.col("_arr_ts_sched_same_day"))
        .alias("arr_ts_sched"),

        pl.when(pl.col("_arr_ts_actual_same_day").is_null() | pl.col("dep_ts_actual").is_null())
        .then(pl.col("_arr_ts_actual_same_day"))
        .when(pl.col("_arr_ts_actual_same_day") < pl.col("dep_ts_actual"))
        .then(pl.col("_arr_ts_actual_same_day") + pl.duration(days=1))
        .otherwise(pl.col("_arr_ts_actual_same_day"))
        .alias("arr_ts_actual"),
    )

    return df.drop(
        [
            "_flight_date",
            "_crs_dep_hour", "_crs_dep_min", "_crs_dep_add_day",
            "_dep_hour", "_dep_min", "_dep_add_day",
            "_crs_arr_hour", "_crs_arr_min", "_crs_arr_add_day",
            "_arr_hour", "_arr_min", "_arr_add_day",
            "_arr_ts_sched_same_day",
            "_arr_ts_actual_same_day",
        ]
    )


def process_bts_month(
    year: int,
    month: int,
    bts_cfg: BTSConfig,
    route_cfg: RouteFilterConfig,
    joins: JoinConfig,
) -> tuple[pl.DataFrame, tuple[Path, Path]]:
    ensure_dir(bts_cfg.out_dir)

    zip_name = bts_cfg.zip_name_template.format(year=year, month=month)
    zip_url = f"{bts_cfg.prezip_base}/{zip_name}"

    zip_path = bts_cfg.out_dir / zip_name
    extract_dir = bts_cfg.out_dir / f"extracted_{year}_{month:02d}"

    session = make_retry_session(
        max_retries=bts_cfg.max_retries,
        user_agent="OR568-BTS-Pipeline/1.0 (polite monthly downloader)",
    )
    try:
        download_file_with_backoff(
            session=session,
            url=zip_url,
            out_path=zip_path,
            timeout=bts_cfg.timeout,
            verify_ssl=bts_cfg.verify_ssl,
            max_retries=bts_cfg.max_retries,
            backoff_base_seconds=bts_cfg.backoff_base_seconds,
        )
    finally:
        session.close()

    csv_path = extract_first_csv(zip_path, extract_dir)
    print(f"Using BTS CSV -> {csv_path}")

    lf = pl.scan_csv(csv_path, infer_schema_length=2000)
    cols = set(lf.collect_schema().names())

    origin_col, dest_col = find_origin_dest_cols(cols)
    date_expr = get_date_expr(cols)
    route_filter = build_route_filter(origin_col, dest_col, route_cfg)

    raw_count = lf.select(pl.len()).collect().item()
    print(f"Raw BTS rows before route filter: {raw_count:,}")

    lf = lf.with_columns(date_expr.alias("_flight_date"))
    if route_filter is not None:
        lf = lf.filter(route_filter)

    filtered_count = lf.select(pl.len()).collect().item()
    print(f"BTS rows after route filter: {filtered_count:,}")

    df = lf.drop("_flight_date").collect(streaming=True)
    df = standardize_bts_columns(df, joins)
    df = enrich_bts_timestamps(df)

    for c in ["dep_ts_sched", "dep_ts_actual", "arr_ts_sched", "arr_ts_actual"]:
        if c in df.columns:
            non_null = df.select(pl.col(c).is_not_null().sum()).item()
            print(f"Non-null {c}: {non_null:,}")

    time.sleep(bts_cfg.chunk_pause_seconds)

    return df, (zip_path, extract_dir)


def write_bts_parquet(df: pl.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.write_parquet(out_path)
    print(f"Wrote BTS -> {out_path}")