from __future__ import annotations

import math
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import polars as pl

from config import WeatherConfig
from utils import ensure_dir, make_retry_session


def build_year_date_window(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year}-12-31"


def _month_windows(year: int) -> list[tuple[str, str, str]]:
    windows: list[tuple[str, str, str]] = []
    for month in range(1, 13):
        start = date(year, month, 1)
        if month == 12:
            end = date(year, 12, 31)
        else:
            next_month = date(year, month + 1, 1)
            end = date.fromordinal(next_month.toordinal() - 1)
        label = f"{year}_{month:02d}"
        windows.append((start.isoformat(), end.isoformat(), label))
    return windows


def _chunk_list(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[i:i + chunk_size] for i in range(0, len(values), chunk_size)]


def _build_weather_url(
    stations: list[str],
    start: str,
    end: str,
    cfg: WeatherConfig,
) -> str:
    params = {
        "dataset": cfg.dataset,
        "stations": ",".join(stations),
        "startDate": start,
        "endDate": end,
        "format": "json",
        "includeAttributes": "false",
        "includeStationName": "true",
        "units": cfg.units,
        "dataTypes": "TMP,WND,CIG",
    }
    return f"{cfg.base_url}?{urlencode(params)}"


def _fetch_weather_json_chunk(
    stations: list[str],
    start: str,
    end: str,
    cfg: WeatherConfig,
) -> list[dict]:
    url = _build_weather_url(stations, start, end, cfg)
    session = make_retry_session(max_retries=cfg.max_retries, user_agent="OR568-Weather/1.0")

    print(
        f"Fetching weather chunk: {len(stations)} stations, "
        f"{start} -> {end}"
    )

    r = session.get(url, timeout=cfg.timeout, verify=cfg.verify_ssl)
    if not r.ok:
        print(f"Weather request failed: status={r.status_code}")
        print(f"URL length: {len(url)}")
        print(f"First 300 chars of URL: {url[:300]}")
        print(f"Response text preview: {r.text[:1000]}")
        r.raise_for_status()

    payload = r.json()

    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]

    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unexpected weather payload type: {type(payload)}")


def _fetch_weather_json(
    stations: list[str],
    start: str,
    end: str,
    cfg: WeatherConfig,
) -> list[dict]:
    """
    Fetch weather in station batches to avoid NOAA request-size failures.
    """
    stations = sorted(set(stations))
    if not stations:
        return []

    station_chunk_size = getattr(cfg, "station_chunk_size", 25)
    station_chunks = _chunk_list(stations, station_chunk_size)

    print(
        f"Fetching weather across {len(stations):,} stations in "
        f"{len(station_chunks)} chunk(s) "
        f"(chunk size={station_chunk_size})"
    )

    all_records: list[dict] = []

    for i, station_chunk in enumerate(station_chunks, start=1):
        print(f"  Station chunk {i}/{len(station_chunks)}")
        records = _fetch_weather_json_chunk(
            stations=station_chunk,
            start=start,
            end=end,
            cfg=cfg,
        )
        all_records.extend(records)

        if cfg.chunk_pause_seconds > 0:
            time.sleep(cfg.chunk_pause_seconds)

    return all_records


def _parse_tmp(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8)
        .str.split(",")
        .list.get(0)
        .cast(pl.Int64, strict=False)
        .truediv(10.0)
    )


def _parse_wnd_dir(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8)
        .str.split(",")
        .list.get(0)
        .cast(pl.Int64, strict=False)
    )


def _parse_wnd_speed(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8)
        .str.split(",")
        .list.get(3)
        .cast(pl.Int64, strict=False)
        .truediv(10.0)
    )


def _parse_cig(expr: pl.Expr) -> pl.Expr:
    return (
        expr.cast(pl.Utf8)
        .str.split(",")
        .list.get(0)
        .cast(pl.Int64, strict=False)
    )


def _normalize_weather_df(df: pl.DataFrame) -> pl.DataFrame:
    rename_map = {}
    for c in df.columns:
        uc = c.upper()
        if uc == "STATION":
            rename_map[c] = "station"
        elif uc == "DATE":
            rename_map[c] = "date"
        elif uc == "TMP":
            rename_map[c] = "TMP"
        elif uc == "WND":
            rename_map[c] = "WND"
        elif uc == "CIG":
            rename_map[c] = "CIG"
        elif uc == "NAME":
            rename_map[c] = "station_name"
    if rename_map:
        df = df.rename(rename_map)

    required = {"station", "date"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required weather columns: {missing}. Available: {df.columns}")

    df = df.with_columns([
        pl.col("station").cast(pl.Utf8).str.strip_chars(),
        pl.col("date").cast(pl.Utf8).str.strptime(pl.Datetime, strict=False).alias("valid_ts"),
    ])

    if "TMP" in df.columns:
        df = df.with_columns(_parse_tmp(pl.col("TMP")).alias("temp_c"))
    if "WND" in df.columns:
        df = df.with_columns([
            _parse_wnd_dir(pl.col("WND")).alias("wind_dir_deg"),
            _parse_wnd_speed(pl.col("WND")).alias("wind_speed_m_s"),
        ])
    if "CIG" in df.columns:
        df = df.with_columns(_parse_cig(pl.col("CIG")).alias("ceiling_height_m"))

    keep_cols = [
        "station",
        "valid_ts",
        "temp_c",
        "wind_speed_m_s",
        "wind_dir_deg",
        "ceiling_height_m",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return (
        df.select(keep_cols)
        .drop_nulls(subset=["station", "valid_ts"])
        .sort(["station", "valid_ts"])
        .unique(subset=["station", "valid_ts"], keep="last")
    )


def pull_weather_for_period(
    stations: list[str],
    start: str,
    end: str,
    cfg: WeatherConfig,
    raw_output_path: Path | None = None,
    clean_output_path: Path | None = None,
) -> pl.DataFrame:
    records = _fetch_weather_json(stations, start, end, cfg)

    if not records:
        print(f"No weather records returned for {start} -> {end}")
        empty_df = pl.DataFrame(
            schema={
                "station": pl.Utf8,
                "valid_ts": pl.Datetime,
                "temp_c": pl.Float64,
                "wind_speed_m_s": pl.Float64,
                "wind_dir_deg": pl.Float64,
                "ceiling_height_m": pl.Float64,
            }
        )
        if clean_output_path is not None:
            ensure_dir(clean_output_path.parent)
            empty_df.write_parquet(clean_output_path)
        return empty_df

    raw_df = pl.DataFrame(records)

    if raw_output_path is not None:
        ensure_dir(raw_output_path.parent)
        raw_df.write_parquet(raw_output_path)
        print(f"Wrote raw weather -> {raw_output_path}")

    clean_df = _normalize_weather_df(raw_df)

    if clean_output_path is not None:
        ensure_dir(clean_output_path.parent)
        clean_df.write_parquet(clean_output_path)
        print(f"Wrote clean weather -> {clean_output_path}")

    return clean_df


def pull_weather_for_year_chunked(
    stations: list[str],
    year: int,
    cfg: WeatherConfig,
    raw_output_path: Path | None = None,
    clean_output_path: Path | None = None,
) -> pl.DataFrame:
    all_months: list[pl.DataFrame] = []

    for start, end, label in _month_windows(year):
        monthly_clean_path = cfg.out_dir / f"weather_clean_{label}.parquet"
        monthly_raw_path = cfg.out_dir / f"weather_raw_{label}.parquet" if cfg.raw_parquet else None

        if cfg.use_monthly_cache and monthly_clean_path.exists():
            print(f"Loading cached monthly weather -> {monthly_clean_path}")
            month_df = pl.read_parquet(monthly_clean_path)
        else:
            month_df = pull_weather_for_period(
                stations=stations,
                start=start,
                end=end,
                cfg=cfg,
                raw_output_path=monthly_raw_path,
                clean_output_path=monthly_clean_path,
            )

        all_months.append(month_df)

        if cfg.chunk_pause_seconds > 0:
            time.sleep(cfg.chunk_pause_seconds)

    weather_df = (
        pl.concat(all_months, how="vertical_relaxed")
        .sort(["station", "valid_ts"])
        .unique(subset=["station", "valid_ts"], keep="last")
    )

    if clean_output_path is not None:
        ensure_dir(clean_output_path.parent)
        weather_df.write_parquet(clean_output_path)
        print(f"Wrote yearly clean weather -> {clean_output_path}")

    return weather_df