from __future__ import annotations

from calendar import monthrange
from pathlib import Path
import random
import time

import requests
import polars as pl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import WeatherConfig
from utils import ensure_dir


def make_session(cfg: WeatherConfig) -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=cfg.max_retries,
        read=cfg.max_retries,
        connect=cfg.max_retries,
        backoff_factor=0.0,  # we handle backoff ourselves
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "OR568-Weather-Pipeline/1.0 (polite monthly chunk downloader)",
            "Accept": "text/csv, text/plain, */*",
            "Connection": "keep-alive",
        }
    )
    return session


def fetch_ncei_csv_text_once(
    session: requests.Session,
    stations: list[str],
    start: str,
    end: str,
    cfg: WeatherConfig,
) -> str:
    params = {
        "dataset": "global-hourly",
        "stations": ",".join(stations),
        "startDate": start,
        "endDate": end,
        "format": "csv",
        "units": cfg.units,
        "includeAttributes": "true" if cfg.include_attributes else "false",
        "includeStationName": "true" if cfg.include_station_name else "false",
    }

    r = session.get(
        cfg.base_url,
        params=params,
        timeout=cfg.timeout,
        verify=cfg.verify_ssl,
    )
    r.raise_for_status()
    return r.text


def fetch_ncei_csv_text_with_backoff(
    session: requests.Session,
    stations: list[str],
    start: str,
    end: str,
    cfg: WeatherConfig,
) -> str:
    last_exc: Exception | None = None

    for attempt in range(1, cfg.max_retries + 1):
        try:
            return fetch_ncei_csv_text_once(
                session=session,
                stations=stations,
                start=start,
                end=end,
                cfg=cfg,
            )
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt == cfg.max_retries:
                break

            sleep_s = cfg.backoff_base_seconds * (2 ** (attempt - 1))
            sleep_s += random.uniform(0.0, 0.75)

            print(
                f"Weather request failed (attempt {attempt}/{cfg.max_retries}) "
                f"for stations={stations}, start={start}, end={end}. "
                f"Retrying in {sleep_s:.1f}s. Error: {exc}"
            )
            time.sleep(sleep_s)

    assert last_exc is not None
    raise last_exc


def clean_global_hourly(df: pl.DataFrame) -> pl.DataFrame:
    cols = set(df.columns)
    needed = {"DATE", "TMP", "WND", "CIG", "STATION"}
    missing = sorted(needed - cols)
    if missing:
        raise KeyError(f"Missing weather columns: {missing}")

    tmp_parts = pl.col("TMP").cast(pl.Utf8).str.split_exact(",", 1)
    wnd_parts = pl.col("WND").cast(pl.Utf8).str.split_exact(",", 4)
    cig_parts = pl.col("CIG").cast(pl.Utf8).str.split_exact(",", 3)

    return (
        df.with_columns(
            tmp_parts.struct.field("field_0").alias("temp_raw"),
            wnd_parts.struct.field("field_0").alias("wind_dir_raw"),
            wnd_parts.struct.field("field_3").alias("wind_speed_raw"),
            cig_parts.struct.field("field_0").alias("ceiling_raw"),
        )
        .with_columns(
            pl.col("DATE").str.strptime(pl.Datetime, strict=False).alias("valid_ts"),
            pl.col("STATION").cast(pl.Utf8).alias("station"),

            pl.when(pl.col("temp_raw") == "+9999")
            .then(None)
            .otherwise(pl.col("temp_raw").cast(pl.Float64, strict=False) / 10.0)
            .alias("temp_c"),

            pl.when(pl.col("wind_speed_raw") == "9999")
            .then(None)
            .otherwise(pl.col("wind_speed_raw").cast(pl.Float64, strict=False) / 10.0)
            .alias("wind_speed_m_s"),

            pl.when(pl.col("wind_dir_raw") == "999")
            .then(None)
            .otherwise(pl.col("wind_dir_raw").cast(pl.Int64, strict=False))
            .alias("wind_dir_deg"),

            pl.when(pl.col("ceiling_raw") == "99999")
            .then(None)
            .otherwise(pl.col("ceiling_raw").cast(pl.Int64, strict=False))
            .alias("ceiling_height_m"),
        )
        .select(
            [
                "station",
                "valid_ts",
                "temp_c",
                "wind_speed_m_s",
                "wind_dir_deg",
                "ceiling_height_m",
            ]
        )
        .sort(["station", "valid_ts"])
    )


def build_month_date_window(year: int, month: int) -> tuple[str, str]:
    last_day = monthrange(year, month)[1]
    start = f"{year}-{month:02d}-01T00:00:00Z"
    end = f"{year}-{month:02d}-{last_day:02d}T23:59:59Z"
    return start, end


def build_year_date_window(year: int) -> tuple[str, str]:
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T23:59:59Z"
    return start, end


def pull_weather_station_month(
    session: requests.Session,
    station: str,
    year: int,
    month: int,
    cfg: WeatherConfig,
) -> pl.DataFrame:
    start, end = build_month_date_window(year, month)

    csv_text = fetch_ncei_csv_text_with_backoff(
        session=session,
        stations=[station],
        start=start,
        end=end,
        cfg=cfg,
    )

    df_raw = pl.read_csv(
        source=csv_text.encode("utf-8"),
        infer_schema_length=2000,
        ignore_errors=True,
    )

    if df_raw.height == 0:
        return pl.DataFrame(
            schema={
                "station": pl.Utf8,
                "valid_ts": pl.Datetime,
                "temp_c": pl.Float64,
                "wind_speed_m_s": pl.Float64,
                "wind_dir_deg": pl.Int64,
                "ceiling_height_m": pl.Int64,
            }
        )

    return clean_global_hourly(df_raw)


def monthly_cache_path(cfg: WeatherConfig, station: str, year: int, month: int) -> Path:
    return cfg.out_dir / "monthly_cache" / f"weather_{station}_{year}_{month:02d}.parquet"


def pull_weather_for_year_chunked(
    stations: list[str],
    year: int,
    cfg: WeatherConfig,
    raw_output_path: Path | None = None,
    clean_output_path: Path | None = None,
) -> pl.DataFrame:
    ensure_dir(cfg.out_dir)
    if raw_output_path is not None:
        ensure_dir(raw_output_path.parent)
    if clean_output_path is not None:
        ensure_dir(clean_output_path.parent)

    session = make_session(cfg)

    monthly_frames: list[pl.DataFrame] = []

    try:
        for station in stations:
            for month in range(1, 13):
                cache_path = monthly_cache_path(cfg, station, year, month)

                if cfg.use_monthly_cache and cache_path.exists():
                    print(f"Loading cached weather chunk -> {cache_path}")
                    df_chunk = pl.read_parquet(cache_path)
                    monthly_frames.append(df_chunk)
                    time.sleep(cfg.chunk_pause_seconds)
                    continue

                print(f"Pulling weather chunk: station={station}, year={year}, month={month:02d}")
                df_chunk = pull_weather_station_month(
                    session=session,
                    station=station,
                    year=year,
                    month=month,
                    cfg=cfg,
                )

                if cfg.use_monthly_cache:
                    ensure_dir(cache_path.parent)
                    df_chunk.write_parquet(cache_path)
                    print(f"Wrote weather chunk cache -> {cache_path}")

                monthly_frames.append(df_chunk)

                time.sleep(cfg.chunk_pause_seconds)
    finally:
        session.close()

    if not monthly_frames:
        df_year = pl.DataFrame(
            schema={
                "station": pl.Utf8,
                "valid_ts": pl.Datetime,
                "temp_c": pl.Float64,
                "wind_speed_m_s": pl.Float64,
                "wind_dir_deg": pl.Int64,
                "ceiling_height_m": pl.Int64,
            }
        )
    else:
        df_year = (
            pl.concat(monthly_frames, how="vertical_relaxed")
            .unique(subset=["station", "valid_ts"], keep="first")
            .sort(["station", "valid_ts"])
        )

    if clean_output_path is not None:
        df_year.write_parquet(clean_output_path)
        print(f"Wrote cleaned yearly weather -> {clean_output_path}")

    # raw_output_path intentionally not used in chunked mode because the source is many requests
    return df_year


def pull_weather_for_period(
    stations: list[str],
    start: str,
    end: str,
    cfg: WeatherConfig,
    raw_output_path: Path | None = None,
    clean_output_path: Path | None = None,
) -> pl.DataFrame:
    session = make_session(cfg)
    try:
        csv_text = fetch_ncei_csv_text_with_backoff(
            session=session,
            stations=stations,
            start=start,
            end=end,
            cfg=cfg,
        )
    finally:
        session.close()

    df_raw = pl.read_csv(
        source=csv_text.encode("utf-8"),
        infer_schema_length=2000,
        ignore_errors=True,
    )

    if raw_output_path is not None:
        ensure_dir(raw_output_path.parent)
        df_raw.write_parquet(raw_output_path)

    df_clean = clean_global_hourly(df_raw)

    if clean_output_path is not None:
        ensure_dir(clean_output_path.parent)
        df_clean.write_parquet(clean_output_path)

    return df_clean