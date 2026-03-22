from __future__ import annotations

from zoneinfo import ZoneInfo

import polars as pl

from config import JoinConfig


def add_station_keys_to_bts(
    df: pl.DataFrame,
    airport_to_station: dict[str, str],
    joins: JoinConfig,
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(joins.origin_col)
        .replace(airport_to_station, default=None)
        .alias(joins.dep_station_col),

        pl.col(joins.dest_col)
        .replace(airport_to_station, default=None)
        .alias(joins.arr_station_col),
    )


def prepare_weather_prefix(df_weather: pl.DataFrame, prefix: str) -> pl.DataFrame:
    keep_cols = {"station", "valid_ts", "valid_ts_utc"}
    rename_map = {
        c: f"{prefix}{c}"
        for c in df_weather.columns
        if c not in keep_cols
    }
    return df_weather.rename(rename_map)


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


def add_weather_utc_timestamps(
    weather_df: pl.DataFrame,
    station_to_timezone: dict[str, str],
) -> pl.DataFrame:
    """
    Convert NOAA weather valid_ts from station-local naive time to UTC naive time.
    """
    if "station" not in weather_df.columns or "valid_ts" not in weather_df.columns:
        raise KeyError(
            f"Weather dataframe must contain ['station', 'valid_ts']; got {weather_df.columns}"
        )

    df = weather_df.with_columns(
        pl.col("station").replace(station_to_timezone, default=None).alias("station_timezone")
    )

    valid_ts_utc = _convert_local_series_to_utc(
        df["valid_ts"],
        df["station_timezone"],
    )

    return df.with_columns(valid_ts_utc.alias("valid_ts_utc"))


def join_weather_to_bts(
    bts_df: pl.DataFrame,
    weather_df: pl.DataFrame,
    joins: JoinConfig,
) -> pl.DataFrame:
    required = [
        joins.dep_ts_col,
        joins.arr_ts_col,
        joins.dep_station_col,
        joins.arr_station_col,
    ]
    missing = [c for c in required if c not in bts_df.columns]
    if missing:
        raise KeyError(
            f"Missing required BTS join columns: {missing}. "
            f"Available columns include: {bts_df.columns[:60]}"
        )

    if "valid_ts_utc" not in weather_df.columns:
        raise KeyError(
            "Weather dataframe must contain 'valid_ts_utc'. "
            "Call add_weather_utc_timestamps() before joining."
        )

    wx_dep = prepare_weather_prefix(weather_df, "dep_").sort(["station", "valid_ts_utc"])
    wx_arr = prepare_weather_prefix(weather_df, "arr_").sort(["station", "valid_ts_utc"])

    dep_sorted = bts_df.sort([joins.dep_station_col, joins.dep_ts_col])

    dep_joined = dep_sorted.join_asof(
        wx_dep,
        left_on=joins.dep_ts_col,
        right_on="valid_ts_utc",
        by_left=joins.dep_station_col,
        by_right="station",
        strategy=joins.weather_strategy,
        tolerance=joins.weather_tolerance,
    )

    arr_sorted = dep_joined.sort([joins.arr_station_col, joins.arr_ts_col])

    fully_joined = arr_sorted.join_asof(
        wx_arr,
        left_on=joins.arr_ts_col,
        right_on="valid_ts_utc",
        by_left=joins.arr_station_col,
        by_right="station",
        strategy=joins.weather_strategy,
        tolerance=joins.weather_tolerance,
    )

    return fully_joined