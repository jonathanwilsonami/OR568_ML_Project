from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from config import FeatureConfig, JoinConfig
from utils import ensure_dir


def join_airport_reference(
    flights_df: pl.DataFrame,
    airport_dim: pl.DataFrame,
    joins: JoinConfig,
) -> pl.DataFrame:
    dep_ref = (
        airport_dim
        .select([
            pl.col("airport").alias("Origin"),
            pl.col("timezone").alias(joins.dep_timezone_col),
            pl.col("icao").alias("OriginICAO"),
        ])
    )
    arr_ref = (
        airport_dim
        .select([
            pl.col("airport").alias("Dest"),
            pl.col("timezone").alias(joins.arr_timezone_col),
            pl.col("icao").alias("DestICAO"),
        ])
    )

    return (
        flights_df
        .join(dep_ref, on="Origin", how="left")
        .join(arr_ref, on="Dest", how="left")
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
        df[joins.dep_ts_sched_local_col], df[joins.dep_timezone_col]
    )
    dep_actual_utc = _convert_local_series_to_utc(
        df[joins.dep_ts_local_col], df[joins.dep_timezone_col]
    )
    arr_sched_utc = _convert_local_series_to_utc(
        df["arr_ts_sched"], df[joins.arr_timezone_col]
    )
    arr_actual_utc = _convert_local_series_to_utc(
        df[joins.arr_ts_local_col], df[joins.arr_timezone_col]
    )

    return df.with_columns([
        dep_sched_utc.alias("dep_ts_sched_utc"),
        dep_actual_utc.alias("dep_ts_actual_utc"),
        arr_sched_utc.alias("arr_ts_sched_utc"),
        arr_actual_utc.alias("arr_ts_actual_utc"),
    ])


def add_flight_id(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.concat_str(
            [
                pl.col("FlightDate").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("Reporting_Airline").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("Flight_Number_Reporting_Airline").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("Tail_Number").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("Origin").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("Dest").cast(pl.Utf8),
                pl.lit("_"),
                pl.col("CRSDepTime").cast(pl.Utf8),
            ]
        ).alias("flight_id")
    ])


def build_flights_canonical(
    flights_df: pl.DataFrame,
    joins: JoinConfig,
) -> pl.DataFrame:
    df = flights_df.sort([joins.tail_col, joins.dep_ts_col])

    df = add_flight_id(df)

    df = df.with_columns([
        pl.col(joins.dep_ts_col).dt.date().alias("dep_date_utc"),
        pl.col(joins.dep_ts_local_col).dt.date().alias("dep_date_local"),
        pl.col("DepDelay").fill_null(0.0).alias("DepDelay"),
        pl.col("ArrDelay").fill_null(0.0).alias("ArrDelay"),
        pl.col("DepDel15").fill_null(0).alias("DepDel15"),
        pl.col("ArrDel15").fill_null(0).alias("ArrDel15"),
        pl.when(pl.col("Cancelled") == 1).then(1).otherwise(0).alias("is_cancelled"),
        pl.when(pl.col("Diverted") == 1).then(1).otherwise(0).alias("is_diverted"),
    ])

    df = df.with_columns([
        pl.col("flight_id").shift(1).over(joins.tail_col).alias("prev_flight_id_same_tail"),
        pl.col("flight_id").shift(-1).over(joins.tail_col).alias("next_flight_id_same_tail"),
        pl.col("Origin").shift(1).over(joins.tail_col).alias("prev_origin"),
        pl.col("Dest").shift(1).over(joins.tail_col).alias("prev_dest"),
        pl.col("Origin").shift(-1).over(joins.tail_col).alias("next_origin"),
        pl.col("Dest").shift(-1).over(joins.tail_col).alias("next_dest"),
        pl.col(joins.arr_ts_col).shift(1).over(joins.tail_col).alias("prev_arr_ts_actual_utc"),
        pl.col(joins.arr_ts_local_col).shift(1).over(joins.tail_col).alias("prev_arr_ts_actual"),
        pl.col(joins.dep_ts_col).shift(-1).over(joins.tail_col).alias("next_dep_ts_actual_utc"),
        pl.col(joins.dep_ts_local_col).shift(-1).over(joins.tail_col).alias("next_dep_ts_actual"),
        pl.col("ArrDelay").shift(1).over(joins.tail_col).alias("prev_arr_delay"),
        pl.col("DepDelay").shift(1).over(joins.tail_col).alias("prev_dep_delay"),
        pl.col("ArrDelay").shift(-1).over(joins.tail_col).alias("next_arr_delay"),
        pl.col("DepDelay").shift(-1).over(joins.tail_col).alias("next_dep_delay"),
        pl.col("DepDel15").shift(1).over(joins.tail_col).alias("prev_dep_del15"),
        pl.col("ArrDel15").shift(1).over(joins.tail_col).alias("prev_arr_del15"),
        pl.col("DepDel15").shift(-1).over(joins.tail_col).alias("next_dep_del15"),
        pl.col("ArrDel15").shift(-1).over(joins.tail_col).alias("next_arr_del15"),
        pl.col("FlightDate").shift(1).over(joins.tail_col).alias("prev_flight_date"),
        pl.col("FlightDate").shift(-1).over(joins.tail_col).alias("next_flight_date"),
    ])

    df = df.with_columns([
        (
            (pl.col(joins.dep_ts_col) - pl.col("prev_arr_ts_actual_utc"))
            .dt.total_minutes()
        ).alias("turnaround_minutes"),
        (
            (pl.col("next_dep_ts_actual_utc") - pl.col(joins.arr_ts_col))
            .dt.total_minutes()
        ).alias("next_turnaround_minutes"),
        pl.when(pl.col("prev_arr_delay") > 15).then(1).otherwise(0).alias("prev_arr_late_15"),
        pl.when(pl.col("prev_dep_delay") > 15).then(1).otherwise(0).alias("prev_dep_late_15"),
        pl.when(pl.col("next_arr_delay") > 15).then(1).otherwise(0).alias("next_arr_late_15"),
        pl.when(pl.col("next_dep_delay") > 15).then(1).otherwise(0).alias("next_dep_late_15"),
        pl.when(pl.col("prev_dest") == pl.col("Origin")).then(1).otherwise(0).alias("rotation_continuity_flag"),
        pl.when(pl.col("Dest") == pl.col("next_origin")).then(1).otherwise(0).alias("next_rotation_continuity_flag"),
    ])

    df = df.with_columns([
        pl.col(joins.dep_ts_col).cum_count().over([joins.tail_col, "dep_date_local"]).alias("aircraft_leg_number_day"),
        pl.col("DepDelay").cum_sum().over([joins.tail_col, "dep_date_local"]).alias("cum_dep_delay_aircraft_day"),
        pl.col("ArrDelay").cum_sum().over([joins.tail_col, "dep_date_local"]).alias("cum_arr_delay_aircraft_day"),
    ])

    df = df.with_columns([
        pl.concat_str([pl.col("Origin"), pl.lit("_"), pl.col("Dest")]).alias("route_key"),
        pl.col(joins.dep_ts_local_col).dt.hour().alias("dep_hour_local"),
        pl.col(joins.dep_ts_local_col).dt.weekday().alias("dep_weekday_local"),
        pl.col(joins.dep_ts_local_col).dt.month().alias("dep_month_local"),
        pl.col(joins.dep_ts_col).dt.hour().alias("dep_hour_utc"),
        pl.col(joins.dep_ts_col).dt.weekday().alias("dep_weekday_utc"),
        pl.col(joins.dep_ts_col).dt.month().alias("dep_month_utc"),
    ])

    return df


def build_aircraft_rotation_table(
    flights_df: pl.DataFrame,
    joins: JoinConfig,
) -> pl.DataFrame:
    keep_cols = [
        "flight_id",
        "FlightDate",
        joins.tail_col,
        "Reporting_Airline",
        "Origin",
        "Dest",
        joins.dep_ts_local_col,
        joins.arr_ts_local_col,
        joins.dep_ts_col,
        joins.arr_ts_col,
        "DepDelay",
        "ArrDelay",
        "DepDel15",
        "ArrDel15",
        "prev_flight_id_same_tail",
        "next_flight_id_same_tail",
        "prev_origin",
        "prev_dest",
        "next_origin",
        "next_dest",
        "prev_arr_ts_actual",
        "prev_arr_ts_actual_utc",
        "next_dep_ts_actual",
        "next_dep_ts_actual_utc",
        "prev_arr_delay",
        "prev_dep_delay",
        "next_arr_delay",
        "next_dep_delay",
        "prev_arr_late_15",
        "prev_dep_late_15",
        "next_arr_late_15",
        "next_dep_late_15",
        "turnaround_minutes",
        "next_turnaround_minutes",
        "rotation_continuity_flag",
        "next_rotation_continuity_flag",
        "aircraft_leg_number_day",
        "cum_dep_delay_aircraft_day",
        "cum_arr_delay_aircraft_day",
    ]
    keep_cols = [c for c in keep_cols if c in flights_df.columns]
    return flights_df.select(keep_cols)


def build_propagation_chains_table(
    flights_df: pl.DataFrame,
    joins: JoinConfig,
) -> pl.DataFrame:
    keep_cols = [
        "flight_id",
        joins.tail_col,
        "Reporting_Airline",
        "FlightDate",
        "Origin",
        "Dest",
        joins.dep_ts_local_col,
        joins.arr_ts_local_col,
        joins.dep_ts_col,
        joins.arr_ts_col,
        "DepDelay",
        "ArrDelay",
        "DepDel15",
        "ArrDel15",
        "prev_flight_id_same_tail",
        "next_flight_id_same_tail",
        "prev_origin",
        "prev_dest",
        "next_origin",
        "next_dest",
        "prev_arr_ts_actual",
        "prev_arr_ts_actual_utc",
        "next_dep_ts_actual",
        "next_dep_ts_actual_utc",
        "prev_arr_delay",
        "prev_dep_delay",
        "next_arr_delay",
        "next_dep_delay",
        "prev_arr_late_15",
        "prev_dep_late_15",
        "next_arr_late_15",
        "next_dep_late_15",
        "turnaround_minutes",
        "next_turnaround_minutes",
        "rotation_continuity_flag",
        "next_rotation_continuity_flag",
        "route_key",
        "aircraft_leg_number_day",
        "cum_dep_delay_aircraft_day",
        "cum_arr_delay_aircraft_day",
    ]
    keep_cols = [c for c in keep_cols if c in flights_df.columns]

    chains = flights_df.select(keep_cols)

    chains = chains.with_columns([
        pl.when(pl.col("prev_flight_id_same_tail").is_not_null()).then(1).otherwise(0).alias("has_prev_leg"),
        pl.when(pl.col("next_flight_id_same_tail").is_not_null()).then(1).otherwise(0).alias("has_next_leg"),
        pl.when(
            pl.col("prev_flight_id_same_tail").is_not_null() &
            pl.col("next_flight_id_same_tail").is_not_null()
        ).then(1).otherwise(0).alias("is_middle_leg"),
        pl.when(pl.col("turnaround_minutes") < 60).then(1).otherwise(0).alias("tight_turnaround_lt_60"),
        pl.when(pl.col("turnaround_minutes") < 90).then(1).otherwise(0).alias("tight_turnaround_lt_90"),
        pl.when(pl.col("next_turnaround_minutes") < 60).then(1).otherwise(0).alias("next_tight_turnaround_lt_60"),
        pl.when(pl.col("next_turnaround_minutes") < 90).then(1).otherwise(0).alias("next_tight_turnaround_lt_90"),
    ])

    return chains


def _add_time_bucket(df: pl.DataFrame, ts_col: str, bucket: str, out_col: str) -> pl.DataFrame:
    return df.with_columns(pl.col(ts_col).dt.truncate(bucket).alias(out_col))


def _mean_exprs(df: pl.DataFrame, cols: list[str]) -> list[pl.Expr]:
    return [pl.col(c).mean().alias(f"{c}_mean") for c in cols if c in df.columns]


def build_airport_time_table(
    flights_df: pl.DataFrame,
    cfg: FeatureConfig,
    joins: JoinConfig,
) -> pl.DataFrame:
    dep_df = _add_time_bucket(flights_df, joins.dep_ts_col, cfg.time_bucket, "time_bucket")
    arr_df = _add_time_bucket(flights_df, joins.arr_ts_col, cfg.time_bucket, "time_bucket")

    dep_aggs: list[pl.Expr] = [
        pl.len().alias("dep_flight_count"),
        pl.col("DepDelay").mean().alias("dep_delay_mean"),
        pl.col("DepDelay").median().alias("dep_delay_median"),
        pl.col("DepDel15").mean().alias("dep_del15_rate"),
        pl.col("Cancelled").mean().alias("dep_cancel_rate"),
    ]
    if "TaxiOut" in dep_df.columns:
        dep_aggs.append(pl.col("TaxiOut").mean().alias("taxi_out_mean"))
    dep_aggs.extend(_mean_exprs(dep_df, ["dep_temp_c", "dep_wind_speed_m_s", "dep_wind_dir_deg", "dep_ceiling_height_m"]))

    arr_aggs: list[pl.Expr] = [
        pl.len().alias("arr_flight_count"),
        pl.col("ArrDelay").mean().alias("arr_delay_mean"),
        pl.col("ArrDelay").median().alias("arr_delay_median"),
        pl.col("ArrDel15").mean().alias("arr_del15_rate"),
        pl.col("Diverted").mean().alias("arr_divert_rate"),
    ]
    if "TaxiIn" in arr_df.columns:
        arr_aggs.append(pl.col("TaxiIn").mean().alias("taxi_in_mean"))
    arr_aggs.extend(_mean_exprs(arr_df, ["arr_temp_c", "arr_wind_speed_m_s", "arr_wind_dir_deg", "arr_ceiling_height_m"]))

    dep_grouped = dep_df.group_by(["Origin", "time_bucket"]).agg(dep_aggs).rename({"Origin": "airport"})
    arr_grouped = arr_df.group_by(["Dest", "time_bucket"]).agg(arr_aggs).rename({"Dest": "airport"})

    airport_time = (
        dep_grouped.join(arr_grouped, on=["airport", "time_bucket"], how="full", coalesce=True)
        .sort(["airport", "time_bucket"])
        .with_columns([
            pl.col("dep_flight_count").fill_null(0),
            pl.col("arr_flight_count").fill_null(0),
        ])
        .with_columns([
            (pl.col("dep_flight_count") + pl.col("arr_flight_count")).alias("total_flight_count"),
        ])
    )

    if "dep_delay_mean" in airport_time.columns and "arr_delay_mean" in airport_time.columns:
        airport_time = airport_time.with_columns(
            (pl.col("dep_delay_mean") - pl.col("arr_delay_mean")).alias("dep_minus_arr_delay_mean")
        )

    for h in cfg.rolling_airport_hours:
        roll_exprs = []
        for col_name in ["dep_delay_mean", "arr_delay_mean", "dep_del15_rate", "arr_del15_rate", "total_flight_count"]:
            if col_name in airport_time.columns:
                roll_exprs.append(
                    pl.col(col_name).rolling_mean(window_size=h, min_samples=1).over("airport").alias(f"{col_name}_roll_{h}")
                )
        if roll_exprs:
            airport_time = airport_time.with_columns(roll_exprs)

    return airport_time


def build_route_time_table(
    flights_df: pl.DataFrame,
    cfg: FeatureConfig,
    joins: JoinConfig,
) -> pl.DataFrame:
    df = _add_time_bucket(flights_df, joins.dep_ts_col, cfg.time_bucket, "time_bucket")

    aggs: list[pl.Expr] = [
        pl.len().alias("flight_count"),
        pl.col("DepDelay").mean().alias("dep_delay_mean"),
        pl.col("ArrDelay").mean().alias("arr_delay_mean"),
        pl.col("DepDel15").mean().alias("dep_del15_rate"),
        pl.col("ArrDel15").mean().alias("arr_del15_rate"),
    ]
    aggs.extend(_mean_exprs(
        df,
        [
            "Distance",
            "ActualElapsedTime",
            "CarrierDelay",
            "LateAircraftDelay",
            "WeatherDelay",
            "NASDelay",
            "dep_temp_c",
            "dep_wind_speed_m_s",
            "dep_wind_dir_deg",
            "dep_ceiling_height_m",
            "arr_temp_c",
            "arr_wind_speed_m_s",
            "arr_wind_dir_deg",
            "arr_ceiling_height_m",
        ],
    ))

    route_time = (
        df.group_by(["Origin", "Dest", "route_key", "time_bucket"])
        .agg(aggs)
        .sort(["route_key", "time_bucket"])
    )

    for h in cfg.rolling_route_hours:
        roll_exprs = []
        for col_name in ["dep_delay_mean", "arr_delay_mean", "dep_del15_rate", "flight_count"]:
            if col_name in route_time.columns:
                roll_exprs.append(
                    pl.col(col_name).rolling_mean(window_size=h, min_samples=1).over("route_key").alias(f"{col_name}_roll_{h}")
                )
        if roll_exprs:
            route_time = route_time.with_columns(roll_exprs)

    return route_time


def write_dataset(df: pl.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.write_parquet(out_path)
    print(f"Wrote -> {out_path}")


def build_all_canonical_feature_tables(
    flights_joined_df: pl.DataFrame,
    airport_dim: pl.DataFrame,
    joins: JoinConfig,
    cfg: FeatureConfig,
    out_dir: Path,
    year_label: str,
) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}

    flights_df = join_airport_reference(flights_joined_df, airport_dim, joins)
    flights_df = add_utc_timestamps(flights_df, joins)
    flights_canonical = build_flights_canonical(flights_df, joins)
    out["flights_canonical"] = flights_canonical

    if cfg.write_flights_canonical:
        write_dataset(flights_canonical, out_dir / f"flights_canonical_{year_label}.parquet")

    if cfg.build_aircraft_rotation_features:
        aircraft_rotation = build_aircraft_rotation_table(flights_canonical, joins)
        out["aircraft_rotation"] = aircraft_rotation
        if cfg.write_aircraft_rotation:
            write_dataset(aircraft_rotation, out_dir / f"aircraft_rotation_{year_label}.parquet")

    if cfg.build_propagation_chain_features:
        propagation_chains = build_propagation_chains_table(flights_canonical, joins)
        out["propagation_chains"] = propagation_chains
        if cfg.write_propagation_chains:
            write_dataset(propagation_chains, out_dir / f"propagation_chains_{year_label}.parquet")

    if cfg.build_airport_time_features:
        airport_time = build_airport_time_table(flights_canonical, cfg, joins)
        out["airport_time"] = airport_time
        if cfg.write_airport_time:
            write_dataset(airport_time, out_dir / f"airport_time_{year_label}.parquet")

    if cfg.build_route_time_features:
        route_time = build_route_time_table(flights_canonical, cfg, joins)
        out["route_time"] = route_time
        if cfg.write_route_time:
            write_dataset(route_time, out_dir / f"route_time_{year_label}.parquet")

    return out