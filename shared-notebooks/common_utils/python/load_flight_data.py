from pathlib import Path
import polars as pl
import urllib.request
import re

URL = "https://or568-flight-delay-data-411750981882-us-east-1-an.s3.us-east-1.amazonaws.com/enriched_flights_2019.parquet"


def _clean_names(columns):
    cleaned = []
    seen = {}

    for col in columns:
        col = col.strip()
        col = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", col)
        col = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
        col = re.sub(r"[^a-zA-Z0-9]+", "_", col)
        col = re.sub(r"_+", "_", col)
        col = col.lower().strip("_")

        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0

        cleaned.append(col)

    return cleaned


def load_flight_data(url: str = URL) -> pl.DataFrame:
    module_file = Path(__file__).resolve()
    shared_notebooks_dir = module_file.parents[2]

    data_dir = shared_notebooks_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    local_file = data_dir / "enriched_flights_2019.parquet"

    if not local_file.exists():
        print("Downloading dataset from S3...")
        urllib.request.urlretrieve(url, local_file)

    df = pl.read_parquet(local_file)
    df.columns = _clean_names(df.columns)
    return df


def _hhmm_to_minutes(col_name: str, alias: str) -> pl.Expr:
    c = pl.col(col_name).cast(pl.Float64, strict=False)
    return (
        pl.when(c.is_not_null())
        .then((c // 100) * 60 + (c % 100))
        .otherwise(None)
        .alias(alias)
    )


def engineer_features(df_work: pl.DataFrame) -> pl.DataFrame:
    numeric_cols = [
        "crs_dep_time", "crs_arr_time", "dep_time", "arr_time",
        "dep_delay_minutes", "arr_delay_minutes",
        "crs_elapsed_time", "air_time",
        "dep_vsby", "dep_gust", "dep_sknt", "dep_p01i", "dep_weather_severity",
    ]

    # Cast first
    df_work = df_work.with_columns(
        [
            pl.col("flight_date").cast(pl.Date, strict=False),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols],
        ]
    )

    # Main feature engineering
    df_work = df_work.with_columns(
        [
            (pl.col("crs_dep_time") // 100).alias("sched_dep_hour"),
            (pl.col("crs_arr_time") // 100).alias("sched_arr_hour"),
            (pl.col("crs_dep_time") // 100).alias("hour_of_day"),

            pl.when(pl.col("day_of_week").is_in([6, 7]))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_weekend"),

            pl.concat_str([pl.col("origin"), pl.lit("_"), pl.col("dest")]).alias("route"),

            pl.when(pl.col("dep_delay_minutes").is_null())
            .then(pl.lit(None, dtype=pl.Int64))
            .when(pl.col("dep_delay_minutes") > 15)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("is_delayed"),

            pl.when(pl.col("dep_delay_minutes").is_null())
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("dep_delay_minutes") <= 0)
            .then(pl.lit("on_time_or_early"))
            .when(pl.col("dep_delay_minutes") <= 15)
            .then(pl.lit("minor_delay"))
            .when(pl.col("dep_delay_minutes") <= 60)
            .then(pl.lit("moderate_delay"))
            .otherwise(pl.lit("severe_delay"))
            .alias("delay_state"),

            _hhmm_to_minutes("crs_dep_time", "sched_dep_min"),
            _hhmm_to_minutes("crs_arr_time", "sched_arr_min"),
            _hhmm_to_minutes("dep_time", "dep_min"),
            _hhmm_to_minutes("arr_time", "arr_min"),

            (pl.col("crs_elapsed_time") - pl.col("air_time")).alias("schedule_buffer"),
        ]
    )

    # Sort before lag features
    df_work = df_work.sort(["tail_number", "flight_date", "sched_dep_min"])

    # Lag features by aircraft
    df_work = df_work.with_columns(
        [
            pl.col("origin").shift(1).over("tail_number").alias("prev_origin"),
            pl.col("dest").shift(1).over("tail_number").alias("prev_dest"),
            pl.col("dep_delay_minutes").shift(1).over("tail_number").alias("prev_dep_delay"),
            pl.col("arr_delay_minutes").shift(1).over("tail_number").alias("prev_arr_delay"),
            pl.col("arr_min").shift(1).over("tail_number").alias("prev_arr_min"),
            pl.col("dep_min").shift(1).over("tail_number").alias("prev_dep_min"),
            pl.col("flight_date").shift(1).over("tail_number").alias("prev_flight_date"),
        ]
    )

    # Within-aircraft/day sequence features
    df_work = df_work.with_columns(
        [
            pl.cum_count("tail_number").over(["tail_number", "flight_date"]).alias("rotation_leg_number"),
            pl.len().over(["tail_number", "flight_date"]).alias("flights_per_aircraft_day"),

            pl.col("dep_delay_minutes")
            .fill_null(0)
            .cum_sum()
            .over(["tail_number", "flight_date"])
            .alias("cum_dep_delay_day"),

            pl.col("arr_delay_minutes")
            .fill_null(0)
            .cum_sum()
            .over(["tail_number", "flight_date"])
            .alias("cum_arr_delay_day"),
        ]
    )

    base_date = df_work.select(pl.col("flight_date").min()).item()

    # Absolute minute features
    df_work = df_work.with_columns(
        [
            (
                (pl.col("flight_date") - pl.lit(base_date)).dt.total_days() * 1440
                + pl.col("sched_dep_min")
            ).alias("curr_sched_dep_abs_min"),

            pl.when(
                pl.col("prev_flight_date").is_not_null() & pl.col("prev_arr_min").is_not_null()
            )
            .then(
                (pl.col("prev_flight_date") - pl.lit(base_date)).dt.total_days() * 1440
                + pl.col("prev_arr_min")
            )
            .otherwise(None)
            .alias("prev_arr_abs_min"),
        ]
    )

    # Create turnaround first
    df_work = df_work.with_columns(
        [
            (pl.col("curr_sched_dep_abs_min") - pl.col("prev_arr_abs_min")).alias("turnaround_minutes"),
        ]
    )

    # Then use turnaround_minutes
    df_work = df_work.with_columns(
        [
            pl.max_horizontal(
                pl.col("prev_arr_delay").fill_null(0) - pl.col("turnaround_minutes").fill_null(0),
                pl.lit(0.0),
            ).alias("inherited_delay"),

            pl.when(pl.col("dep_vsby") < 3)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("bad_visibility"),

            pl.when((pl.col("dep_gust") > 20) | (pl.col("dep_sknt") > 20))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("high_wind"),

            pl.when(pl.col("dep_p01i") > 0)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("precipitation"),

            pl.when(pl.col("dep_weather_severity") >= 2)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("severe_weather"),
        ]
    )

    return df_work