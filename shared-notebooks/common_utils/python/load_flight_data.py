from pathlib import Path
from urllib.parse import urljoin, urlparse
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
        col = re.sub(r"[^a-zA-Z0-9]+", r"_", col)
        col = re.sub(r"_+", "_", col)
        col = col.lower().strip("_")

        if col in seen:
            seen[col] += 1
            col = f"{col}_{seen[col]}"
        else:
            seen[col] = 0

        cleaned.append(col)

    return cleaned


def _resolve_url(url: str, file_name: str | None = None) -> str:
    """
    Build the final URL.

    Rules:
    - If file_name is not provided, use url as-is.
    - If file_name is provided, replace the file portion of the default/base URL
      with that file name so callers only need to pass something like
      'enriched_flights_2018.parquet'.
    """
    if not file_name:
        return url

    base_url = url.rsplit("/", 1)[0] + "/"
    return urljoin(base_url, file_name)


def _infer_local_filename(resolved_url: str) -> str:
    parsed = urlparse(resolved_url)
    file_name = Path(parsed.path).name
    if not file_name:
        raise ValueError("Could not infer a file name from the resolved URL.")
    return file_name


def load_flight_data(
    url: str = URL,
    file_name: str | None = None,
    lazy: bool = True
):
    """
    Load flight data from local cache or S3.

    Returns:
        LazyFrame (default) or DataFrame if lazy=False
    """

    module_file = Path(__file__).resolve()
    shared_notebooks_dir = module_file.parents[2]

    data_dir = shared_notebooks_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    resolved_url = _resolve_url(url, file_name)
    local_file_name = _infer_local_filename(resolved_url)
    local_file = data_dir / local_file_name

    if not local_file.exists():
        print(f"Downloading dataset from S3: {resolved_url}")
        urllib.request.urlretrieve(resolved_url, local_file)

    if lazy:
        # LAZY LOAD (SAFE FOR LARGE DATA)
        lf = pl.scan_parquet(local_file)

        # rename columns lazily
        cleaned = _clean_names(lf.columns)
        lf = lf.rename(dict(zip(lf.columns, cleaned)))

        return lf

    else:
        # FULL LOAD (ONLY USE IF DATA IS SMALL)
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

    df_work = df_work.with_columns(
        [
            pl.col("flight_date").cast(pl.Date, strict=False),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols],
        ]
    )

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

    df_work = df_work.sort(["tail_number", "flight_date", "sched_dep_min"])

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

    df_work = df_work.with_columns(
        [
            (pl.col("curr_sched_dep_abs_min") - pl.col("prev_arr_abs_min")).alias("turnaround_minutes"),
        ]
    )

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