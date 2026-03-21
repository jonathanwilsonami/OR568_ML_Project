from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BTSConfig:
    prezip_base: str = "https://transtats.bts.gov/PREZIP"
    zip_name_template: str = (
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    )
    out_dir: Path = Path("data/bts")
    timeout: int = 180

    max_retries: int = 6
    backoff_base_seconds: float = 2.0
    chunk_pause_seconds: float = 1.0
    verify_ssl: bool = True

    keep_zip_files: bool = True
    keep_extracted_csvs: bool = True
    cleanup_downloads_if_final_has_data: bool = False


@dataclass
class WeatherConfig:
    base_url: str = "https://www.ncei.noaa.gov/access/services/data/v1"
    dataset: str = "global-hourly"
    out_dir: Path = Path("data/weather")
    units: str = "metric"
    timeout: int = 180

    chunk_by_month: bool = True
    chunk_pause_seconds: float = 1.5
    max_retries: int = 6
    backoff_base_seconds: float = 2.0
    verify_ssl: bool = True

    use_monthly_cache: bool = True
    raw_parquet: bool = True


@dataclass
class MappingConfig:
    out_dir: Path = Path("data/reference")

    unique_airports_path: Path = Path("data/reference/unique_airports_2019.parquet")
    airport_station_json_path: Path = Path("data/reference/airport_to_station_2019.json")
    airport_station_csv_path: Path = Path("data/reference/airport_to_station_2019.csv")
    unmapped_airports_path: Path = Path("data/reference/unmapped_airports_2019.parquet")

    ourairports_url: str = "https://davidmegginson.github.io/ourairports-data/airports.csv"
    isd_history_url: str = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

    airports_cache_path: Path = Path("data/reference/raw/ourairports_airports.csv")
    isd_history_cache_path: Path = Path("data/reference/raw/isd-history.csv")

    mapping_year: int = 2019


@dataclass
class JoinConfig:
    dep_ts_col: str = "dep_ts_actual"
    arr_ts_col: str = "arr_ts_actual"

    origin_col: str = "Origin"
    dest_col: str = "Dest"
    tail_col: str = "Tail_Number"
    carrier_col: str = "Reporting_Airline"

    dep_station_col: str = "dep_station"
    arr_station_col: str = "arr_station"

    weather_ts_col: str = "valid_ts"
    weather_strategy: str = "backward"
    weather_tolerance: str = "2h"


@dataclass
class RouteFilterConfig:
    airports: list[str] | None = None
    airport_pairs: list[tuple[str, str]] | None = None
    origin_filter: list[str] | None = None
    dest_filter: list[str] | None = None


@dataclass
class FeatureConfig:
    enabled: bool = True
    time_bucket: str = "1h"

    rolling_airport_hours: list[int] = field(default_factory=lambda: [1, 3, 6])
    rolling_route_hours: list[int] = field(default_factory=lambda: [1, 3, 6])

    build_aircraft_rotation_features: bool = True
    build_airport_time_features: bool = True
    build_route_time_features: bool = True

    write_flights_enriched: bool = True
    write_aircraft_rotation: bool = True
    write_airport_time: bool = True
    write_route_time: bool = True


@dataclass
class PostProcessConfig:
    enabled: bool = True
    write_filtered_monthly: bool = False
    write_filtered_yearly: bool = True
    write_filtered_all_years: bool = True
    write_all_years_full: bool = True
    write_all_years_filtered: bool = True
    strict_missing_columns: bool = False

    selected_columns: list[str] = field(default_factory=lambda: [
        "FlightDate",
        "Reporting_Airline",
        "Tail_Number",
        "Origin",
        "Dest",
        "CRSDepTime",
        "DepTime",
        "DepDelay",
        "DepDelayMinutes",
        "DepDel15",
        "CRSArrTime",
        "ArrTime",
        "ArrDelay",
        "ArrDelayMinutes",
        "ArrDel15",
        "TaxiOut",
        "TaxiIn",
        "Cancelled",
        "Diverted",
        "Distance",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "LateAircraftDelay",
        "dep_ts_actual",
        "arr_ts_actual",
        "dep_station",
        "arr_station",
    ])


@dataclass
class PipelineConfig:
    years: list[int] = field(default_factory=lambda: [2019])
    months_by_year: dict[int, list[int]] | None = None

    route_filter: RouteFilterConfig = field(default_factory=RouteFilterConfig)

    bts: BTSConfig = field(default_factory=BTSConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    joins: JoinConfig = field(default_factory=JoinConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)

    final_out_dir: Path = Path("data/final")
    feature_out_dir: Path = Path("data/features")

    use_cached_bts_months: bool = True
    use_cached_weather: bool = True
    use_cached_monthly_joined: bool = False

    write_monthly_joined: bool = True
    write_yearly_joined: bool = True

    run_bts_stage: bool = True
    run_mapping_stage: bool = True
    run_weather_stage: bool = True
    run_join_stage: bool = True
    run_feature_stage: bool = True


CONFIG = PipelineConfig(
    years=[2019],
    months_by_year={
        2019: list(range(1, 13)),
    },
    route_filter=RouteFilterConfig(
        airports=None,
        airport_pairs=None,
        origin_filter=None,
        dest_filter=None,
    ),
)