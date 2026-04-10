from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# =========================
# BTS CONFIG
# =========================
@dataclass
class BTSConfig:
    prezip_base: str = "https://transtats.bts.gov/PREZIP"
    zip_name_template: str = (
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    )
    out_dir: Path = Path("data/bts")
    timeout: int = 180
    cleanup_downloads_if_final_has_data: bool = False

    # polite downloading
    max_retries: int = 6
    backoff_base_seconds: float = 2.0
    chunk_pause_seconds: float = 1.0
    verify_ssl: bool = True

    # caching
    keep_zip_files: bool = True
    keep_extracted_csvs: bool = True


# =========================
# WEATHER CONFIG (NOAA)
# =========================
@dataclass
class WeatherConfig:
    base_url: str = "https://www.ncei.noaa.gov/access/services/data/v1"
    out_dir: Path = Path("data/weather")
    units: str = "metric"
    include_attributes: bool = False
    include_station_name: bool = True
    raw_parquet: bool = True
    timeout: int = 180

    chunk_by_month: bool = True
    chunk_pause_seconds: float = 1.5
    max_retries: int = 6
    backoff_base_seconds: float = 2.0
    verify_ssl: bool = True
    use_monthly_cache: bool = True


# =========================
# JOIN CONFIG
# =========================
@dataclass
class JoinConfig:
    dep_ts_col: str = "dep_ts_actual"
    arr_ts_col: str = "arr_ts_actual"

    origin_col: str = "Origin"
    dest_col: str = "Dest"
    tail_col: str = "Tail_Number"
    carrier_col: str = "Reporting_Airline"

    weather_station_col: str = "station"
    weather_ts_col: str = "valid_ts"

    dep_station_col: str = "dep_station"
    arr_station_col: str = "arr_station"

    weather_strategy: str = "backward"
    weather_tolerance: str = "2h"


# =========================
# ROUTE FILTER CONFIG
# =========================
@dataclass
class RouteFilterConfig:
    airports: list[str] | None = None
    airport_pairs: list[tuple[str, str]] | None = None
    origin_filter: list[str] | None = None
    dest_filter: list[str] | None = None


# =========================
# FEATURE ENGINEERING CONFIG
# =========================
@dataclass
class FeatureConfig:
    enabled: bool = True

    # Time resolution for network modeling
    time_bucket: str = "1h"

    # Rolling windows (important for propagation)
    rolling_airport_hours: list[int] = field(default_factory=lambda: [1, 3, 6])
    rolling_route_hours: list[int] = field(default_factory=lambda: [1, 3, 6])

    # Feature toggles
    build_aircraft_rotation_features: bool = True
    build_airport_time_features: bool = True
    build_route_time_features: bool = True

    # Outputs
    write_flights_enriched: bool = True
    write_aircraft_rotation: bool = True
    write_airport_time: bool = True
    write_route_time: bool = True


# =========================
# POST PROCESS CONFIG
# =========================
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


# =========================
# PIPELINE CONFIG
# =========================
@dataclass
class PipelineConfig:
    years: list[int] = field(default_factory=lambda: [2019])
    months_by_year: dict[int, list[int]] | None = None

    # NETWORK AIRPORTS (CORE)
    airport_to_station: dict[str, str] = field(
        default_factory=lambda: {
            "BWI": "72406093721",
            "ATL": "72219013874",
            "ORD": "72530094846",
            "DFW": "72259003927",
            "DEN": "72565003017",
            "BOS": "72509014739",
            "CLT": "72314013881",
            "MCO": "72205012815",
            "JFK": "74486094789",
            "LGA": "72503014732",
            "EWR": "72502014734",
        }
    )

    route_filter: RouteFilterConfig = field(
        default_factory=lambda: RouteFilterConfig(
            airports=[
                "BWI",
                "ATL",
                "ORD",
                "DFW",
                "DEN",
                "BOS",
                "CLT",
                "MCO",
                "JFK",
                "LGA",
                "EWR",
            ]
        )
    )

    bts: BTSConfig = field(default_factory=BTSConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    joins: JoinConfig = field(default_factory=JoinConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)

    final_out_dir: Path = Path("data/final")
    feature_out_dir: Path = Path("data/features")

    write_monthly_joined: bool = True
    write_yearly_joined: bool = True

    use_cached_weather: bool = True
    use_cached_bts_months: bool = True
    use_cached_monthly_joined: bool = False

    run_weather_stage: bool = True
    run_bts_stage: bool = True
    run_join_stage: bool = True
    run_feature_stage: bool = True


# =========================
# FINAL CONFIG INSTANCE
# =========================
CONFIG = PipelineConfig(
    years=[2019],
    months_by_year={
        2019: list(range(1, 13)),
    },
)