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
    cleanup_downloads_if_final_has_data: bool = False

    # Safer / more polite downloading
    max_retries: int = 6
    backoff_base_seconds: float = 2.0
    chunk_pause_seconds: float = 1.0
    verify_ssl: bool = True

    # Cache behavior
    keep_zip_files: bool = True
    keep_extracted_csvs: bool = True


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


@dataclass
class JoinConfig:
    dep_ts_col: str = "dep_ts_actual"
    arr_ts_col: str = "arr_ts_actual"

    origin_col: str = "Origin"
    dest_col: str = "Dest"
    tail_col: str = "Tail_Number"

    weather_station_col: str = "station"
    weather_ts_col: str = "valid_ts"

    dep_station_col: str = "dep_station"
    arr_station_col: str = "arr_station"

    weather_strategy: str = "backward"
    weather_tolerance: str = "2h"


@dataclass
class RouteFilterConfig:
    airports: list[str] | None = None
    airport_pairs: list[tuple[str, str]] | None = None
    origin_filter: list[str] | None = None
    dest_filter: list[str] | None = None


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
        "Tail_Number",
        "Origin",
        "OriginState",
        "OriginStateFips",
        "OriginWac",
        "Dest",
        "DestState",
        "DestStateFips",
        "DestWac",
        "CRSDepTime",
        "DepTime",
        "DepDelay",
        "DepDelayMinutes",
        "DepDel15",
        "DepartureDelayGroups",
        "CRSArrTime",
        "ArrTime",
        "ArrDelay",
        "ArrDelayMinutes",
        "ArrDel15",
        "ArrivalDelayGroups",
        "TaxiOut",
        "WheelsOff",
        "WheelsOn",
        "TaxiIn",
        "Cancelled",
        "CancellationCode",
        "Diverted",
        "CRSElapsedTime",
        "ActualElapsedTime",
        "AirTime",
        "Flights",
        "Distance",
        "DistanceGroup",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
        "crs_arr_date",
        "act_arr_date",
        "dep_ts_sched",
        "arr_ts_sched",
        "dep_ts_actual",
        "arr_ts_actual",
        "dep_drct",
        "dep_sknt",
        "dep_p01i",
        "dep_vsby",
        "dep_gust",
        "dep_wxcodes",
        "dep_ice_accretion_1hr",
        "dep_ice_accretion_3hr",
        "dep_ice_accretion_6hr",
        "dep_peak_wind_gust",
        "dep_peak_wind_drct",
        "dep_peak_wind_time",
        "dep_weather_severity",
        "dep_wx_intensity",
        "dep_wx_has_ra",
        "dep_wx_has_ts",
        "dep_wx_has_sn",
        "dep_wx_has_fg",
        "dep_wx_has_br",
        "dep_wx_has_hz",
        "arr_drct",
        "arr_sknt",
        "arr_p01i",
        "arr_vsby",
        "arr_gust",
        "arr_wxcodes",
        "arr_ice_accretion_1hr",
        "arr_ice_accretion_3hr",
        "arr_ice_accretion_6hr",
        "arr_peak_wind_gust",
        "arr_peak_wind_drct",
        "arr_peak_wind_time",
        "arr_weather_severity",
        "arr_wx_intensity",
        "arr_wx_has_ra",
        "arr_wx_has_ts",
        "arr_wx_has_sn",
        "arr_wx_has_fg",
        "arr_wx_has_br",
        "arr_wx_has_hz",
    ])


@dataclass
class PipelineConfig:
    years: list[int] = field(default_factory=lambda: [2019])
    months_by_year: dict[int, list[int]] | None = None

    airport_to_station: dict[str, str] = field(
        default_factory=lambda: {
            "BWI": "72406093721",
            "ATL": "72219013874",
            "EWR": "72502014734",
        }
    )

    route_filter: RouteFilterConfig = field(
        default_factory=lambda: RouteFilterConfig(
            airport_pairs=[("BWI", "EWR"), ("ATL", "BWI")]
        )
    )

    bts: BTSConfig = field(default_factory=BTSConfig)
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    joins: JoinConfig = field(default_factory=JoinConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)

    final_out_dir: Path = Path("data/final")
    write_monthly_joined: bool = True
    write_yearly_joined: bool = True

    use_cached_weather: bool = True
    use_cached_bts_months: bool = True
    use_cached_monthly_joined: bool = False

    run_weather_stage: bool = True
    run_bts_stage: bool = True
    run_join_stage: bool = True


CONFIG = PipelineConfig(
    years=[2015, 2016, 2017, 2018, 2019],
    months_by_year={
        2015: list(range(1, 13)),
        2016: list(range(1, 13)),
        2017: list(range(1, 13)),
        2018: list(range(1, 13)),
        2019: list(range(1, 13)),
    },
    airport_to_station={
        "BWI": "72406093721",
        "ATL": "72219013874",
        "EWR": "72502014734",
    },
    route_filter=RouteFilterConfig(
        airport_pairs=[("BWI", "EWR"), ("ATL", "BWI")]
    ),
)