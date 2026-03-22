from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from config import ReferenceConfig
from utils import ensure_dir, make_retry_session, download_file_with_backoff


def load_or_download_reference_file(url: str, out_path: Path, timeout: int = 180) -> Path:
    ensure_dir(out_path.parent)
    if out_path.exists():
        print(f"Using cached reference file -> {out_path}")
        return out_path

    session = make_retry_session(max_retries=6, user_agent="OR568-Reference/1.0")
    download_file_with_backoff(
        session=session,
        url=url,
        out_path=out_path,
        timeout=timeout,
        verify_ssl=True,
        max_retries=6,
        backoff_base_seconds=2.0,
    )
    return out_path


def extract_unique_airports_from_bts(
    bts_df: pl.DataFrame,
    out_path: Path,
) -> pl.DataFrame:
    ensure_dir(out_path.parent)

    unique_airports = (
        pl.concat(
            [
                bts_df.select(pl.col("Origin").alias("airport")),
                bts_df.select(pl.col("Dest").alias("airport")),
            ],
            how="vertical",
        )
        .drop_nulls()
        .filter(pl.col("airport").str.len_chars() == 3)
        .unique()
        .sort("airport")
    )

    unique_airports.write_parquet(out_path)
    print(f"Wrote unique airports -> {out_path}")
    print(f"Unique airports discovered: {unique_airports.height:,}")
    return unique_airports


def _load_ourairports_airports(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=5000, ignore_errors=True)

    keep = [
        "ident",
        "type",
        "name",
        "iata_code",
        "gps_code",
        "municipality",
        "iso_country",
        "scheduled_service",
        "latitude_deg",
        "longitude_deg",
        "timezone",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df.select(keep)

    if "iata_code" in df.columns:
        df = df.filter(pl.col("iata_code").is_not_null() & (pl.col("iata_code") != ""))

    return df


def _load_isd_history(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path, infer_schema_length=5000, ignore_errors=True)

    rename_map = {}
    for c in df.columns:
        c_norm = c.strip().upper().replace(" ", "_")
        if c_norm != c:
            rename_map[c] = c_norm
    if rename_map:
        df = df.rename(rename_map)

    needed = [
        "USAF",
        "WBAN",
        "STATION_NAME",
        "CTRY",
        "STATE",
        "ICAO",
        "LAT",
        "LON",
        "BEGIN",
        "END",
    ]
    needed = [c for c in needed if c in df.columns]
    df = df.select(needed)

    return df.with_columns([
        pl.col("USAF").cast(pl.Utf8).str.strip_chars().alias("USAF"),
        pl.col("WBAN").cast(pl.Utf8).str.strip_chars().alias("WBAN"),
        pl.col("ICAO").cast(pl.Utf8).str.strip_chars().alias("ICAO"),
        pl.col("STATION_NAME").cast(pl.Utf8).str.strip_chars().alias("STATION_NAME"),
        pl.col("BEGIN").cast(pl.Utf8).str.strip_chars().alias("BEGIN"),
        pl.col("END").cast(pl.Utf8).str.strip_chars().alias("END"),
    ]).with_columns([
        (pl.col("USAF").str.zfill(6) + pl.col("WBAN").str.zfill(5)).alias("station"),
    ])


def _timezone_from_lon_lat(lon: float | None, lat: float | None) -> str | None:
    if lon is None:
        return None

    try:
        lon = float(lon)
    except (TypeError, ValueError):
        return None

    try:
        lat = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat = None

    if lon < -154:
        return "Pacific/Honolulu"
    if lon <= -130:
        return "America/Anchorage"
    if lon <= -114:
        return "America/Los_Angeles"
    if lon <= -101:
        if lat is not None and 31 <= lat <= 38 and -115 <= lon <= -109:
            return "America/Phoenix"
        return "America/Denver"
    if lon <= -85:
        return "America/Chicago"
    return "America/New_York"


def _pick_best_station_for_icao(
    isd_df: pl.DataFrame,
    icao: str,
    mapping_year: int,
) -> str | None:
    begin_floor = f"{mapping_year}0101"
    end_floor = f"{mapping_year}1231"

    candidates = (
        isd_df
        .filter(pl.col("ICAO") == icao)
        .with_columns([
            (pl.col("BEGIN") <= end_floor).alias("begins_before_end_of_year"),
            (pl.col("END") >= begin_floor).alias("ends_after_start_of_year"),
        ])
        .filter(pl.col("begins_before_end_of_year") & pl.col("ends_after_start_of_year"))
    )

    if candidates.height == 0:
        return None

    candidates = candidates.with_columns([
        pl.when(pl.col("STATION_NAME").str.to_lowercase().str.contains("intl"))
        .then(2)
        .when(pl.col("STATION_NAME").str.to_lowercase().str.contains("airport"))
        .then(1)
        .otherwise(0)
        .alias("name_score"),
        pl.col("END").alias("end_score"),
    ])

    best = candidates.sort(
        by=["name_score", "end_score", "station"],
        descending=[True, True, False],
    ).head(1)

    return best["station"].item()


def build_reference_dimensions(
    unique_airports_df: pl.DataFrame,
    cfg: ReferenceConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, str], dict[str, str], dict[str, str]]:
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.airports_cache_path.parent)
    ensure_dir(cfg.isd_history_cache_path.parent)

    airports_ref_path = load_or_download_reference_file(cfg.ourairports_url, cfg.airports_cache_path)
    isd_ref_path = load_or_download_reference_file(cfg.isd_history_url, cfg.isd_history_cache_path)

    airports_ref = _load_ourairports_airports(airports_ref_path)
    isd_history = _load_isd_history(isd_ref_path)

    exprs = [
        pl.col("iata_code").str.to_uppercase().alias("airport"),
        pl.when(pl.col("gps_code").is_not_null() & (pl.col("gps_code") != ""))
        .then(pl.col("gps_code"))
        .otherwise(pl.col("ident"))
        .cast(pl.Utf8)
        .str.to_uppercase()
        .alias("icao"),
        pl.col("name").alias("airport_name"),
        pl.col("type").alias("airport_type"),
        pl.col("municipality"),
        pl.col("iso_country"),
    ]

    if "latitude_deg" in airports_ref.columns:
        exprs.append(pl.col("latitude_deg").cast(pl.Float64, strict=False).alias("latitude_deg"))
    if "longitude_deg" in airports_ref.columns:
        exprs.append(pl.col("longitude_deg").cast(pl.Float64, strict=False).alias("longitude_deg"))
    if "timezone" in airports_ref.columns:
        exprs.append(pl.col("timezone").cast(pl.Utf8).alias("timezone"))

    airport_ref = (
        airports_ref
        .filter(pl.col("iata_code").is_not_null())
        .select(exprs)
        .unique(subset=["airport"], keep="first")
    )

    airport_dim = (
        unique_airports_df
        .join(airport_ref, on="airport", how="left")
    )

    # Only use source timezone column if it actually exists
    if "timezone" in airport_dim.columns:
        airport_dim = airport_dim.with_columns([
            pl.when(pl.col("timezone").is_not_null())
            .then(pl.col("timezone"))
            .otherwise(
                pl.struct(["longitude_deg", "latitude_deg"]).map_elements(
                    lambda x: _timezone_from_lon_lat(x["longitude_deg"], x["latitude_deg"]),
                    return_dtype=pl.Utf8,
                )
            )
            .alias("timezone")
        ])
    else:
        airport_dim = airport_dim.with_columns([
            pl.struct(["longitude_deg", "latitude_deg"]).map_elements(
                lambda x: _timezone_from_lon_lat(x["longitude_deg"], x["latitude_deg"]),
                return_dtype=pl.Utf8,
            ).alias("timezone")
        ])

    airport_dim = airport_dim.sort("airport")

    station_rows: list[dict[str, str | float | None]] = []
    for row in airport_dim.iter_rows(named=True):
        airport = row["airport"]
        icao = row.get("icao")
        station = _pick_best_station_for_icao(isd_history, icao, cfg.mapping_year) if icao else None

        station_rows.append(
            {
                "airport": airport,
                "icao": icao,
                "station": station,
                "airport_timezone": row.get("timezone"),
            }
        )

    airport_station_bridge = pl.DataFrame(station_rows).sort("airport")

    station_dim = (
        airport_station_bridge
        .filter(pl.col("station").is_not_null())
        .join(
            isd_history.select(["station", "STATION_NAME", "ICAO", "LAT", "LON", "BEGIN", "END"]).unique(),
            on="station",
            how="left",
        )
        .rename({
            "STATION_NAME": "station_name",
            "ICAO": "station_icao",
            "LAT": "station_lat",
            "LON": "station_lon",
            "BEGIN": "station_begin",
            "END": "station_end",
        })
        .with_columns([
            pl.col("station_lat").cast(pl.Float64, strict=False).alias("station_lat"),
            pl.col("station_lon").cast(pl.Float64, strict=False).alias("station_lon"),
        ])
        .with_columns([
            pl.struct(["station_lon", "station_lat"]).map_elements(
                lambda x: _timezone_from_lon_lat(x["station_lon"], x["station_lat"]),
                return_dtype=pl.Utf8,
            ).alias("station_timezone")
        ])
        .unique(subset=["station"])
        .sort("station")
    )

    airport_dim.write_parquet(cfg.airport_dim_path)
    station_dim.write_parquet(cfg.station_dim_path)
    airport_station_bridge.write_parquet(cfg.airport_station_bridge_path)

    unmapped = airport_station_bridge.filter(pl.col("station").is_null())
    unmapped.write_parquet(cfg.unmapped_airports_path)

    airport_to_station = {
        row["airport"]: row["station"]
        for row in airport_station_bridge.filter(pl.col("station").is_not_null())
        .select(["airport", "station"]).iter_rows(named=True)
    }

    airport_to_timezone = {
        row["airport"]: row["timezone"]
        for row in airport_dim.filter(pl.col("timezone").is_not_null())
        .select(["airport", "timezone"]).iter_rows(named=True)
    }

    station_to_timezone = {
        row["station"]: row["station_timezone"]
        for row in station_dim.filter(pl.col("station_timezone").is_not_null())
        .select(["station", "station_timezone"]).iter_rows(named=True)
    }

    with open(cfg.airport_station_json_path, "w", encoding="utf-8") as f:
        json.dump(airport_to_station, f, indent=2, sort_keys=True)

    with open(cfg.airport_timezone_json_path, "w", encoding="utf-8") as f:
        json.dump(airport_to_timezone, f, indent=2, sort_keys=True)

    print(f"Wrote airport_dim -> {cfg.airport_dim_path}")
    print(f"Wrote station_dim -> {cfg.station_dim_path}")
    print(f"Wrote airport_station_bridge -> {cfg.airport_station_bridge_path}")
    print(f"Mapped stations: {len(airport_to_station):,}")
    print(f"Mapped airport timezones: {len(airport_to_timezone):,}")
    print(f"Mapped station timezones: {len(station_to_timezone):,}")
    print(f"Unmapped stations: {unmapped.height:,}")

    return airport_dim, station_dim, airport_station_bridge, airport_to_station, airport_to_timezone, station_to_timezone