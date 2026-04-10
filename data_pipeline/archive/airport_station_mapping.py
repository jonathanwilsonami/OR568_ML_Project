from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from config import MappingConfig
from utils import ensure_dir, make_retry_session, download_file_with_backoff


def load_or_download_reference_file(url: str, out_path: Path, timeout: int = 180) -> Path:
    ensure_dir(out_path.parent)
    if out_path.exists():
        print(f"Using cached reference file -> {out_path}")
        return out_path

    session = make_retry_session(max_retries=6, user_agent="OR568-Mapping/1.0")
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

    origins = bts_df.select(pl.col("Origin").alias("airport"))
    dests = bts_df.select(pl.col("Dest").alias("airport"))

    unique_airports = (
        pl.concat([origins, dests], how="vertical")
        .drop_nulls()
        .filter(pl.col("airport").str.len_chars() == 3)
        .unique()
        .sort("airport")
    )

    unique_airports.write_parquet(out_path)
    print(f"Wrote unique airports -> {out_path}")
    print(f"Unique airports discovered: {unique_airports.height:,}")
    return unique_airports


def build_two_hop_airport_sets(
    bts_df: pl.DataFrame,
    core_airports: list[str],
    cfg: MappingConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    core_set = set(core_airports)

    hop1 = (
        bts_df
        .filter(~pl.col("Origin").is_in(core_set) & pl.col("Dest").is_in(core_set))
        .select(pl.col("Origin").alias("airport"))
        .drop_nulls()
        .unique()
        .sort("airport")
    )

    hop1_set = set(hop1["airport"].to_list())

    hop2 = (
        bts_df
        .filter(~pl.col("Origin").is_in(core_set) & pl.col("Dest").is_in(hop1_set))
        .select(pl.col("Origin").alias("airport"))
        .drop_nulls()
        .unique()
        .sort("airport")
    )

    hop1.write_parquet(cfg.hop1_airports_path)
    hop2.write_parquet(cfg.hop2_airports_path)

    print(f"Wrote hop1 airports -> {cfg.hop1_airports_path}")
    print(f"Wrote hop2 airports -> {cfg.hop2_airports_path}")
    print(f"Hop1 airports: {hop1.height:,}")
    print(f"Hop2 airports: {hop2.height:,}")

    return hop1, hop2


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


def build_airport_to_station_and_timezone_mapping(
    unique_airports_df: pl.DataFrame,
    cfg: MappingConfig,
) -> tuple[dict[str, str], dict[str, str], pl.DataFrame]:
    ensure_dir(cfg.out_dir)
    ensure_dir(cfg.airports_cache_path.parent)
    ensure_dir(cfg.isd_history_cache_path.parent)

    airports_ref_path = load_or_download_reference_file(
        cfg.ourairports_url,
        cfg.airports_cache_path,
    )
    isd_ref_path = load_or_download_reference_file(
        cfg.isd_history_url,
        cfg.isd_history_cache_path,
    )

    airports_ref = _load_ourairports_airports(airports_ref_path)
    isd_history = _load_isd_history(isd_ref_path)

    iata_to_ref = (
        airports_ref
        .filter(pl.col("iata_code").is_not_null())
        .with_columns([
            pl.col("iata_code").str.to_uppercase().alias("IATA"),
            pl.when(pl.col("gps_code").is_not_null() & (pl.col("gps_code") != ""))
            .then(pl.col("gps_code"))
            .otherwise(pl.col("ident"))
            .cast(pl.Utf8)
            .str.to_uppercase()
            .alias("ICAO"),
            pl.col("timezone").cast(pl.Utf8).alias("timezone"),
        ])
        .select(["IATA", "ICAO", "timezone", "name", "type", "municipality", "iso_country"])
        .unique(subset=["IATA"], keep="first")
    )

    discovered = (
        unique_airports_df
        .with_columns(pl.col("airport").str.to_uppercase())
        .join(iata_to_ref, left_on="airport", right_on="IATA", how="left")
    )

    mapping_rows: list[dict[str, str | None]] = []

    for row in discovered.iter_rows(named=True):
        airport = row["airport"]
        icao = row.get("ICAO")
        timezone = row.get("timezone")

        station = None
        if icao is not None:
            station = _pick_best_station_for_icao(isd_history, icao, cfg.mapping_year)

        mapping_rows.append(
            {
                "airport": airport,
                "icao": icao,
                "timezone": timezone,
                "station": station,
                "airport_name": row.get("name"),
                "airport_type": row.get("type"),
                "municipality": row.get("municipality"),
                "iso_country": row.get("iso_country"),
            }
        )

    mapping_df = pl.DataFrame(mapping_rows).sort("airport")

    mapped_station_df = mapping_df.filter(pl.col("station").is_not_null())
    mapped_timezone_df = mapping_df.filter(pl.col("timezone").is_not_null())
    unmapped_df = mapping_df.filter(pl.col("station").is_null())

    airport_to_station = {
        row["airport"]: row["station"]
        for row in mapped_station_df.select(["airport", "station"]).iter_rows(named=True)
    }

    airport_to_timezone = {
        row["airport"]: row["timezone"]
        for row in mapped_timezone_df.select(["airport", "timezone"]).iter_rows(named=True)
    }

    with open(cfg.airport_station_json_path, "w", encoding="utf-8") as f:
        json.dump(airport_to_station, f, indent=2, sort_keys=True)

    with open(cfg.airport_timezone_json_path, "w", encoding="utf-8") as f:
        json.dump(airport_to_timezone, f, indent=2, sort_keys=True)

    mapped_station_df.write_csv(cfg.airport_station_csv_path)
    mapping_df.select(["airport", "timezone"]).write_csv(cfg.airport_timezone_csv_path)
    unmapped_df.write_parquet(cfg.unmapped_airports_path)

    print(f"Wrote airport_to_station JSON -> {cfg.airport_station_json_path}")
    print(f"Wrote airport_to_timezone JSON -> {cfg.airport_timezone_json_path}")

    return airport_to_station, airport_to_timezone, unmapped_df