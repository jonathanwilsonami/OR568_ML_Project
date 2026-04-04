from pathlib import Path
import io
import re
import zipfile
import urllib.request
import urllib.error
import polars as pl
import csv

FAA_ZIP_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"


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


def _find_shared_notebooks_dir() -> Path:
    module_file = Path(__file__).resolve()
    current = module_file.parent

    while current.name != "shared-notebooks":
        if current.parent == current:
            raise RuntimeError("Could not locate shared-notebooks directory")
        current = current.parent

    return current


def _normalize_tail_expr(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_uppercase()
        .str.replace_all(r"[^A-Z0-9]", "")
    )


def _download_file(url: str, destination: Path) -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/zip,application/octet-stream,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.faa.gov/",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response, open(destination, "wb") as f:
            f.write(response.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            "FAA download failed. If the FAA site blocks the request, "
            "download ReleasableAircraft.zip manually and place it at:\n"
            f"{destination}\n\n"
            f"Original error: HTTP {e.code} - {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"FAA download failed due to a network error: {e}"
        ) from e




def _read_text_file_from_zip(zip_path: Path, member_name: str) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name) as f:
            text = f.read().decode("utf-8", errors="replace")

    lines = text.splitlines()
    reader = csv.reader(lines)

    rows = list(reader)
    if not rows:
        raise RuntimeError(f"{member_name} is empty")

    header = rows[0]
    data = rows[1:]

    # normalize row lengths
    width = len(header)
    cleaned_data = []
    for row in data:
        if len(row) < width:
            row = row + [None] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        cleaned_data.append(row)

    return pl.DataFrame(cleaned_data, schema=header, orient="row")


def _build_faa_registry_from_zip(zip_path: Path) -> pl.DataFrame:
    master = _read_text_file_from_zip(zip_path, "MASTER.txt")
    acftref = _read_text_file_from_zip(zip_path, "ACFTREF.txt")

    master.columns = _clean_names(master.columns)
    acftref.columns = _clean_names(acftref.columns)

    # Clean core FAA tables
    master = master.with_columns([
        _normalize_tail_expr("n_number").alias("tail_number"),
        pl.col("year_mfr").cast(pl.Int32, strict=False),
        pl.col("mfr_mdl_code").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("type_registrant").cast(pl.Int32, strict=False),
        pl.col("status_code").cast(pl.Utf8, strict=False).str.strip_chars(),
    ])

    acftref = acftref.with_columns([
        pl.col("code").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("type_acft").cast(pl.Int32, strict=False),
        pl.col("type_eng").cast(pl.Int32, strict=False),
        pl.col("ac_cat").cast(pl.Int32, strict=False),
        pl.col("build_cert_ind").cast(pl.Utf8, strict=False).str.strip_chars(),
        pl.col("no_eng").cast(pl.Int32, strict=False),
        pl.col("no_seats").cast(pl.Int32, strict=False),
        pl.col("ac_weight").cast(pl.Int32, strict=False),
        pl.col("speed").cast(pl.Int32, strict=False),
        pl.col("tc_data_sheet").cast(pl.Utf8, strict=False),
        pl.col("tc_data_holder").cast(pl.Utf8, strict=False),
    ])

    faa = (
        master
        .join(
            acftref,
            left_on="mfr_mdl_code",
            right_on="code",
            how="left",
            suffix="_acftref",
        )
        .select([
            "tail_number",
            "serial_number",
            "mfr_mdl_code",
            "year_mfr",
            "type_registrant",
            "name",
            "street",
            "street2",
            "city",
            "state",
            "zip_code",
            "region",
            "county",
            "country",
            "last_action_date",
            "cert_issue_date",
            "status_code",
            "mode_s_code",
            "fract_owner",
            "air_worth_date",
            "other_names_1",
            "expiration_date",

            # ACFTREF fields
            "mfr",
            "model",
            "type_acft",
            "type_eng",
            "ac_cat",
            "build_cert_ind",
            "no_eng",
            "no_seats",
            "ac_weight",
            "speed",
            "tc_data_sheet",
            "tc_data_holder",
        ])
    )

    return faa


def _map_registrant_type() -> pl.Expr:
    return (
        pl.when(pl.col("type_registrant") == 1).then(pl.lit("individual"))
        .when(pl.col("type_registrant") == 2).then(pl.lit("partnership"))
        .when(pl.col("type_registrant") == 3).then(pl.lit("corporation"))
        .when(pl.col("type_registrant") == 4).then(pl.lit("co_owned"))
        .when(pl.col("type_registrant") == 5).then(pl.lit("government"))
        .when(pl.col("type_registrant") == 7).then(pl.lit("llc"))
        .when(pl.col("type_registrant") == 8).then(pl.lit("non_citizen_corporation"))
        .when(pl.col("type_registrant") == 9).then(pl.lit("non_citizen_co_owned"))
        .otherwise(pl.lit("unknown"))
        .alias("registrant_type_class")
    )


def _map_aircraft_type() -> pl.Expr:
    return (
        pl.when(pl.col("type_acft") == 1).then(pl.lit("glider"))
        .when(pl.col("type_acft") == 2).then(pl.lit("balloon"))
        .when(pl.col("type_acft") == 3).then(pl.lit("blimp_dirigible"))
        .when(pl.col("type_acft") == 4).then(pl.lit("fixed_wing_single_engine"))
        .when(pl.col("type_acft") == 5).then(pl.lit("fixed_wing_multi_engine"))
        .when(pl.col("type_acft") == 6).then(pl.lit("rotorcraft"))
        .when(pl.col("type_acft") == 7).then(pl.lit("weight_shift_control"))
        .when(pl.col("type_acft") == 8).then(pl.lit("powered_parachute"))
        .when(pl.col("type_acft") == 9).then(pl.lit("gyroplane"))
        .when(pl.col("type_acft") == 10).then(pl.lit("hybrid_lift"))
        .otherwise(pl.lit("unknown"))
        .alias("aircraft_type_class")
    )


def _map_engine_type() -> pl.Expr:
    return (
        pl.when(pl.col("type_eng") == 0).then(pl.lit("none"))
        .when(pl.col("type_eng") == 1).then(pl.lit("reciprocating"))
        .when(pl.col("type_eng") == 2).then(pl.lit("turbo_prop"))
        .when(pl.col("type_eng") == 3).then(pl.lit("turbo_shaft"))
        .when(pl.col("type_eng") == 4).then(pl.lit("turbo_jet"))
        .when(pl.col("type_eng") == 5).then(pl.lit("turbo_fan"))
        .when(pl.col("type_eng") == 6).then(pl.lit("ramjet"))
        .when(pl.col("type_eng") == 7).then(pl.lit("two_cycle"))
        .when(pl.col("type_eng") == 8).then(pl.lit("four_cycle"))
        .when(pl.col("type_eng") == 9).then(pl.lit("unknown_engine"))
        .when(pl.col("type_eng") == 10).then(pl.lit("electric"))
        .when(pl.col("type_eng") == 11).then(pl.lit("rotary"))
        .otherwise(pl.lit("unknown"))
        .alias("engine_class")
    )


def _map_weight_class() -> pl.Expr:
    return (
        pl.when(pl.col("ac_weight") == 1).then(pl.lit("up_to_12499"))
        .when(pl.col("ac_weight") == 2).then(pl.lit("12500_to_19999"))
        .when(pl.col("ac_weight") == 3).then(pl.lit("20000_and_over"))
        .otherwise(pl.lit("unknown"))
        .alias("aircraft_weight_class")
    )


def engineer_faa_features(faa_df: pl.DataFrame) -> pl.DataFrame:
    df = (
        faa_df
        .with_columns([
            pl.col("type_registrant").cast(pl.Int32, strict=False),
            pl.col("type_acft").cast(pl.Int32, strict=False),
            pl.col("type_eng").cast(pl.Int32, strict=False),
            pl.col("ac_weight").cast(pl.Int32, strict=False),
            pl.col("no_eng").cast(pl.Int32, strict=False),
            pl.col("no_seats").cast(pl.Int32, strict=False),
            pl.col("speed").cast(pl.Int32, strict=False),
            pl.col("year_mfr").cast(pl.Int32, strict=False),
        ])
        .with_columns([
            _map_registrant_type(),
            _map_aircraft_type(),
            _map_engine_type(),
            _map_weight_class(),

            pl.when(pl.col("no_eng").is_null()).then(None)
            .when(pl.col("no_eng") == 1).then(pl.lit("single_engine"))
            .when(pl.col("no_eng") == 2).then(pl.lit("twin_engine"))
            .when(pl.col("no_eng") >= 3).then(pl.lit("multi_engine_3_plus"))
            .otherwise(pl.lit("unknown"))
            .alias("engine_count_class"),

            pl.when(pl.col("no_seats").is_null()).then(None)
            .when(pl.col("no_seats") <= 6).then(pl.lit("very_small"))
            .when(pl.col("no_seats") <= 12).then(pl.lit("small"))
            .when(pl.col("no_seats") <= 50).then(pl.lit("regional"))
            .when(pl.col("no_seats") <= 150).then(pl.lit("narrowbody_like"))
            .when(pl.col("no_seats") > 150).then(pl.lit("large"))
            .otherwise(pl.lit("unknown"))
            .alias("seat_class"),

            pl.when(pl.col("speed").is_null()).then(None)
            .when(pl.col("speed") < 200).then(pl.lit("slow"))
            .when(pl.col("speed") < 400).then(pl.lit("medium"))
            .when(pl.col("speed") >= 400).then(pl.lit("fast"))
            .otherwise(pl.lit("unknown"))
            .alias("speed_class"),

            pl.col("mfr").cast(pl.Utf8, strict=False).str.to_uppercase().alias("mfr_upper"),
            pl.col("model").cast(pl.Utf8, strict=False).str.to_uppercase().alias("model_upper"),
        ])
        .with_columns([
            pl.col("mfr_upper").str.contains("BOEING", literal=True).fill_null(False).cast(pl.Int8).alias("is_boeing"),
            pl.col("mfr_upper").str.contains("AIRBUS", literal=True).fill_null(False).cast(pl.Int8).alias("is_airbus"),
            pl.col("mfr_upper").str.contains("EMBRAER", literal=True).fill_null(False).cast(pl.Int8).alias("is_embraer"),
            pl.col("mfr_upper").str.contains("BOMBARDIER", literal=True).fill_null(False).cast(pl.Int8).alias("is_bombardier"),
            pl.col("mfr_upper").str.contains("MCDONNELL DOUGLAS", literal=True).fill_null(False).cast(pl.Int8).alias("is_mcdonnell_douglas"),
            pl.col("country").is_not_null().cast(pl.Int8).alias("has_country"),
            pl.col("state").is_not_null().cast(pl.Int8).alias("has_state"),
            pl.col("mode_s_code").is_not_null().cast(pl.Int8).alias("has_mode_s_code"),
            pl.col("tc_data_sheet").is_not_null().cast(pl.Int8).alias("has_type_certificate_data"),
        ])
        .drop(["mfr_upper", "model_upper"])
    )

    return df


def load_faa_registry(
    url: str = FAA_ZIP_URL,
    lazy: bool = True,
    refresh: bool = False,
    download_if_missing: bool = True,
):
    shared_notebooks_dir = _find_shared_notebooks_dir()
    data_dir = shared_notebooks_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "ReleasableAircraft.zip"
    faa_parquet = data_dir / "faa_registry_enriched.parquet"

    if faa_parquet.exists() and not refresh:
        print(f"Using cached FAA parquet: {faa_parquet}")
        return pl.scan_parquet(faa_parquet) if lazy else pl.read_parquet(faa_parquet)

    if not zip_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"FAA ZIP not found at {zip_path} and download_if_missing=False"
            )
        print(f"Downloading FAA registry ZIP to: {zip_path}")
        _download_file(url, zip_path)
    else:
        print(f"Using cached FAA ZIP: {zip_path}")

    faa_raw = _build_faa_registry_from_zip(zip_path)
    faa_enriched = engineer_faa_features(faa_raw)

    faa_enriched.write_parquet(faa_parquet)
    print(f"Wrote FAA enriched parquet: {faa_parquet}")

    return pl.scan_parquet(faa_parquet) if lazy else pl.read_parquet(faa_parquet)


def join_faa_registry(
    flights,
    faa=None,
    flight_tail_col: str = "tail_number",
    faa_tail_col: str = "tail_number",
    how: str = "left",
):
    """
    Join FAA registry enrichment to a flight DataFrame or LazyFrame.

    flights: Polars DataFrame or LazyFrame
    faa: optional Polars DataFrame or LazyFrame. If None, loads from cache.
    """

    if faa is None:
        faa = load_faa_registry(lazy=isinstance(flights, pl.LazyFrame))

    faa_feature_cols = [
        faa_tail_col,
        "serial_number",
        "mfr_mdl_code",
        "year_mfr",
        "type_registrant",
        "name",
        "city",
        "state",
        "country",
        "status_code",
        "mode_s_code",
        "mfr",
        "model",
        "type_acft",
        "type_eng",
        "ac_cat",
        "build_cert_ind",
        "no_eng",
        "no_seats",
        "ac_weight",
        "speed",
        "tc_data_sheet",
        "tc_data_holder",
        "registrant_type_class",
        "aircraft_type_class",
        "engine_class",
        "aircraft_weight_class",
        "engine_count_class",
        "seat_class",
        "speed_class",
        "is_boeing",
        "is_airbus",
        "is_embraer",
        "is_bombardier",
        "is_mcdonnell_douglas",
        "has_country",
        "has_state",
        "has_mode_s_code",
        "has_type_certificate_data",
    ]

    if isinstance(flights, pl.LazyFrame):
        flights_norm = flights.with_columns([
            _normalize_tail_expr(flight_tail_col).alias(flight_tail_col)
        ])
        faa_norm = faa.with_columns([
            _normalize_tail_expr(faa_tail_col).alias(faa_tail_col)
        ])
        return flights_norm.join(
            faa_norm.select(faa_feature_cols),
            left_on=flight_tail_col,
            right_on=faa_tail_col,
            how=how,
        )

    flights_norm = flights.with_columns([
        _normalize_tail_expr(flight_tail_col).alias(flight_tail_col)
    ])
    faa_norm = faa.with_columns([
        _normalize_tail_expr(faa_tail_col).alias(faa_tail_col)
    ])
    return flights_norm.join(
        faa_norm.select(faa_feature_cols),
        left_on=flight_tail_col,
        right_on=faa_tail_col,
        how=how,
    )


def add_dynamic_aircraft_features(
    flights,
    flight_date_col: str = "flight_date",
    year_mfr_col: str = "year_mfr",
):
    """
    Add aircraft features that depend on the year of the flight.
    Works with either DataFrame or LazyFrame.
    """

    return (
        flights
        .with_columns([
            pl.col(flight_date_col).cast(pl.Date, strict=False).dt.year().alias("flight_year")
        ])
        .with_columns([
            pl.when(
                pl.col("flight_year").is_null() | pl.col(year_mfr_col).is_null()
            ).then(None)
            .otherwise(pl.col("flight_year") - pl.col(year_mfr_col))
            .alias("aircraft_age"),

            pl.when(
                pl.col("flight_year").is_null() | pl.col(year_mfr_col).is_null()
            ).then(None)
            .when(pl.col(year_mfr_col) > pl.col("flight_year")).then(1)
            .otherwise(0)
            .alias("year_mfr_after_flight_flag"),
        ])
        .with_columns([
            pl.when(pl.col("aircraft_age").is_null()).then(None)
            .when(pl.col("aircraft_age") < 0).then(pl.lit("invalid_future_mfr"))
            .when(pl.col("aircraft_age") <= 5).then(pl.lit("new"))
            .when(pl.col("aircraft_age") <= 10).then(pl.lit("modern"))
            .when(pl.col("aircraft_age") <= 20).then(pl.lit("mid_life"))
            .when(pl.col("aircraft_age") <= 30).then(pl.lit("old"))
            .otherwise(pl.lit("very_old"))
            .alias("age_class"),

            pl.when(pl.col("aircraft_age").is_null()).then(None)
            .when(pl.col("aircraft_age") <= 3).then(pl.lit("0_3"))
            .when(pl.col("aircraft_age") <= 7).then(pl.lit("4_7"))
            .when(pl.col("aircraft_age") <= 12).then(pl.lit("8_12"))
            .when(pl.col("aircraft_age") <= 20).then(pl.lit("13_20"))
            .when(pl.col("aircraft_age") <= 30).then(pl.lit("21_30"))
            .otherwise(pl.lit("31_plus"))
            .alias("age_band"),

            pl.when(pl.col("aircraft_age").is_null()).then(None)
            .when(pl.col("aircraft_age") >= 20).then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_older_aircraft"),
        ])
        .with_columns([
            pl.when(
                pl.col("turnaround_minutes").is_null() | pl.col("aircraft_age").is_null()
            ).then(None)
            .otherwise(pl.col("turnaround_minutes") / (pl.col("aircraft_age") + 1))
            .alias("turnaround_per_aircraft_age"),

            pl.when(
                pl.col("prev_arr_delay").is_null() | pl.col("aircraft_age").is_null()
            ).then(None)
            .otherwise(pl.col("prev_arr_delay") * pl.col("aircraft_age"))
            .alias("prev_arr_delay_x_aircraft_age"),

            pl.when(
                pl.col("aircraft_leg_number_day").is_null() | pl.col("aircraft_age").is_null()
            ).then(None)
            .otherwise(pl.col("aircraft_leg_number_day") * pl.col("aircraft_age"))
            .alias("daily_utilization_x_age"),
        ])
    )