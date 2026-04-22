from __future__ import annotations

"""
Fetches and caches FAA aircraft registry data (MASTER + DEREG),
cleans it, and exposes a join function for use in build_modeling_table.

All logic is self-contained so the rest of the pipeline only needs
a one-line call: enrich_with_aircraft_features(lf, config)
"""

import io
import logging
import zipfile
from pathlib import Path

import polars as pl
import requests

logger = logging.getLogger(__name__)

FAA_ZIP_URL = "https://registry.faa.gov/database/ReleasableAircraft.zip"

# Columns we keep from MASTER/DEREG after cleaning
_REGISTRY_KEEP = [
    "n_number",       # tail number (no leading N)
    "mfr_mdl_code",   # joins to ACFTREF
    "year_mfr",       # year manufactured
    "type_acft",      # 1=fixed-wing single, 4=fixed-wing multi, 6=heli, etc.
    "no_eng",         # number of engines
    "no_seats",       # max seats (from ACFTREF join)
    "ac_weight",      # weight class: CLASS 1-4 or A/B/C/D
]

# ACFTREF columns
_ACFTREF_KEEP = [
    "code",
    "mfr",
    "model",
    "type_acft",
    "no_eng",
    "no_seats",
    "ac_weight",
]


def _download_faa_zip(cache_path: Path) -> bytes:
    if cache_path.exists():
        logger.info("[FAA] Using cached zip at %s", cache_path)
        return cache_path.read_bytes()

    logger.info("[FAA] Downloading FAA registry from %s", FAA_ZIP_URL)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(FAA_ZIP_URL, headers=headers, timeout=120)
    resp.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)
    logger.info("[FAA] Saved zip to %s (%.1f MB)", cache_path, len(resp.content) / 1e6)
    return resp.content


def _read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pl.DataFrame:
    import pandas as pd
    with zf.open(name) as f:
        raw = f.read().decode("latin-1")
    pdf = pd.read_csv(
        io.StringIO(raw),
        dtype=str,
        keep_default_na=False,
        on_bad_lines="skip",
        quoting=3,          # csv.QUOTE_NONE — ignore quoting entirely
    )
    # Strip whitespace from all string values (FAA pads fields heavily)
    pdf = pdf.apply(lambda col: col.str.strip() if col.dtype == object else col)
    return pl.from_pandas(pdf)


def _normalize_cols(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({c: c.strip().lower().replace(" ", "_") for c in df.columns})


def _clean_n_number(df: pl.DataFrame, col: str = "n_number") -> pl.DataFrame:
    """Strip whitespace; the FAA omits the leading 'N' in the file."""
    return df.with_columns(pl.col(col).str.strip_chars().alias(col))


def _build_tail_lookup(zf: zipfile.ZipFile) -> pl.DataFrame:
    master  = _normalize_cols(_read_csv_from_zip(zf, "MASTER.txt"))
    dereg   = _normalize_cols(_read_csv_from_zip(zf, "DEREG.txt"))
    acftref = _normalize_cols(_read_csv_from_zip(zf, "ACFTREF.txt"))

    # Strip BOM and normalize hyphens→underscores on all three frames
    def fix_cols(df: pl.DataFrame) -> pl.DataFrame:
        return df.rename({
            c: c.replace("ï»¿", "").replace("-", "_")
            for c in df.columns
        })

    master  = fix_cols(master)
    dereg   = fix_cols(dereg)
    acftref = fix_cols(acftref)

    logger.info("[FAA] ACFTREF columns (fixed): %s", acftref.columns)
    logger.info("[FAA] MASTER columns (fixed): %s", master.columns)
    logger.info("[FAA] DEREG columns (fixed): %s", dereg.columns)

    # ACFTREF first column is the type code key
    acftref_join_col = acftref.columns[0]
    acftref = acftref.rename({acftref_join_col: "mfr_mdl_code"})

    acftref_slim = acftref.select(
        [c for c in ["mfr_mdl_code", "mfr", "model", "no_seats", "ac_weight", "no_eng"]
         if c in acftref.columns]
    )
    

    # MASTER uses type_aircraft, DEREG may differ — normalize to type_acft
    for df_name, df in [("master", master), ("dereg", dereg)]:
        if "type_aircraft" in df.columns and "type_acft" not in df.columns:
            if df_name == "master":
                master = master.rename({"type_aircraft": "type_acft"})
            else:
                dereg = dereg.rename({"type_aircraft": "type_acft"})

    keep_cols = ["n_number", "mfr_mdl_code", "year_mfr", "type_acft", "no_eng"]

    def slim(df: pl.DataFrame) -> pl.DataFrame:
        present = [c for c in keep_cols if c in df.columns]
        return df.select(present)

    combined = pl.concat([slim(master), slim(dereg)], how="diagonal_relaxed")
    combined = combined.unique(subset=["n_number"], keep="first")
    combined = combined.join(acftref_slim, on="mfr_mdl_code", how="left")

    for num_col in ["year_mfr", "no_eng", "no_seats"]:
        if num_col in combined.columns:
            combined = combined.with_columns(
                pl.col(num_col).cast(pl.Int32, strict=False)
            )

    # Normalize n_number and rename to tail_number for the join
    combined = combined.with_columns(
        pl.col("n_number").str.strip_chars().str.replace(r"^N", "").alias("n_number")
    )
    combined = combined.rename({"n_number": "tail_number"})

    logger.info(
        "[FAA] tail lookup built | rows=%s cols=%s",
        f"{combined.height:,}",
        combined.columns,
    )
    return combined


def load_aircraft_lookup(cache_path: Path) -> pl.DataFrame:
    """Public entry point: returns the cleaned tail-number lookup table."""
    raw_bytes = _download_faa_zip(cache_path)
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        return _build_tail_lookup(zf)


def enrich_with_aircraft_features(
    lf: pl.LazyFrame,
    cache_path: Path,
) -> pl.LazyFrame:
    lookup = load_aircraft_lookup(cache_path)

    # Do NOT prefix tail_number — it's the join key, not a feature column
    # Only rename the actual feature columns
    rename_map = {
        c: f"aircraft_{c}"
        for c in lookup.columns
        if c != "tail_number"
    }
    lookup = lookup.rename(rename_map)

    lookup_lf = lookup.lazy()

    schema = lf.collect_schema().names()

    if "tail_number" not in schema:
        logger.warning("[FAA] tail_number column not found — skipping aircraft enrichment")
        return lf

    # Normalize flight-side tail number: strip whitespace and leading N
    lf = lf.with_columns(
        pl.col("tail_number")
        .str.strip_chars()
        .str.replace(r"^N", "")
        .alias("_faa_tail")
    )

    lf = lf.join(lookup_lf, left_on="_faa_tail", right_on="tail_number", how="left")
    lf = lf.drop("_faa_tail")

    # Derive aircraft age if year column is present
    schema_after = lf.collect_schema().names()
    if "year" in schema_after and "aircraft_year_mfr" in schema_after:
        lf = lf.with_columns(
            (pl.col("year") - pl.col("aircraft_year_mfr"))
            .clip(0, 60)
            .alias("aircraft_age")
        )

    return lf