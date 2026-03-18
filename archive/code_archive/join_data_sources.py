import zipfile
from pathlib import Path
import requests
import polars as pl

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
YEAR = 2019

# Choose ONE of these modes:
MODE = "year"        # "month" or "year"
MONTH = 1            # used only if MODE == "month"

# If MODE == "year", set this:
MONTHS = list(range(1, 13))  # 1..12 (full year)

OUT_DIR = Path("bts_ontime_downloads")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- ROUTE FILTER: ONLY BWI <-> JFK -------------------------
AIRPORT_PAIR = ("BWI", "JFK")  # keeps BWI->JFK and JFK->BWI
# ----------------------------------------------------------------------------------

# Output options:
WRITE_MONTHLY_FILES = True   # if MODE=="year", also write per-month outputs
WRITE_YEAR_FILE = True       # if MODE=="year", write combined year output

# ------------------------- CLEANUP OPTIONS -------------------------
# Clean up downloaded zip + extracted CSVs ONLY if we produced final output with data
CLEANUP_DOWNLOADS_IF_FINAL_HAS_DATA = True
# ---------------------------------------------------------------------------------

# ------------------------- BTS PREZIP NAMING -------------------------
BTS_PREZIP_BASE = "https://transtats.bts.gov/PREZIP"
ZIP_NAME_TEMPLATE = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
# ---------------------------------------------------------------------------------


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def download_zip(zip_url: str, zip_path: Path) -> None:
    """Download zip if not present."""
    if zip_path.exists():
        print(f"Zip already exists -> {zip_path}")
        return

    print(f"Downloading: {zip_url}")
    with requests.get(zip_url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"Saved zip -> {zip_path}")


def extract_zip(zip_path: Path, extract_dir: Path) -> Path:
    """Extract and return the first CSV path inside."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    csv_files = sorted(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {extract_dir}")

    return csv_files[0]


def get_date_expr(cols: set[str]) -> pl.Expr:
    """Return a Polars expression that parses the flight date."""
    if "FL_DATE" in cols:
        return pl.col("FL_DATE").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    if "FlightDate" in cols:
        # Some TranStats variants store yyyymmdd
        return pl.col("FlightDate").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
    raise KeyError("Could not find a date column: expected 'FL_DATE' or 'FlightDate'.")


def write_outputs(df: pl.DataFrame, out_stem: str) -> Path:
    """Write parquet only; return parquet path."""
    out_parquet = OUT_DIR / f"{out_stem}.parquet"
    df.write_parquet(out_parquet)
    print(f"Wrote -> {out_parquet}")
    return out_parquet


def cleanup_path(p: Path) -> None:
    """Delete a file or directory tree (best-effort)."""
    try:
        if not p.exists():
            return
        if p.is_file():
            p.unlink()
            return
        # directory: delete contents then the dir
        for child in sorted(p.rglob("*"), reverse=True):
            try:
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            except Exception:
                pass
        try:
            p.rmdir()
        except Exception:
            pass
    except Exception:
        pass


def cleanup_downloads(download_records: list[tuple[Path, Path]]) -> None:
    """Delete all zip files + extracted directories recorded during processing."""
    for zip_path, extract_dir in download_records:
        cleanup_path(zip_path)
        cleanup_path(extract_dir)
        print(f"Cleaned -> {zip_path} and {extract_dir}")


def process_month(year: int, month: int, download_records: list[tuple[Path, Path]]) -> pl.DataFrame:
    """Download, extract, read, filter one month; return DataFrame."""
    zip_name = ZIP_NAME_TEMPLATE.format(year=year, month=month)
    zip_url = f"{BTS_PREZIP_BASE}/{zip_name}"

    zip_path = OUT_DIR / zip_name
    extract_dir = OUT_DIR / f"extracted_{year}_{month:02d}"

    download_zip(zip_url, zip_path)
    csv_path = extract_zip(zip_path, extract_dir)

    # Record for potential end-of-run cleanup
    download_records.append((zip_path, extract_dir))

    print(f"Using CSV: {csv_path}")

    lf = pl.scan_csv(csv_path, infer_schema_length=2000)
    cols = set(lf.collect_schema().names())

    # Normalize column names (some files use ORIGIN, DEST)
    origin_col = "Origin" if "Origin" in cols else ("ORIGIN" if "ORIGIN" in cols else None)
    dest_col = "Dest" if "Dest" in cols else ("DEST" if "DEST" in cols else None)
    if origin_col is None or dest_col is None:
        raise KeyError("Could not find Origin/Dest columns (expected Origin/Dest or ORIGIN/DEST).")

    # (kept) date parsing helper
    date_expr = get_date_expr(cols)

    # Route filter: ONLY BWI <-> JFK (either direction)
    a, b = AIRPORT_PAIR
    route_filter = ((pl.col(origin_col) == a) & (pl.col(dest_col) == b)) | ((pl.col(origin_col) == b) & (pl.col(dest_col) == a))

    df = (
        lf.with_columns(date_expr.alias("_flight_date"))
          .filter(route_filter)
          .drop("_flight_date")
          .collect(streaming=True)
    )

    return df


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    download_records: list[tuple[Path, Path]] = []
    final_parquet_path: Path | None = None
    final_rows: int = 0

    if MODE == "month":
        df = process_month(YEAR, MONTH, download_records)
        print(df.head())
        final_rows = df.height
        final_parquet_path = write_outputs(df, out_stem=f"flight_delay_{YEAR}_{MONTH:02d}")

    elif MODE == "year":
        dfs: list[pl.DataFrame] = []

        for m in MONTHS:
            df_m = process_month(YEAR, m, download_records)
            print(f"Month {m:02d}: rows={df_m.height:,}")

            if WRITE_MONTHLY_FILES:
                write_outputs(df_m, out_stem=f"flight_delay_{YEAR}_{m:02d}")

            dfs.append(df_m)

        if WRITE_YEAR_FILE:
            df_year = pl.concat(dfs, how="vertical_relaxed")
            print(f"Year {YEAR}: rows={df_year.height:,}")
            final_rows = df_year.height
            final_parquet_path = write_outputs(df_year, out_stem=f"flight_delay_{YEAR}")

    else:
        raise ValueError("MODE must be 'month' or 'year'.")

    # ------------------------- CLEANUP AT END (ONLY IF FINAL HAS DATA) -------------------------
    if CLEANUP_DOWNLOADS_IF_FINAL_HAS_DATA and (final_parquet_path is not None) and (final_rows > 0):
        cleanup_downloads(download_records)
    else:
        print("Skipping cleanup (final file missing or had 0 rows).")
    # ------------------------------------------------------------------------------------------