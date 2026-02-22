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

# Filter: only flights where Origin == BWI AND Dest == BWI (as you requested)
AIRPORT = "BWI"

# Output options:
WRITE_MONTHLY_FILES = True   # if MODE=="year", also write per-month outputs
WRITE_YEAR_FILE = True       # if MODE=="year", write combined year output

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


def process_month(year: int, month: int) -> pl.DataFrame:
    """Download, extract, read, filter one month; return DataFrame."""
    zip_name = f"flight_delays_{year}_{month}.zip"
    zip_url = f"https://transtats.bts.gov/PREZIP/{zip_name}"

    zip_path = OUT_DIR / zip_name
    extract_dir = OUT_DIR / f"extracted_{year}_{month:02d}"

    download_zip(zip_url, zip_path)
    csv_path = extract_zip(zip_path, extract_dir)

    print(f"Using CSV: {csv_path}")

    lf = pl.scan_csv(csv_path, infer_schema_length=2000)
    cols = set(lf.collect_schema().names())

    # Normalize column names (some files use ORIGIN, DEST)
    origin_col = "Origin" if "Origin" in cols else ("ORIGIN" if "ORIGIN" in cols else None)
    dest_col = "Dest" if "Dest" in cols else ("DEST" if "DEST" in cols else None)
    if origin_col is None or dest_col is None:
        raise KeyError("Could not find Origin/Dest columns (expected Origin/Dest or ORIGIN/DEST).")

    date_expr = get_date_expr(cols)

    # Filter to only Origin==BWI AND Dest==BWI
    df = (
        lf.with_columns(date_expr.alias("_flight_date"))
          .filter((pl.col(origin_col) == AIRPORT) & (pl.col(dest_col) == AIRPORT))
          .drop("_flight_date")
          .collect(streaming=True)
    )

    return df


def write_outputs(df: pl.DataFrame, out_stem: str) -> None:
    """Write parquet + csv with consistent naming."""
    out_parquet = OUT_DIR / f"{out_stem}.parquet"
    out_csv = OUT_DIR / f"{out_stem}.csv"

    df.write_parquet(out_parquet)
    df.write_csv(out_csv)

    print(f"Wrote -> {out_parquet}")
    print(f"Wrote -> {out_csv}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    if MODE == "month":
        df = process_month(YEAR, MONTH)
        print(df.head())
        write_outputs(df, out_stem=f"flight_delay_{YEAR}_{MONTH:02d}")

    elif MODE == "year":
        dfs: list[pl.DataFrame] = []

        for m in MONTHS:
            df_m = process_month(YEAR, m)
            print(f"Month {m:02d}: rows={df_m.height:,}")

            if WRITE_MONTHLY_FILES:
                write_outputs(df_m, out_stem=f"flight_delay_{YEAR}_{m:02d}")

            dfs.append(df_m)

        if WRITE_YEAR_FILE:
            df_year = pl.concat(dfs, how="vertical_relaxed")
            print(f"Year {YEAR}: rows={df_year.height:,}")
            write_outputs(df_year, out_stem=f"flight_delay_{YEAR}")

    else:
        raise ValueError("MODE must be 'month' or 'year'.")