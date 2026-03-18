import os
import shutil
import subprocess
from pathlib import Path
import datetime as dt
import tarfile

import polars as pl

# ===============================
# HARD-CODED SETTINGS
# ===============================

START_MONDAY = dt.date(2021, 1, 18)   # MUST be a Monday
NUM_WEEKS = 3

# IAD terminal area bbox
LAT_MIN, LAT_MAX = 38.3, 39.6
LON_MIN, LON_MAX = -78.3, -76.5

S3_ENDPOINT = "https://s3.opensky-network.org"
S3_PREFIX = "s3://data-samples/states"

TMP_DIR = Path("./tmp_opensky")
OUT_DIR = Path("./iad_dataset")

TMP_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ===============================
# HELPERS
# ===============================

def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["AWS_EC2_METADATA_DISABLED"] = "true"
    subprocess.run(cmd, check=True, env=env)

def download_avro_tars(monday_str: str, dest: Path) -> None:
    """Download only *.avro.tar for the given Monday."""
    run([
        "aws", "s3", "sync",
        "--no-sign-request",
        "--endpoint-url", S3_ENDPOINT,
        f"{S3_PREFIX}/{monday_str}/",
        str(dest),
        "--exclude", "*",
        "--include", "*.avro.tar"
    ])

def extract_avro_tar(tar_path: Path, out_dir: Path) -> list[Path]:
    """Extract a single .avro.tar and return avro files inside."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:") as tf:
        tf.extractall(out_dir)

    return sorted(out_dir.rglob("*.avro"))

def resolve_lat_lon_cols(cols: list[str]) -> tuple[str, str]:
    """
    OpenSky sometimes uses different names. Handle common cases.
    """
    c = set(cols)
    lat = "latitude" if "latitude" in c else ("lat" if "lat" in c else None)
    lon = "longitude" if "longitude" in c else ("lon" if "lon" in c else None)

    if lat is None or lon is None:
        # Print some columns for debugging
        raise KeyError(f"Could not find lat/lon columns. Example columns: {sorted(list(c))[:50]}")
    return lat, lon

# ===============================
# MAIN LOOP
# ===============================

print("Running from:", Path.cwd().resolve())
print("OUT_DIR:", OUT_DIR.resolve())

for i in range(NUM_WEEKS):
    monday = START_MONDAY + dt.timedelta(days=7 * i)
    monday_str = monday.isoformat()

    print(f"\n=== Processing {monday_str} ===")

    week_tmp = TMP_DIR / monday_str
    if week_tmp.exists():
        shutil.rmtree(week_tmp)
    week_tmp.mkdir(parents=True, exist_ok=True)

    # 1) Download only *.avro.tar
    download_avro_tars(monday_str, week_tmp)
    tar_files = sorted(week_tmp.rglob("*.avro.tar"))
    print(f"Downloaded {len(tar_files)} *.avro.tar files")

    if not tar_files:
        print("No avro.tar files found; leaving tmp for inspection:", week_tmp)
        continue

    # We'll write per-avro parquet parts then optionally merge.
    parts_dir = week_tmp / "_parts_parquet"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part_paths: list[Path] = []

    lat_col = None
    lon_col = None
    total_rows_kept = 0
    part_idx = 0

    # 2) Extract each tar, read each avro with read_avro, filter immediately, write parquet part
    for tar_path in tar_files:
        extracted_dir = tar_path.parent / (tar_path.name + "_extracted")
        avro_files = extract_avro_tar(tar_path, extracted_dir)

        if not avro_files:
            continue

        for avro_path in avro_files:
            # Read one avro
            df = pl.read_avro(str(avro_path))

            # Resolve column names once
            if lat_col is None or lon_col is None:
                lat_col, lon_col = resolve_lat_lon_cols(df.columns)

            # Filter to IAD bbox
            df_iad = df.filter(
                (pl.col(lat_col).is_between(LAT_MIN, LAT_MAX)) &
                (pl.col(lon_col).is_between(LON_MIN, LON_MAX))
            )

            if df_iad.height == 0:
                continue

            df_iad = df_iad.with_columns(pl.lit(monday_str).alias("monday_date"))

            # Write a small parquet part
            part_idx += 1
            part_path = parts_dir / f"part_{part_idx:04d}.parquet"
            df_iad.write_parquet(part_path)
            part_paths.append(part_path)

            total_rows_kept += df_iad.height

    print(f"Kept total rows in bbox: {total_rows_kept:,}")
    print(f"Parquet parts written: {len(part_paths)}")

    if not part_paths:
        print("No filtered data for this day; leaving tmp for inspection:", week_tmp)
        continue

    # 3) Merge parts into one parquet for the day (optional but usually what you want)
    out_file = OUT_DIR / f"iad_{monday_str}.parquet"
    pl.scan_parquet([str(p) for p in part_paths]).collect(streaming=True).write_parquet(out_file)
    print("Wrote:", out_file.resolve())

    # 4) Cleanup only after successful write
    shutil.rmtree(week_tmp, ignore_errors=True)

print("\nDONE")