from pathlib import Path
import tarfile
import polars as pl

BASE_DIR = Path(__file__).resolve().parent  

data_dir = BASE_DIR / "raw_data" / "states" / "2022_06_27"
extract_dir = BASE_DIR / "raw_data" / "extracted"
extract_dir.mkdir(parents=True, exist_ok=True)

# BWI terminal area bbox
LAT_MIN, LAT_MAX = 38.9191667, 39.4166667
LON_MIN, LON_MAX = -77.0597222, -76.3075000
# $$c(W 76°56'39"-W 76°22'05"/N 39°21'00"-N 38°58'44")

extract_dir = BASE_DIR / "raw_data" / "extracted" / "states" / "2022_06_27"
extract_dir.mkdir(parents=True, exist_ok=True)

# Find tar files (your example ends with .csv.tar)
tar_files = sorted(data_dir.glob("*.tar"))

print(f"data_dir: {data_dir}")
print(f"found tar files: {len(tar_files)}")
if not tar_files:
    raise FileNotFoundError(f"No tar files found in: {data_dir}")

# Extract all tar files
for tf in tar_files:
    with tarfile.open(tf, "r:*") as tar:
        tar.extractall(path=extract_dir)

# Read ALL extracted CSVs
lf = pl.scan_csv(str(extract_dir / "**" / "*.csv*"))
df = lf.collect()

# filter on IAD terminal area 
df = df.filter(
    (pl.col("lat") >= LAT_MIN) & (pl.col("lat") <= LAT_MAX) &
    (pl.col("lon") >= LON_MIN) & (pl.col("lon") <= LON_MAX)
)

df.write_csv(BASE_DIR / "raw_data" / "bwi_states_2022_06_27_raw.csv")