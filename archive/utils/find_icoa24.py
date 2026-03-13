# import polars as pl

# INPUT_FILE = "faa_aircraft_dim.parquet"
# OUTPUT_FILE = "icoas_found.csv"

# ICAO24_TARGET = "a3b124"

# # -----------------------------
# # READ + FILTER
# # -----------------------------
# df = (
#     pl.read_parquet(INPUT_FILE)
#       .filter(pl.col("icao24") == ICAO24_TARGET)
# )

# # Show result
# print(df.head())
# print(f"Rows found: {df.height}")

# # -----------------------------
# # SAVE (optional)
# # -----------------------------
# df.write_csv(OUTPUT_FILE)

# print(f"Filtered file saved → {OUTPUT_FILE}")


import polars as pl

# -----------------------------
# CONFIG
# -----------------------------
REGISTRY_FILE = "faa_aircraft_dim.parquet"   # FAA registry
SOURCE_FILE = "/home/jon/Documents/grad_school/OR568/project/OR568_ML_Project/data_pipeline/raw_data/iad_states_2022_06_27.csv"         # file containing icao24 values

OUTPUT_FILE = "icoas_found.csv"

ICAO_COL = "icao24"

# -----------------------------
# READ DATA
# -----------------------------
# FAA registry
registry_df = pl.read_parquet(REGISTRY_FILE)

# Source data (OpenSky or other)
source_df = pl.read_csv(SOURCE_FILE)

# -----------------------------
# GET UNIQUE ICAO24 VALUES
# -----------------------------
source_icaos = (
    source_df
    .select(pl.col(ICAO_COL))
    .drop_nulls()
    .unique()
)

print(f"Unique ICAO24s in source: {source_icaos.height}")

# -----------------------------
# FIND MATCHES IN REGISTRY
# -----------------------------
matches = registry_df.join(
    source_icaos,
    on=ICAO_COL,
    how="inner"
)

# -----------------------------
# COUNTS
# -----------------------------
found_count = matches.select(pl.col(ICAO_COL).n_unique()).item()

total_source = source_icaos.height
missing_count = total_source - found_count

print(f"Found in registry: {found_count}")
print(f"Missing: {missing_count}")
print(f"Match %: {found_count / total_source:.2%}")

# -----------------------------
# SAVE RESULTS
# -----------------------------
matches.write_csv(OUTPUT_FILE)

print(f"Matches saved → {OUTPUT_FILE}")


