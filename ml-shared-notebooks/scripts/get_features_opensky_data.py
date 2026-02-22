from pathlib import Path
import pandas as pd

BASE_DIR = Path.cwd()
# file_path = BASE_DIR / ".." / "data" / "iad_dca_states_smoketest.parquet"
file_path = BASE_DIR / ".." / "data" / "iad_dca_states_10min_30s.parquet"



df = pd.read_parquet(file_path)

# Convert to datetime
df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)

# Sort and de-dupe at the true grain
df = df.sort_values(["icao24", "snapshot_time"])
df = df.drop_duplicates(subset=["icao24", "snapshot_time"])

# ✅ MultiIndex prevents duplicate-index assignment issues
df = df.set_index(["icao24", "snapshot_time"]).sort_index()

# Basic cleaning
df = df.ffill()
df = df[df["velocity"] >= 0]
df = df[df["geo_altitude"] >= 0]

# -----------------------------
# Feature config
# -----------------------------
features_10s = ["latitude", "longitude", "velocity", "geo_altitude", "vertical_rate", "true_track"]
features_1min = ["velocity", "geo_altitude", "vertical_rate"]

stats = {
    "mean": "mean",
    "std":  "std",
    "min":  "min",
    "max":  "max",
    "median": "median",
    "count": "count",
}

windows = {
    "10s": ("10s", features_10s),
    "1min": ("1min", features_1min),
}

g = df.groupby(level=0)  # group by icao24 (index level 0)

# -----------------------------
# Add resampled stats as columns
# -----------------------------
for win_name, (rule, cols) in windows.items():
    for col in cols:
        rs = g[col].resample(rule, level=1)  # resample on snapshot_time (index level 1)
        for stat_name, func in stats.items():
            new_col = f"{col}_{stat_name}_{win_name}"
            df[new_col] = rs.transform(func)

# Optional: bring index back to columns for downstream ML
df = df.reset_index()

print(df.head())

output_path = BASE_DIR / ".." / "data" / "iad_states_10min_30s.csv"

df.to_csv(output_path, index=False)
