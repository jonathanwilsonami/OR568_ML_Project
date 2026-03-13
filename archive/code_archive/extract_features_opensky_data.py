import pandas as pd


df = pd.read_csv("raw_data/bwi_states_2022_06_27_raw.csv")

# Convert Unix/POSIX epoch -> human-readable datetime
time_series = pd.to_numeric(df["time"], errors="coerce")
unit = "ms" if time_series.dropna().median() > 1e12 else "s"
df["time"] = pd.to_datetime(time_series, unit=unit, utc=True)

# Sort and de-dupe at the true grain
df = df.sort_values(["icao24", "time"])
df = df.drop_duplicates(subset=["icao24", "time"])

# MultiIndex prevents duplicate-index assignment issues
df = df.set_index(["icao24", "time"]).sort_index()

# Basic cleaning
df = df.ffill()
df = df[df["velocity"] >= 0]
df = df[df["geoaltitude"] >= 0]

# -----------------------------
# Features
# -----------------------------
features_10s = ["lat", "lon", "velocity", "geoaltitude", "vertrate", "heading"]
features_1min = ["velocity", "geoaltitude", "vertrate"]

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
        rs = g[col].resample(rule, level=1)  # resample on time (index level 1)
        for stat_name, func in stats.items():
            new_col = f"{col}_{stat_name}_{win_name}"
            df[new_col] = rs.transform(func)

# Optional: bring index back to columns for downstream ML
df = df.reset_index()

df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day

df["hour"] = df["time"].dt.hour

# yyyy-mm-dd date (string)
df["date"] = df["time"].dt.strftime("%Y-%m-%d")

df.to_csv("raw_data/bwi_states_smoothed_resampled.csv", index=True)