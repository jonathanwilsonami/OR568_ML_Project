from __future__ import annotations

import time
from dataclasses import dataclass
from pyopensky.rest import REST
import pandas as pd

@dataclass(frozen=True)
class Airport:
    icao: str
    lat: float
    lon: float

# Rough airport coordinates (sufficient for bounding-box filtering)
IAD = Airport("KIAD", 38.9475, -77.4599)
# DCA = Airport("KDCA", 38.8514, -77.0377)

def make_bbox(buffer_deg: float = 0.20) -> tuple[float, float, float, float]:
    """
    Returns (west, south, east, north). Keep buffer small to minimize credits.
    buffer_deg=0.20 is a conservative "small" box around IAD+DCA.
    """
    lats = [IAD.lat] #, DCA.lat]
    lons = [IAD.lon] #, DCA.lon]
    south = min(lats) - buffer_deg
    north = max(lats) + buffer_deg
    west  = min(lons) - buffer_deg
    east  = max(lons) + buffer_deg
    return (west, south, east, north)

def fetch_states_snapshot(rest, ts: pd.Timestamp, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    west, south, east, north = bounds
    ts_int = int(ts.timestamp())

    url = (
        "https://opensky-network.org/api/states/all"
        f"?time={ts_int}"
        f"&lamin={south}&lamax={north}&lomin={west}&lomax={east}"
        "&extended=1"
    )
    json = rest.get(url)

    cols = [
        "icao24","callsign","origin_country","time_position","last_contact",
        "longitude","latitude","baro_altitude","on_ground","velocity","true_track",
        "vertical_rate","sensors","geo_altitude","squawk","spi","position_source",
        "category"
    ]

    states = json.get("states", []) or []
    df = pd.DataFrame.from_records(states, columns=cols)
    if df.empty:
        return df

    df["snapshot_time"] = ts
    df["callsign"] = df["callsign"].astype("string").str.strip()
    df["time_position"] = pd.to_datetime(df["time_position"], utc=True, unit="s", errors="coerce")
    df["last_contact"]  = pd.to_datetime(df["last_contact"],  utc=True, unit="s", errors="coerce")
    return df

def get_iad_dca_last_10min_smoketest(rest) -> pd.DataFrame:
    """
    SAFE test: only 3 snapshots, bounded area, minimal credits.
    This validates extraction + downstream aggregation without heavy API usage.
    """
    bounds = make_bbox(buffer_deg=0.20)
    now = pd.Timestamp.now(tz="utc").floor("s")

    # Only 3 snapshots (very light usage)
    sample_offsets = [0, 60, 120]  # seconds ago
    frames = []
    for offset in sample_offsets:
        ts = now - pd.Timedelta(seconds=offset)
        frames.append(fetch_states_snapshot(rest, ts, bounds))
        time.sleep(1)  # be polite; also helps if near rate limits

    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["icao24", "callsign", "snapshot_time", "longitude", "latitude"]
    )

import time

def collect_states_for_minutes(rest, minutes=10, interval_seconds=30, buffer_deg=0.20):
    bounds = make_bbox(buffer_deg=buffer_deg)
    end = pd.Timestamp.now(tz="utc").floor("s")
    start = end - pd.Timedelta(minutes=minutes)
    times = pd.date_range(start=start, end=end, freq=f"{interval_seconds}s", tz="utc")

    frames = []
    for ts in times:
        frames.append(fetch_states_snapshot(rest, ts, bounds))
        time.sleep(1)  # polite
    return pd.concat(frames, ignore_index=True)

rest = REST()



def main():
    rest = REST()  # uses your configured credentials if present
    # df = get_iad_dca_last_10min_smoketest(rest)

    # print(df.shape)
    # print(df.head(10))

    # Save output
    # df.to_parquet("iad_dca_states_smoketest.parquet", index=False)
    # df.to_csv("iad_dca_states_smoketest.csv", index=False)

    df_more = collect_states_for_minutes(rest, minutes=10, interval_seconds=30)
    df_more.to_parquet("iad_dca_states_10min_30s.parquet", index=False)

if __name__ == "__main__":
    main()

# TODO 
# Get 1 hour of data with 5 sec inttervals 
# Get 1 day of data wtih 5 sec intervals 
# Start joining the data on 2 other data sets 