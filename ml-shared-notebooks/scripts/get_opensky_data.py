
# test_trino_connection.py

from pyopensky.trino import Trino

trino = Trino()

print("Testing connection...")

df = trino.query("SELECT 1 AS test")

print(df)




# from opensky_pull import Airport, fetch_opensky_history_airport

# IAD = Airport(icao="KIAD", lat=38.9531, lon=-77.4565)

# df = fetch_opensky_history_airport(
#     start="2022-01-15 12:00:00",
#     stop="2022-01-15 12:10:00",   # 10 minutes only
#     airport=IAD,
#     radius_nm=15,                 # smaller area = fewer rows
#     mode="none",                  # just spatial
#     chunk_hours=1,                # > window so it will run as a single chunk
#     limit=20000,                  # optional safety cap (remove if you want full)
# )

# out_csv = "../data/opensky_iad_20220115_1200_1210.csv"
# df.to_csv(out_csv, index=False)
# print(f"Wrote {len(df):,} rows to {out_csv}")
