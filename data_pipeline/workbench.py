import polars as pl

data_names = [
    "aircraft_rotation_2019",
    "airport_time_2019",
    "flights_canonical_2019",
    "route_time_2019",
    "propagation_chains_2019"
]

for data_name in data_names:
    df = pl.read_parquet(
        f"/home/jon/projects/OR568_ML_Project/data_pipeline/data/features/{data_name}.parquet"
    )

    print(f"{data_name}:", df.shape)
    df = df.sample(n=100000, seed=42)
    df.write_csv(
        f"/home/jon/projects/OR568_ML_Project/data/v4/{data_name}.csv"
    )
