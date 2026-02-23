#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import polars as pl

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PARQUET_PATH = BASE_DIR / "outputs" / "enriched_flights.parquet"

# If your parquet is elsewhere, just change PARQUET_PATH above.
# ------------------------------------------------------------


def pct_nulls(df: pl.DataFrame, col: str) -> float:
    n = df.height
    if n == 0:
        return 0.0
    return float(df[col].null_count() / n) * 100.0


def main() -> None:
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet not found: {PARQUET_PATH}")

    print(f"\n📦 Reading parquet: {PARQUET_PATH}\n")

    # Lazy scan for performance
    lf = pl.scan_parquet(PARQUET_PATH)

    # Basic shape + schema (cheap)
    schema = lf.collect_schema()
    cols = schema.names()
    print("=== BASIC INFO ===")
    print(f"Columns: {len(cols)}")
    print("Sample of schema:")
    for name, dtype in list(schema.items())[:30]:
        print(f"  - {name}: {dtype}")
    if len(cols) > 30:
        print(f"  ... ({len(cols)-30} more)\n")
    else:
        print()

    # Materialize small samples only
    df_head = lf.head(10).collect()
    print("=== HEAD (10 rows) ===")
    print(df_head)

    # Row count (requires scan but still light)
    n_rows = lf.select(pl.len().alias("n_rows")).collect().item()
    print("\n=== ROW COUNT ===")
    print(f"Rows: {n_rows:,}\n")

    # -----------------------------
    # Missingness summary
    # -----------------------------
    # Compute null counts per column without collecting full dataset into memory
    nulls = (
        lf.select([pl.col(c).null_count().alias(c) for c in cols])
          .collect(engine="streaming")
    )

    # Convert to a tidy table
    null_counts = []
    for c in cols:
        null_counts.append((c, int(nulls[0, c])))

    null_df = pl.DataFrame(null_counts, schema=["column", "null_count"])
    null_df = null_df.with_columns(
        (pl.col("null_count") / pl.lit(n_rows) * 100.0).alias("null_pct")
    ).sort("null_pct", descending=True)

    print("=== TOP 25 NULL % COLUMNS ===")
    print(null_df.head(25))

    # -----------------------------
    # Key summary stats (numeric)
    # -----------------------------
    numeric_cols = [
        name for name, dtype in schema.items()
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                     pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                     pl.Float32, pl.Float64)
    ]

    print("\n=== NUMERIC COLUMNS ===")
    print(f"Numeric columns: {len(numeric_cols)}")

    # If you want to focus on your core delay/weather features, specify them here:
    focus_numeric = [
        c for c in [
            "DepDelay", "DepDelayMinutes", "ArrDelay", "ArrDelayMinutes",
            "TaxiOut", "TaxiIn", "AirTime",
            "dep_tmpf", "dep_sknt", "dep_vsby", "dep_p01i", "dep_skyl1", "dep_weather_severity",
            "arr_tmpf", "arr_sknt", "arr_vsby", "arr_p01i", "arr_skyl1", "arr_weather_severity",
        ]
        if c in cols
    ]

    numeric_to_describe = focus_numeric if focus_numeric else numeric_cols

    if numeric_to_describe:
        # Polars describe gives count, null_count, mean, std, min, 25%, 50%, 75%, max
        # for numeric columns.
        desc = lf.select(numeric_to_describe).collect(engine="streaming").describe()
        print("\n=== DESCRIBE (numeric focus) ===")
        print(desc)
    else:
        print("\n(No numeric columns found to describe.)")

    # -----------------------------
    # Categorical summaries
    # -----------------------------
    print("\n=== BASIC CATEGORICAL COUNTS ===")
    cat_cols = [c for c in ["Origin", "Dest", "Reporting_Airline", "Tail_Number", "Cancelled", "Diverted"] if c in cols]

    for c in cat_cols:
        top = (
            lf.group_by(pl.col(c))
              .agg(pl.len().alias("count"))
              .sort("count", descending=True)
              .head(10)
              .collect(engine="streaming")
        )
        print(f"\nTop values for {c}:")
        print(top)

    # -----------------------------
    # Delay target distribution (helpful for classification)
    # -----------------------------
    print("\n=== DELAY TARGET QUICK CHECKS ===")

    if "ArrDelayMinutes" in cols:
        delay_targets = (
            lf.select([
                pl.col("ArrDelayMinutes").cast(pl.Float64, strict=False).alias("ArrDelayMinutes"),
                (pl.col("ArrDelayMinutes").cast(pl.Float64, strict=False) >= 15).alias("ArrDel15_calc"),
            ])
            .select([
                pl.col("ArrDelayMinutes").count().alias("n_non_null"),
                pl.col("ArrDelayMinutes").null_count().alias("n_null"),
                pl.col("ArrDelayMinutes").mean().alias("mean_delay_min"),
                pl.col("ArrDelayMinutes").median().alias("median_delay_min"),
                pl.col("ArrDelayMinutes").min().alias("min_delay_min"),
                pl.col("ArrDelayMinutes").max().alias("max_delay_min"),
                pl.col("ArrDel15_calc").mean().alias("pct_delayed_15plus"),
            ])
            .collect(engine="streaming")
        )
        print(delay_targets)

    elif "ArrDelay" in cols:
        delay_targets = (
            lf.select([
                pl.col("ArrDelay").cast(pl.Float64, strict=False).alias("ArrDelay"),
                (pl.col("ArrDelay").cast(pl.Float64, strict=False) >= 15).alias("ArrDel15_calc"),
            ])
            .select([
                pl.col("ArrDelay").count().alias("n_non_null"),
                pl.col("ArrDelay").null_count().alias("n_null"),
                pl.col("ArrDelay").mean().alias("mean_delay_min"),
                pl.col("ArrDelay").median().alias("median_delay_min"),
                pl.col("ArrDelay").min().alias("min_delay_min"),
                pl.col("ArrDelay").max().alias("max_delay_min"),
                pl.col("ArrDel15_calc").mean().alias("pct_delayed_15plus"),
            ])
            .collect(engine="streaming")
        )
        print(delay_targets)
    else:
        print("No ArrDelay/ArrDelayMinutes found.")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()