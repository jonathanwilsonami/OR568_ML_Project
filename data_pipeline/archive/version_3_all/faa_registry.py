from __future__ import annotations

import polars as pl

from data_pipeline.archive.version_3_all.config import FAAConfig, JoinConfig


def load_faa_registry(cfg: FAAConfig) -> pl.DataFrame:
    path = cfg.registry_file
    if not path.exists():
        raise FileNotFoundError(f"FAA registry file not found: {path}")

    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix == ".csv":
        df = pl.read_csv(path, infer_schema_length=5000)
    else:
        raise ValueError(f"Unsupported FAA registry format: {path.suffix}")

    return df


def normalize_tail_number_expr(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_uppercase()
        .alias(col_name)
    )


def standardize_faa_registry(df: pl.DataFrame, cfg: FAAConfig) -> pl.DataFrame:
    if cfg.tail_number_col not in df.columns:
        raise KeyError(
            f"FAA registry missing tail number column: {cfg.tail_number_col}. "
            f"Available columns: {df.columns[:60]}"
        )

    df = df.with_columns(normalize_tail_number_expr(cfg.tail_number_col))

    if cfg.tail_number_col != "faa_tail_number":
        df = df.rename({cfg.tail_number_col: "faa_tail_number"})

    return df


def join_faa_registry(
    flights_df: pl.DataFrame,
    faa_df: pl.DataFrame,
    joins: JoinConfig,
) -> pl.DataFrame:
    if joins.tail_col not in flights_df.columns:
        print(f"Skipping FAA join: BTS tail column not found -> {joins.tail_col}")
        return flights_df

    flights_df = flights_df.with_columns(
        normalize_tail_number_expr(joins.tail_col)
    )

    out = flights_df.join(
        faa_df,
        left_on=joins.tail_col,
        right_on="faa_tail_number",
        how="left",
    )

    return out