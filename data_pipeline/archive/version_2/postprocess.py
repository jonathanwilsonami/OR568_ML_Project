from __future__ import annotations

from pathlib import Path
import polars as pl

from data_pipeline.archive.version_2.config import PostProcessConfig
from data_pipeline.archive.version_2.utils import ensure_dir


def filter_selected_columns(
    df: pl.DataFrame,
    cfg: PostProcessConfig,
    dataset_name: str = "dataset",
) -> pl.DataFrame:
    """
    Keep only requested columns. Missing columns are skipped unless strict_missing_columns=True.
    """
    requested = cfg.selected_columns
    available = set(df.columns)

    keep = [c for c in requested if c in available]
    missing = [c for c in requested if c not in available]

    print(f"{dataset_name}: requested {len(requested)} columns")
    print(f"{dataset_name}: keeping   {len(keep)} columns")

    if missing:
        print(f"{dataset_name}: missing  {len(missing)} columns")
        print(missing)
        if cfg.strict_missing_columns:
            raise KeyError(
                f"{dataset_name}: missing requested columns: {missing}"
            )

    return df.select(keep)


def write_parquet(df: pl.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.write_parquet(out_path)
    print(f"Wrote -> {out_path}")


def maybe_write_filtered(
    df: pl.DataFrame,
    cfg: PostProcessConfig,
    out_path: Path,
    dataset_name: str,
) -> pl.DataFrame:
    filtered_df = filter_selected_columns(df, cfg, dataset_name=dataset_name)
    write_parquet(filtered_df, out_path)
    return filtered_df