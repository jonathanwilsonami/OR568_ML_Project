from __future__ import annotations

import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from feature_definitions import TARGET_CLASS, TARGET_REG, XGB_FEATURE_SETS
from config import CONFIG
from aircraft_features import enrich_with_aircraft_features

logger = logging.getLogger(__name__)


# ============================================================
# GENERAL HELPERS
# ============================================================

def timer_log(label: str, start_time: float, time_module=time) -> None:
    elapsed = time_module.perf_counter() - start_time
    print(f"[TIMER] {label}: {elapsed:.2f}s")
    try:
        logger.info("[TIMER] %s: %.2fs", label, elapsed)
    except Exception:
        pass


def clean_name(col: str) -> str:
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    col = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", col)
    col = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_").lower()


def maybe_sample(df: pl.DataFrame, sample_fraction: float | None, seed: int) -> pl.DataFrame:
    """Optional in-memory subsample. Only used when sampling is explicitly configured."""
    if sample_fraction is None or sample_fraction >= 1.0:
        return df
    if sample_fraction <= 0:
        raise ValueError("sample_fraction must be > 0 when provided.")
    return df.sample(fraction=sample_fraction, seed=seed, shuffle=True)


def rank_results(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = []
    ascending = []

    for col in ["cv_auc_mean", "cv_f1_mean"]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(False)

    for col in ["cv_mae_mean", "cv_rmse_mean"]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(True)

    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    return df


def summarize_cv_metrics(
    metrics_list: list[dict[str, float]], prefix: str = "cv"
) -> dict[str, float]:
    if not metrics_list:
        return {}

    out: dict[str, float] = {}
    skip_keys = {"fold", "train_years", "val_year"}
    metric_names = [k for k in metrics_list[0].keys() if k not in skip_keys]

    for name in metric_names:
        vals = [float(m[name]) for m in metrics_list]
        out[f"{prefix}_{name}_mean"] = float(sum(vals) / len(vals))
        out[f"{prefix}_{name}_min"] = float(min(vals))
        out[f"{prefix}_{name}_max"] = float(max(vals))

    return out


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
    }


def safe_fill_and_to_pandas(df_pl: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df_pl.select(cols).to_pandas()

    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            median_val = out[c].median()
            out[c] = out[c].fillna(median_val)
            if out[c].dtype.kind == "f":
                out[c] = pd.to_numeric(out[c], downcast="float")
            else:
                out[c] = pd.to_numeric(out[c], downcast="integer")
        else:
            out[c] = out[c].fillna("missing")

    return out


# ============================================================
# LSTM HELPERS  (unchanged)
# ============================================================

STEP_FEATURES = [
    "arr_delay_prev", "dep_delay_prev", "arr_del15_prev", "dep_del15_prev",
    "gap_minutes", "distance", "dep_hour_local", "dep_weekday_local",
    "dep_month_local", "dep_time_bucket", "is_weekend", "is_holiday",
    "days_to_nearest_holiday", "crs_elapsed_time", "dep_temp_c",
    "dep_wind_speed_m_s", "dep_wind_dir_deg", "dep_ceiling_height_m",
    "arr_temp_c", "arr_wind_speed_m_s", "arr_wind_dir_deg",
    "arr_ceiling_height_m", "route_frequency", "origin_flight_volume",
    "dest_flight_volume", "tight_turnaround_flag", "relative_leg_position",
    "cum_dep_delay_aircraft_day", "cum_arr_delay_aircraft_day",
]

LSTM_CONTEXT_CURRENT = [
    "distance", "dep_hour_local", "dep_weekday_local", "dep_month_local",
    "dep_time_bucket", "is_weekend", "is_holiday", "days_to_nearest_holiday",
    "crs_elapsed_time", "dep_temp_c", "dep_wind_speed_m_s", "dep_wind_dir_deg",
    "dep_ceiling_height_m", "arr_temp_c", "arr_wind_speed_m_s",
    "arr_wind_dir_deg", "arr_ceiling_height_m", "route_frequency",
    "origin_flight_volume", "dest_flight_volume", "tight_turnaround_flag",
    "relative_leg_position", "cum_dep_delay_aircraft_day",
    "cum_arr_delay_aircraft_day",
]


def build_lstm_step_matrix(df_pl: pl.DataFrame, variant_name: str = "context_full"):
    pdf = df_pl.to_pandas().copy()

    numeric_fill_cols = [
        "prev2_arr_delay", "prev2_dep_delay", "prev2_arr_del15", "prev2_dep_del15",
        "prev1_arr_delay", "prev1_dep_delay", "prev1_arr_del15", "prev1_dep_del15",
        "prev1_turnaround_minutes", "time_since_prev2_arrival_minutes",
        "distance", "dep_hour_local", "dep_weekday_local", "dep_month_local",
        "dep_time_bucket", "is_weekend", "is_holiday", "days_to_nearest_holiday",
        "crs_elapsed_time", "dep_temp_c", "dep_wind_speed_m_s", "dep_wind_dir_deg",
        "dep_ceiling_height_m", "arr_temp_c", "arr_wind_speed_m_s", "arr_wind_dir_deg",
        "arr_ceiling_height_m", "route_frequency", "origin_flight_volume",
        "dest_flight_volume", "tight_turnaround_flag", "relative_leg_position",
        "cum_dep_delay_aircraft_day", "cum_arr_delay_aircraft_day",
    ]
    for c in numeric_fill_cols:
        if c in pdf.columns:
            pdf[c] = pdf[c].fillna(pdf[c].median())

    current_context_cols = (
        ["distance", "dep_hour_local", "dep_weekday_local", "dep_month_local",
         "dep_time_bucket", "is_weekend", "is_holiday", "days_to_nearest_holiday",
         "crs_elapsed_time"]
        if variant_name == "delay_only"
        else LSTM_CONTEXT_CURRENT
    )

    X = np.zeros((len(pdf), 3, len(STEP_FEATURES)), dtype="float32")
    X[:, 0, STEP_FEATURES.index("arr_delay_prev")] = pdf["prev2_arr_delay"].values
    X[:, 0, STEP_FEATURES.index("dep_delay_prev")] = pdf["prev2_dep_delay"].values
    X[:, 0, STEP_FEATURES.index("arr_del15_prev")] = pdf["prev2_arr_del15"].values
    X[:, 0, STEP_FEATURES.index("dep_del15_prev")] = pdf["prev2_dep_del15"].values
    X[:, 0, STEP_FEATURES.index("gap_minutes")] = pdf["time_since_prev2_arrival_minutes"].values
    X[:, 1, STEP_FEATURES.index("arr_delay_prev")] = pdf["prev1_arr_delay"].values
    X[:, 1, STEP_FEATURES.index("dep_delay_prev")] = pdf["prev1_dep_delay"].values
    X[:, 1, STEP_FEATURES.index("arr_del15_prev")] = pdf["prev1_arr_del15"].values
    X[:, 1, STEP_FEATURES.index("dep_del15_prev")] = pdf["prev1_dep_del15"].values
    X[:, 1, STEP_FEATURES.index("gap_minutes")] = pdf["prev1_turnaround_minutes"].values
    for col in current_context_cols:
        if col in STEP_FEATURES and col in pdf.columns:
            X[:, 2, STEP_FEATURES.index(col)] = pdf[col].values

    y_cls = pdf[TARGET_CLASS].astype(int).values
    y_reg = pdf[TARGET_REG].astype(float).values
    return X, y_cls, y_reg, pdf


def scale_lstm(X_train, X_test):
    n_train, timesteps, n_features = X_train.shape
    n_test = X_test.shape[0]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, n_features)
    ).reshape(n_train, timesteps, n_features)
    X_test_scaled = scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(n_test, timesteps, n_features)
    return X_train_scaled.astype("float32"), X_test_scaled.astype("float32"), scaler


# ============================================================
# LAZY FRAME BUILDER
# ============================================================

def load_years_lazy(
    canonical_dir: str | Path,
    file_pattern: str,
    years: list[int],
) -> pl.LazyFrame:
    canonical_dir = Path(canonical_dir)
    lazy_frames: list[pl.LazyFrame] = []

    for year in years:
        path = canonical_dir / file_pattern.format(year=year)
        if not path.exists():
            raise FileNotFoundError(f"Missing input parquet: {path}")
        print(f"[LOAD] {path}")
        logger.info("[LOAD] %s", path)
        lf = pl.scan_parquet(path)
        schema_names = lf.collect_schema().names()
        rename_map = {c: clean_name(c) for c in schema_names}
        lf = lf.rename(rename_map)
        lazy_frames.append(lf)

    if not lazy_frames:
        raise ValueError("No parquet files were loaded.")

    return pl.concat(lazy_frames, how="vertical_relaxed")


def build_modeling_table(lf: pl.LazyFrame) -> pl.LazyFrame:
    required_filter_cols = [
        "is_cancelled", "is_diverted", "arr_del15",
        "tail_number", "dep_ts_actual_utc", "arr_ts_actual_utc",
    ]
    missing = [c for c in required_filter_cols if c not in lf.collect_schema().names()]
    if missing:
        raise ValueError(f"Missing required columns before build_modeling_table: {missing}")

    us_holidays = [
        "2022-01-01", "2022-01-17", "2022-02-21", "2022-05-30", "2022-07-04",
        "2022-09-05", "2022-10-10", "2022-11-11", "2022-11-24", "2022-12-25",
        "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29", "2023-07-04",
        "2023-09-04", "2023-10-09", "2023-11-10", "2023-11-23", "2023-12-25",
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27", "2024-07-04",
        "2024-09-02", "2024-10-14", "2024-11-11", "2024-11-28", "2024-12-25",
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-07-04",
        "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27", "2025-12-25",
    ]

    cols = set(lf.collect_schema().names())
    exprs: list[pl.Expr] = []

    if "dep_hour_local" in cols:
        exprs.append(
            pl.when(pl.col("dep_hour_local") < 6).then(1)
            .when(pl.col("dep_hour_local") < 11).then(2)
            .when(pl.col("dep_hour_local") < 14).then(3)
            .when(pl.col("dep_hour_local") < 18).then(4)
            .when(pl.col("dep_hour_local") < 21).then(5)
            .otherwise(6)
            .alias("dep_time_bucket")
        )
    if "dep_weekday_local" in cols:
        exprs.append(
            pl.col("dep_weekday_local").is_in([6, 7]).cast(pl.Int8).alias("is_weekend")
        )
    if "flight_date" in cols:
        exprs.append(
            pl.col("flight_date").cast(pl.Utf8).is_in(us_holidays).cast(pl.Int8).alias("is_holiday")
        )
        exprs.append(
            pl.min_horizontal([
                (pl.col("flight_date").cast(pl.Date) - pl.lit(h).str.strptime(pl.Date))
                .abs().dt.total_days()
                for h in us_holidays
            ]).alias("days_to_nearest_holiday")
        )
    if "route_key" in cols:
        exprs.append(pl.len().over("route_key").alias("route_frequency"))
    if "origin" in cols:
        exprs.append(pl.len().over("origin").alias("origin_flight_volume"))
    if "dest" in cols:
        exprs.append(pl.len().over("dest").alias("dest_flight_volume"))
    if "prev_arr_delay" in cols:
        exprs.append((pl.col("prev_arr_delay") > 15).cast(pl.Int8).alias("prev_arr_delayed_flag"))
    if {"prev_arr_delay", "prev_dep_delay"}.issubset(cols):
        exprs.append((pl.col("prev_arr_delay") + pl.col("prev_dep_delay")).alias("prev_total_delay"))
    if "turnaround_minutes" in cols:
        exprs.append((pl.col("turnaround_minutes") < 60).cast(pl.Int8).alias("tight_turnaround_flag"))
    if {"aircraft_leg_number_day", "tail_number"}.issubset(cols):
        exprs.append(
            (pl.col("aircraft_leg_number_day") / pl.max("aircraft_leg_number_day").over("tail_number"))
            .alias("relative_leg_position")
        )

    out = lf.filter(
        (pl.col("is_cancelled") == 0)
        & (pl.col("is_diverted") == 0)
        & pl.col("arr_del15").is_not_null()
        & pl.col("tail_number").is_not_null()
        & pl.col("dep_ts_actual_utc").is_not_null()
        & pl.col("arr_ts_actual_utc").is_not_null()
    )
    if exprs:
        out = out.with_columns(exprs)

    out_cols = set(out.collect_schema().names())
    needed_two_hop = {
        "prev1_arr_delay", "prev1_dep_delay", "prev1_arr_del15", "prev1_dep_del15",
        "prev2_arr_delay", "prev2_dep_delay", "prev2_arr_del15", "prev2_dep_del15",
        "prev1_turnaround_minutes", "time_since_prev2_arrival_minutes",
    }
    can_build_two_hop = {
        "tail_number", "dep_ts_actual_utc", "arr_ts_actual_utc",
        "arr_delay", "dep_delay", "arr_del15", "dep_del15",
    }.issubset(out_cols)

    if can_build_two_hop and not needed_two_hop.issubset(out_cols):
        out = (
            out
            .sort(["tail_number", "dep_ts_actual_utc"])
            .with_columns([
                pl.col("arr_delay").shift(1).over("tail_number").alias("prev1_arr_delay"),
                pl.col("dep_delay").shift(1).over("tail_number").alias("prev1_dep_delay"),
                pl.col("arr_del15").shift(1).over("tail_number").alias("prev1_arr_del15"),
                pl.col("dep_del15").shift(1).over("tail_number").alias("prev1_dep_del15"),
                pl.col("arr_delay").shift(2).over("tail_number").alias("prev2_arr_delay"),
                pl.col("dep_delay").shift(2).over("tail_number").alias("prev2_dep_delay"),
                pl.col("arr_del15").shift(2).over("tail_number").alias("prev2_arr_del15"),
                pl.col("dep_del15").shift(2).over("tail_number").alias("prev2_dep_del15"),
                (pl.col("dep_ts_actual_utc") - pl.col("arr_ts_actual_utc").shift(1).over("tail_number"))
                .dt.total_minutes().alias("prev1_turnaround_minutes"),
                (pl.col("dep_ts_actual_utc") - pl.col("arr_ts_actual_utc").shift(2).over("tail_number"))
                .dt.total_minutes().alias("time_since_prev2_arrival_minutes"),
            ])
        )

    return out


# ============================================================
# REQUIRED COLUMNS
# ============================================================

LSTM_REQUIRED_COLUMNS = [
    "year", "dep_ts_actual_utc", TARGET_CLASS, TARGET_REG,
    "prev2_arr_delay", "prev2_dep_delay", "prev2_arr_del15", "prev2_dep_del15",
    "prev1_arr_delay", "prev1_dep_delay", "prev1_arr_del15", "prev1_dep_del15",
    "prev1_turnaround_minutes", "time_since_prev2_arrival_minutes",
    "distance", "dep_hour_local", "dep_weekday_local", "dep_month_local",
    "dep_time_bucket", "is_weekend", "is_holiday", "days_to_nearest_holiday",
    "crs_elapsed_time", "dep_temp_c", "dep_wind_speed_m_s", "dep_wind_dir_deg",
    "dep_ceiling_height_m", "arr_temp_c", "arr_wind_speed_m_s", "arr_wind_dir_deg",
    "arr_ceiling_height_m", "route_frequency", "origin_flight_volume",
    "dest_flight_volume", "tight_turnaround_flag", "relative_leg_position",
    "cum_dep_delay_aircraft_day", "cum_arr_delay_aircraft_day",
]


def resolve_required_columns(
    run_xgb: bool,
    run_lstm: bool,
    xgb_feature_set_name: str | None = None,
) -> list[str]:
    required = {"year", "dep_ts_actual_utc", TARGET_CLASS, TARGET_REG}

    if run_xgb:
        if xgb_feature_set_name is None:
            raise ValueError("xgb_feature_set_name is required when run_xgb=True")
        feature_set_names = (
            [xgb_feature_set_name]
            if isinstance(xgb_feature_set_name, str)
            else list(xgb_feature_set_name)
        )
        for fs_name in feature_set_names:
            if fs_name not in XGB_FEATURE_SETS:
                raise ValueError(f"Unknown XGB feature set: {fs_name}")
            required.update(XGB_FEATURE_SETS[fs_name])

    if run_lstm:
        required.update(LSTM_REQUIRED_COLUMNS)

    return sorted(required)


# ============================================================
# STREAMING LAZY FRAME — the central data accessor
#
# Instead of collecting full DataFrames into RAM, callers receive a
# LazyFrame + year lists and stream one year at a time.  Only
# _collect_year() ever materialises data, and it is discarded
# immediately after use.
# ============================================================

def build_lazy_modeling_frame(
    canonical_dir: str | Path,
    file_pattern: str,
    all_years: list[int],
    required_columns: list[str],
) -> pl.LazyFrame:
    """Build and return the fully-prepared lazy frame.

    Nothing is collected here — the caller decides which years to
    materialise and when.
    """
    t0 = time.perf_counter()
    lf = load_years_lazy(
        canonical_dir=canonical_dir,
        file_pattern=file_pattern,
        years=all_years,
    )
    timer_log("Load years lazy", t0, time)

    t0 = time.perf_counter()
    lf = build_modeling_table(lf)
    if CONFIG.aircraft.enabled:
        logger.info("[DATA] Aircraft enrichment enabled — joining FAA registry")
    lf = enrich_with_aircraft_features(lf, CONFIG.aircraft.cache_path)
    timer_log("Build modeling table lazy", t0, time)

    schema_names = set(lf.collect_schema().names())
    missing = [c for c in required_columns if c not in schema_names]
    if missing:
        logger.error("[DATA] available columns=%s", sorted(schema_names))
        raise ValueError(f"Missing required modeling columns: {missing}")

    return lf.select(required_columns)


def collect_year(
    lf: pl.LazyFrame,
    year: int,
    label: str = "",
    sample_fraction: float | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Collect a single year from the lazy frame and optionally sample it.

    This is the only place data is ever materialised.  Callers should
    del the result and gc.collect() when done.

    sample_fraction is provided only when the user explicitly opts in via
    config (train_sample_fraction / val_sample_fraction).  By default it
    is None and the full year is returned.
    """
    df = lf.filter(pl.col("year") == year).collect(streaming=True)
    raw = df.height

    if sample_fraction is not None and 0 < sample_fraction < 1.0:
        df = df.sample(fraction=sample_fraction, seed=seed, shuffle=True)

    logger.info(
        "[STREAM] %s year=%s raw=%s kept=%s%s",
        label or "collect",
        year,
        f"{raw:,}",
        f"{df.height:,}",
        f" (sampled {sample_fraction:.0%})" if sample_fraction and sample_fraction < 1.0 else "",
    )
    return df


# ============================================================
# ROLLING CV SPLIT DESCRIPTORS
# (year lists only — no DataFrames held in memory)
# ============================================================

def make_rolling_year_cv_descriptors(
    train_years: list[int],
    min_train_years: int = 4,
) -> list[dict[str, Any]]:
    """Return fold descriptors containing only year lists, not DataFrames.

    The caller streams the actual data per fold using collect_year().
    """
    years = sorted(train_years)
    folds: list[dict[str, Any]] = []

    if len(years) < min_train_years + 1:
        logger.warning(
            "[CV] Not enough train years for rolling CV. years=%s min_train_years=%s",
            years, min_train_years,
        )
        return folds

    for fold_num, idx in enumerate(range(min_train_years, len(years)), start=1):
        folds.append({
            "fold": fold_num,
            "train_years": years[:idx],
            "val_year": years[idx],
        })

    return folds


# ============================================================
# LEGACY: collect_modeling_splits
# Kept for backward compatibility and LSTM which still needs
# full DataFrames.  For XGBoost use build_lazy_modeling_frame
# + stream per year.
# ============================================================

def _collect_year_partition(
    lf: pl.LazyFrame,
    years: list[int],
    label: str,
    sample_fraction: float | None = None,
    seed: int = 42,
) -> pl.DataFrame | None:
    if not years:
        logger.info("[SPLIT] %s skipped (no years)", label)
        return None

    if sample_fraction is not None and sample_fraction <= 0:
        raise ValueError(f"sample_fraction must be > 0, got {sample_fraction}")

    t0 = time.perf_counter()
    sampled = sample_fraction is not None and sample_fraction < 1.0
    year_frames: list[pl.DataFrame] = []

    for year in sorted(years):
        ydf = lf.filter(pl.col("year") == year).collect(streaming=True)
        raw = ydf.height
        if sampled:
            ydf = ydf.sample(fraction=sample_fraction, seed=seed, shuffle=True)
        logger.info(
            "[SPLIT] %s year=%s raw=%s kept=%s%s",
            label, year, f"{raw:,}", f"{ydf.height:,}",
            f" (sampled {sample_fraction:.0%})" if sampled else "",
        )
        year_frames.append(ydf)
        del ydf
        gc.collect()

    df = pl.concat(year_frames, how="vertical_relaxed")
    del year_frames
    gc.collect()

    timer_log(f"Collect {label}", t0, time)
    try:
        est_mb = df.estimated_size("mb")
        logger.info(
            "[SPLIT] %s collected | years=%s rows=%s cols=%s est_size≈%.2f MB%s",
            label, years, f"{df.height:,}", df.width, est_mb,
            f" (sampled {sample_fraction:.0%})" if sampled else "",
        )
    except Exception:
        logger.info(
            "[SPLIT] %s collected | years=%s rows=%s cols=%s",
            label, years, f"{df.height:,}", df.width,
        )
    return df


def collect_modeling_splits(
    canonical_dir: str | Path,
    file_pattern: str,
    train_years: list[int],
    validation_years: list[int],
    test_years: list[int],
    use_validation_holdout: bool,
    run_xgb: bool,
    run_lstm: bool,
    xgb_feature_set_name: str | None = None,
    train_sample_fraction: float | None = None,
    val_sample_fraction: float | None = None,
    test_sample_fraction: float | None = None,
    sample_seed: int = 42,
) -> dict[str, pl.DataFrame | None]:
    """Legacy: collect full DataFrames.  Only used by LSTM."""
    all_years = sorted(set(train_years + validation_years + test_years))
    required_columns = resolve_required_columns(
        run_xgb=run_xgb,
        run_lstm=run_lstm,
        xgb_feature_set_name=xgb_feature_set_name,
    )

    logger.info("[DATA] years=%s", all_years)
    logger.info("[DATA] projecting %s required columns", len(required_columns))

    lf = build_lazy_modeling_frame(
        canonical_dir=canonical_dir,
        file_pattern=file_pattern,
        all_years=all_years,
        required_columns=required_columns,
    )

    t0 = time.perf_counter()
    train_df = _collect_year_partition(lf, train_years, "train", train_sample_fraction, sample_seed)
    gc.collect()
    val_df = (
        _collect_year_partition(lf, validation_years, "validation", val_sample_fraction, sample_seed)
        if use_validation_holdout else None
    )
    gc.collect()
    test_df = _collect_year_partition(lf, test_years, "test", test_sample_fraction, sample_seed)
    gc.collect()
    timer_log("Collect all split partitions", t0, time)

    return {"train_df": train_df, "val_df": val_df, "test_df": test_df}


# ============================================================
# LEGACY: make_rolling_year_cv_splits (returns DataFrames)
# Only used by LSTM.  XGBoost uses make_rolling_year_cv_descriptors.
# ============================================================

def make_rolling_year_cv_splits(
    train_df: pl.DataFrame,
    min_train_years: int = 4,
) -> list[dict[str, Any]]:
    years = sorted(train_df["year"].unique().to_list())
    folds: list[dict[str, Any]] = []

    if len(years) < min_train_years + 1:
        logger.warning(
            "[CV] Not enough train years for rolling CV. years=%s min_train_years=%s",
            years, min_train_years,
        )
        return folds

    for fold_num, idx in enumerate(range(min_train_years, len(years)), start=1):
        fold_train_years = years[:idx]
        val_year = years[idx]
        folds.append({
            "fold": fold_num,
            "train_years": fold_train_years,
            "val_year": val_year,
            "train_df": train_df.filter(pl.col("year").is_in(fold_train_years)),
            "val_df": train_df.filter(pl.col("year") == val_year),
        })

    return folds


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results_summary(results_payload: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2, default=str)
    logger.info("[SAVE] Wrote %s", json_path)