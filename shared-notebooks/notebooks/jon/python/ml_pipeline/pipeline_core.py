from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from feature_definitions import (
    TARGET_CLASS,
    TARGET_REG,
    TIME_COL,
    YEAR_COL,
    LSTM_CONTEXT_CURRENT,
    LSTM_DELAY_ONLY_CURRENT,
    STEP_FEATURES,
)


def timer_log(label: str, start_time: float, time_module) -> None:
    elapsed = time_module.perf_counter() - start_time
    print(f"[TIMER] {label}: {elapsed:.2f}s")


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def summarize_cv_metrics(metric_list: list[dict[str, float]], prefix: str = "cv") -> dict[str, float]:
    keys = metric_list[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_list], dtype=float)
        out[f"{prefix}_{k}_mean"] = float(vals.mean())
        out[f"{prefix}_{k}_std"] = float(vals.std())
    return out


def rank_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(
            by=["cv_auc_mean", "cv_f1_mean", "cv_mae_mean", "cv_rmse_mean"],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )


def safe_fill_and_to_pandas(df_pl: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df_pl.select(cols).to_pandas()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            med = out[c].median()
            if pd.isna(med):
                med = 0
            out[c] = out[c].fillna(med)
        else:
            out[c] = out[c].fillna("missing")
    return out


def maybe_sample(df: pl.DataFrame, fraction: float | None, seed: int) -> pl.DataFrame:
    if fraction is None or fraction >= 1.0:
        return df
    if fraction <= 0:
        raise ValueError("sample fraction must be > 0")
    return df.sample(fraction=fraction, seed=seed, shuffle=True)


def load_years_lazy(canonical_dir: str, file_pattern: str, years: list[int]) -> pl.LazyFrame:
    lfs: list[pl.LazyFrame] = []
    for year in years:
        path = Path(canonical_dir) / file_pattern.format(year=year)
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical file: {path}")
        print(f"[LOAD] {path}")
        lfs.append(pl.scan_parquet(path))
    return pl.concat(lfs, how="vertical_relaxed")


def build_modeling_table(lf: pl.LazyFrame) -> pl.LazyFrame:
    # Assumes the canonical datasets already include the major columns you engineered.
    # Add only lightweight and reusable derived fields here.

    holidays = [
        "2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01",
        "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01",
    ]

    return (
        lf
        .filter(
            (pl.col("is_cancelled") == 0) &
            (pl.col("is_diverted") == 0) &
            pl.col(TARGET_CLASS).is_not_null() &
            pl.col("tail_number").is_not_null() &
            pl.col(TIME_COL).is_not_null()
        )
        .with_columns([
            pl.col(TIME_COL).cast(pl.Datetime).alias(TIME_COL),
            pl.col(TIME_COL).cast(pl.Datetime).dt.year().alias(YEAR_COL),

            pl.when(pl.col("dep_hour_local") < 6).then(1)
            .when(pl.col("dep_hour_local") < 11).then(2)
            .when(pl.col("dep_hour_local") < 14).then(3)
            .when(pl.col("dep_hour_local") < 18).then(4)
            .when(pl.col("dep_hour_local") < 21).then(5)
            .otherwise(6)
            .alias("dep_time_bucket"),

            pl.col("dep_weekday_local").is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),

            pl.col("flight_date").cast(pl.Utf8).is_in(holidays).cast(pl.Int8).alias("is_holiday"),

            pl.min_horizontal([
                *[
                    (pl.col("flight_date").cast(pl.Date) - pl.lit(h).str.strptime(pl.Date)).abs().dt.total_days()
                    for h in holidays
                ]
            ]).alias("days_to_nearest_holiday"),

            pl.len().over("route_key").alias("route_frequency"),
            pl.len().over("origin").alias("origin_flight_volume"),
            pl.len().over("dest").alias("dest_flight_volume"),

            (pl.col("prev_arr_delay") > 15).cast(pl.Int8).alias("prev_arr_delayed_flag"),
            (pl.col("prev_arr_delay") + pl.col("prev_dep_delay")).alias("prev_total_delay"),

            (pl.col("turnaround_minutes") < 60).cast(pl.Int8).alias("tight_turnaround_flag"),

            (
                pl.col("aircraft_leg_number_day") /
                pl.max("aircraft_leg_number_day").over(["tail_number", "flight_date"])
            ).alias("relative_leg_position"),
        ])
    )


def collect_full_modeling_data(
    canonical_dir: str,
    file_pattern: str,
    years: list[int],
) -> pl.DataFrame:
    lf = load_years_lazy(canonical_dir=canonical_dir, file_pattern=file_pattern, years=years)
    lf = build_modeling_table(lf)
    # streaming=True helps when possible
    df = lf.collect(streaming=True)
    print(f"[DATA] collected rows: {df.height:,}")
    return df


def split_train_val_test(
    df: pl.DataFrame,
    train_years: list[int],
    validation_years: list[int],
    test_years: list[int],
    use_validation_holdout: bool,
) -> dict[str, pl.DataFrame | None]:
    train_df = df.filter(pl.col(YEAR_COL).is_in(train_years))
    val_df = df.filter(pl.col(YEAR_COL).is_in(validation_years)) if use_validation_holdout else None
    test_df = df.filter(pl.col(YEAR_COL).is_in(test_years))

    print(f"[SPLIT] train rows: {train_df.height:,}")
    print(f"[SPLIT] val rows  : {0 if val_df is None else val_df.height:,}")
    print(f"[SPLIT] test rows : {test_df.height:,}")

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def make_rolling_year_cv_splits(
    train_df: pl.DataFrame,
    min_train_years: int = 4,
) -> list[dict[str, Any]]:
    all_years = sorted(train_df[YEAR_COL].unique().to_list())
    if len(all_years) < min_train_years + 1:
        raise ValueError(
            f"Not enough years for rolling CV. Need at least {min_train_years + 1}, got {len(all_years)}"
        )

    splits: list[dict[str, Any]] = []

    for i in range(min_train_years, len(all_years)):
        train_years = all_years[:i]
        val_year = all_years[i]

        train_fold = train_df.filter(pl.col(YEAR_COL).is_in(train_years))
        val_fold = train_df.filter(pl.col(YEAR_COL) == val_year)

        if train_fold.height == 0 or val_fold.height == 0:
            continue

        splits.append({
            "fold": len(splits) + 1,
            "train_years": train_years,
            "val_year": val_year,
            "train_df": train_fold,
            "val_df": val_fold,
        })

    print(f"[CV] built {len(splits)} rolling year folds")
    for s in splits:
        print(
            f"[CV] fold={s['fold']} "
            f"train_years={s['train_years']} "
            f"val_year={s['val_year']} "
            f"train_rows={s['train_df'].height:,} "
            f"val_rows={s['val_df'].height:,}"
        )

    return splits


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
            med = pdf[c].median()
            if pd.isna(med):
                med = 0
            pdf[c] = pdf[c].fillna(med)

    if variant_name == "delay_only":
        current_context_cols = LSTM_DELAY_ONLY_CURRENT
    elif variant_name == "context_full":
        current_context_cols = LSTM_CONTEXT_CURRENT
    else:
        raise ValueError(f"Unsupported LSTM variant: {variant_name}")

    X = np.zeros((len(pdf), 3, len(STEP_FEATURES)), dtype=np.float32)

    # prev2 step
    X[:, 0, STEP_FEATURES.index("arr_delay_prev")] = pdf["prev2_arr_delay"].values
    X[:, 0, STEP_FEATURES.index("dep_delay_prev")] = pdf["prev2_dep_delay"].values
    X[:, 0, STEP_FEATURES.index("arr_del15_prev")] = pdf["prev2_arr_del15"].values
    X[:, 0, STEP_FEATURES.index("dep_del15_prev")] = pdf["prev2_dep_del15"].values
    X[:, 0, STEP_FEATURES.index("gap_minutes")] = pdf["time_since_prev2_arrival_minutes"].values

    # prev1 step
    X[:, 1, STEP_FEATURES.index("arr_delay_prev")] = pdf["prev1_arr_delay"].values
    X[:, 1, STEP_FEATURES.index("dep_delay_prev")] = pdf["prev1_dep_delay"].values
    X[:, 1, STEP_FEATURES.index("arr_del15_prev")] = pdf["prev1_arr_del15"].values
    X[:, 1, STEP_FEATURES.index("dep_del15_prev")] = pdf["prev1_dep_del15"].values
    X[:, 1, STEP_FEATURES.index("gap_minutes")] = pdf["prev1_turnaround_minutes"].values

    # current step context
    for col in current_context_cols:
        if col in STEP_FEATURES:
            X[:, 2, STEP_FEATURES.index(col)] = pdf[col].values

    y_cls = pdf[TARGET_CLASS].astype(int).values
    y_reg = pdf[TARGET_REG].astype(float).values
    return X, y_cls, y_reg, pdf


def scale_lstm(X_train: np.ndarray, X_val: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    n_train, timesteps, n_features = X_train.shape
    n_val = X_val.shape[0]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, timesteps, n_features)
    X_val_scaled = scaler.transform(X_val_2d).reshape(n_val, timesteps, n_features)

    return X_train_scaled.astype(np.float32), X_val_scaled.astype(np.float32), scaler


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj


def save_results_summary(output_dir: str, filename: str, payload: dict) -> Path:
    output_path = Path(output_dir) / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(payload), f, indent=2)
    print(f"[SAVE] {output_path}")
    return output_path