# ============================================================
# XGBoost vs LSTM on flight delay classification + regression
# Adapted to your 2-hop propagation modeling table
# ============================================================

from pathlib import Path
import sys
import importlib
import warnings
warnings.filterwarnings("ignore")

import os
import time

# ============================================================
# MULTI-CORE SETUP  ← ADDED
# ============================================================
import multiprocessing
N_CORES = multiprocessing.cpu_count()
print(f"[SYSTEM] Detected {N_CORES} CPU cores")

# Tell NumPy/MKL/OpenBLAS/TensorFlow to use all cores
# NOTE: TF thread env vars MUST be set before importing TensorFlow
os.environ["OMP_NUM_THREADS"]             = str(N_CORES)
os.environ["MKL_NUM_THREADS"]             = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"]        = str(N_CORES)
os.environ["NUMEXPR_NUM_THREADS"]         = str(N_CORES)
os.environ["TF_NUM_INTEROP_THREADS"]      = str(N_CORES)   # ← FIXED: env var instead of post-init call
os.environ["TF_NUM_INTRAOP_THREADS"]      = str(N_CORES)   # ← FIXED: env var instead of post-init call

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier, XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print(f"[TF] inter_op threads: {tf.config.threading.get_inter_op_parallelism_threads()}")
print(f"[TF] intra_op threads: {tf.config.threading.get_intra_op_parallelism_threads()}")

# ============================================================
# TIMER HELPERS
# ============================================================

GLOBAL_START = time.perf_counter()

def timer_log(label, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"[TIMER] {label}: {elapsed:.2f}s")


# ============================================================
# TENSORFLOW DEVICE SETUP
# ============================================================

# If running in Jupyter notebook make sure to restart the kernal before running 
USE_GPU = False

try:
    if USE_GPU:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # don't grab all VRAM at once
            print(f"[TF] GPU mode enabled. Found {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        else:
            print("[TF] WARNING: USE_GPU=True but no GPU found — falling back to CPU.")
    else:
        tf.config.set_visible_devices([], "GPU")
        print("[TF] CPU-only mode enabled.")
except RuntimeError as e:
    print(f"[TF WARNING] Device config must be set before TF runtime init: {e}")


# ============================================================
# 1. LOAD DATA THE SAME WAY YOU ALREADY DO
# ============================================================

t0 = time.perf_counter()

pl.Config.set_tbl_rows(1000)
pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_width_chars(200)

current = Path.cwd()
while current.name != "shared-notebooks":
    if current.parent == current:
        raise RuntimeError("Could not locate shared-notebooks directory")
    current = current.parent

utils_path = current / "common_utils" / "python"
if str(utils_path) not in sys.path:
    sys.path.append(str(utils_path))

import load_flight_data
importlib.reload(load_flight_data)

lf = load_flight_data.load_flight_data(file_name="flights_canonical_2019.parquet")

timer_log("Load data", t0)


# ============================================================
# 2. REBUILD YOUR MODELING TABLE
#    (based directly on your uploaded code)
# ============================================================

t0 = time.perf_counter()

RAW_FEATURES = [
    "flight_id",
    "tail_number",
    "reporting_airline",
    "origin",
    "dest",
    "route_key",
    "distance",
    "flight_date",

    # local calendar/time features
    "dep_hour_local",
    "dep_weekday_local",
    "dep_month_local",

    # UTC timestamps for sequencing
    "dep_ts_sched_utc",
    "dep_ts_actual_utc",
    "arr_ts_sched_utc",
    "arr_ts_actual_utc",

    # weather
    "dep_temp_c",
    "dep_wind_speed_m_s",
    "dep_wind_dir_deg",
    "dep_ceiling_height_m",
    "arr_temp_c",
    "arr_wind_speed_m_s",
    "arr_wind_dir_deg",
    "arr_ceiling_height_m",

    # existing same-tail linkage
    "prev_flight_id_same_tail",
    "next_flight_id_same_tail",
    "prev_origin",
    "prev_dest",
    "next_origin",
    "next_dest",
    "prev_arr_ts_actual_utc",
    "next_dep_ts_actual_utc",

    # delays
    "dep_delay",
    "dep_del15",
    "arr_delay",
    "arr_del15",

    # existing lag features
    "prev_arr_delay",
    "prev_dep_delay",
    "next_arr_delay",
    "next_dep_delay",
    "prev_arr_del15",
    "prev_dep_del15",
    "next_dep_del15",
    "next_arr_del15",
    "prev_arr_late_15",
    "prev_dep_late_15",
    "next_arr_late_15",
    "next_dep_late_15",

    # rotation features
    "turnaround_minutes",
    "next_turnaround_minutes",
    "rotation_continuity_flag",
    "next_rotation_continuity_flag",
    "aircraft_leg_number_day",
    "cum_dep_delay_aircraft_day",
    "cum_arr_delay_aircraft_day",

    # status flags
    "is_cancelled",
    "is_diverted",

    # schedule block features
    "crs_elapsed_time",
    "dep_time_blk",
    "arr_time_blk",
]

US_HOLIDAYS_2019 = [
    "2019-01-01",
    "2019-01-21",
    "2019-02-18",
    "2019-05-27",
    "2019-07-04",
    "2019-09-02",
    "2019-10-14",
    "2019-11-11",
    "2019-11-28",
    "2019-12-25",
]

ml_lf = (
    lf
    .select(RAW_FEATURES)
    .filter(
        (pl.col("is_cancelled") == 0) &
        (pl.col("is_diverted") == 0) &
        pl.col("arr_del15").is_not_null() &
        pl.col("tail_number").is_not_null() &
        pl.col("dep_ts_actual_utc").is_not_null() &
        pl.col("arr_ts_actual_utc").is_not_null()
    )
)

lf_features = (
    ml_lf
    .with_columns([
        # Time bucket
        pl.when(pl.col("dep_hour_local") < 6).then(1)
        .when(pl.col("dep_hour_local") < 11).then(2)
        .when(pl.col("dep_hour_local") < 14).then(3)
        .when(pl.col("dep_hour_local") < 18).then(4)
        .when(pl.col("dep_hour_local") < 21).then(5)
        .otherwise(6)
        .alias("dep_time_bucket"),

        # Weekend flag
        pl.col("dep_weekday_local").is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),

        # Holiday flag
        pl.col("flight_date").cast(pl.Utf8).is_in(US_HOLIDAYS_2019).cast(pl.Int8).alias("is_holiday"),

        # Days to nearest holiday
        pl.min_horizontal([
            *[
                (pl.col("flight_date").cast(pl.Date) - pl.lit(h).str.strptime(pl.Date)).abs().dt.total_days()
                for h in US_HOLIDAYS_2019
            ]
        ]).alias("days_to_nearest_holiday"),

        # Traffic proxies
        pl.len().over("route_key").alias("route_frequency"),
        pl.len().over("origin").alias("origin_flight_volume"),
        pl.len().over("dest").alias("dest_flight_volume"),

        # Propagation strength
        (pl.col("prev_arr_delay") > 15).cast(pl.Int8).alias("prev_arr_delayed_flag"),
        (pl.col("prev_arr_delay") + pl.col("prev_dep_delay")).alias("prev_total_delay"),

        # Turnaround pressure
        (pl.col("turnaround_minutes") < 60).cast(pl.Int8).alias("tight_turnaround_flag"),

        # Relative position in day
        (
            pl.col("aircraft_leg_number_day") /
            pl.max("aircraft_leg_number_day").over("tail_number")
        ).alias("relative_leg_position"),
    ])
)

usa_2hop_lf = (
    lf_features
    .sort(["tail_number", "dep_ts_actual_utc"])
    .with_columns([
        # 1 hop back
        pl.col("flight_id").shift(1).over("tail_number").alias("prev1_flight_id"),
        pl.col("origin").shift(1).over("tail_number").alias("prev1_origin"),
        pl.col("dest").shift(1).over("tail_number").alias("prev1_dest"),
        pl.col("dep_ts_actual_utc").shift(1).over("tail_number").alias("prev1_dep_ts_utc"),
        pl.col("arr_ts_actual_utc").shift(1).over("tail_number").alias("prev1_arr_ts_utc"),
        pl.col("arr_delay").shift(1).over("tail_number").alias("prev1_arr_delay"),
        pl.col("dep_delay").shift(1).over("tail_number").alias("prev1_dep_delay"),
        pl.col("arr_del15").shift(1).over("tail_number").alias("prev1_arr_del15"),
        pl.col("dep_del15").shift(1).over("tail_number").alias("prev1_dep_del15"),

        # 2 hops back
        pl.col("flight_id").shift(2).over("tail_number").alias("prev2_flight_id"),
        pl.col("origin").shift(2).over("tail_number").alias("prev2_origin"),
        pl.col("dest").shift(2).over("tail_number").alias("prev2_dest"),
        pl.col("dep_ts_actual_utc").shift(2).over("tail_number").alias("prev2_dep_ts_utc"),
        pl.col("arr_ts_actual_utc").shift(2).over("tail_number").alias("prev2_arr_ts_utc"),
        pl.col("arr_delay").shift(2).over("tail_number").alias("prev2_arr_delay"),
        pl.col("dep_delay").shift(2).over("tail_number").alias("prev2_dep_delay"),
        pl.col("arr_del15").shift(2).over("tail_number").alias("prev2_arr_del15"),
        pl.col("dep_del15").shift(2).over("tail_number").alias("prev2_dep_del15"),

        # timing gaps
        (
            pl.col("dep_ts_actual_utc") -
            pl.col("arr_ts_actual_utc").shift(1).over("tail_number")
        ).dt.total_minutes().alias("prev1_turnaround_minutes"),

        (
            pl.col("dep_ts_actual_utc") -
            pl.col("arr_ts_actual_utc").shift(2).over("tail_number")
        ).dt.total_minutes().alias("time_since_prev2_arrival_minutes"),
    ])
    .filter(
        pl.col("prev1_flight_id").is_not_null() &
        pl.col("prev2_flight_id").is_not_null() &
        pl.col("prev1_turnaround_minutes").is_not_null() &
        pl.col("time_since_prev2_arrival_minutes").is_not_null() &
        pl.col("prev1_turnaround_minutes").is_between(0, 12 * 60) &
        pl.col("time_since_prev2_arrival_minutes").is_between(0, 24 * 60)
    )
)

flights = usa_2hop_lf.collect()
print("Rows in final modeling table:", flights.height)
print(flights.select(["flight_id", "tail_number", "origin", "dest", "arr_delay", "arr_del15"]).head())

timer_log("Feature engineering + collect", t0)


# ============================================================
# 3. DEFINE FEATURE SETS
#    Avoid next_* columns to prevent leakage
# ============================================================

xgb_schedule_features = [
    "distance",
    "dep_hour_local",
    "dep_weekday_local",
    "dep_month_local",
    "dep_time_bucket",
    "is_weekend",
    "is_holiday",
    "days_to_nearest_holiday",
    "crs_elapsed_time",
]

xgb_weather_traffic_features = xgb_schedule_features + [
    "dep_temp_c",
    "dep_wind_speed_m_s",
    "dep_wind_dir_deg",
    "dep_ceiling_height_m",
    "arr_temp_c",
    "arr_wind_speed_m_s",
    "arr_wind_dir_deg",
    "arr_ceiling_height_m",
    "route_frequency",
    "origin_flight_volume",
    "dest_flight_volume",
]

xgb_full_features = xgb_weather_traffic_features + [
    "turnaround_minutes",
    "tight_turnaround_flag",
    "rotation_continuity_flag",
    "aircraft_leg_number_day",
    "relative_leg_position",
    "cum_dep_delay_aircraft_day",
    "cum_arr_delay_aircraft_day",
    "prev1_arr_delay",
    "prev1_dep_delay",
    "prev1_arr_del15",
    "prev1_dep_del15",
    "prev2_arr_delay",
    "prev2_dep_delay",
    "prev2_arr_del15",
    "prev2_dep_del15",
    "prev1_turnaround_minutes",
    "time_since_prev2_arrival_minutes",
]

lstm_delay_only_current = [
    "distance",
    "dep_hour_local",
    "dep_weekday_local",
    "dep_month_local",
    "dep_time_bucket",
    "is_weekend",
    "is_holiday",
    "days_to_nearest_holiday",
    "crs_elapsed_time",
]

lstm_context_current = lstm_delay_only_current + [
    "dep_temp_c",
    "dep_wind_speed_m_s",
    "dep_wind_dir_deg",
    "dep_ceiling_height_m",
    "arr_temp_c",
    "arr_wind_speed_m_s",
    "arr_wind_dir_deg",
    "arr_ceiling_height_m",
    "route_frequency",
    "origin_flight_volume",
    "dest_flight_volume",
    "tight_turnaround_flag",
    "relative_leg_position",
    "cum_dep_delay_aircraft_day",
    "cum_arr_delay_aircraft_day",
]

TARGET_CLASS = "arr_del15"
TARGET_REG = "arr_delay"


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

t0 = time.perf_counter()

split_date = pd.Timestamp("2019-11-01")

flights = flights.with_columns(
    pl.col("dep_ts_actual_utc").cast(pl.Datetime).alias("dep_ts_actual_utc")
)

train_df = flights.filter(pl.col("dep_ts_actual_utc") < split_date)
test_df = flights.filter(pl.col("dep_ts_actual_utc") >= split_date)

print("Train rows:", train_df.height)
print("Test rows :", test_df.height)

timer_log("Train/test split", t0)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def safe_fill_and_to_pandas(df_pl: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df_pl.select(cols).to_pandas()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = out[c].fillna(out[c].median())
        else:
            out[c] = out[c].fillna("missing")
    return out

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

def build_eval_table(results_dict):
    rows = []
    for model_name, res in results_dict.items():
        row = {"model": model_name}
        row.update(res)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


# ============================================================
# 6. XGBOOST TRAINING  ← n_jobs=-1 already set, kept as-is
# ============================================================

def train_xgb_models(train_df, test_df, feature_sets):
    t_total = time.perf_counter()

    cls_results = {}
    reg_results = {}
    cls_preds = {}
    reg_preds = {}
    fitted_models = {}

    y_train_cls = train_df[TARGET_CLASS].to_pandas().astype(int)
    y_test_cls = test_df[TARGET_CLASS].to_pandas().astype(int)
    y_train_reg = train_df[TARGET_REG].to_pandas().astype(float)
    y_test_reg = test_df[TARGET_REG].to_pandas().astype(float)

    for model_name, cols in feature_sets.items():
        t_model = time.perf_counter()

        X_train = safe_fill_and_to_pandas(train_df, cols)
        X_test = safe_fill_and_to_pandas(test_df, cols)

        xgb_device = "cuda" if USE_GPU else "cpu"   # ← GPU for XGBoost if available

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            device=xgb_device,  # ← ADDED
        )
        clf.fit(X_train, y_train_cls)
        y_prob = clf.predict_proba(X_test)[:, 1]

        reg = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            device=xgb_device,  # ← ADDED
        )
        reg.fit(X_train, y_train_reg)
        y_pred_reg = reg.predict(X_test)

        cls_results[model_name] = classification_metrics(y_test_cls, y_prob)
        reg_results[model_name] = regression_metrics(y_test_reg, y_pred_reg)
        cls_preds[model_name] = y_prob
        reg_preds[model_name] = y_pred_reg
        fitted_models[model_name] = {"classifier": clf, "regressor": reg, "features": cols}

        print(f"Done: {model_name}")
        timer_log(f"XGB {model_name}", t_model)

    timer_log("Total XGBoost training", t_total)
    return cls_results, reg_results, cls_preds, reg_preds, fitted_models


xgb_feature_sets = {
    "xgb_schedule": xgb_schedule_features,
    "xgb_weather_traffic": xgb_weather_traffic_features,
    "xgb_full_propagation": xgb_full_features,
}

t0 = time.perf_counter()
xgb_cls_results, xgb_reg_results, xgb_cls_preds, xgb_reg_preds, xgb_models = train_xgb_models(
    train_df, test_df, xgb_feature_sets
)
timer_log("XGBoost block", t0)


# ============================================================
# 7. LSTM DATA BUILDERS
# ============================================================

STEP_FEATURES = [
    "arr_delay_prev",
    "dep_delay_prev",
    "arr_del15_prev",
    "dep_del15_prev",
    "gap_minutes",
    "distance",
    "dep_hour_local",
    "dep_weekday_local",
    "dep_month_local",
    "dep_time_bucket",
    "is_weekend",
    "is_holiday",
    "days_to_nearest_holiday",
    "crs_elapsed_time",
    "dep_temp_c",
    "dep_wind_speed_m_s",
    "dep_wind_dir_deg",
    "dep_ceiling_height_m",
    "arr_temp_c",
    "arr_wind_speed_m_s",
    "arr_wind_dir_deg",
    "arr_ceiling_height_m",
    "route_frequency",
    "origin_flight_volume",
    "dest_flight_volume",
    "tight_turnaround_flag",
    "relative_leg_position",
    "cum_dep_delay_aircraft_day",
    "cum_arr_delay_aircraft_day",
]

def build_lstm_step_matrix(df_pl: pl.DataFrame, current_variant="delay_only"):
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
        "cum_dep_delay_aircraft_day", "cum_arr_delay_aircraft_day"
    ]
    for c in numeric_fill_cols:
        if c in pdf.columns:
            pdf[c] = pdf[c].fillna(pdf[c].median())

    if current_variant == "delay_only":
        current_context_cols = [
            "distance", "dep_hour_local", "dep_weekday_local", "dep_month_local",
            "dep_time_bucket", "is_weekend", "is_holiday",
            "days_to_nearest_holiday", "crs_elapsed_time"
        ]
    else:
        current_context_cols = lstm_context_current

    X = np.zeros((len(pdf), 3, len(STEP_FEATURES)), dtype=np.float32)

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
        if col in STEP_FEATURES:
            X[:, 2, STEP_FEATURES.index(col)] = pdf[col].values

    y_cls = pdf[TARGET_CLASS].astype(int).values
    y_reg = pdf[TARGET_REG].astype(float).values

    return X, y_cls, y_reg, pdf

def scale_lstm(X_train, X_test):
    n_train, timesteps, n_features = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, timesteps, n_features)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, timesteps, n_features)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), scaler

def build_lstm_model(input_shape, units1=32, units2=16, dropout=0.2):
    inp = Input(shape=input_shape)
    x = LSTM(units1, return_sequences=True)(inp)
    x = Dropout(dropout)(x)
    x = LSTM(units2)(x)
    x = Dropout(dropout)(x)

    cls_out = Dense(1, activation="sigmoid", name="cls")(x)
    reg_out = Dense(1, activation="linear", name="reg")(x)

    model = Model(inputs=inp, outputs=[cls_out, reg_out])
    model.compile(
        optimizer="adam",
        loss={"cls": "binary_crossentropy", "reg": "mse"},
        metrics={"cls": [tf.keras.metrics.AUC(name="auc")], "reg": ["mae"]},
    )
    return model

def train_lstm_variant(train_df, test_df, variant_name, current_variant="delay_only", units1=32, units2=16, epochs=10):
    t_variant = time.perf_counter()

    X_train, y_train_cls, y_train_reg, train_pdf = build_lstm_step_matrix(train_df, current_variant=current_variant)
    X_test, y_test_cls, y_test_reg, test_pdf = build_lstm_step_matrix(test_df, current_variant=current_variant)

    X_train_scaled, X_test_scaled, scaler = scale_lstm(X_train, X_test)

    y_train_cls = np.asarray(y_train_cls, dtype=np.float32)
    y_train_reg = np.asarray(y_train_reg, dtype=np.float32)
    y_test_cls = np.asarray(y_test_cls, dtype=np.float32)
    y_test_reg = np.asarray(y_test_reg, dtype=np.float32)

    model = build_lstm_model(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        units1=units1,
        units2=units2,
        dropout=0.2,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled,
        {"cls": y_train_cls, "reg": y_train_reg},
        validation_split=0.15,
        epochs=epochs,
        batch_size=256,         # increase if you have RAM headroom (e.g. 512 or 1024)
        callbacks=[early_stop],
        verbose=1,
    )

    pred_cls, pred_reg = model.predict(X_test_scaled, verbose=0)
    pred_cls = pred_cls.ravel()
    pred_reg = pred_reg.ravel()

    cls_metrics = classification_metrics(y_test_cls.astype(int), pred_cls)
    reg_metrics = regression_metrics(y_test_reg, pred_reg)

    timer_log(f"LSTM {variant_name}", t_variant)

    return {
        "variant_name": variant_name,
        "model": model,
        "history": history,
        "X_test": X_test_scaled,
        "test_pdf": test_pdf,
        "y_test_cls": y_test_cls,
        "y_test_reg": y_test_reg,
        "pred_cls": pred_cls,
        "pred_reg": pred_reg,
        "cls_metrics": cls_metrics,
        "reg_metrics": reg_metrics,
        "scaler": scaler,
    }


# ============================================================
# 8. TRAIN LSTM VARIANTS
# ============================================================

t0 = time.perf_counter()

lstm_runs = {}

lstm_runs["lstm_delay_small"] = train_lstm_variant(
    train_df, test_df,
    variant_name="lstm_delay_small",
    current_variant="delay_only",
    units1=32, units2=16, epochs=10
)

lstm_runs["lstm_delay_medium"] = train_lstm_variant(
    train_df, test_df,
    variant_name="lstm_delay_medium",
    current_variant="delay_only",
    units1=64, units2=32, epochs=10
)

lstm_runs["lstm_context_full"] = train_lstm_variant(
    train_df, test_df,
    variant_name="lstm_context_full",
    current_variant="context_full",
    units1=64, units2=32, epochs=10
)

timer_log("Total LSTM training", t0)


# ============================================================
# 9. COLLECT RESULTS
# ============================================================

t0 = time.perf_counter()

all_cls_results = {}
all_reg_results = {}

for k, v in xgb_cls_results.items():
    all_cls_results[k] = v
for k, v in xgb_reg_results.items():
    all_reg_results[k] = v

for name, run in lstm_runs.items():
    all_cls_results[name] = run["cls_metrics"]
    all_reg_results[name] = run["reg_metrics"]

cls_table = build_eval_table(all_cls_results).sort_values("auc", ascending=False)
reg_table = build_eval_table(all_reg_results).sort_values("mae", ascending=True)

print("\nClassification Results")
print(cls_table)

print("\nRegression Results")
print(reg_table)

timer_log("Collect results tables", t0)


# ============================================================
# 10. VISUALIZATION: MODEL COMPARISON BARS
# ============================================================

def plot_metric_bars(df, metric, title, sort_ascending=False):
    plot_df = df.sort_values(metric, ascending=sort_ascending).copy()

    plt.figure(figsize=(12, 5))
    plt.bar(plot_df["model"], plot_df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.show()

t0 = time.perf_counter()
plot_metric_bars(cls_table, "auc", "Classification Comparison: AUC", sort_ascending=False)
plot_metric_bars(cls_table, "f1", "Classification Comparison: F1", sort_ascending=False)
plot_metric_bars(reg_table, "mae", "Regression Comparison: MAE (lower is better)", sort_ascending=True)
plot_metric_bars(reg_table, "rmse", "Regression Comparison: RMSE (lower is better)", sort_ascending=True)
timer_log("Metric bar visualizations", t0)


# ============================================================
# 11. VISUALIZATION: ROC CURVES FOR BEST MODELS
# ============================================================

t0 = time.perf_counter()

best_cls_models = cls_table["model"].head(3).tolist()

plt.figure(figsize=(8, 6))

y_test_cls = test_df[TARGET_CLASS].to_pandas().astype(int)

for model_name in best_cls_models:
    if model_name.startswith("xgb"):
        y_prob = xgb_cls_preds[model_name]
    else:
        y_prob = lstm_runs[model_name]["pred_cls"]

    fpr, tpr, _ = roc_curve(y_test_cls, y_prob)
    auc_val = roc_auc_score(y_test_cls, y_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Best Classification Models")
plt.legend()
plt.tight_layout()
plt.show()

timer_log("ROC curve visualization", t0)


# ============================================================
# 12. VISUALIZATION: ACTUAL VS PREDICTED DELAY
# ============================================================

t0 = time.perf_counter()

best_reg_model = reg_table.iloc[0]["model"]
y_test_reg = test_df[TARGET_REG].to_pandas().astype(float)

if best_reg_model.startswith("xgb"):
    best_pred_reg = xgb_reg_preds[best_reg_model]
else:
    best_pred_reg = lstm_runs[best_reg_model]["pred_reg"]

sample_n = min(5000, len(y_test_reg))
idx = np.random.RandomState(42).choice(len(y_test_reg), size=sample_n, replace=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_reg.iloc[idx], np.array(best_pred_reg)[idx], alpha=0.3)
plt.xlabel("Actual Arrival Delay (minutes)")
plt.ylabel("Predicted Arrival Delay (minutes)")
plt.title(f"Actual vs Predicted Delay: {best_reg_model}")
plt.tight_layout()
plt.show()

timer_log("Actual vs predicted visualization", t0)


# ============================================================
# 13. FEATURE IMPORTANCE FOR BEST XGBOOST MODEL
# ============================================================

t0 = time.perf_counter()

best_xgb_cls_model_name = cls_table[cls_table["model"].str.startswith("xgb")].iloc[0]["model"]
best_xgb = xgb_models[best_xgb_cls_model_name]["classifier"]
best_xgb_features = xgb_models[best_xgb_cls_model_name]["features"]

importance_df = pd.DataFrame({
    "feature": best_xgb_features,
    "importance": best_xgb.feature_importances_
}).sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(10, 7))
plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
plt.title(f"Top Feature Importances: {best_xgb_cls_model_name}")
plt.tight_layout()
plt.show()

timer_log("Feature importance visualization", t0)


# ============================================================
# 14. DEMO TEST CASE EXPLANATION
# ============================================================

t0 = time.perf_counter()

# get the test hold outs 
test_pdf_full = test_df.to_pandas().copy()

""" 
We intentionally found a find a strong delay propagation example so 
that it is easy to see how the model might work. 
"""
demo_candidates = test_pdf_full[
    (test_pdf_full["arr_del15"] == 1) &
    (test_pdf_full["prev1_arr_delay"] > 15) &
    (test_pdf_full["prev2_arr_delay"] > 15)
].copy()

"""
 if no such strong propagation case exists in the test set,
 then just pick any delayed test flight
"""
if len(demo_candidates) == 0:
    demo_candidates = test_pdf_full[test_pdf_full["arr_del15"] == 1].copy()

demo_row = demo_candidates.iloc[0]
demo_flight_id = demo_row["flight_id"]

print("\n================ DEMO TEST CASE ================\n")
print(f"Flight ID: {demo_row['flight_id']}")
print(f"Tail Number: {demo_row['tail_number']}")
print(f"Route: {demo_row['origin']} -> {demo_row['dest']}")
print(f"Scheduled Distance: {demo_row['distance']}")
print(f"Departure hour local: {demo_row['dep_hour_local']}")
print(f"Actual target: arr_del15={demo_row['arr_del15']}, arr_delay={demo_row['arr_delay']:.1f} minutes")
print()
print("Propagation context:")
print(f"prev2_arr_delay={demo_row['prev2_arr_delay']:.1f}, prev2_dep_delay={demo_row['prev2_dep_delay']:.1f}")
print(f"prev1_arr_delay={demo_row['prev1_arr_delay']:.1f}, prev1_dep_delay={demo_row['prev1_dep_delay']:.1f}")
print(f"prev1_turnaround_minutes={demo_row['prev1_turnaround_minutes']:.1f}")
print(f"time_since_prev2_arrival_minutes={demo_row['time_since_prev2_arrival_minutes']:.1f}")
print()
print("Current context:")
print(f"dep_temp_c={demo_row['dep_temp_c']:.2f}, dep_wind_speed_m_s={demo_row['dep_wind_speed_m_s']:.2f}")
print(f"route_frequency={demo_row['route_frequency']}, origin_flight_volume={demo_row['origin_flight_volume']}, dest_flight_volume={demo_row['dest_flight_volume']}")
print(f"is_holiday={demo_row['is_holiday']}, is_weekend={demo_row['is_weekend']}, dep_time_bucket={demo_row['dep_time_bucket']}")

demo_index = test_pdf_full.index[test_pdf_full["flight_id"] == demo_flight_id][0]

print("\nModel predictions for this test case:\n")
for model_name in cls_table["model"].tolist():
    if model_name.startswith("xgb"):
        prob = xgb_cls_preds[model_name][demo_index]
        delay_pred = xgb_reg_preds[model_name][demo_index]
    else:
        prob = lstm_runs[model_name]["pred_cls"][demo_index]
        delay_pred = lstm_runs[model_name]["pred_reg"][demo_index]

    print(f"{model_name:24s}  P(delay>=15)={prob:.3f}   Predicted delay={delay_pred:.1f} min")

timer_log("Demo test case explanation", t0)


# ============================================================
# 15. VISUAL DEMO: PRIOR HOPS -> CURRENT FLIGHT
# ============================================================

t0 = time.perf_counter()

demo_chain = pd.DataFrame([
    {
        "step": "prev2",
        "flight_id": demo_row["prev2_flight_id"],
        "route": f"{demo_row['prev2_origin']} -> {demo_row['prev2_dest']}",
        "arr_delay": demo_row["prev2_arr_delay"],
        "dep_delay": demo_row["prev2_dep_delay"],
        "gap_minutes": demo_row["time_since_prev2_arrival_minutes"],
    },
    {
        "step": "prev1",
        "flight_id": demo_row["prev1_flight_id"],
        "route": f"{demo_row['prev1_origin']} -> {demo_row['prev1_dest']}",
        "arr_delay": demo_row["prev1_arr_delay"],
        "dep_delay": demo_row["prev1_dep_delay"],
        "gap_minutes": demo_row["prev1_turnaround_minutes"],
    },
    {
        "step": "current",
        "flight_id": demo_row["flight_id"],
        "route": f"{demo_row['origin']} -> {demo_row['dest']}",
        "arr_delay": demo_row["arr_delay"],
        "dep_delay": demo_row["dep_delay"],
        "gap_minutes": np.nan,
    },
])

print("\nPropagation chain for demo case:\n")
print(demo_chain)

plt.figure(figsize=(9, 4))
plt.plot(demo_chain["step"], demo_chain["arr_delay"], marker="o", label="Arrival Delay")
plt.plot(demo_chain["step"], demo_chain["dep_delay"], marker="o", label="Departure Delay")
plt.title("Demo Flight: Propagation Across Previous Two Hops")
plt.ylabel("Delay (minutes)")
plt.legend()
plt.tight_layout()
plt.show()

timer_log("Propagation chain visualization", t0)

print()
timer_log("TOTAL PIPELINE TIME", GLOBAL_START)