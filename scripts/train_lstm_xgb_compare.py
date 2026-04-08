from pathlib import Path
import sys
import importlib
import warnings
import time
from datetime import datetime

warnings.filterwarnings("ignore")

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
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier, XGBRegressor

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 0. Logging helpers
# ============================================================

GLOBAL_START = time.time()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    elapsed = time.time() - GLOBAL_START
    print(f"[{now_str()} | +{elapsed:10.2f}s] {msg}", flush=True)


def fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m {rem:.2f}s"


def time_block(name: str):
    class _Timer:
        def __enter__(self):
            self.start = time.time()
            log(f"START: {name}")
            return self

        def __exit__(self, exc_type, exc, tb):
            elapsed = time.time() - self.start
            if exc is None:
                log(f"END:   {name} | duration={fmt_seconds(elapsed)}")
            else:
                log(f"FAIL:  {name} | duration={fmt_seconds(elapsed)} | error={exc}")
            return False
    return _Timer()


# ============================================================
# 1. TensorFlow / GPU setup
# ============================================================

with time_block("TensorFlow GPU detection"):
    log(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    log(f"GPUs detected: {gpus}")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log("Enabled memory growth for detected GPUs")
        except RuntimeError as e:
            log(f"Could not enable memory growth: {e}")


# ============================================================
# 2. Load project utility
# ============================================================

with time_block("Project path setup"):
    pl.Config.set_tbl_rows(1000)
    pl.Config.set_tbl_cols(100)
    pl.Config.set_tbl_width_chars(200)

    current = Path.cwd()
    project_root = current

    if project_root.name == "scripts":
        project_root = project_root.parent
    elif project_root.name == "notebooks":
        project_root = project_root.parent

    shared_notebooks = project_root / "shared-notebooks"
    utils_path = shared_notebooks / "common_utils" / "python"

    log(f"Current working directory: {current}")
    log(f"Resolved project root: {project_root}")
    log(f"Python utils path: {utils_path}")

    if str(utils_path) not in sys.path:
        sys.path.append(str(utils_path))
        log("Added utils path to sys.path")

with time_block("Import load_flight_data utility"):
    import load_flight_data
    importlib.reload(load_flight_data)

with time_block("Load canonical flight dataset (lazy)"):
    lf = load_flight_data.load_flight_data(file_name="flights_canonical_2019.parquet")
    log(f"LazyFrame loaded: {type(lf)}")


# ============================================================
# 3. Rebuild modeling table
# ============================================================

RAW_FEATURES = [
    "flight_id",
    "tail_number",
    "reporting_airline",
    "origin",
    "dest",
    "route_key",
    "distance",
    "flight_date",
    "dep_hour_local",
    "dep_weekday_local",
    "dep_month_local",
    "dep_ts_sched_utc",
    "dep_ts_actual_utc",
    "arr_ts_sched_utc",
    "arr_ts_actual_utc",
    "dep_temp_c",
    "dep_wind_speed_m_s",
    "dep_wind_dir_deg",
    "dep_ceiling_height_m",
    "arr_temp_c",
    "arr_wind_speed_m_s",
    "arr_wind_dir_deg",
    "arr_ceiling_height_m",
    "prev_flight_id_same_tail",
    "next_flight_id_same_tail",
    "prev_origin",
    "prev_dest",
    "next_origin",
    "next_dest",
    "prev_arr_ts_actual_utc",
    "next_dep_ts_actual_utc",
    "dep_delay",
    "dep_del15",
    "arr_delay",
    "arr_del15",
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
    "turnaround_minutes",
    "next_turnaround_minutes",
    "rotation_continuity_flag",
    "next_rotation_continuity_flag",
    "aircraft_leg_number_day",
    "cum_dep_delay_aircraft_day",
    "cum_arr_delay_aircraft_day",
    "is_cancelled",
    "is_diverted",
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

with time_block("Select and filter raw modeling rows"):
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
    log("Created filtered LazyFrame for modeling")

with time_block("Engineer derived tabular features"):
    lf_features = (
        ml_lf
        .with_columns([
            pl.when(pl.col("dep_hour_local") < 6).then(1)
            .when(pl.col("dep_hour_local") < 11).then(2)
            .when(pl.col("dep_hour_local") < 14).then(3)
            .when(pl.col("dep_hour_local") < 18).then(4)
            .when(pl.col("dep_hour_local") < 21).then(5)
            .otherwise(6)
            .alias("dep_time_bucket"),

            pl.col("dep_weekday_local").is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),

            pl.col("flight_date").cast(pl.Utf8).is_in(US_HOLIDAYS_2019).cast(pl.Int8).alias("is_holiday"),

            pl.min_horizontal([
                *[
                    (pl.col("flight_date").cast(pl.Date) - pl.lit(h).str.strptime(pl.Date)).abs().dt.total_days()
                    for h in US_HOLIDAYS_2019
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
                pl.max("aircraft_leg_number_day").over("tail_number")
            ).alias("relative_leg_position"),
        ])
    )
    log("Added calendar, traffic, and propagation helper features")

with time_block("Build 2-hop propagation table"):
    usa_2hop_lf = (
        lf_features
        .sort(["tail_number", "dep_ts_actual_utc"])
        .with_columns([
            pl.col("flight_id").shift(1).over("tail_number").alias("prev1_flight_id"),
            pl.col("origin").shift(1).over("tail_number").alias("prev1_origin"),
            pl.col("dest").shift(1).over("tail_number").alias("prev1_dest"),
            pl.col("dep_ts_actual_utc").shift(1).over("tail_number").alias("prev1_dep_ts_utc"),
            pl.col("arr_ts_actual_utc").shift(1).over("tail_number").alias("prev1_arr_ts_utc"),
            pl.col("arr_delay").shift(1).over("tail_number").alias("prev1_arr_delay"),
            pl.col("dep_delay").shift(1).over("tail_number").alias("prev1_dep_delay"),
            pl.col("arr_del15").shift(1).over("tail_number").alias("prev1_arr_del15"),
            pl.col("dep_del15").shift(1).over("tail_number").alias("prev1_dep_del15"),

            pl.col("flight_id").shift(2).over("tail_number").alias("prev2_flight_id"),
            pl.col("origin").shift(2).over("tail_number").alias("prev2_origin"),
            pl.col("dest").shift(2).over("tail_number").alias("prev2_dest"),
            pl.col("dep_ts_actual_utc").shift(2).over("tail_number").alias("prev2_dep_ts_utc"),
            pl.col("arr_ts_actual_utc").shift(2).over("tail_number").alias("prev2_arr_ts_utc"),
            pl.col("arr_delay").shift(2).over("tail_number").alias("prev2_arr_delay"),
            pl.col("dep_delay").shift(2).over("tail_number").alias("prev2_dep_delay"),
            pl.col("arr_del15").shift(2).over("tail_number").alias("prev2_arr_del15"),
            pl.col("dep_del15").shift(2).over("tail_number").alias("prev2_dep_del15"),

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
    log("Constructed final 2-hop LazyFrame")

with time_block("Collect 2-hop propagation table into memory"):
    flights = usa_2hop_lf.collect()
    log(f"Rows in final modeling table: {flights.height}")
    log(f"Columns in final modeling table: {len(flights.columns)}")


# ============================================================
# 4. Feature sets
# ============================================================

TARGET_CLASS = "arr_del15"
TARGET_REG = "arr_delay"

xgb_schedule_features = [
    "distance", "dep_hour_local", "dep_weekday_local", "dep_month_local",
    "dep_time_bucket", "is_weekend", "is_holiday",
    "days_to_nearest_holiday", "crs_elapsed_time",
]

xgb_weather_traffic_features = xgb_schedule_features + [
    "dep_temp_c", "dep_wind_speed_m_s", "dep_wind_dir_deg", "dep_ceiling_height_m",
    "arr_temp_c", "arr_wind_speed_m_s", "arr_wind_dir_deg", "arr_ceiling_height_m",
    "route_frequency", "origin_flight_volume", "dest_flight_volume",
]

xgb_full_features = xgb_weather_traffic_features + [
    "prev_arr_delay", "prev_dep_delay", "prev_arr_del15", "prev_dep_del15",
    "prev_arr_delayed_flag", "prev_total_delay", "turnaround_minutes",
    "tight_turnaround_flag", "rotation_continuity_flag", "aircraft_leg_number_day",
    "relative_leg_position", "cum_dep_delay_aircraft_day", "cum_arr_delay_aircraft_day",
    "prev1_arr_delay", "prev1_dep_delay", "prev1_arr_del15", "prev1_dep_del15",
    "prev2_arr_delay", "prev2_dep_delay", "prev2_arr_del15", "prev2_dep_del15",
    "prev1_turnaround_minutes", "time_since_prev2_arrival_minutes",
]


# ============================================================
# 5. Time split
# ============================================================

with time_block("Train/test split"):
    split_date = pd.Timestamp("2019-11-01")
    flights = flights.with_columns(pl.col("dep_ts_actual_utc").cast(pl.Datetime))

    train_df = flights.filter(pl.col("dep_ts_actual_utc") < split_date)
    test_df = flights.filter(pl.col("dep_ts_actual_utc") >= split_date)

    log(f"Train rows: {train_df.height}")
    log(f"Test rows : {test_df.height}")


# ============================================================
# 6. Helpers
# ============================================================

def safe_fill_and_to_pandas(df_pl, cols, label="data"):
    with time_block(f"Convert {label} -> pandas for {len(cols)} columns"):
        out = df_pl.select(cols).to_pandas()
        log(f"{label} pandas shape before fill: {out.shape}")

        for c in out.columns:
            if out[c].dtype.kind in "biufc":
                out[c] = out[c].fillna(out[c].median())
            else:
                out[c] = out[c].fillna("missing")

        log(f"{label} pandas shape after fill: {out.shape}")
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


# ============================================================
# 7. XGBoost
# ============================================================

def train_xgb_model(train_df, test_df, features, name):
    log(f"Preparing XGBoost model: {name}")
    log(f"Feature count for {name}: {len(features)}")

    X_train = safe_fill_and_to_pandas(train_df, features, label=f"{name} train features")
    X_test = safe_fill_and_to_pandas(test_df, features, label=f"{name} test features")

    with time_block(f"Prepare targets for {name}"):
        y_train_cls = train_df[TARGET_CLASS].to_pandas().astype(int)
        y_test_cls = test_df[TARGET_CLASS].to_pandas().astype(int)
        y_train_reg = train_df[TARGET_REG].to_pandas().astype(float)
        y_test_reg = test_df[TARGET_REG].to_pandas().astype(float)
        log(f"{name} target lengths: cls_train={len(y_train_cls)}, cls_test={len(y_test_cls)}, reg_train={len(y_train_reg)}, reg_test={len(y_test_reg)}")

    with time_block(f"Fit XGBClassifier for {name}"):
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train_cls)

    with time_block(f"Predict classifier probabilities for {name}"):
        cls_prob = clf.predict_proba(X_test)[:, 1]

    with time_block(f"Fit XGBRegressor for {name}"):
        reg = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        reg.fit(X_train, y_train_reg)

    with time_block(f"Predict regressor outputs for {name}"):
        reg_pred = reg.predict(X_test)

    cls_metrics = classification_metrics(y_test_cls, cls_prob)
    reg_metrics = regression_metrics(y_test_reg, reg_pred)

    log(f"{name} classification metrics: {cls_metrics}")
    log(f"{name} regression metrics: {reg_metrics}")

    return {
        "name": name,
        "cls_metrics": cls_metrics,
        "reg_metrics": reg_metrics,
        "cls_prob": cls_prob,
        "reg_pred": reg_pred,
        "clf": clf,
        "reg": reg,
    }


xgb_runs = []
for features, name in [
    (xgb_schedule_features, "xgb_schedule"),
    (xgb_weather_traffic_features, "xgb_weather_traffic"),
    (xgb_full_features, "xgb_full_propagation"),
]:
    with time_block(f"Full XGBoost pipeline: {name}"):
        xgb_runs.append(train_xgb_model(train_df, test_df, features, name))


# ============================================================
# 8. LSTM prep
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


def build_lstm_step_matrix(df_pl, full_context=True, label="dataset"):
    with time_block(f"Build LSTM step matrix for {label} | full_context={full_context}"):
        pdf = df_pl.to_pandas().copy()
        log(f"{label} pandas shape for LSTM: {pdf.shape}")

        numeric_cols = [
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
        for c in numeric_cols:
            pdf[c] = pdf[c].fillna(pdf[c].median())

        current_basic = [
            "distance", "dep_hour_local", "dep_weekday_local", "dep_month_local",
            "dep_time_bucket", "is_weekend", "is_holiday",
            "days_to_nearest_holiday", "crs_elapsed_time",
        ]
        current_full = current_basic + [
            "dep_temp_c", "dep_wind_speed_m_s", "dep_wind_dir_deg", "dep_ceiling_height_m",
            "arr_temp_c", "arr_wind_speed_m_s", "arr_wind_dir_deg", "arr_ceiling_height_m",
            "route_frequency", "origin_flight_volume", "dest_flight_volume",
            "tight_turnaround_flag", "relative_leg_position",
            "cum_dep_delay_aircraft_day", "cum_arr_delay_aircraft_day",
        ]
        current_cols = current_full if full_context else current_basic

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

        for col in current_cols:
            if col in STEP_FEATURES:
                X[:, 2, STEP_FEATURES.index(col)] = pdf[col].values

        y_cls = pdf[TARGET_CLASS].astype(int).values
        y_reg = pdf[TARGET_REG].astype(float).values

        log(f"{label} LSTM tensor shape: X={X.shape}, y_cls={y_cls.shape}, y_reg={y_reg.shape}")
        return X, y_cls, y_reg


def scale_lstm(X_train, X_test, label="lstm"):
    with time_block(f"Scale LSTM tensors for {label}"):
        n_train, t, f = X_train.shape
        n_test = X_test.shape[0]
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, f)).reshape(n_train, t, f)
        X_test_scaled = scaler.transform(X_test.reshape(-1, f)).reshape(n_test, t, f)

        log(f"{label} scaled shapes: train={X_train_scaled.shape}, test={X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled


def build_lstm_model(input_shape, units1=64, units2=32, dropout=0.2):
    log(f"Building LSTM model | input_shape={input_shape}, units1={units1}, units2={units2}, dropout={dropout}")
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


def train_lstm_variant(train_df, test_df, name, full_context=True, units1=64, units2=32, epochs=10):
    log(f"Preparing LSTM model: {name}")

    X_train, y_train_cls, y_train_reg = build_lstm_step_matrix(
        train_df, full_context=full_context, label=f"{name} train"
    )
    X_test, y_test_cls, y_test_reg = build_lstm_step_matrix(
        test_df, full_context=full_context, label=f"{name} test"
    )

    X_train, X_test = scale_lstm(X_train, X_test, label=name)

    with time_block(f"Build model object for {name}"):
        model = build_lstm_model(
            (X_train.shape[1], X_train.shape[2]),
            units1=units1,
            units2=units2
        )

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    with time_block(f"Fit LSTM for {name}"):
        history = model.fit(
            X_train,
            {"cls": y_train_cls, "reg": y_train_reg},
            validation_split=0.15,
            epochs=epochs,
            batch_size=256,
            callbacks=[early_stop],
            verbose=1,
        )

    with time_block(f"Predict with LSTM for {name}"):
        pred_cls, pred_reg = model.predict(X_test, verbose=0)
        pred_cls = pred_cls.ravel()
        pred_reg = pred_reg.ravel()

    cls_metrics = classification_metrics(y_test_cls, pred_cls)
    reg_metrics = regression_metrics(y_test_reg, pred_reg)

    log(f"{name} classification metrics: {cls_metrics}")
    log(f"{name} regression metrics: {reg_metrics}")

    return {
        "name": name,
        "cls_metrics": cls_metrics,
        "reg_metrics": reg_metrics,
        "pred_cls": pred_cls,
        "pred_reg": pred_reg,
        "model": model,
        "history": history.history,
    }


lstm_runs = []
for cfg in [
    {"name": "lstm_basic", "full_context": False, "units1": 32, "units2": 16, "epochs": 10},
    {"name": "lstm_full_context", "full_context": True, "units1": 64, "units2": 32, "epochs": 10},
]:
    with time_block(f"Full LSTM pipeline: {cfg['name']}"):
        lstm_runs.append(train_lstm_variant(train_df, test_df, **cfg))


# ============================================================
# 9. Summaries
# ============================================================

with time_block("Build summary tables"):
    cls_rows = []
    reg_rows = []

    for run in xgb_runs + lstm_runs:
        cls_row = {"model": run["name"], **run["cls_metrics"]}
        reg_row = {"model": run["name"], **run["reg_metrics"]}
        cls_rows.append(cls_row)
        reg_rows.append(reg_row)

    cls_df = pd.DataFrame(cls_rows).sort_values("auc", ascending=False)
    reg_df = pd.DataFrame(reg_rows).sort_values("mae", ascending=True)

    print("\nClassification Results")
    print(cls_df)

    print("\nRegression Results")
    print(reg_df)


# ============================================================
# 10. Plots
# ============================================================

with time_block("Render plots"):
    plt.figure(figsize=(10, 5))
    plt.bar(cls_df["model"], cls_df["auc"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Classification AUC Comparison")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(reg_df["model"], reg_df["mae"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Regression MAE Comparison")
    plt.tight_layout()
    plt.show()


# ============================================================
# 11. Final runtime
# ============================================================

TOTAL_RUNTIME = time.time() - GLOBAL_START
log(f"TOTAL SCRIPT RUNTIME: {fmt_seconds(TOTAL_RUNTIME)}")