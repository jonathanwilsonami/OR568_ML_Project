from __future__ import annotations

import gc
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
from xgboost import XGBClassifier, XGBRegressor

from feature_definitions import TARGET_CLASS, TARGET_REG, XGB_FEATURE_SETS
from pipeline_core import (
    build_lstm_step_matrix,
    classification_metrics,
    maybe_sample,
    rank_results,
    regression_metrics,
    safe_fill_and_to_pandas,
    scale_lstm,
    summarize_cv_metrics,
)

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ============================================================
# XGBOOST
# ============================================================

def make_xgb_model(params: dict, task: str, use_gpu: bool, n_jobs: int):
    common = dict(
        random_state=42,
        n_jobs=n_jobs,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        **params,
    )
    if task == "classification":
        return XGBClassifier(eval_metric="logloss", **common)
    elif task == "regression":
        return XGBRegressor(**common)
    raise ValueError(f"Unsupported task: {task}")


def run_xgb_time_cv(
    cv_splits: list[dict],
    feature_set_name: str,
    param_grid: list[dict],
    use_gpu: bool,
    n_jobs: int,
    sample_fraction: float | None,
    seed: int,
) -> pd.DataFrame:
    if feature_set_name not in XGB_FEATURE_SETS:
        raise ValueError(f"Unknown XGB feature set: {feature_set_name}")

    feature_cols = XGB_FEATURE_SETS[feature_set_name]
    results_rows = []

    for config_id, params in enumerate(param_grid, start=1):
        cls_fold_metrics = []
        reg_fold_metrics = []

        for split in cv_splits:
            fold = split["fold"]

            train_fold = maybe_sample(split["train_df"], sample_fraction, seed)
            val_fold = maybe_sample(split["val_df"], sample_fraction, seed)

            X_train = safe_fill_and_to_pandas(train_fold, feature_cols)
            X_val = safe_fill_and_to_pandas(val_fold, feature_cols)

            y_train_cls = train_fold[TARGET_CLASS].to_pandas().astype("int8")
            y_val_cls = val_fold[TARGET_CLASS].to_pandas().astype("int8")

            y_train_reg = train_fold[TARGET_REG].to_pandas().astype("float32")
            y_val_reg = val_fold[TARGET_REG].to_pandas().astype("float32")

            clf = make_xgb_model(params, "classification", use_gpu=use_gpu, n_jobs=n_jobs)
            clf.fit(X_train, y_train_cls)
            val_prob = clf.predict_proba(X_val)[:, 1]
            cls_metrics = classification_metrics(y_val_cls, val_prob)

            reg = make_xgb_model(params, "regression", use_gpu=use_gpu, n_jobs=n_jobs)
            reg.fit(X_train, y_train_reg)
            val_pred_reg = reg.predict(X_val)
            reg_metrics = regression_metrics(y_val_reg, val_pred_reg)

            # --- Annotate each fold metric dict with fold number and train/val year info ---
            cls_metrics["fold"] = fold
            cls_metrics["train_years"] = str(split["train_years"])
            cls_metrics["val_year"] = split["val_year"]

            reg_metrics["fold"] = fold
            reg_metrics["train_years"] = str(split["train_years"])
            reg_metrics["val_year"] = split["val_year"]

            cls_fold_metrics.append(cls_metrics)
            reg_fold_metrics.append(reg_metrics)

            print(
                f"[XGB cfg {config_id:02d}] fold={fold} "
                f"AUC={cls_metrics['auc']:.4f} "
                f"F1={cls_metrics['f1']:.4f} "
                f"MAE={reg_metrics['mae']:.4f} "
                f"RMSE={reg_metrics['rmse']:.4f}"
            )

            del clf, reg
            del X_train, X_val
            del y_train_cls, y_val_cls, y_train_reg, y_val_reg
            gc.collect()

        row = {
            "model_family": "xgb",
            "feature_set_name": feature_set_name,
            "config_id": f"xgb_{config_id:02d}",
            "params": deepcopy(params),
            # --- Per-fold metric lists preserved for downstream saving ---
            "fold_cls_metrics": deepcopy(cls_fold_metrics),
            "fold_reg_metrics": deepcopy(reg_fold_metrics),
        }
        row.update(summarize_cv_metrics(cls_fold_metrics, prefix="cv"))
        row.update(summarize_cv_metrics(reg_fold_metrics, prefix="cv"))
        results_rows.append(row)

    return rank_results(pd.DataFrame(results_rows))


def refit_best_xgb(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    feature_set_name: str,
    best_params: dict,
    use_gpu: bool,
    n_jobs: int,
) -> dict:
    feature_cols = XGB_FEATURE_SETS[feature_set_name]

    X_train = safe_fill_and_to_pandas(train_df, feature_cols)
    X_test = safe_fill_and_to_pandas(test_df, feature_cols)

    print(f"[XGB FINAL] X_train shape={X_train.shape}")
    print(f"[XGB FINAL] X_test shape={X_test.shape}")

    y_train_cls = train_df[TARGET_CLASS].to_pandas().astype("int8")
    y_test_cls = test_df[TARGET_CLASS].to_pandas().astype("int8")

    y_train_reg = train_df[TARGET_REG].to_pandas().astype("float32")
    y_test_reg = test_df[TARGET_REG].to_pandas().astype("float32")

    clf = make_xgb_model(best_params, "classification", use_gpu=use_gpu, n_jobs=n_jobs)
    clf.fit(X_train, y_train_cls)
    test_prob = clf.predict_proba(X_test)[:, 1]

    reg = make_xgb_model(best_params, "regression", use_gpu=use_gpu, n_jobs=n_jobs)
    reg.fit(X_train, y_train_reg)
    test_pred_reg = reg.predict(X_test)

    # Free large training arrays before returning
    del X_train, y_train_cls, y_train_reg
    gc.collect()

    return {
        "model_family": "xgb",
        "feature_set_name": feature_set_name,
        "params": deepcopy(best_params),
        "classifier": clf,
        "regressor": reg,
        "test_pred_cls": test_prob,
        "test_pred_reg": test_pred_reg,
        "test_cls_metrics": classification_metrics(y_test_cls, test_prob),
        "test_reg_metrics": regression_metrics(y_test_reg, test_pred_reg),
        "y_test_cls": y_test_cls,
        "y_test_reg": y_test_reg,
    }


# ============================================================
# LSTM
# ============================================================

def build_lstm_model(input_shape, units1=64, units2=32, dropout=0.2, learning_rate=1e-3):
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras is not available in this environment.")

    inp = Input(shape=input_shape)
    x = LSTM(units1, return_sequences=True)(inp)
    x = Dropout(dropout)(x)
    x = LSTM(units2)(x)
    x = Dropout(dropout)(x)

    cls_out = Dense(1, activation="sigmoid", name="cls")(x)
    reg_out = Dense(1, activation="linear", name="reg")(x)

    model = Model(inputs=inp, outputs=[cls_out, reg_out])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"cls": "binary_crossentropy", "reg": "mse"},
        metrics={"cls": [tf.keras.metrics.AUC(name="auc")], "reg": ["mae"]},
    )
    return model


def run_lstm_time_cv(
    cv_splits: list[dict],
    variant_name: str,
    param_grid: list[dict],
    sample_fraction: float | None,
    seed: int,
) -> pd.DataFrame:
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras is not available. Cannot run LSTM.")

    results_rows = []

    for config_id, params in enumerate(param_grid, start=1):
        cls_fold_metrics = []
        reg_fold_metrics = []

        for split in cv_splits:
            fold = split["fold"]

            train_fold = maybe_sample(split["train_df"], sample_fraction, seed)
            val_fold = maybe_sample(split["val_df"], sample_fraction, seed)

            X_train, y_train_cls, y_train_reg, _ = build_lstm_step_matrix(
                train_fold, variant_name=variant_name
            )
            X_val, y_val_cls, y_val_reg, _ = build_lstm_step_matrix(
                val_fold, variant_name=variant_name
            )

            X_train_scaled, X_val_scaled, scaler = scale_lstm(X_train, X_val)

            y_train_cls = np.asarray(y_train_cls, dtype=np.float32)
            y_train_reg = np.asarray(y_train_reg, dtype=np.float32)
            y_val_cls = np.asarray(y_val_cls, dtype=np.float32)
            y_val_reg = np.asarray(y_val_reg, dtype=np.float32)

            tf.keras.backend.clear_session()

            model = build_lstm_model(
                input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                units1=params["units1"],
                units2=params["units2"],
                dropout=params["dropout"],
                learning_rate=params["learning_rate"],
            )

            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )

            model.fit(
                X_train_scaled,
                {"cls": y_train_cls, "reg": y_train_reg},
                validation_data=(X_val_scaled, {"cls": y_val_cls, "reg": y_val_reg}),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=[early_stop],
                verbose=0,
            )

            pred_cls, pred_reg = model.predict(X_val_scaled, verbose=0)
            pred_cls = pred_cls.ravel()
            pred_reg = pred_reg.ravel()

            cls_metrics = classification_metrics(y_val_cls.astype(int), pred_cls)
            reg_metrics = regression_metrics(y_val_reg, pred_reg)

            # Annotate with fold info
            cls_metrics["fold"] = fold
            cls_metrics["train_years"] = str(split["train_years"])
            cls_metrics["val_year"] = split["val_year"]

            reg_metrics["fold"] = fold
            reg_metrics["train_years"] = str(split["train_years"])
            reg_metrics["val_year"] = split["val_year"]

            cls_fold_metrics.append(cls_metrics)
            reg_fold_metrics.append(reg_metrics)

            print(
                f"[LSTM cfg {config_id:02d}] fold={fold} "
                f"AUC={cls_metrics['auc']:.4f} "
                f"F1={cls_metrics['f1']:.4f} "
                f"MAE={reg_metrics['mae']:.4f} "
                f"RMSE={reg_metrics['rmse']:.4f}"
            )

            del model, X_train, X_val, X_train_scaled, X_val_scaled, scaler
            gc.collect()

        row = {
            "model_family": "lstm",
            "variant_name": variant_name,
            "config_id": f"lstm_{config_id:02d}",
            "params": deepcopy(params),
            "fold_cls_metrics": deepcopy(cls_fold_metrics),
            "fold_reg_metrics": deepcopy(reg_fold_metrics),
        }
        row.update(summarize_cv_metrics(cls_fold_metrics, prefix="cv"))
        row.update(summarize_cv_metrics(reg_fold_metrics, prefix="cv"))
        results_rows.append(row)

    return rank_results(pd.DataFrame(results_rows))


def refit_best_lstm(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    variant_name: str,
    best_params: dict,
) -> dict:
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow/Keras is not available. Cannot refit LSTM.")

    X_train, y_train_cls, y_train_reg, train_pdf = build_lstm_step_matrix(
        train_df, variant_name=variant_name
    )
    X_test, y_test_cls, y_test_reg, test_pdf = build_lstm_step_matrix(
        test_df, variant_name=variant_name
    )

    X_train_scaled, X_test_scaled, scaler = scale_lstm(X_train, X_test)

    y_train_cls = np.asarray(y_train_cls, dtype=np.float32)
    y_train_reg = np.asarray(y_train_reg, dtype=np.float32)
    y_test_cls = np.asarray(y_test_cls, dtype=np.float32)
    y_test_reg = np.asarray(y_test_reg, dtype=np.float32)

    tf.keras.backend.clear_session()

    model = build_lstm_model(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        units1=best_params["units1"],
        units2=best_params["units2"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )

    model.fit(
        X_train_scaled,
        {"cls": y_train_cls, "reg": y_train_reg},
        validation_split=0.10,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        callbacks=[early_stop],
        verbose=1,
    )

    pred_cls, pred_reg = model.predict(X_test_scaled, verbose=0)
    pred_cls = pred_cls.ravel()
    pred_reg = pred_reg.ravel()

    return {
        "model_family": "lstm",
        "variant_name": variant_name,
        "params": deepcopy(best_params),
        "model": model,
        "scaler": scaler,
        "test_pdf": test_pdf,
        "X_test": X_test_scaled,
        "test_pred_cls": pred_cls,
        "test_pred_reg": pred_reg,
        "test_cls_metrics": classification_metrics(y_test_cls.astype(int), pred_cls),
        "test_reg_metrics": regression_metrics(y_test_reg, pred_reg),
        "y_test_cls": y_test_cls,
        "y_test_reg": y_test_reg,
    }