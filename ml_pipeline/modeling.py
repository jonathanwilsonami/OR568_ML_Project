from __future__ import annotations

import gc
import logging
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

from feature_definitions import TARGET_CLASS, TARGET_REG, XGB_FEATURE_SETS
from pipeline_core import (
    build_lstm_step_matrix,
    classification_metrics,
    collect_year,
    maybe_sample,
    rank_results,
    regression_metrics,
    safe_fill_and_to_pandas,
    scale_lstm,
    summarize_cv_metrics,
)

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ============================================================
# HELPERS
# ============================================================

def _params_to_xgb_native(params: dict, task: str, use_gpu: bool, n_jobs: int) -> dict:
    """Convert sklearn-style param dict to xgb.train() compatible params."""
    p = dict(params)
    p["tree_method"] = "hist"
    p["device"] = "cuda" if use_gpu else "cpu"
    p["nthread"] = n_jobs
    p["seed"] = 42

    if task == "classification":
        p["objective"] = "binary:logistic"
        p["eval_metric"] = "logloss"
    elif task == "regression":
        p["objective"] = "reg:squarederror"
        p["eval_metric"] = "rmse"
    else:
        raise ValueError(f"Unsupported task: {task}")

    # n_estimators is handled as num_boost_round; remove from param dict
    num_boost_round = p.pop("n_estimators", 100)
    return p, num_boost_round


def _fill_pandas(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Select feature columns and median-fill missing values."""
    out = df[feature_cols].copy()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = out[c].fillna(out[c].median())
            if out[c].dtype.kind == "f":
                out[c] = pd.to_numeric(out[c], downcast="float")
            else:
                out[c] = pd.to_numeric(out[c], downcast="integer")
        else:
            out[c] = out[c].fillna("missing")
    return out


# ============================================================
# STREAMING XGB TRAINING — core primitive
#
# Trains one XGBoost booster (classifier OR regressor) by
# iterating over a list of years one at a time.  Each year is
# collected, converted to DMatrix, used for one round of
# boosting, then discarded before the next year is loaded.
#
# Peak RAM = one year of data converted to DMatrix (~1-2 GB)
# ============================================================

def _train_xgb_streaming(
    lf: pl.LazyFrame,
    years: list[int],
    feature_cols: list[str],
    target_col: str,
    params: dict,
    num_boost_round: int,
    use_gpu: bool,
    n_jobs: int,
    label: str = "",
    sample_fraction: float | None = None,
    seed: int = 42,
) -> xgb.Booster:
    """Train an XGBoost booster by streaming years one at a time.

    Each year is loaded, converted to DMatrix, and used for
    num_boost_round / len(years) boosting rounds (rounded), then
    freed before the next year loads.  The booster is warm-started
    across years so the full tree budget is distributed over all data.

    Parameters
    ----------
    lf : pl.LazyFrame
        Prepared lazy frame (all years, columns already selected).
    years : list[int]
        Ordered list of years to train on.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Target column name (TARGET_CLASS or TARGET_REG).
    params : dict
        Native xgb.train() params (no n_estimators key).
    num_boost_round : int
        Total boosting rounds across all years.
    label : str
        Human-readable label for log messages.
    sample_fraction : float | None
        If set, each year is randomly subsampled to this fraction
        before training.  None = use full year.
    seed : int
        Random seed for sampling.
    """
    booster = None
    rounds_per_year = max(1, num_boost_round // len(years))

    for i, year in enumerate(years):
        # Last year gets any remaining rounds so total = num_boost_round
        if i == len(years) - 1:
            rounds = num_boost_round - rounds_per_year * (len(years) - 1)
        else:
            rounds = rounds_per_year

        year_df = collect_year(
            lf, year, label=label or "train",
            sample_fraction=sample_fraction, seed=seed,
        )

        X = _fill_pandas(year_df.to_pandas(), feature_cols)
        y = year_df[target_col].to_pandas().values
        del year_df
        gc.collect()

        dtrain = xgb.DMatrix(X, label=y)
        del X, y
        gc.collect()

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            xgb_model=booster,   # warm-start from previous years
            verbose_eval=False,
        )

        del dtrain
        gc.collect()

        logger.info(
            "[XGB STREAM] %s year=%s rounds=%s (cumulative)",
            label or "train",
            year,
            rounds_per_year * (i + 1) if i < len(years) - 1 else num_boost_round,
        )

    return booster


def _predict_xgb_streaming(
    booster: xgb.Booster,
    lf: pl.LazyFrame,
    years: list[int],
    feature_cols: list[str],
    target_col: str,
    sample_fraction: float | None = None,
    seed: int = 42,
    label: str = "predict",
) -> tuple[np.ndarray, np.ndarray]:
    """Stream years through a trained booster and collect predictions + truth."""
    all_preds: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []

    for year in years:
        year_df = collect_year(
            lf, year, label=label,
            sample_fraction=sample_fraction, seed=seed,
        )
        X = _fill_pandas(year_df.to_pandas(), feature_cols)
        y = year_df[target_col].to_pandas().values
        del year_df
        gc.collect()

        dmat = xgb.DMatrix(X)
        del X
        gc.collect()

        preds = booster.predict(dmat)
        del dmat
        gc.collect()

        all_preds.append(preds)
        all_truth.append(y)

    return np.concatenate(all_preds), np.concatenate(all_truth)


# ============================================================
# CV — streaming, no DataFrames held between folds
# ============================================================

def run_xgb_time_cv(
    cv_fold_descriptors: list[dict],
    lf: pl.LazyFrame,
    feature_set_name: str,
    param_grid: list[dict],
    use_gpu: bool,
    n_jobs: int,
    sample_fraction: float | None,
    seed: int,
) -> pd.DataFrame:
    """Rolling time-series CV using streaming year-by-year training.

    Parameters
    ----------
    cv_fold_descriptors : list[dict]
        From make_rolling_year_cv_descriptors().  Each entry has
        'fold', 'train_years', 'val_year' — no DataFrames.
    lf : pl.LazyFrame
        The prepared lazy modeling frame.
    """
    if feature_set_name not in XGB_FEATURE_SETS:
        raise ValueError(f"Unknown XGB feature set: {feature_set_name}")

    feature_cols = XGB_FEATURE_SETS[feature_set_name]
    results_rows = []

    for config_id, params in enumerate(param_grid, start=1):
        cls_fold_metrics: list[dict] = []
        reg_fold_metrics: list[dict] = []

        for fold_desc in cv_fold_descriptors:
            fold = fold_desc["fold"]
            train_years = fold_desc["train_years"]
            val_year = fold_desc["val_year"]

            cls_params, num_rounds = _params_to_xgb_native(params, "classification", use_gpu, n_jobs)
            reg_params, _ = _params_to_xgb_native(params, "regression", use_gpu, n_jobs)

            # --- Train streaming ---
            clf_booster = _train_xgb_streaming(
                lf=lf,
                years=train_years,
                feature_cols=feature_cols,
                target_col=TARGET_CLASS,
                params=cls_params,
                num_boost_round=num_rounds,
                use_gpu=use_gpu,
                n_jobs=n_jobs,
                label=f"cv_fold{fold}_cls",
                sample_fraction=sample_fraction,
                seed=seed,
            )
            reg_booster = _train_xgb_streaming(
                lf=lf,
                years=train_years,
                feature_cols=feature_cols,
                target_col=TARGET_REG,
                params=reg_params,
                num_boost_round=num_rounds,
                use_gpu=use_gpu,
                n_jobs=n_jobs,
                label=f"cv_fold{fold}_reg",
                sample_fraction=sample_fraction,
                seed=seed,
            )

            # --- Validate (single year, always full) ---
            val_df = collect_year(lf, val_year, label=f"cv_fold{fold}_val")
            X_val = _fill_pandas(val_df.to_pandas(), feature_cols)
            y_val_cls = val_df[TARGET_CLASS].to_pandas().values.astype(int)
            y_val_reg = val_df[TARGET_REG].to_pandas().values.astype("float32")
            del val_df
            gc.collect()

            dval = xgb.DMatrix(X_val)
            del X_val
            gc.collect()

            val_prob = clf_booster.predict(dval)
            val_pred_reg = reg_booster.predict(dval)
            del dval
            gc.collect()

            cls_metrics = classification_metrics(y_val_cls, val_prob)
            reg_metrics = regression_metrics(y_val_reg, val_pred_reg)

            cls_metrics["fold"] = fold
            cls_metrics["train_years"] = str(train_years)
            cls_metrics["val_year"] = val_year

            reg_metrics["fold"] = fold
            reg_metrics["train_years"] = str(train_years)
            reg_metrics["val_year"] = val_year

            cls_fold_metrics.append(cls_metrics)
            reg_fold_metrics.append(reg_metrics)

            print(
                f"[XGB cfg {config_id:02d}] fold={fold} "
                f"AUC={cls_metrics['auc']:.4f} "
                f"F1={cls_metrics['f1']:.4f} "
                f"MAE={reg_metrics['mae']:.4f} "
                f"RMSE={reg_metrics['rmse']:.4f}"
            )

            del clf_booster, reg_booster
            del y_val_cls, y_val_reg, val_prob, val_pred_reg
            gc.collect()

        row = {
            "model_family": "xgb",
            "feature_set_name": feature_set_name,
            "config_id": f"xgb_{config_id:02d}",
            "params": deepcopy(params),
            "fold_cls_metrics": deepcopy(cls_fold_metrics),
            "fold_reg_metrics": deepcopy(reg_fold_metrics),
        }
        row.update(summarize_cv_metrics(cls_fold_metrics, prefix="cv"))
        row.update(summarize_cv_metrics(reg_fold_metrics, prefix="cv"))
        results_rows.append(row)

    return rank_results(pd.DataFrame(results_rows))


# ============================================================
# FINAL REFIT — streaming, predictions collected year by year
# ============================================================

def refit_best_xgb(
    lf: pl.LazyFrame,
    train_years: list[int],
    test_years: list[int],
    feature_set_name: str,
    best_params: dict,
    use_gpu: bool,
    n_jobs: int,
    sample_fraction: float | None = None,
    seed: int = 42,
) -> dict:
    """Refit XGBoost on all training years (streaming) then evaluate on test years.

    Parameters
    ----------
    lf : pl.LazyFrame
        Prepared lazy modeling frame.
    train_years : list[int]
        All years to train on (typically train + validation combined).
    test_years : list[int]
        Years to evaluate on (full, no sampling).
    """
    feature_cols = XGB_FEATURE_SETS[feature_set_name]

    cls_params, num_rounds = _params_to_xgb_native(best_params, "classification", use_gpu, n_jobs)
    reg_params, _ = _params_to_xgb_native(best_params, "regression", use_gpu, n_jobs)

    logger.info(
        "[XGB FINAL] Training classifier streaming over years=%s feature_set=%s",
        train_years, feature_set_name,
    )
    clf_booster = _train_xgb_streaming(
        lf=lf,
        years=train_years,
        feature_cols=feature_cols,
        target_col=TARGET_CLASS,
        params=cls_params,
        num_boost_round=num_rounds,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
        label="final_train_cls",
        sample_fraction=sample_fraction,
        seed=seed,
    )

    logger.info(
        "[XGB FINAL] Training regressor streaming over years=%s feature_set=%s",
        train_years, feature_set_name,
    )
    reg_booster = _train_xgb_streaming(
        lf=lf,
        years=train_years,
        feature_cols=feature_cols,
        target_col=TARGET_REG,
        params=reg_params,
        num_boost_round=num_rounds,
        use_gpu=use_gpu,
        n_jobs=n_jobs,
        label="final_train_reg",
        sample_fraction=sample_fraction,
        seed=seed,
    )

    logger.info("[XGB FINAL] Predicting on test years=%s", test_years)
    test_pred_cls, y_test_cls = _predict_xgb_streaming(
        booster=clf_booster,
        lf=lf,
        years=test_years,
        feature_cols=feature_cols,
        target_col=TARGET_CLASS,
        label="final_test_cls",
    )
    test_pred_reg, y_test_reg = _predict_xgb_streaming(
        booster=reg_booster,
        lf=lf,
        years=test_years,
        feature_cols=feature_cols,
        target_col=TARGET_REG,
        label="final_test_reg",
    )

    # Extract feature importances from the native booster directly.
    # Newer XGBoost versions do not allow setting sklearn wrapper
    # attributes (n_features_in_, classes_) on unfitted objects,
    # so we pass native boosters to artifact saving instead.
    clf_importances = clf_booster.get_score(importance_type="gain")
    # get_score only includes features that appeared in at least one
    # split — fill zeros for any unused feature.
    feature_importances = np.array(
        [clf_importances.get(f, 0.0) for f in feature_cols], dtype=np.float32
    )
    total = feature_importances.sum()
    if total > 0:
        feature_importances = feature_importances / total

    return {
        "model_family": "xgb",
        "feature_set_name": feature_set_name,
        "params": deepcopy(best_params),
        # Native xgb.Booster objects — serialisable with joblib
        "classifier": clf_booster,
        "regressor": reg_booster,
        "feature_importances": feature_importances,
        "clf_booster": clf_booster,
        "reg_booster": reg_booster,
        "test_pred_cls": test_pred_cls,
        "test_pred_reg": test_pred_reg,
        "test_cls_metrics": classification_metrics(y_test_cls.astype(int), test_pred_cls),
        "test_reg_metrics": regression_metrics(y_test_reg, test_pred_reg),
        "y_test_cls": y_test_cls,
        "y_test_reg": y_test_reg,
    }


# ============================================================
# LSTM  (unchanged — uses DataFrame path for now)
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
        cls_fold_metrics: list[dict] = []
        reg_fold_metrics: list[dict] = []

        for split in cv_splits:
            fold = split["fold"]
            train_fold = maybe_sample(split["train_df"], sample_fraction, seed)
            val_fold = maybe_sample(split["val_df"], sample_fraction, seed)

            X_train, y_train_cls, y_train_reg, _ = build_lstm_step_matrix(train_fold, variant_name=variant_name)
            X_val, y_val_cls, y_val_reg, _ = build_lstm_step_matrix(val_fold, variant_name=variant_name)
            X_train_scaled, X_val_scaled, scaler = scale_lstm(X_train, X_val)

            y_train_cls = np.asarray(y_train_cls, dtype=np.float32)
            y_train_reg = np.asarray(y_train_reg, dtype=np.float32)
            y_val_cls = np.asarray(y_val_cls, dtype=np.float32)
            y_val_reg = np.asarray(y_val_reg, dtype=np.float32)

            tf.keras.backend.clear_session()
            model = build_lstm_model(
                input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                units1=params["units1"], units2=params["units2"],
                dropout=params["dropout"], learning_rate=params["learning_rate"],
            )
            early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
            model.fit(
                X_train_scaled, {"cls": y_train_cls, "reg": y_train_reg},
                validation_data=(X_val_scaled, {"cls": y_val_cls, "reg": y_val_reg}),
                epochs=params["epochs"], batch_size=params["batch_size"],
                callbacks=[early_stop], verbose=0,
            )

            pred_cls, pred_reg = model.predict(X_val_scaled, verbose=0)
            pred_cls = pred_cls.ravel()
            pred_reg = pred_reg.ravel()

            cls_metrics = classification_metrics(y_val_cls.astype(int), pred_cls)
            reg_metrics = regression_metrics(y_val_reg, pred_reg)
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

    X_train, y_train_cls, y_train_reg, _ = build_lstm_step_matrix(train_df, variant_name=variant_name)
    X_test, y_test_cls, y_test_reg, test_pdf = build_lstm_step_matrix(test_df, variant_name=variant_name)
    X_train_scaled, X_test_scaled, scaler = scale_lstm(X_train, X_test)

    y_train_cls = np.asarray(y_train_cls, dtype=np.float32)
    y_train_reg = np.asarray(y_train_reg, dtype=np.float32)
    y_test_cls = np.asarray(y_test_cls, dtype=np.float32)
    y_test_reg = np.asarray(y_test_reg, dtype=np.float32)

    tf.keras.backend.clear_session()
    model = build_lstm_model(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        units1=best_params["units1"], units2=best_params["units2"],
        dropout=best_params["dropout"], learning_rate=best_params["learning_rate"],
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(
        X_train_scaled, {"cls": y_train_cls, "reg": y_train_reg},
        validation_split=0.10,
        epochs=best_params["epochs"], batch_size=best_params["batch_size"],
        callbacks=[early_stop], verbose=1,
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