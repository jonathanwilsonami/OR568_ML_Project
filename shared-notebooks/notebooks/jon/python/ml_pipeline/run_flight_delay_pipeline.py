from __future__ import annotations

import gc
import multiprocessing
import os
import time
from pathlib import Path

import pandas as pd
import polars as pl

from config import CONFIG
from feature_definitions import XGB_FEATURE_SETS
from modeling import (
    TF_AVAILABLE,
    refit_best_lstm,
    refit_best_xgb,
    run_lstm_time_cv,
    run_xgb_time_cv,
)
from pipeline_core import (
    collect_full_modeling_data,
    make_rolling_year_cv_splits,
    save_results_summary,
    split_train_val_test,
    timer_log,
)

# ============================================================
# THREAD / DEVICE SETUP
# ============================================================

N_CORES = multiprocessing.cpu_count()
print(f"[SYSTEM] Detected {N_CORES} CPU cores")

os.environ["OMP_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_CORES)
os.environ["TF_NUM_INTEROP_THREADS"] = str(N_CORES)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(N_CORES)

if TF_AVAILABLE:
    import tensorflow as tf

    try:
        if CONFIG.runtime.use_gpu:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[TF] GPU mode enabled. Found {len(gpus)} GPU(s).")
            else:
                print("[TF] WARNING: use_gpu=True but no GPU found. Using CPU.")
        else:
            tf.config.set_visible_devices([], "GPU")
            print("[TF] CPU-only mode enabled.")
    except RuntimeError as e:
        print(f"[TF WARNING] Device config must be set before TF runtime init: {e}")


def main():
    global_start = time.perf_counter()

    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_cols(100)
    pl.Config.set_tbl_width_chars(180)

    train_years = list(range(CONFIG.split.train_start_year, CONFIG.split.train_end_year + 1))
    all_years = sorted(set(train_years + CONFIG.split.validation_years + CONFIG.split.test_years))

    # ============================================================
    # LOAD AND BUILD MODELING TABLE
    # ============================================================
    t0 = time.perf_counter()
    full_df = collect_full_modeling_data(
        canonical_dir=CONFIG.data.canonical_dir,
        file_pattern=CONFIG.data.file_pattern,
        years=all_years,
    )
    timer_log("Load + collect all modeling data", t0, time)

    # ============================================================
    # SPLIT
    # ============================================================
    t0 = time.perf_counter()
    split_dict = split_train_val_test(
        df=full_df,
        train_years=train_years,
        validation_years=CONFIG.split.validation_years,
        test_years=CONFIG.split.test_years,
        use_validation_holdout=CONFIG.split.use_validation_holdout,
    )
    train_df = split_dict["train_df"]
    val_df = split_dict["val_df"]
    test_df = split_dict["test_df"]
    timer_log("Year split", t0, time)

    # ============================================================
    # CV SPLITS INSIDE TRAIN YEARS
    # ============================================================
    cv_splits = []
    if CONFIG.cv.enabled:
        t0 = time.perf_counter()
        cv_splits = make_rolling_year_cv_splits(
            train_df=train_df,
            min_train_years=CONFIG.cv.min_train_years,
        )
        timer_log("Build rolling CV splits", t0, time)

    # ============================================================
    # OPTIONAL FINAL TRAINING DATA = train + validation
    # ============================================================
    if CONFIG.split.use_validation_holdout and val_df is not None:
        final_train_df = pl.concat([train_df, val_df], how="vertical_relaxed")
    else:
        final_train_df = train_df

    print(f"[FINAL TRAIN] rows used for final refit: {final_train_df.height:,}")

    results_payload = {
        "config": {
            "train_years": train_years,
            "validation_years": CONFIG.split.validation_years,
            "test_years": CONFIG.split.test_years,
            "use_validation_holdout": CONFIG.split.use_validation_holdout,
            "cv_enabled": CONFIG.cv.enabled,
            "cv_min_train_years": CONFIG.cv.min_train_years,
            "run_xgb": CONFIG.models.run_xgb,
            "run_lstm": CONFIG.models.run_lstm,
            "tune_xgb": CONFIG.models.tune_xgb,
            "tune_lstm": CONFIG.models.tune_lstm,
            "xgb_feature_set_name": CONFIG.models.xgb_feature_set_name,
            "lstm_variant_name": CONFIG.models.lstm_variant_name,
            "use_gpu": CONFIG.runtime.use_gpu,
        },
        "row_counts": {
            "train_rows": train_df.height,
            "validation_rows": 0 if val_df is None else val_df.height,
            "test_rows": test_df.height,
            "final_train_rows": final_train_df.height,
        },
        "cv_splits": [
            {
                "fold": s["fold"],
                "train_years": s["train_years"],
                "val_year": s["val_year"],
                "train_rows": s["train_df"].height,
                "val_rows": s["val_df"].height,
            }
            for s in cv_splits
        ],
    }

    # ============================================================
    # XGBOOST
    # ============================================================
    xgb_cv_results = None
    best_xgb_row = None
    best_xgb_final = None

    if CONFIG.models.run_xgb:
        feature_set_name = CONFIG.models.xgb_feature_set_name
        if feature_set_name not in XGB_FEATURE_SETS:
            raise ValueError(f"Unknown XGB feature set name: {feature_set_name}")

        if CONFIG.models.tune_xgb:
            if not cv_splits:
                raise RuntimeError("XGBoost tuning requested, but no CV splits were built.")

            t0 = time.perf_counter()
            xgb_cv_results = run_xgb_time_cv(
                cv_splits=cv_splits,
                feature_set_name=feature_set_name,
                param_grid=CONFIG.xgb_search.param_grid,
                use_gpu=CONFIG.runtime.use_gpu,
                n_jobs=CONFIG.runtime.n_jobs,
                sample_fraction=CONFIG.runtime.sample_fraction_for_tuning,
                seed=CONFIG.runtime.random_seed,
            )
            timer_log("XGB time-aware CV search", t0, time)
            print("\nTop XGB configs")
            print(xgb_cv_results.head(10))
            best_xgb_row = xgb_cv_results.iloc[0].to_dict()
        else:
            best_xgb_row = {
                "feature_set_name": feature_set_name,
                "params": CONFIG.xgb_search.param_grid[0],
            }

        t0 = time.perf_counter()
        best_xgb_final = refit_best_xgb(
            train_df=final_train_df,
            test_df=test_df,
            feature_set_name=feature_set_name,
            best_params=best_xgb_row["params"],
            use_gpu=CONFIG.runtime.use_gpu,
            n_jobs=CONFIG.runtime.n_jobs,
        )
        timer_log("Final XGB refit + test", t0, time)

        print("\n[XGB FINAL] Classification")
        print(best_xgb_final["test_cls_metrics"])
        print("[XGB FINAL] Regression")
        print(best_xgb_final["test_reg_metrics"])

        results_payload["xgb_cv_results"] = None if xgb_cv_results is None else xgb_cv_results.to_dict(orient="records")
        results_payload["best_xgb_row"] = best_xgb_row
        results_payload["best_xgb_final"] = {
            "model_family": best_xgb_final["model_family"],
            "feature_set_name": best_xgb_final["feature_set_name"],
            "params": best_xgb_final["params"],
            "test_cls_metrics": best_xgb_final["test_cls_metrics"],
            "test_reg_metrics": best_xgb_final["test_reg_metrics"],
        }

    # ============================================================
    # LSTM
    # ============================================================
    lstm_cv_results = None
    best_lstm_row = None
    best_lstm_final = None

    if CONFIG.models.run_lstm:
        if not TF_AVAILABLE:
            raise RuntimeError("LSTM requested but TensorFlow is not installed/available.")

        variant_name = CONFIG.models.lstm_variant_name

        if CONFIG.models.tune_lstm:
            if not cv_splits:
                raise RuntimeError("LSTM tuning requested, but no CV splits were built.")

            t0 = time.perf_counter()
            lstm_cv_results = run_lstm_time_cv(
                cv_splits=cv_splits,
                variant_name=variant_name,
                param_grid=CONFIG.lstm_search.param_grid,
                sample_fraction=CONFIG.runtime.sample_fraction_for_lstm_tuning,
                seed=CONFIG.runtime.random_seed,
            )
            timer_log("LSTM time-aware CV search", t0, time)
            print("\nTop LSTM configs")
            print(lstm_cv_results.head(10))
            best_lstm_row = lstm_cv_results.iloc[0].to_dict()
        else:
            best_lstm_row = {
                "variant_name": variant_name,
                "params": CONFIG.lstm_search.param_grid[0],
            }

        t0 = time.perf_counter()
        best_lstm_final = refit_best_lstm(
            train_df=final_train_df,
            test_df=test_df,
            variant_name=variant_name,
            best_params=best_lstm_row["params"],
        )
        timer_log("Final LSTM refit + test", t0, time)

        print("\n[LSTM FINAL] Classification")
        print(best_lstm_final["test_cls_metrics"])
        print("[LSTM FINAL] Regression")
        print(best_lstm_final["test_reg_metrics"])

        results_payload["lstm_cv_results"] = None if lstm_cv_results is None else lstm_cv_results.to_dict(orient="records")
        results_payload["best_lstm_row"] = best_lstm_row
        results_payload["best_lstm_final"] = {
            "model_family": best_lstm_final["model_family"],
            "variant_name": best_lstm_final["variant_name"],
            "params": best_lstm_final["params"],
            "test_cls_metrics": best_lstm_final["test_cls_metrics"],
            "test_reg_metrics": best_lstm_final["test_reg_metrics"],
        }

    # ============================================================
    # COMBINED FINAL RESULTS
    # ============================================================
    final_rows = []

    if best_xgb_final is not None:
        final_rows.append({
            "model_family": "xgb",
            "details": best_xgb_final["feature_set_name"],
            "test_auc": best_xgb_final["test_cls_metrics"]["auc"],
            "test_f1": best_xgb_final["test_cls_metrics"]["f1"],
            "test_precision": best_xgb_final["test_cls_metrics"]["precision"],
            "test_recall": best_xgb_final["test_cls_metrics"]["recall"],
            "test_accuracy": best_xgb_final["test_cls_metrics"]["accuracy"],
            "test_mae": best_xgb_final["test_reg_metrics"]["mae"],
            "test_rmse": best_xgb_final["test_reg_metrics"]["rmse"],
        })

    if best_lstm_final is not None:
        final_rows.append({
            "model_family": "lstm",
            "details": best_lstm_final["variant_name"],
            "test_auc": best_lstm_final["test_cls_metrics"]["auc"],
            "test_f1": best_lstm_final["test_cls_metrics"]["f1"],
            "test_precision": best_lstm_final["test_cls_metrics"]["precision"],
            "test_recall": best_lstm_final["test_cls_metrics"]["recall"],
            "test_accuracy": best_lstm_final["test_cls_metrics"]["accuracy"],
            "test_mae": best_lstm_final["test_reg_metrics"]["mae"],
            "test_rmse": best_lstm_final["test_reg_metrics"]["rmse"],
        })

    if final_rows:
        final_test_results = (
            pd.DataFrame(final_rows)
            .sort_values(
                by=["test_auc", "test_f1", "test_mae", "test_rmse"],
                ascending=[False, False, True, True],
            )
            .reset_index(drop=True)
        )

        print("\n================================================")
        print("FINAL TEST RESULTS")
        print("================================================")
        print(final_test_results)

        best_overall_model = final_test_results.iloc[0].to_dict()

        print("\n================================================")
        print("BEST OVERALL MODEL")
        print("================================================")
        print(best_overall_model)

        results_payload["final_test_results"] = final_test_results.to_dict(orient="records")
        results_payload["best_overall_model"] = best_overall_model
    else:
        print("\n[INFO] No models were run, so no final comparison table was created.")
        final_test_results = pd.DataFrame()
        best_overall_model = None
        results_payload["final_test_results"] = []
        results_payload["best_overall_model"] = None

    # ============================================================
    # SAVE SUMMARY
    # ============================================================
    t0 = time.perf_counter()

    try:
        output_dir = None

        if hasattr(CONFIG, "output") and hasattr(CONFIG.output, "results_dir"):
            output_dir = Path(CONFIG.output.results_dir)
        elif hasattr(CONFIG, "paths") and hasattr(CONFIG.paths, "results_dir"):
            output_dir = Path(CONFIG.paths.results_dir)
        elif hasattr(CONFIG, "runtime") and hasattr(CONFIG.runtime, "results_dir"):
            output_dir = Path(CONFIG.runtime.results_dir)
        else:
            output_dir = Path(CONFIG.data.canonical_dir).resolve().parent / "pipeline_results"

        output_dir.mkdir(parents=True, exist_ok=True)

        save_succeeded = False

        try:
            save_results_summary(results_payload=results_payload, output_dir=output_dir)
            save_succeeded = True
        except TypeError:
            pass

        if not save_succeeded:
            try:
                save_results_summary(payload=results_payload, output_dir=output_dir)
                save_succeeded = True
            except TypeError:
                pass

        if not save_succeeded:
            try:
                save_results_summary(results_payload, output_dir)
                save_succeeded = True
            except TypeError:
                pass

        if not save_succeeded:
            try:
                save_results_summary(results_payload)
                save_succeeded = True
            except TypeError:
                pass

        if save_succeeded:
            print(f"\n[SAVE] Results summary written under: {output_dir}")
        else:
            print("\n[WARN] save_results_summary was available, but its signature did not match any expected call pattern.")

    except Exception as e:
        print(f"\n[WARN] Could not save results summary automatically: {e}")

    timer_log("Save results summary", t0, time)

    # ============================================================
    # CLEANUP / FINAL TIMER
    # ============================================================
    gc.collect()
    timer_log("TOTAL PIPELINE TIME", global_start, time)

    return {
        "results_payload": results_payload,
        "final_test_results": final_test_results,
        "best_overall_model": best_overall_model,
    }


if __name__ == "__main__":
    main()