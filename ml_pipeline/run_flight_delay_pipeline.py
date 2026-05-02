from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import sys
import time
from datetime import datetime
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
    collect_modeling_splits,
    make_rolling_year_cv_splits,
    save_results_summary,
    timer_log,
    maybe_sample,
)
from artifact_utils import build_run_version, init_run_directories, save_xgb_artifacts
from visualize_results import generate_visualizations

# ============================================================
# LOGGING / DIAGNOSTICS HELPERS
# ============================================================

def setup_logging(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"pipeline_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logging.info("Logging initialized")
    logging.info("Log file: %s", log_path)
    return log_path


def memory_snapshot(label: str) -> None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_raw = usage.ru_maxrss
        rss_mb = rss_raw / 1024 if rss_raw > 10_000 else rss_raw / (1024 * 1024)
        logging.info("[MEM] %s | maxrss≈%.2f MB", label, rss_mb)
    except Exception:
        logging.info("[MEM] %s | maxrss unavailable", label)


def log_df_info(name: str, df: pl.DataFrame | None, sample_cols: int = 12) -> None:
    if df is None:
        logging.info("[DF] %s = None", name)
        return

    try:
        est_mb = df.estimated_size("mb")
    except Exception:
        est_mb = None

    cols = df.columns
    preview_cols = cols[:sample_cols]
    suffix = "" if len(cols) <= sample_cols else " ..."

    if est_mb is not None:
        logging.info(
            "[DF] %s | rows=%s cols=%s est_size≈%.2f MB | columns=%s%s",
            name,
            f"{df.height:,}",
            len(cols),
            est_mb,
            preview_cols,
            suffix,
        )
    else:
        logging.info(
            "[DF] %s | rows=%s cols=%s | columns=%s%s",
            name,
            f"{df.height:,}",
            len(cols),
            preview_cols,
            suffix,
        )


def safe_timer_log(label: str, start: float) -> None:
    try:
        timer_log(label, start, time)
    except TypeError:
        try:
            timer_log(label, start)
        except Exception:
            elapsed = time.perf_counter() - start
            print(f"[TIMER] {label}: {elapsed:.2f}s")

    elapsed = time.perf_counter() - start
    logging.info("[TIMER] %s: %.2fs", label, elapsed)


def resolve_results_output_dir() -> Path:
    if hasattr(CONFIG, "output") and hasattr(CONFIG.output, "results_dir"):
        return Path(CONFIG.output.results_dir)
    if hasattr(CONFIG, "paths") and hasattr(CONFIG.paths, "results_dir"):
        return Path(CONFIG.paths.results_dir)
    if hasattr(CONFIG, "runtime") and hasattr(CONFIG.runtime, "results_dir"):
        return Path(CONFIG.runtime.results_dir)

    return Path(CONFIG.data.output_dir)


def validate_runtime_config() -> int:
    n_cores = multiprocessing.cpu_count()
    requested_n_jobs = getattr(CONFIG.runtime, "n_jobs", -1)

    if requested_n_jobs in (-1, 0, None):
        worker_cores = n_cores
    else:
        worker_cores = max(1, min(int(requested_n_jobs), n_cores))

    return worker_cores


# ============================================================
# THREAD / DEVICE SETUP
# ============================================================

N_CORES = multiprocessing.cpu_count()
WORKER_CORES = validate_runtime_config()

os.environ["OMP_NUM_THREADS"] = str(WORKER_CORES)
os.environ["MKL_NUM_THREADS"] = str(WORKER_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(WORKER_CORES)
os.environ["NUMEXPR_NUM_THREADS"] = str(WORKER_CORES)
os.environ["TF_NUM_INTEROP_THREADS"] = str(WORKER_CORES)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(WORKER_CORES)

if TF_AVAILABLE:
    import tensorflow as tf

    try:
        if CONFIG.runtime.use_gpu:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        else:
            tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass


# ============================================================
# MAIN
# ============================================================

def main():
    global_start = time.perf_counter()

    run_config_stub = {
        "train_years": list(range(CONFIG.split.train_start_year, CONFIG.split.train_end_year + 1)),
        "test_years": CONFIG.split.test_years,
        "run_xgb": CONFIG.models.run_xgb,
        "run_lstm": CONFIG.models.run_lstm,
    }
    run_version = build_run_version(run_config_stub)
    run_paths = init_run_directories(Path(CONFIG.data.output_dir), run_version)

    setup_logging(run_paths["logs_dir"])

    logging.info("=" * 60)
    logging.info("FLIGHT DELAY ML PIPELINE")
    logging.info("=" * 60)
    logging.info("[RUN] version=%s", run_version)
    logging.info("[RUN] run_dir=%s", run_paths["run_dir"])
    logging.info("[SYSTEM] Detected %s CPU cores", N_CORES)
    logging.info("[SYSTEM] Using %s worker cores", WORKER_CORES)
    logging.info("[SYSTEM] use_gpu=%s", CONFIG.runtime.use_gpu)
    logging.info("[PATH] canonical_dir=%s", CONFIG.data.canonical_dir)
    logging.info("[PATH] output_dir=%s", CONFIG.data.output_dir)

    if TF_AVAILABLE:
        try:
            if CONFIG.runtime.use_gpu:
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    logging.info("[TF] GPU mode enabled. Found %s GPU(s).", len(gpus))
                else:
                    logging.warning("[TF] use_gpu=True but no GPU found. Using CPU.")
            else:
                logging.info("[TF] CPU-only mode enabled.")
        except Exception as e:
            logging.warning("[TF] Could not inspect device configuration: %s", e)
    else:
        logging.info("[TF] TensorFlow not available.")

    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_cols(100)
    pl.Config.set_tbl_width_chars(180)

    train_years = list(range(CONFIG.split.train_start_year, CONFIG.split.train_end_year + 1))
    all_years = sorted(set(train_years + CONFIG.split.validation_years + CONFIG.split.test_years))

    # ------------------------------------------------------------
    # Resolve XGB feature sets once
    # ------------------------------------------------------------
    # Supports both modes:
    #
    # Single model:
    #   xgb_feature_set_name = "xgb_full_aircraft"
    #   xgb_feature_set_names = []
    #
    # Multiple models:
    #   xgb_feature_set_names = [
    #       "xgb_schedule",
    #       "xgb_context",
    #       "xgb_2hop_propagation",
    #       "xgb_full_aircraft",
    #   ]
    # ------------------------------------------------------------
    configured_xgb_feature_sets = (
        CONFIG.models.xgb_feature_set_names
        if getattr(CONFIG.models, "xgb_feature_set_names", None)
        else [CONFIG.models.xgb_feature_set_name]
    )

    logging.info("[CONFIG] train_years=%s", train_years)
    logging.info("[CONFIG] validation_years=%s", CONFIG.split.validation_years)
    logging.info("[CONFIG] test_years=%s", CONFIG.split.test_years)
    logging.info("[CONFIG] all_years=%s", all_years)
    logging.info("[CONFIG] cv_enabled=%s", CONFIG.cv.enabled)
    logging.info("[CONFIG] use_validation_holdout=%s", CONFIG.split.use_validation_holdout)
    logging.info("[CONFIG] xgb_feature_sets=%s", configured_xgb_feature_sets)
    logging.info("[CONFIG] sample_fraction_for_tuning=%s", CONFIG.runtime.sample_fraction_for_tuning)
    logging.info("[CONFIG] sample_fraction_for_lstm_tuning=%s", CONFIG.runtime.sample_fraction_for_lstm_tuning)
    logging.info("[CONFIG] sample_fraction_for_final_train=%s", CONFIG.runtime.sample_fraction_for_final_train)

    memory_snapshot("startup")

    train_df = None
    val_df = None
    test_df = None
    final_train_df = None

    try:
        # ============================================================
        # LOAD / SPLIT MODELING DATA
        # ============================================================
        t0 = time.perf_counter()
        logging.info("[STEP] Starting memory-safe collect_modeling_splits")

        split_dict = collect_modeling_splits(
            canonical_dir=CONFIG.data.canonical_dir,
            file_pattern=CONFIG.data.file_pattern,
            train_years=train_years,
            validation_years=CONFIG.split.validation_years,
            test_years=CONFIG.split.test_years,
            use_validation_holdout=CONFIG.split.use_validation_holdout,
            run_xgb=CONFIG.models.run_xgb,
            run_lstm=CONFIG.models.run_lstm,
            xgb_feature_set_name=configured_xgb_feature_sets,
        )

        train_df = split_dict["train_df"]
        val_df = split_dict["val_df"]
        test_df = split_dict["test_df"]

        safe_timer_log("Collect split modeling data", t0)

        logging.info(
            "[ROWS] train=%s validation=%s test=%s",
            f"{train_df.height:,}" if train_df is not None else "None",
            f"{val_df.height:,}" if val_df is not None else "None",
            f"{test_df.height:,}" if test_df is not None else "None",
        )

        log_df_info("train_df", train_df)
        log_df_info("val_df", val_df)
        log_df_info("test_df", test_df)
        memory_snapshot("after split collection")

        if train_df is None or train_df.height == 0:
            raise RuntimeError("train_df is empty after split collection.")
        if test_df is None or test_df.height == 0:
            raise RuntimeError("test_df is empty after split collection.")

        # ============================================================
        # CV SPLITS INSIDE TRAIN YEARS
        # ============================================================
        cv_splits = []
        if CONFIG.cv.enabled:
            t0 = time.perf_counter()
            logging.info("[STEP] Building rolling CV splits")

            cv_splits = make_rolling_year_cv_splits(
                train_df=train_df,
                min_train_years=CONFIG.cv.min_train_years,
            )

            safe_timer_log("Build rolling CV splits", t0)
            logging.info("[CV] built %s folds", len(cv_splits))

            for s in cv_splits:
                logging.info(
                    "[CV] fold=%s train_years=%s val_year=%s",
                    s["fold"],
                    s["train_years"],
                    s["val_year"],
                )

            memory_snapshot("after CV split build")

        # ============================================================
        # OPTIONAL FINAL TRAINING DATA = train + validation
        # ============================================================
        logging.info("[STEP] Building final training dataframe")
        if CONFIG.split.use_validation_holdout and val_df is not None:
            final_train_df = pl.concat([train_df, val_df], how="vertical_relaxed")
        else:
            final_train_df = train_df

        logging.info("[FINAL TRAIN] rows used for final refit: %s", f"{final_train_df.height:,}")
        log_df_info("final_train_df", final_train_df)
        memory_snapshot("after final_train_df creation")

        # Cache row counts before any large references are released
        train_rows = 0 if train_df is None else train_df.height
        validation_rows = 0 if val_df is None else val_df.height
        test_rows = 0 if test_df is None else test_df.height
        final_train_rows = 0 if final_train_df is None else final_train_df.height

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
                "worker_cores": WORKER_CORES,
            },
            "row_counts": {
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "test_rows": test_rows,
                "final_train_rows": final_train_rows,
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
        xgb_cv_results_by_feature_set = {}
        best_xgb_rows_by_feature_set = {}
        best_xgb_finals_by_feature_set = {}
        artifact_paths_by_feature_set = {}

        if CONFIG.models.run_xgb:
            feature_set_names = configured_xgb_feature_sets

            logging.info("[XGB] Feature sets to run: %s", feature_set_names)

            # Release train/val references after CV split construction
            train_df = None
            val_df = None
            gc.collect()
            memory_snapshot("after releasing train_df/val_df before XGB loop")

            for feature_set_name in feature_set_names:
                logging.info("=" * 60)
                logging.info("[XGB] Starting feature set: %s", feature_set_name)
                logging.info("=" * 60)

                if feature_set_name not in XGB_FEATURE_SETS:
                    raise ValueError(f"Unknown XGB feature set name: {feature_set_name}")

                logging.info(
                    "[XGB] number of features=%s",
                    len(XGB_FEATURE_SETS[feature_set_name]),
                )
                logging.info("[XGB] tune_xgb=%s", CONFIG.models.tune_xgb)

                xgb_cv_results = None
                best_xgb_row = None
                best_xgb_final = None

                # ----------------------------------------------------
                # Optional tuning
                # ----------------------------------------------------
                if CONFIG.models.tune_xgb and cv_splits:
                    t0 = time.perf_counter()

                    logging.info(
                        "[XGB:%s] Starting time-aware CV search with %s configs",
                        feature_set_name,
                        len(CONFIG.xgb_search.param_grid),
                    )

                    xgb_cv_results = run_xgb_time_cv(
                        cv_splits=cv_splits,
                        feature_set_name=feature_set_name,
                        param_grid=CONFIG.xgb_search.param_grid,
                        use_gpu=CONFIG.runtime.use_gpu,
                        n_jobs=CONFIG.runtime.n_jobs,
                        sample_fraction=CONFIG.runtime.sample_fraction_for_tuning,
                        seed=CONFIG.runtime.random_seed,
                    )

                    safe_timer_log(
                        f"XGB time-aware CV search: {feature_set_name}",
                        t0,
                    )

                    logging.info(
                        "[XGB:%s] Top configs:\n%s",
                        feature_set_name,
                        xgb_cv_results.head(10),
                    )

                    best_xgb_row = xgb_cv_results.iloc[0].to_dict()

                    logging.info(
                        "[XGB:%s] Best config row=%s",
                        feature_set_name,
                        best_xgb_row,
                    )

                    memory_snapshot(
                        f"after XGB CV search: {feature_set_name}"
                    )

                    xgb_cv_results_by_feature_set[
                        feature_set_name
                    ] = xgb_cv_results.to_dict(orient="records")

                    del xgb_cv_results
                    xgb_cv_results = None
                    gc.collect()

                    memory_snapshot(
                        f"after releasing CV results: {feature_set_name}"
                    )

                else:
                    if CONFIG.models.tune_xgb and not cv_splits:
                        logging.warning(
                            "[XGB:%s] tune_xgb=True but no CV splits "
                            "were built. Falling back to first parameter set.",
                            feature_set_name,
                        )

                    best_xgb_row = {
                        "model_family": "xgb",
                        "feature_set_name": feature_set_name,
                        "config_id": "manual_01",
                        "params": CONFIG.xgb_search.param_grid[0],
                    }

                    xgb_cv_results_by_feature_set[
                        feature_set_name
                    ] = None

                    logging.info(
                        "[XGB:%s] Using params=%s",
                        feature_set_name,
                        best_xgb_row["params"],
                    )

                best_xgb_rows_by_feature_set[
                    feature_set_name
                ] = best_xgb_row

                # ----------------------------------------------------
                # Final fit
                # ----------------------------------------------------
                t0 = time.perf_counter()

                logging.info(
                    "[XGB:%s] Starting final refit + test",
                    feature_set_name,
                )

                final_fit_df = maybe_sample(
                    final_train_df,
                    CONFIG.runtime.sample_fraction_for_final_train,
                    CONFIG.runtime.random_seed,
                )

                logging.info(
                    "[XGB:%s] final_fit_df rows=%s "
                    "from original final_train_df rows=%s",
                    feature_set_name,
                    f"{final_fit_df.height:,}",
                    f"{final_train_df.height:,}",
                )

                best_xgb_final = refit_best_xgb(
                    train_df=final_fit_df,
                    test_df=test_df,
                    feature_set_name=feature_set_name,
                    best_params=best_xgb_row["params"],
                    use_gpu=CONFIG.runtime.use_gpu,
                    n_jobs=CONFIG.runtime.n_jobs,
                )

                safe_timer_log(
                    f"Final XGB refit + test: {feature_set_name}",
                    t0,
                )

                logging.info(
                    "[XGB FINAL:%s] Classification=%s",
                    feature_set_name,
                    best_xgb_final["test_cls_metrics"],
                )

                logging.info(
                    "[XGB FINAL:%s] Regression=%s",
                    feature_set_name,
                    best_xgb_final["test_reg_metrics"],
                )

                # ----------------------------------------------------
                # Keep only lightweight metrics in memory
                # ----------------------------------------------------
                best_xgb_finals_by_feature_set[
                    feature_set_name
                ] = {
                    "model_family": best_xgb_final["model_family"],
                    "feature_set_name": best_xgb_final["feature_set_name"],
                    "params": best_xgb_final["params"],
                    "test_cls_metrics": best_xgb_final["test_cls_metrics"],
                    "test_reg_metrics": best_xgb_final["test_reg_metrics"],
                }

                # ----------------------------------------------------
                # Save artifacts per feature set
                # ----------------------------------------------------
                if CONFIG.artifacts.enabled:
                    artifact_paths = save_xgb_artifacts(
                        run_paths=run_paths,
                        version=f"{run_version}_{feature_set_name}",
                        feature_set_name=feature_set_name,
                        best_params=best_xgb_final["params"],
                        classifier=best_xgb_final["classifier"],
                        regressor=best_xgb_final["regressor"],
                        results_payload=results_payload,
                        feature_names=XGB_FEATURE_SETS[
                            feature_set_name
                        ],
                        y_test_cls=best_xgb_final.get("y_test_cls"),
                        test_pred_cls=best_xgb_final.get(
                            "test_pred_cls"
                        ),
                        y_test_reg=best_xgb_final.get("y_test_reg"),
                        test_pred_reg=best_xgb_final.get(
                            "test_pred_reg"
                        ),
                    )

                    artifact_paths_by_feature_set[
                        feature_set_name
                    ] = artifact_paths

                    logging.info(
                        "[ARTIFACTS:%s] Saved artifacts: %s",
                        feature_set_name,
                        artifact_paths,
                    )

                # ----------------------------------------------------
                # Memory cleanup after each feature set
                # ----------------------------------------------------
                del final_fit_df
                del best_xgb_final
                best_xgb_final = None

                gc.collect()

                memory_snapshot(
                    f"after cleanup for feature set: {feature_set_name}"
                )

            results_payload[
                "xgb_cv_results_by_feature_set"
            ] = xgb_cv_results_by_feature_set

            results_payload[
                "best_xgb_rows_by_feature_set"
            ] = best_xgb_rows_by_feature_set

            results_payload[
                "best_xgb_finals_by_feature_set"
            ] = best_xgb_finals_by_feature_set

            results_payload[
                "artifact_paths_by_feature_set"
            ] = artifact_paths_by_feature_set

            results_payload["model_version"] = run_version


        # ============================================================
        # LSTM
        # ============================================================
        lstm_cv_results = None
        best_lstm_row = None
        best_lstm_final = None

        if CONFIG.models.run_lstm:
            logging.info("[LSTM] run_lstm=True")

            if not TF_AVAILABLE:
                raise RuntimeError("LSTM requested but TensorFlow is not installed/available.")

            variant_name = CONFIG.models.lstm_variant_name
            logging.info("[LSTM] variant_name=%s", variant_name)
            logging.info("[LSTM] tune_lstm=%s", CONFIG.models.tune_lstm)

            if CONFIG.models.tune_lstm and cv_splits:
                t0 = time.perf_counter()
                logging.info(
                    "[LSTM] Starting time-aware CV search with %s configs",
                    len(CONFIG.lstm_search.param_grid),
                )

                lstm_cv_results = run_lstm_time_cv(
                    cv_splits=cv_splits,
                    variant_name=variant_name,
                    param_grid=CONFIG.lstm_search.param_grid,
                    sample_fraction=CONFIG.runtime.sample_fraction_for_lstm_tuning,
                    seed=CONFIG.runtime.random_seed,
                )

                safe_timer_log("LSTM time-aware CV search", t0)
                logging.info("[LSTM] Top configs:\n%s", lstm_cv_results.head(10))
                best_lstm_row = lstm_cv_results.iloc[0].to_dict()
                logging.info("[LSTM] Best config row=%s", best_lstm_row)
                memory_snapshot("after LSTM CV search")
            else:
                if CONFIG.models.tune_lstm and not cv_splits:
                    logging.warning(
                        "[LSTM] tune_lstm=True but no CV splits were built. Falling back to first parameter set."
                    )

                best_lstm_row = {
                    "variant_name": variant_name,
                    "params": CONFIG.lstm_search.param_grid[0],
                }
                logging.info("[LSTM] Using params=%s", best_lstm_row["params"])

            t0 = time.perf_counter()
            logging.info("[LSTM] Starting final refit + test")

            best_lstm_final = refit_best_lstm(
                train_df=final_train_df,
                test_df=test_df,
                variant_name=variant_name,
                best_params=best_lstm_row["params"],
            )

            safe_timer_log("Final LSTM refit + test", t0)
            logging.info("[LSTM FINAL] Classification=%s", best_lstm_final["test_cls_metrics"])
            logging.info("[LSTM FINAL] Regression=%s", best_lstm_final["test_reg_metrics"])
            memory_snapshot("after final LSTM")

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

        # XGBoost multi-feature-set results
        for feature_set_name, xgb_result in best_xgb_finals_by_feature_set.items():
            final_rows.append({
                "model_family": "xgb",
                "details": feature_set_name,
                "test_auc": xgb_result["test_cls_metrics"]["auc"],
                "test_f1": xgb_result["test_cls_metrics"]["f1"],
                "test_precision": xgb_result["test_cls_metrics"]["precision"],
                "test_recall": xgb_result["test_cls_metrics"]["recall"],
                "test_accuracy": xgb_result["test_cls_metrics"]["accuracy"],
                "test_mae": xgb_result["test_reg_metrics"]["mae"],
                "test_rmse": xgb_result["test_reg_metrics"]["rmse"],
            })

        # LSTM result, if enabled
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

            logging.info("================================================")
            logging.info("FINAL TEST RESULTS")
            logging.info("================================================")
            logging.info("\n%s", final_test_results)

            best_overall_model = final_test_results.iloc[0].to_dict()

            logging.info("================================================")
            logging.info("BEST OVERALL MODEL")
            logging.info("================================================")
            logging.info("%s", best_overall_model)

            results_payload["final_test_results"] = final_test_results.to_dict(orient="records")
            results_payload["best_overall_model"] = best_overall_model

        else:
            logging.info("[INFO] No models were run, so no final comparison table was created.")
            final_test_results = pd.DataFrame()
            best_overall_model = None
            results_payload["final_test_results"] = []
            results_payload["best_overall_model"] = None

        # ============================================================
        # SAVE SUMMARY
        # ============================================================
        t0 = time.perf_counter()

        try:
            output_dir = run_paths["evaluations_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)

            logging.info("[SAVE] Attempting to write results summary under: %s", output_dir)

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
                logging.info("[SAVE] Results summary written under: %s", output_dir)
            else:
                logging.warning(
                    "[SAVE] save_results_summary was available, but its signature did not match any expected call pattern."
                )

        except Exception as e:
            logging.exception("[SAVE] Could not save results summary automatically: %s", e)

        safe_timer_log("Save results summary", t0)

        # ============================================================
        # GENERATE VISUALIZATIONS
        # ============================================================
        if CONFIG.visualizations.enabled:
            viz_t0 = time.perf_counter()
            try:
                summary_json_path = run_paths["evaluations_dir"] / "results_summary.json"
                predictions_path = run_paths["evaluations_dir"] / "test_predictions.parquet"
                feature_importance_csv_path = run_paths["evaluations_dir"] / "feature_importance.csv"

                generate_visualizations(
                    summary_json_path=summary_json_path,
                    plots_dir=run_paths["plots_dir"],
                    tables_dir=run_paths["tables_dir"],
                    predictions_path=predictions_path if predictions_path.exists() else None,
                    feature_importance_csv_path=feature_importance_csv_path if feature_importance_csv_path.exists() else None,
                    make_summary_table=CONFIG.visualizations.make_summary_table,
                    make_cv_table=CONFIG.visualizations.make_cv_table,
                    make_metric_bar_charts=CONFIG.visualizations.make_metric_bar_charts,
                    make_cv_metric_bar_charts=CONFIG.visualizations.make_cv_metric_bar_charts,
                    make_feature_importance_chart=CONFIG.visualizations.make_feature_importance_chart,
                    make_roc_curve_chart=CONFIG.visualizations.make_roc_curve,
                    make_actual_vs_predicted_chart=CONFIG.visualizations.make_actual_vs_predicted,
                    top_n_features=CONFIG.visualizations.top_n_features,
                    scatter_sample_n=CONFIG.visualizations.scatter_sample_n,
                    dpi=CONFIG.visualizations.figure_dpi,
                )

                logging.info("[VIZ] Generated evaluation plots and tables under %s", run_paths["run_dir"])

            except Exception as e:
                logging.exception("[VIZ] Failed to generate visualizations: %s", e)

            safe_timer_log("Generate visualizations", viz_t0)

        gc.collect()
        memory_snapshot("before return")
        safe_timer_log("TOTAL PIPELINE TIME", global_start)

        return {
            "results_payload": results_payload,
            "final_test_results": final_test_results,
            "best_overall_model": best_overall_model,
        }

    except MemoryError:
        logging.exception("[FATAL] Python raised MemoryError.")
        raise
    except Exception as e:
        logging.exception("[FATAL] Pipeline failed: %s", e)
        raise
    finally:
        gc.collect()
        memory_snapshot("final cleanup")


if __name__ == "__main__":
    main()