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
    build_lazy_modeling_frame,
    collect_modeling_splits,
    make_rolling_year_cv_descriptors,
    make_rolling_year_cv_splits,
    resolve_required_columns,
    save_results_summary,
    timer_log,
    maybe_sample,
)
from artifact_utils import build_run_version, init_run_directories, save_xgb_artifacts
from visualize_results import generate_visualizations

# ============================================================
# LOGGING / DIAGNOSTICS
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
    logging.info("Logging initialized — log file: %s", log_path)
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
    preview = cols[:sample_cols]
    suffix = "" if len(cols) <= sample_cols else " ..."
    if est_mb is not None:
        logging.info(
            "[DF] %s | rows=%s cols=%s est_size≈%.2f MB | columns=%s%s",
            name, f"{df.height:,}", len(cols), est_mb, preview, suffix,
        )
    else:
        logging.info(
            "[DF] %s | rows=%s cols=%s | columns=%s%s",
            name, f"{df.height:,}", len(cols), preview, suffix,
        )


def safe_timer_log(label: str, start: float) -> None:
    try:
        timer_log(label, start, time)
    except TypeError:
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {label}: {elapsed:.2f}s")
    elapsed = time.perf_counter() - start
    logging.info("[TIMER] %s: %.2fs", label, elapsed)


def validate_runtime_config() -> int:
    n_cores = multiprocessing.cpu_count()
    requested = getattr(CONFIG.runtime, "n_jobs", -1)
    if requested in (-1, 0, None):
        return n_cores
    return max(1, min(int(requested), n_cores))


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

    train_years = list(range(CONFIG.split.train_start_year, CONFIG.split.train_end_year + 1))

    run_config_stub = {
        "train_years": train_years,
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
    logging.info("[SYSTEM] %s CPU cores detected, using %s", N_CORES, WORKER_CORES)
    logging.info("[SYSTEM] use_gpu=%s", CONFIG.runtime.use_gpu)

    if TF_AVAILABLE:
        try:
            if CONFIG.runtime.use_gpu:
                gpus = tf.config.list_physical_devices("GPU")
                logging.info("[TF] GPU mode. Found %s GPU(s).", len(gpus))
            else:
                logging.info("[TF] CPU-only mode.")
        except Exception as e:
            logging.warning("[TF] Device check failed: %s", e)
    else:
        logging.info("[TF] TensorFlow not available.")

    pl.Config.set_tbl_rows(50)
    pl.Config.set_tbl_cols(100)
    pl.Config.set_tbl_width_chars(180)

    all_years = sorted(set(
        train_years + CONFIG.split.validation_years + CONFIG.split.test_years
    ))

    # Final training uses train + validation years combined
    final_train_years = sorted(set(train_years + CONFIG.split.validation_years))

    configured_xgb_feature_sets = (
        CONFIG.models.xgb_feature_set_names
        if getattr(CONFIG.models, "xgb_feature_set_names", None)
        else [CONFIG.models.xgb_feature_set_name]
    )

    logging.info("[CONFIG] train_years=%s", train_years)
    logging.info("[CONFIG] validation_years=%s", CONFIG.split.validation_years)
    logging.info("[CONFIG] test_years=%s", CONFIG.split.test_years)
    logging.info("[CONFIG] final_train_years=%s", final_train_years)
    logging.info("[CONFIG] cv_enabled=%s", CONFIG.cv.enabled)
    logging.info("[CONFIG] tune_xgb=%s", CONFIG.models.tune_xgb)
    logging.info("[CONFIG] xgb_feature_sets=%s", configured_xgb_feature_sets)
    logging.info(
        "[CONFIG] train_sample_fraction=%s (None = full data)",
        CONFIG.runtime.train_sample_fraction,
    )

    memory_snapshot("startup")

    try:
        # ============================================================
        # BUILD LAZY FRAME — nothing collected yet
        # Resolve the union of all required columns across all feature sets
        # so the lazy frame is built once and reused for every feature set.
        # ============================================================
        required_columns = resolve_required_columns(
            run_xgb=CONFIG.models.run_xgb,
            run_lstm=CONFIG.models.run_lstm,
            xgb_feature_set_name=configured_xgb_feature_sets,
        )

        logging.info("[DATA] Building lazy modeling frame for years=%s", all_years)
        logging.info("[DATA] Projecting %s columns", len(required_columns))

        t0 = time.perf_counter()
        lf = build_lazy_modeling_frame(
            canonical_dir=CONFIG.data.canonical_dir,
            file_pattern=CONFIG.data.file_pattern,
            all_years=all_years,
            required_columns=required_columns,
        )
        safe_timer_log("Build lazy modeling frame", t0)
        memory_snapshot("after building lazy frame")

        # ============================================================
        # CV FOLD DESCRIPTORS — year lists only, no DataFrames
        # ============================================================
        cv_fold_descriptors = []
        if CONFIG.cv.enabled and CONFIG.models.run_xgb:
            cv_fold_descriptors = make_rolling_year_cv_descriptors(
                train_years=train_years,
                min_train_years=CONFIG.cv.min_train_years,
            )
            logging.info("[CV] Built %s fold descriptors", len(cv_fold_descriptors))
            for fd in cv_fold_descriptors:
                logging.info(
                    "[CV] fold=%s train_years=%s val_year=%s",
                    fd["fold"], fd["train_years"], fd["val_year"],
                )

        results_payload = {
            "config": {
                "train_years": train_years,
                "validation_years": CONFIG.split.validation_years,
                "test_years": CONFIG.split.test_years,
                "final_train_years": final_train_years,
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
                "train_sample_fraction": CONFIG.runtime.train_sample_fraction,
                "val_sample_fraction": CONFIG.runtime.val_sample_fraction,
                "test_sample_fraction": CONFIG.runtime.test_sample_fraction,
            },
            "cv_splits": [
                {
                    "fold": fd["fold"],
                    "train_years": fd["train_years"],
                    "val_year": fd["val_year"],
                }
                for fd in cv_fold_descriptors
            ],
        }

        # ============================================================
        # XGBOOST — fully streaming, no full DataFrame in RAM
        # ============================================================
        xgb_cv_results_by_feature_set = {}
        best_xgb_rows_by_feature_set = {}
        best_xgb_finals_by_feature_set = {}
        artifact_paths_by_feature_set = {}

        if CONFIG.models.run_xgb:
            logging.info("[XGB] Feature sets to run: %s", configured_xgb_feature_sets)

            for feature_set_name in configured_xgb_feature_sets:
                logging.info("=" * 60)
                logging.info("[XGB] Feature set: %s", feature_set_name)
                logging.info("=" * 60)

                if feature_set_name not in XGB_FEATURE_SETS:
                    raise ValueError(f"Unknown XGB feature set: {feature_set_name}")

                logging.info("[XGB] %s features", len(XGB_FEATURE_SETS[feature_set_name]))

                best_xgb_fold_cls_metrics: list[dict] = []
                best_xgb_fold_reg_metrics: list[dict] = []
                best_xgb_row = None

                # --------------------------------------------------
                # CV (always runs when enabled — tune_xgb controls
                # whether results drive param selection)
                # --------------------------------------------------
                if cv_fold_descriptors:
                    t0 = time.perf_counter()

                    cv_param_grid = (
                        CONFIG.xgb_search.param_grid
                        if CONFIG.models.tune_xgb
                        else CONFIG.xgb_search.param_grid[:1]
                    )

                    logging.info(
                        "[XGB:%s] CV (%s) — %s config(s) × %s folds | "
                        "sample_fraction=%s",
                        feature_set_name,
                        "tuning" if CONFIG.models.tune_xgb else "diagnostics",
                        len(cv_param_grid),
                        len(cv_fold_descriptors),
                        CONFIG.runtime.train_sample_fraction,
                    )

                    xgb_cv_results = run_xgb_time_cv(
                        cv_fold_descriptors=cv_fold_descriptors,
                        lf=lf,
                        feature_set_name=feature_set_name,
                        param_grid=cv_param_grid,
                        use_gpu=CONFIG.runtime.use_gpu,
                        n_jobs=CONFIG.runtime.n_jobs,
                        sample_fraction=CONFIG.runtime.train_sample_fraction,
                        seed=CONFIG.runtime.random_seed,
                    )

                    safe_timer_log(
                        f"XGB CV ({'tuning' if CONFIG.models.tune_xgb else 'diagnostics'}): {feature_set_name}",
                        t0,
                    )
                    logging.info("[XGB:%s] CV results:\n%s", feature_set_name, xgb_cv_results.head(10))

                    best_cv_row = xgb_cv_results.iloc[0].to_dict()
                    best_xgb_fold_cls_metrics = best_cv_row.pop("fold_cls_metrics", [])
                    best_xgb_fold_reg_metrics = best_cv_row.pop("fold_reg_metrics", [])

                    if CONFIG.models.tune_xgb:
                        best_xgb_row = best_cv_row
                        logging.info("[XGB:%s] Best config (tuned): %s", feature_set_name, best_xgb_row)
                    else:
                        best_xgb_row = {
                            "model_family": "xgb",
                            "feature_set_name": feature_set_name,
                            "config_id": "manual_01",
                            "params": CONFIG.xgb_search.param_grid[0],
                        }
                        logging.info(
                            "[XGB:%s] Manual params (tune_xgb=False): %s",
                            feature_set_name, best_xgb_row["params"],
                        )

                    memory_snapshot(f"after XGB CV: {feature_set_name}")

                    summary_df = xgb_cv_results.drop(
                        columns=["fold_cls_metrics", "fold_reg_metrics"], errors="ignore"
                    )
                    xgb_cv_results_by_feature_set[feature_set_name] = {
                        "summary": summary_df.to_dict(orient="records"),
                        "best_config_fold_cls_metrics": best_xgb_fold_cls_metrics,
                        "best_config_fold_reg_metrics": best_xgb_fold_reg_metrics,
                    }
                    del xgb_cv_results
                    gc.collect()
                    memory_snapshot(f"after releasing CV results: {feature_set_name}")

                else:
                    logging.warning(
                        "[XGB:%s] No CV fold descriptors — using first param set.",
                        feature_set_name,
                    )
                    best_xgb_row = {
                        "model_family": "xgb",
                        "feature_set_name": feature_set_name,
                        "config_id": "manual_01",
                        "params": CONFIG.xgb_search.param_grid[0],
                    }
                    xgb_cv_results_by_feature_set[feature_set_name] = None

                best_xgb_rows_by_feature_set[feature_set_name] = best_xgb_row

                # --------------------------------------------------
                # Final refit — streaming over final_train_years
                # --------------------------------------------------
                t0 = time.perf_counter()
                logging.info("[XGB:%s] Final refit over years=%s", feature_set_name, final_train_years)

                best_xgb_final = refit_best_xgb(
                    lf=lf,
                    train_years=final_train_years,
                    test_years=CONFIG.split.test_years,
                    feature_set_name=feature_set_name,
                    best_params=best_xgb_row["params"],
                    use_gpu=CONFIG.runtime.use_gpu,
                    n_jobs=CONFIG.runtime.n_jobs,
                    sample_fraction=CONFIG.runtime.train_sample_fraction,
                    seed=CONFIG.runtime.random_seed,
                )

                safe_timer_log(f"Final XGB refit + test: {feature_set_name}", t0)
                logging.info(
                    "[XGB FINAL:%s] Classification=%s",
                    feature_set_name, best_xgb_final["test_cls_metrics"],
                )
                logging.info(
                    "[XGB FINAL:%s] Regression=%s",
                    feature_set_name, best_xgb_final["test_reg_metrics"],
                )

                best_xgb_finals_by_feature_set[feature_set_name] = {
                    "model_family": best_xgb_final["model_family"],
                    "feature_set_name": best_xgb_final["feature_set_name"],
                    "params": best_xgb_final["params"],
                    "test_cls_metrics": best_xgb_final["test_cls_metrics"],
                    "test_reg_metrics": best_xgb_final["test_reg_metrics"],
                }

                if CONFIG.artifacts.enabled:
                    artifact_paths = save_xgb_artifacts(
                        run_paths=run_paths,
                        version=f"{run_version}_{feature_set_name}",
                        feature_set_name=feature_set_name,
                        best_params=best_xgb_final["params"],
                        classifier=best_xgb_final["classifier"],
                        regressor=best_xgb_final["regressor"],
                        results_payload=results_payload,
                        feature_names=XGB_FEATURE_SETS[feature_set_name],
                        y_test_cls=best_xgb_final.get("y_test_cls"),
                        test_pred_cls=best_xgb_final.get("test_pred_cls"),
                        y_test_reg=best_xgb_final.get("y_test_reg"),
                        test_pred_reg=best_xgb_final.get("test_pred_reg"),
                        fold_cls_metrics=best_xgb_fold_cls_metrics,
                        fold_reg_metrics=best_xgb_fold_reg_metrics,
                        feature_importances=best_xgb_final.get("feature_importances"),
                    )
                    artifact_paths_by_feature_set[feature_set_name] = artifact_paths
                    logging.info("[ARTIFACTS:%s] Saved: %s", feature_set_name, artifact_paths)

                del best_xgb_final
                gc.collect()
                memory_snapshot(f"after cleanup: {feature_set_name}")

            results_payload["xgb_cv_results_by_feature_set"] = xgb_cv_results_by_feature_set
            results_payload["best_xgb_rows_by_feature_set"] = best_xgb_rows_by_feature_set
            results_payload["best_xgb_finals_by_feature_set"] = best_xgb_finals_by_feature_set
            results_payload["artifact_paths_by_feature_set"] = artifact_paths_by_feature_set
            results_payload["model_version"] = run_version

        # ============================================================
        # LSTM — uses legacy DataFrame path (collect full data)
        # ============================================================
        best_lstm_final = None

        if CONFIG.models.run_lstm:
            logging.info("[LSTM] Loading full DataFrames for LSTM training")
            if not TF_AVAILABLE:
                raise RuntimeError("LSTM requested but TensorFlow is not available.")

            lstm_required = resolve_required_columns(
                run_xgb=False, run_lstm=True,
            )
            from pipeline_core import collect_modeling_splits as _cms
            split_dict = _cms(
                canonical_dir=CONFIG.data.canonical_dir,
                file_pattern=CONFIG.data.file_pattern,
                train_years=train_years,
                validation_years=CONFIG.split.validation_years,
                test_years=CONFIG.split.test_years,
                use_validation_holdout=CONFIG.split.use_validation_holdout,
                run_xgb=False,
                run_lstm=True,
                train_sample_fraction=CONFIG.runtime.train_sample_fraction,
                val_sample_fraction=CONFIG.runtime.val_sample_fraction,
                test_sample_fraction=CONFIG.runtime.test_sample_fraction,
                sample_seed=CONFIG.runtime.random_seed,
            )
            lstm_train_df = split_dict["train_df"]
            lstm_val_df = split_dict["val_df"]
            lstm_test_df = split_dict["test_df"]

            if CONFIG.split.use_validation_holdout and lstm_val_df is not None:
                lstm_final_train_df = pl.concat([lstm_train_df, lstm_val_df], how="vertical_relaxed")
            else:
                lstm_final_train_df = lstm_train_df

            variant_name = CONFIG.models.lstm_variant_name
            lstm_cv_results = None
            best_lstm_row = None

            if CONFIG.cv.enabled:
                lstm_cv_splits = make_rolling_year_cv_splits(
                    lstm_train_df, min_train_years=CONFIG.cv.min_train_years
                )
                lstm_cv_param_grid = (
                    CONFIG.lstm_search.param_grid
                    if CONFIG.models.tune_lstm
                    else CONFIG.lstm_search.param_grid[:1]
                )
                lstm_cv_results = run_lstm_time_cv(
                    cv_splits=lstm_cv_splits,
                    variant_name=variant_name,
                    param_grid=lstm_cv_param_grid,
                    sample_fraction=CONFIG.runtime.sample_fraction_for_lstm_tuning,
                    seed=CONFIG.runtime.random_seed,
                )
                best_lstm_cv_row = lstm_cv_results.iloc[0].to_dict()
                best_lstm_cv_row.pop("fold_cls_metrics", None)
                best_lstm_cv_row.pop("fold_reg_metrics", None)
                best_lstm_row = best_lstm_cv_row if CONFIG.models.tune_lstm else {
                    "variant_name": variant_name,
                    "params": CONFIG.lstm_search.param_grid[0],
                }
            else:
                best_lstm_row = {
                    "variant_name": variant_name,
                    "params": CONFIG.lstm_search.param_grid[0],
                }

            best_lstm_final = refit_best_lstm(
                train_df=lstm_final_train_df,
                test_df=lstm_test_df,
                variant_name=variant_name,
                best_params=best_lstm_row["params"],
            )

            logging.info("[LSTM FINAL] Classification=%s", best_lstm_final["test_cls_metrics"])
            logging.info("[LSTM FINAL] Regression=%s", best_lstm_final["test_reg_metrics"])

            results_payload["best_lstm_final"] = {
                "model_family": best_lstm_final["model_family"],
                "variant_name": best_lstm_final["variant_name"],
                "params": best_lstm_final["params"],
                "test_cls_metrics": best_lstm_final["test_cls_metrics"],
                "test_reg_metrics": best_lstm_final["test_reg_metrics"],
            }

            del lstm_train_df, lstm_val_df, lstm_test_df, lstm_final_train_df
            gc.collect()

        # ============================================================
        # COMBINED FINAL RESULTS
        # ============================================================
        final_rows = []

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
            logging.info("FINAL TEST RESULTS\n%s", final_test_results)
            logging.info("================================================")

            best_overall_model = final_test_results.iloc[0].to_dict()
            logging.info("BEST OVERALL MODEL: %s", best_overall_model)

            results_payload["final_test_results"] = final_test_results.to_dict(orient="records")
            results_payload["best_overall_model"] = best_overall_model
        else:
            logging.info("[INFO] No models were run.")
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
            save_succeeded = False
            for call in [
                lambda: save_results_summary(results_payload=results_payload, output_dir=output_dir),
                lambda: save_results_summary(results_payload, output_dir),
            ]:
                if save_succeeded:
                    break
                try:
                    call()
                    save_succeeded = True
                except TypeError:
                    pass
            if save_succeeded:
                logging.info("[SAVE] Results summary written to %s", output_dir)
            else:
                logging.warning("[SAVE] Could not write results summary.")
        except Exception as e:
            logging.exception("[SAVE] Error saving results summary: %s", e)
        safe_timer_log("Save results summary", t0)

        # ============================================================
        # GENERATE VISUALIZATIONS
        # ============================================================
        if CONFIG.visualizations.enabled:
            viz_t0 = time.perf_counter()
            try:
                generate_visualizations(
                    summary_json_path=run_paths["evaluations_dir"] / "results_summary.json",
                    plots_dir=run_paths["plots_dir"],
                    tables_dir=run_paths["tables_dir"],
                    predictions_path=None,
                    feature_importance_csv_path=None,
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
                logging.info("[VIZ] Plots and tables written to %s", run_paths["run_dir"])
            except Exception as e:
                logging.exception("[VIZ] Visualization failed: %s", e)
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
        logging.exception("[FATAL] MemoryError.")
        raise
    except Exception as e:
        logging.exception("[FATAL] Pipeline failed: %s", e)
        raise
    finally:
        gc.collect()
        memory_snapshot("final cleanup")


if __name__ == "__main__":
    main()