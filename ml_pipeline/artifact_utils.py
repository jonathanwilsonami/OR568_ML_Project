from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_run_version(config: dict) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_years = config["train_years"]
    test_years = config["test_years"]

    train_label = f"{min(train_years)}_{max(train_years)}"
    test_label = f"{min(test_years)}_{max(test_years)}"

    model_family = "xgb" if config.get("run_xgb") else "lstm"
    return f"{model_family}_train_{train_label}_test_{test_label}_{timestamp}"


def init_run_directories(base_output_dir: Path, run_version: str) -> dict[str, Path]:
    run_dir = ensure_dir(base_output_dir / "runs" / run_version)

    paths = {
        "run_dir": run_dir,
        "logs_dir": ensure_dir(run_dir / "logs"),
        "models_dir": ensure_dir(run_dir / "models"),
        "evaluations_dir": ensure_dir(run_dir / "evaluations"),
        "plots_dir": ensure_dir(run_dir / "plots"),
        "tables_dir": ensure_dir(run_dir / "tables"),
    }
    return paths


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def save_xgb_artifacts(
    run_paths: dict[str, Path],
    version: str,
    feature_set_name: str,
    best_params: dict,
    classifier,
    regressor,
    results_payload: dict,
    feature_names: list[str],
    y_test_cls=None,
    test_pred_cls=None,
    y_test_reg=None,
    test_pred_reg=None,
) -> dict[str, str]:
    models_dir = run_paths["models_dir"]
    eval_dir = run_paths["evaluations_dir"]

    classifier_path = models_dir / f"xgb_classifier_{feature_set_name}.joblib"
    regressor_path = models_dir / f"xgb_regressor_{feature_set_name}.joblib"
    metadata_path = models_dir / f"metadata_{feature_set_name}.json"
    summary_json_path = eval_dir / "results_summary.json"
    feature_importance_csv_path = eval_dir / f"feature_importance_{feature_set_name}.csv"
    predictions_path = eval_dir / f"test_predictions_{feature_set_name}.parquet"

    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    metadata = {
        "version": version,
        "model_family": "xgb",
        "feature_set_name": feature_set_name,
        "params": best_params,
        "feature_names": feature_names,
    }
    save_json(metadata, metadata_path)
    save_json(results_payload, summary_json_path)

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_,
    }).sort_values("importance", ascending=False)
    feature_importance_df.to_csv(feature_importance_csv_path, index=False)

    if (
        y_test_cls is not None
        and test_pred_cls is not None
        and y_test_reg is not None
        and test_pred_reg is not None
    ):
        pred_df = pd.DataFrame({
            "y_test_cls": y_test_cls,
            "test_pred_cls": test_pred_cls,
            "y_test_reg": y_test_reg,
            "test_pred_reg": test_pred_reg,
        })
        pred_df.to_parquet(predictions_path, index=False)

    return {
    "run_dir": str(run_paths["run_dir"]),
    "logs_dir": str(run_paths["logs_dir"]),
    "models_dir": str(models_dir),
    "evaluations_dir": str(eval_dir),
    "plots_dir": str(run_paths["plots_dir"]),
    "tables_dir": str(run_paths["tables_dir"]),
    "classifier_path": str(classifier_path),
    "regressor_path": str(regressor_path),
    "metadata_path": str(metadata_path),
    "summary_json_path": str(summary_json_path),
    "feature_importance_csv_path": str(feature_importance_csv_path),
    "predictions_path": str(predictions_path) if (
        y_test_cls is not None
        and test_pred_cls is not None
        and y_test_reg is not None
        and test_pred_reg is not None
    ) else None,
}