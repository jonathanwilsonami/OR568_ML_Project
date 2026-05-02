from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_results(json_path: str | Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metric_bars(
    df: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    sort_ascending: bool = False,
    dpi: int = 150,
):
    plot_df = df.sort_values(metric, ascending=sort_ascending).copy()

    plt.figure(figsize=(12, 5))
    plt.bar(plot_df["model"], plot_df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    output_path: Path,
    top_n: int = 20,
    dpi: int = 150,
):
    plot_df = importance_df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"][::-1], plot_df["importance"][::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    predictions_df: pd.DataFrame,
    output_path: Path,
    title: str = "ROC Curve",
    dpi: int = 150,
):
    y_true = predictions_df["y_test_cls"].astype(int).values
    y_prob = predictions_df["test_pred_cls"].astype(float).values

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    output_path: Path,
    title: str = "Actual vs Predicted Delay",
    sample_n: int = 5000,
    dpi: int = 150,
):
    y_true = predictions_df["y_test_reg"].astype(float).values
    y_pred = predictions_df["test_pred_reg"].astype(float).values

    n = len(y_true)
    if n == 0:
        return

    sample_n = min(sample_n, n)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=sample_n, replace=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true[idx], y_pred[idx], alpha=0.3)
    plt.xlabel("Actual Arrival Delay (minutes)")
    plt.ylabel("Predicted Arrival Delay (minutes)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def build_summary_tables(results: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    final_rows = results.get("final_test_results", [])
    final_df = pd.DataFrame(final_rows)

    if not final_df.empty:
        final_df = final_df.rename(columns={
            "model_family": "model",
            "details": "variant",
        })

    cv_rows = results.get("xgb_cv_results")
    cv_df = pd.DataFrame(cv_rows) if cv_rows else None

    return final_df, cv_df


def generate_visualizations(
    summary_json_path: str | Path,
    plots_dir: str | Path,
    tables_dir: str | Path,
    predictions_path: str | Path | None = None,
    feature_importance_csv_path: str | Path | None = None,
    make_summary_table: bool = True,
    make_cv_table: bool = True,
    make_metric_bar_charts: bool = True,
    make_cv_metric_bar_charts: bool = True,
    make_feature_importance_chart: bool = True,
    make_roc_curve_chart: bool = True,
    make_actual_vs_predicted_chart: bool = True,
    top_n_features: int = 20,
    scatter_sample_n: int = 5000,
    dpi: int = 150,
):
    plots_dir = ensure_dir(Path(plots_dir))
    tables_dir = ensure_dir(Path(tables_dir))
    evaluations_dir = Path(summary_json_path).parent

    results = load_results(summary_json_path)
    final_df, cv_df = build_summary_tables(results)

    # -----------------------------
    # Tables
    # -----------------------------
    if make_summary_table and final_df is not None and not final_df.empty:
        final_df.to_csv(tables_dir / "final_test_summary.csv", index=False)
        final_df.to_markdown(tables_dir / "final_test_summary.md", index=False)

    if make_cv_table and cv_df is not None and not cv_df.empty:
        cv_df.to_csv(tables_dir / "xgb_cv_summary.csv", index=False)
        cv_df.to_markdown(tables_dir / "xgb_cv_summary.md", index=False)

    # -----------------------------
    # Final metric plots
    # -----------------------------
    if make_metric_bar_charts and final_df is not None and not final_df.empty:
        plot_metric_bars(
            final_df,
            "test_auc",
            "Final Test Comparison: AUC",
            plots_dir / "final_auc.png",
            sort_ascending=False,
            dpi=dpi,
        )

        plot_metric_bars(
            final_df,
            "test_f1",
            "Final Test Comparison: F1",
            plots_dir / "final_f1.png",
            sort_ascending=False,
            dpi=dpi,
        )

        plot_metric_bars(
            final_df,
            "test_mae",
            "Final Test Comparison: MAE",
            plots_dir / "final_mae.png",
            sort_ascending=True,
            dpi=dpi,
        )

        plot_metric_bars(
            final_df,
            "test_rmse",
            "Final Test Comparison: RMSE",
            plots_dir / "final_rmse.png",
            sort_ascending=True,
            dpi=dpi,
        )

        if len(final_df) > 1:
            plot_multi_metric_bar_chart(
                final_df=final_df,
                metrics=["test_auc", "test_f1", "test_accuracy"],
                output_path=plots_dir / "classification_metrics_side_by_side.png",
                title="Classification Metrics by Model",
                dpi=dpi,
            )

            plot_multi_metric_bar_chart(
                final_df=final_df,
                metrics=["test_mae", "test_rmse"],
                output_path=plots_dir / "regression_metrics_side_by_side.png",
                title="Regression Metrics by Model",
                dpi=dpi,
            )

            plot_multi_metric_line_chart(
                final_df=final_df,
                metrics=["test_auc", "test_f1", "test_accuracy"],
                output_path=plots_dir / "classification_metrics_overlay.png",
                title="Classification Metrics Overlay",
                dpi=dpi,
            )

            plot_multi_metric_line_chart(
                final_df=final_df,
                metrics=["test_mae", "test_rmse"],
                output_path=plots_dir / "regression_metrics_overlay.png",
                title="Regression Metrics Overlay",
                dpi=dpi,
            )

    # -----------------------------
    # CV plots
    # -----------------------------
    if make_cv_metric_bar_charts and cv_df is not None and not cv_df.empty:
        cv_plot_df = cv_df.copy()
        cv_plot_df["model"] = cv_plot_df["config_id"]

        plot_metric_bars(cv_plot_df, "cv_auc_mean", "CV Comparison: AUC", plots_dir / "cv_auc.png", sort_ascending=False, dpi=dpi)
        plot_metric_bars(cv_plot_df, "cv_f1_mean", "CV Comparison: F1", plots_dir / "cv_f1.png", sort_ascending=False, dpi=dpi)
        plot_metric_bars(cv_plot_df, "cv_mae_mean", "CV Comparison: MAE", plots_dir / "cv_mae.png", sort_ascending=True, dpi=dpi)
        plot_metric_bars(cv_plot_df, "cv_rmse_mean", "CV Comparison: RMSE", plots_dir / "cv_rmse.png", sort_ascending=True, dpi=dpi)

    # -----------------------------
    # Feature importance
    # -----------------------------
    if make_feature_importance_chart:
        importance_files = sorted(evaluations_dir.glob("feature_importance_*.csv"))

        for path in importance_files:
            model_name = path.stem.replace("feature_importance_", "")
            feature_importance_df = pd.read_csv(path)

            plot_feature_importance(
                feature_importance_df,
                title=f"Top Feature Importances: {model_name}",
                output_path=plots_dir / f"feature_importance_{model_name}.png",
                top_n=top_n_features,
                dpi=dpi,
            )

    # -----------------------------
    # Multi-model ROC curves
    # -----------------------------
    if make_roc_curve_chart:
        plot_multi_model_roc_curves(
            evaluations_dir=evaluations_dir,
            output_path=plots_dir / "roc_curves_all_models.png",
            dpi=dpi,
        )

    # -----------------------------
    # Actual vs predicted scatter
    # -----------------------------
    if make_actual_vs_predicted_chart:
        prediction_files = sorted(evaluations_dir.glob("test_predictions_*.parquet"))

        for pred_path in prediction_files:
            model_name = pred_path.stem.replace("test_predictions_", "")
            predictions_df = pd.read_parquet(pred_path)

            plot_actual_vs_predicted(
                predictions_df=predictions_df,
                output_path=plots_dir / f"actual_vs_predicted_delay_{model_name}.png",
                title=f"Actual vs Predicted Delay: {model_name}",
                sample_n=scatter_sample_n,
                dpi=dpi,
            )

def plot_multi_metric_bar_chart(
    final_df: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    title: str = "Model Comparison",
    dpi: int = 150,
):
    plot_df = final_df.copy()
    plot_df["model_label"] = plot_df["model"] + " - " + plot_df["variant"]

    metric_df = plot_df.set_index("model_label")[metrics]

    ax = metric_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric Value")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_multi_metric_line_chart(
    final_df: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    title: str = "Model Metric Overlay",
    dpi: int = 150,
):
    plot_df = final_df.copy()
    plot_df["model_label"] = plot_df["model"] + " - " + plot_df["variant"]

    plt.figure(figsize=(12, 6))

    x = range(len(plot_df))
    for metric in metrics:
        plt.plot(x, plot_df[metric], marker="o", label=metric)

    plt.xticks(x, plot_df["model_label"], rotation=45, ha="right")
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def plot_multi_model_roc_curves(
    evaluations_dir: Path,
    output_path: Path,
    dpi: int = 150,
):
    prediction_files = sorted(evaluations_dir.glob("test_predictions_*.parquet"))

    if not prediction_files:
        return

    plt.figure(figsize=(8, 6))

    for pred_path in prediction_files:
        model_name = pred_path.stem.replace("test_predictions_", "")
        df = pd.read_parquet(pred_path)

        y_true = df["y_test_cls"].astype(int).values
        y_prob = df["test_pred_cls"].astype(float).values

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        plt.plot(fpr, tpr, label=f"{model_name} AUC={auc_val:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves by Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()