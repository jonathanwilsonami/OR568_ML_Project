"""
Regenerate the actual-vs-predicted delay scatter plot with a perfect-prediction
reference line, using only the saved predictions parquet.

No retraining required.

Usage:
    python plot_actual_vs_predicted.py \
        --predictions path/to/test_predictions_xgb_full_aircraft.parquet \
        --output      path/to/actuals-v-predicted-final.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd


def plot_actual_vs_predicted(
    predictions_path: str | Path,
    output_path: str | Path,
    title: str = "Actual vs Predicted Arrival Delay\n(xgb_full_aircraft, 2025 holdout)",
    sample_n: int = 10_000,
    dpi: int = 150,
    xlim: tuple[float, float] = (-60, 300),
    ylim: tuple[float, float] = (-60, 300),
) -> None:
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(predictions_path)
    y_true = df["y_test_reg"].astype(float).values
    y_pred = df["test_pred_reg"].astype(float).values

    # ---------------------------------------------------------------
    # Subsample for plotting density (all points at 6.8 M overplot)
    # ---------------------------------------------------------------
    n = len(y_true)
    sample_n = min(sample_n, n)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, size=sample_n, replace=False)
    y_true_s = y_true[idx]
    y_pred_s = y_pred[idx]

    # ---------------------------------------------------------------
    # Summary metrics for annotation
    # ---------------------------------------------------------------
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))

    # Scatter — semi-transparent to show density
    ax.scatter(
        y_true_s, y_pred_s,
        alpha=0.15, s=6, color="#3a86ff", linewidths=0,
        label=f"Predictions (n = {sample_n:,} sampled)",
        rasterized=True,
    )

    # Perfect-prediction reference line  y_pred = y_actual
    ref_min = max(xlim[0], ylim[0])
    ref_max = min(xlim[1], ylim[1])
    ax.plot(
        [ref_min, ref_max], [ref_min, ref_max],
        color="#d62828", linewidth=1.8, linestyle="--",
        label="Perfect prediction (y = x)",
        zorder=5,
    )

    # ---------------------------------------------------------------
    # Zero-delay reference lines (dashed grey)
    # ---------------------------------------------------------------
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)

    # ---------------------------------------------------------------
    # Annotation box
    # ---------------------------------------------------------------
    ax.text(
        0.97, 0.05,
        f"MAE  = {mae:.1f} min\nRMSE = {rmse:.1f} min\nn = {n:,}",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="lightgrey", alpha=0.9),
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Actual Arrival Delay (minutes)", fontsize=11)
    ax.set_ylabel("Predicted Arrival Delay (minutes)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved → {output_path}")
    print(f"  MAE : {mae:.2f} min")
    print(f"  RMSE: {rmse:.2f} min")
    print(f"  n   : {n:,} total  ({sample_n:,} plotted)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot actual vs predicted delay with a perfect-prediction guide line."
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to test_predictions_xgb_full_aircraft.parquet",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output image path (e.g. images/paper/actuals-v-predicted-final.png)",
    )
    parser.add_argument(
        "--title", default="Actual vs Predicted Arrival Delay\n(xgb_full_aircraft, 2025 holdout)",
    )
    parser.add_argument("--sample-n",  type=int,   default=10_000)
    parser.add_argument("--dpi",       type=int,   default=150)
    parser.add_argument("--xlim-min",  type=float, default=-60)
    parser.add_argument("--xlim-max",  type=float, default=300)
    parser.add_argument("--ylim-min",  type=float, default=-60)
    parser.add_argument("--ylim-max",  type=float, default=300)
    args = parser.parse_args()

    plot_actual_vs_predicted(
        predictions_path=args.predictions,
        output_path=args.output,
        title=args.title,
        sample_n=args.sample_n,
        dpi=args.dpi,
        xlim=(args.xlim_min, args.xlim_max),
        ylim=(args.ylim_min, args.ylim_max),
    )