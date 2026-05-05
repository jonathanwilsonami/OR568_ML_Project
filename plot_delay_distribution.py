"""
Plot the distribution of actual arrival delays vs predicted arrival delays
as overlaid KDE curves, so the reader can directly compare how closely the
model's output distribution matches reality.

No retraining required — reads the saved predictions parquet only.

Usage:
    python plot_delay_distribution.py \
        --predictions path/to/test_predictions_xgb_full_aircraft.parquet \
        --output      path/to/delay-distribution-comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_delay_distribution(
    predictions_path: str | Path,
    output_path: str | Path,
    title: str = "Distribution of Actual vs Predicted Arrival Delay\n(xgb_full_aircraft, 2025 holdout)",
    xlim: tuple[float, float] = (-60, 240),
    dpi: int = 150,
    sample_n: int | None = 50_000,
) -> None:
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(predictions_path)
    y_true = df["y_test_reg"].astype(float).values
    y_pred = df["test_pred_reg"].astype(float).values

    n_total = len(y_true)

    # ---------------------------------------------------------------
    # Subsample for KDE speed (KDE is O(n²) — 50k is plenty)
    # ---------------------------------------------------------------
    if sample_n is not None and sample_n < n_total:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_total, size=sample_n, replace=False)
        y_true_s = y_true[idx]
        y_pred_s = y_pred[idx]
    else:
        y_true_s = y_true
        y_pred_s = y_pred

    # Clip to xlim for KDE so extreme outliers don't flatten the curves
    mask_t = (y_true_s >= xlim[0]) & (y_true_s <= xlim[1])
    mask_p = (y_pred_s >= xlim[0]) & (y_pred_s <= xlim[1])
    y_true_clipped = y_true_s[mask_t]
    y_pred_clipped = y_pred_s[mask_p]

    # ---------------------------------------------------------------
    # KDE
    # ---------------------------------------------------------------
    x_grid = np.linspace(xlim[0], xlim[1], 1000)
    kde_true = gaussian_kde(y_true_clipped, bw_method="scott")
    kde_pred = gaussian_kde(y_pred_clipped, bw_method="scott")
    density_true = kde_true(x_grid)
    density_pred = kde_pred(x_grid)

    # ---------------------------------------------------------------
    # Summary stats for annotation
    # ---------------------------------------------------------------
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mean_actual = float(np.mean(y_true))
    mean_pred   = float(np.mean(y_pred))

    # ---------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filled area under curves for readability
    ax.fill_between(x_grid, density_true, alpha=0.15, color="#d62828")
    ax.fill_between(x_grid, density_pred, alpha=0.15, color="#3a86ff")

    # Main lines
    ax.plot(x_grid, density_true, color="#d62828", linewidth=2.0,
            label="Actual delay")
    ax.plot(x_grid, density_pred, color="#3a86ff", linewidth=2.0,
            label="Predicted delay")

    # Vertical mean lines
    ax.axvline(mean_actual, color="#d62828", linewidth=1.2,
               linestyle="--", alpha=0.8,
               label=f"Mean actual  = {mean_actual:.1f} min")
    ax.axvline(mean_pred, color="#3a86ff", linewidth=1.2,
               linestyle="--", alpha=0.8,
               label=f"Mean predicted = {mean_pred:.1f} min")

    # Zero-delay reference
    ax.axvline(0, color="grey", linewidth=0.9, linestyle=":",
               alpha=0.6, label="On-time boundary (0 min)")

    # Annotation box
    ax.text(
        0.97, 0.97,
        f"MAE  = {mae:.1f} min\nRMSE = {rmse:.1f} min\nn = {n_total:,}",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="lightgrey", alpha=0.9),
    )

    ax.set_xlim(xlim)
    ax.set_xlabel("Arrival Delay (minutes)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved → {output_path}")
    print(f"  Mean actual    : {mean_actual:.2f} min")
    print(f"  Mean predicted : {mean_pred:.2f} min")
    print(f"  MAE            : {mae:.2f} min")
    print(f"  RMSE           : {rmse:.2f} min")
    print(f"  n total        : {n_total:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare actual vs predicted delay distributions as overlaid KDE curves."
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to test_predictions_xgb_full_aircraft.parquet",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        default="Distribution of Actual vs Predicted Arrival Delay\n(xgb_full_aircraft, 2025 holdout)",
    )
    parser.add_argument("--xlim-min",  type=float, default=-60)
    parser.add_argument("--xlim-max",  type=float, default=240)
    parser.add_argument("--dpi",       type=int,   default=150)
    parser.add_argument("--sample-n",  type=int,   default=50_000)
    args = parser.parse_args()

    plot_delay_distribution(
        predictions_path=args.predictions,
        output_path=args.output,
        title=args.title,
        xlim=(args.xlim_min, args.xlim_max),
        dpi=args.dpi,
        sample_n=args.sample_n,
    )