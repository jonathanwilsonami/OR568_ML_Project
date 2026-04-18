"""
models/loader.py
----------------
Loads XGBoost .joblib model artifacts from a directory and runs inference.

Expected file naming convention (from the pipeline):
    xgb_classifier_<feature_set_name>.joblib   e.g. xgb_classifier_xgb_full.joblib
    xgb_regressor_<feature_set_name>.joblib    e.g. xgb_regressor_xgb_full.joblib

Both files must exist for a model to appear in the selector.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data import XGB_FULL_FEATURES


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(model_dir: str | Path) -> dict[str, dict]:
    """
    Scan a directory for matched classifier/regressor pairs.
    Returns { display_name: {"clf_path": ..., "reg_path": ..., "feature_set": ...} }
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return {}

    clf_files = {f.stem: f for f in model_dir.glob("xgb_classifier_*.joblib")}
    reg_files = {f.stem: f for f in model_dir.glob("xgb_regressor_*.joblib")}

    models = {}
    for clf_stem, clf_path in clf_files.items():
        # xgb_classifier_xgb_full → xgb_full
        feature_set = clf_stem.replace("xgb_classifier_", "")
        reg_stem = f"xgb_regressor_{feature_set}"

        if reg_stem in reg_files:
            display = feature_set.replace("_", " ").title()
            models[display] = {
                "clf_path":    clf_path,
                "reg_path":    reg_files[reg_stem],
                "feature_set": feature_set,
            }

    return models


# ---------------------------------------------------------------------------
# Loading (cached in memory across requests)
# ---------------------------------------------------------------------------

_cache: dict[str, tuple] = {}


def load_model_pair(clf_path: Path, reg_path: Path) -> tuple:
    key = str(clf_path) + "|" + str(reg_path)
    if key not in _cache:
        _cache[key] = (joblib.load(clf_path), joblib.load(reg_path))
    return _cache[key]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_row(
    row: pd.Series,
    clf,
    reg,
    feature_cols: list[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Run classifier + regressor on a single row (as a 1-row DataFrame).
    Returns a dict with prob, delay_est, verdict, and feature values used.
    """
    cols = feature_cols or XGB_FULL_FEATURES
    X = pd.DataFrame([row[cols].values], columns=cols).fillna(0)

    prob       = float(clf.predict_proba(X)[0, 1])
    delay_est  = float(reg.predict(X)[0])
    verdict    = "Delayed" if prob >= threshold else "On time"

    return {
        "prob":       round(prob, 4),
        "delay_est":  round(delay_est, 1),
        "verdict":    verdict,
    }


def predict_dataframe(
    df: pd.DataFrame,
    clf,
    reg,
    feature_cols: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Run predictions on an entire DataFrame, return df with added columns."""
    cols = feature_cols or XGB_FULL_FEATURES
    X = df[cols].fillna(0)

    probs     = clf.predict_proba(X)[:, 1]
    delays    = reg.predict(X)
    verdicts  = ["Delayed" if p >= threshold else "On time" for p in probs]

    out = df.copy()
    out["pred_prob"]    = probs.round(4)
    out["pred_delay"]   = delays.round(1)
    out["pred_verdict"] = verdicts
    return out