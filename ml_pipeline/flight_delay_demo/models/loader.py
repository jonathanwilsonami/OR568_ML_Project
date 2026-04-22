"""
models/loader.py
----------------
Multi-model inference layer supporting:

  XGBoost      xgb_classifier_<set>.joblib + xgb_regressor_<set>.joblib
  Sklearn      lr_classifier_<set>.joblib  + lr_regressor_<set>.joblib
               (logistic regression, random forest, any sklearn estimator)
  LSTM         lstm_<variant>.h5  (Keras/TF, expects 3-D timestep input)

File naming convention
──────────────────────
  <family>_classifier_<name>.joblib   paired with
  <family>_regressor_<name>.joblib

  lstm_<name>_cls.h5   paired with
  lstm_<name>_reg.h5

  family is anything: xgb, lr, rf, lgbm, …
  name  is anything: xgb_full, xgb_full_aircraft, schedule, …

Feature column resolution (in priority order)
──────────────────────────────────────────────
  1. Explicit feature_cols argument
  2. XGBoost: clf.get_booster().feature_names
  3. Sklearn:  clf.feature_names_in_
  4. LSTM:     reads a sidecar  lstm_<name>_features.json  next to the .h5
  5. Falls back to XGB_FULL_FEATURES from data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import XGB_FULL_FEATURES, STEP_FEATURES, LSTM_CONTEXT_CURRENT

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import joblib as _joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import tensorflow as _tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ---------------------------------------------------------------------------
# In-memory model cache
# ---------------------------------------------------------------------------

_cache: dict[str, object] = {}


def _load(path: Path) -> object:
    key = str(path)
    if key not in _cache:
        suffix = path.suffix.lower()
        if suffix in (".h5", ".keras"):
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow is required to load LSTM models.")
            _cache[key] = _tf.keras.models.load_model(str(path))
        else:
            if not JOBLIB_AVAILABLE:
                raise ImportError("joblib is required to load sklearn/XGBoost models.")
            _cache[key] = _joblib.load(path)
    return _cache[key]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_models(model_dir: str | Path) -> dict[str, dict]:
    """
    Scan model_dir for all supported model pairs.
    Returns:
        {
          "Xgb Full":          {"clf_path": ..., "reg_path": ..., "family": "xgb", "name": "xgb_full"},
          "Xgb Full Aircraft": {...},
          "Lr Schedule":       {..., "family": "lr"},
          "Lstm Context Full": {"clf_path": ..., "reg_path": ..., "family": "lstm", "name": "context_full"},
        }
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return {}

    models: dict[str, dict] = {}

    # ── joblib pairs  (<family>_classifier_<name> + <family>_regressor_<name>) ──
    clf_files: dict[str, Path] = {}
    reg_files: dict[str, Path] = {}
    for f in model_dir.glob("*.joblib"):
        stem = f.stem
        if "_classifier_" in stem:
            clf_files[stem] = f
        elif "_regressor_" in stem:
            reg_files[stem] = f

    for clf_stem, clf_path in sorted(clf_files.items()):
        # e.g. xgb_classifier_xgb_full → family=xgb, name=xgb_full
        parts    = clf_stem.split("_classifier_", 1)
        family   = parts[0]
        name     = parts[1]
        reg_stem = f"{family}_regressor_{name}"
        if reg_stem in reg_files:
            display = f"{name.replace('_', ' ').title()} ({family.upper()})"
            models[display] = {
                "clf_path": clf_path,
                "reg_path": reg_files[reg_stem],
                "family":   family,
                "name":     name,
            }

    # ── LSTM h5 pairs  (lstm_<name>_cls.h5 + lstm_<name>_reg.h5) ────────────
    cls_h5 = {f.stem: f for f in model_dir.glob("lstm_*_cls.h5")}
    reg_h5 = {f.stem: f for f in model_dir.glob("lstm_*_reg.h5")}

    for cls_stem, cls_path in sorted(cls_h5.items()):
        # lstm_context_full_cls → name = context_full
        name     = cls_stem.replace("lstm_", "").replace("_cls", "")
        reg_stem = f"lstm_{name}_reg"
        if reg_stem in reg_h5:
            display = f"{name.replace('_', ' ').title()} (LSTM)"
            models[display] = {
                "clf_path": cls_path,
                "reg_path": reg_h5[reg_stem],
                "family":   "lstm",
                "name":     name,
            }

    return models


def load_model_pair(clf_path: Path, reg_path: Path) -> tuple:
    return _load(clf_path), _load(reg_path)


# ---------------------------------------------------------------------------
# Feature name introspection
# ---------------------------------------------------------------------------

def get_feature_names(clf, model_info: dict | None = None) -> list[str]:
    """
    Return the feature list the classifier was trained on.
    Tries multiple strategies depending on model family.
    """
    # XGBoost
    try:
        names = clf.get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass

    # Sklearn (LogisticRegression, RandomForest, Pipeline, etc.)
    try:
        names = list(clf.feature_names_in_)
        if names:
            return names
    except Exception:
        pass

    # sklearn Pipeline wrapping XGB
    try:
        names = clf.named_steps["clf"].get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass

    # LSTM sidecar JSON  lstm_<name>_features.json
    if model_info and model_info.get("family") == "lstm":
        sidecar = Path(model_info["clf_path"]).parent / \
                  f"lstm_{model_info['name']}_features.json"
        if sidecar.exists():
            try:
                with open(sidecar) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    return data
                if "step_features" in data:
                    return data["step_features"]
            except Exception:
                pass

    # Last resort
    return list(XGB_FULL_FEATURES)


# ---------------------------------------------------------------------------
# LSTM input builder
# ---------------------------------------------------------------------------

def _build_lstm_input(row: pd.Series, step_features: list[str]) -> np.ndarray:
    """
    Build the (1, 3, n_features) array the LSTM expects from a flat row.

    Timestep 0 = prev2 context
    Timestep 1 = prev1 context
    Timestep 2 = current flight context

    Mirrors build_lstm_step_matrix from your pipeline.
    """
    n = len(step_features)
    X = np.zeros((1, 3, n), dtype="float32")

    def _set(timestep: int, feat: str, col: str) -> None:
        if feat in step_features and col in row.index:
            val = row[col]
            X[0, timestep, step_features.index(feat)] = float(val) if pd.notna(val) else 0.0

    # prev2 timestep
    _set(0, "arr_delay_prev", "prev2_arr_delay")
    _set(0, "dep_delay_prev", "prev2_dep_delay")
    _set(0, "arr_del15_prev", "prev2_arr_del15")
    _set(0, "dep_del15_prev", "prev2_dep_del15")
    _set(0, "gap_minutes",    "time_since_prev2_arrival_minutes")

    # prev1 timestep
    _set(1, "arr_delay_prev", "prev1_arr_delay")
    _set(1, "dep_delay_prev", "prev1_dep_delay")
    _set(1, "arr_del15_prev", "prev1_arr_del15")
    _set(1, "dep_del15_prev", "prev1_dep_del15")
    _set(1, "gap_minutes",    "prev1_turnaround_minutes")

    # current flight context (all shared context features)
    context_cols = LSTM_CONTEXT_CURRENT
    for col in context_cols:
        if col in step_features and col in row.index:
            val = row[col]
            X[0, 2, step_features.index(col)] = float(val) if pd.notna(val) else 0.0

    return X


# ---------------------------------------------------------------------------
# Unified predict_row
# ---------------------------------------------------------------------------

def predict_row(
    row: pd.Series,
    clf,
    reg,
    feature_cols: list[str] | None = None,
    threshold: float = 0.5,
    model_info: dict | None = None,
) -> dict:
    """
    Run classifier + regressor on one row.
    Handles XGBoost, sklearn, and LSTM transparently.
    """
    family = (model_info or {}).get("family", "xgb")

    # ── LSTM path ────────────────────────────────────────────────────────────
    if family == "lstm":
        step_features = feature_cols or get_feature_names(clf, model_info)
        if not step_features:
            step_features = list(STEP_FEATURES)

        X_seq = _build_lstm_input(row, step_features)

        raw_cls = clf.predict(X_seq, verbose=0)
        raw_reg = reg.predict(X_seq, verbose=0)

        # Handle various output shapes
        prob      = float(np.squeeze(raw_cls))
        if prob > 1.0:                         # sigmoid not applied
            from scipy.special import expit
            prob = float(expit(prob))
        delay_est = float(np.squeeze(raw_reg))
        verdict   = "Delayed" if prob >= threshold else "On time"

        return {
            "prob":          round(prob, 4),
            "delay_est":     round(delay_est, 1),
            "verdict":       verdict,
            "features_used": step_features,
            "missing_cols":  [],
        }

    # ── Flat model path (XGBoost / sklearn) ──────────────────────────────────
    cols = feature_cols or get_feature_names(clf, model_info)
    if not cols:
        raise ValueError(
            "Cannot determine feature columns: model has no stored feature names "
            "and no feature_cols were passed."
        )

    row_dict = row.to_dict()
    missing  = [c for c in cols if c not in row_dict or pd.isna(row_dict.get(c))]
    X = pd.DataFrame(
        [{c: float(row_dict.get(c, 0) or 0) for c in cols}],
        columns=cols,
    )

    prob      = float(clf.predict_proba(X)[0, 1])
    delay_est = float(reg.predict(X)[0])
    verdict   = "Delayed" if prob >= threshold else "On time"

    return {
        "prob":          round(prob, 4),
        "delay_est":     round(delay_est, 1),
        "verdict":       verdict,
        "features_used": cols,
        "missing_cols":  missing,
    }


# ---------------------------------------------------------------------------
# collect_all_model_features  (used by data.py cache builder)
# ---------------------------------------------------------------------------

def collect_all_model_features(models_dir: str | Path) -> list[str]:
    """
    Union of all flat feature lists across all discovered models.
    LSTM models are skipped (they use timestep arrays, not flat columns).
    """
    models_dir = Path(models_dir)
    all_features: set[str] = set()

    for info in discover_models(models_dir).values():
        if info["family"] == "lstm":
            continue
        try:
            clf = _load(info["clf_path"])
            names = get_feature_names(clf, info)
            all_features.update(names)
            print(f"  [features] {info['clf_path'].name}: {len(names)} features")
        except Exception as e:
            print(f"  [features] Could not read {info['clf_path'].name}: {e}")

    return sorted(all_features)