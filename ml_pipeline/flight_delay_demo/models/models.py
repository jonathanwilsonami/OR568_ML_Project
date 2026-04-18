"""
models/models.py
----------------
Synthetic predictions that mirror your real pipeline output.

When you're ready to plug in real models, replace _synthetic_predict()
with your actual XGBoost / LSTM inference. Everything else stays the same.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Feature container  (mirrors pipeline step-14 fields)
# ---------------------------------------------------------------------------

@dataclass
class FlightFeatures:
    flight_id: str
    tail_number: str
    origin: str
    dest: str
    distance: float
    dep_hour_local: int

    prev2_arr_delay: float
    prev2_dep_delay: float
    prev1_arr_delay: float
    prev1_dep_delay: float
    prev1_turnaround_minutes: float
    time_since_prev2_arrival_minutes: float

    dep_temp_c: float
    dep_wind_speed_m_s: float
    actual_arr_delay: float
    actual_dep_delay: float

    # Route chain airports — used for the geographic map
    prev2_origin: str = ""   # 2 hops back departure airport
    prev2_dest:   str = ""   # 2 hops back arrival  (= prev1 departure)

    is_holiday: bool = False
    is_weekend: bool = False
    dep_time_bucket: str = "afternoon"


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    model_name: str
    prob: float        # P(arr_del15 == 1)
    delay_est: float   # regression estimate in minutes
    verdict: str       # "Delayed" | "On time"
    correct: bool


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

AVAILABLE_MODELS: dict[str, str] = {
    "xgb_base":  "XGBoost base",
    "xgb_full":  "XGBoost full",
    "lstm_seq":  "LSTM sequential",
    "lstm_attn": "LSTM + attention",
    "lgbm":      "LightGBM",
    "rf":        "Random forest",
    "logreg":    "Logistic regression",
}

_BASE_PROBS: dict[str, float] = {
    "xgb_base":  0.61,
    "xgb_full":  0.74,
    "lstm_seq":  0.69,
    "lstm_attn": 0.77,
    "lgbm":      0.72,
    "rf":        0.58,
    "logreg":    0.53,
}


# ---------------------------------------------------------------------------
# Inference  — swap this function for your real model calls
# ---------------------------------------------------------------------------

def _synthetic_predict(model_name: str, f: FlightFeatures) -> tuple[float, float]:
    """Deterministic stand-in. Returns (prob, delay_est_minutes)."""
    base = _BASE_PROBS.get(model_name, 0.62)
    prop   = min((f.prev2_arr_delay + f.prev1_arr_delay) / 160.0, 1.0) * 0.18
    weather = (0.04 if abs(f.dep_temp_c) > 5 else 0.0) + (0.03 if f.dep_wind_speed_m_s > 10 else 0.0)
    noise  = math.sin(len(model_name) * 7 + f.prev1_arr_delay) * 0.06
    prob   = min(0.97, max(0.05, base + prop + weather + noise))
    delay  = max(0.0, f.prev1_arr_delay * 0.55 + f.prev2_arr_delay * 0.15 + (prob - 0.5) * 40 + 8)
    return round(prob, 4), round(delay, 1)


def predict(model_name: str, features: FlightFeatures, threshold: float = 0.50) -> Prediction:
    prob, delay_est = _synthetic_predict(model_name, features)
    verdict = "Delayed" if prob >= threshold else "On time"
    correct = (verdict == "Delayed") == (features.actual_arr_delay >= 15)
    return Prediction(model_name=model_name, prob=prob, delay_est=delay_est,
                      verdict=verdict, correct=correct)


def run_all(model_names: list[str], features: FlightFeatures,
            threshold: float = 0.50) -> list[Prediction]:
    return [predict(m, features, threshold) for m in model_names]


def ensemble(predictions: list[Prediction]) -> tuple[float, float]:
    """Returns (mean_prob, mean_delay_est)."""
    avg_prob  = sum(p.prob      for p in predictions) / len(predictions)
    avg_delay = sum(p.delay_est for p in predictions) / len(predictions)
    return round(avg_prob, 4), round(avg_delay, 1)