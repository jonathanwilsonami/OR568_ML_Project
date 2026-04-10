from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # Directory containing flights_canonical_YYYY.parquet
    canonical_dir: str = "/home/jon/projects/OR568_ML_Project/data_pipeline/features"
    file_pattern: str = "flights_canonical_{year}.parquet"

    # Where to save results / artifacts
    output_dir: str = "/home/jon/projects/OR568_ML_Project/model_outputs"


@dataclass
class SplitConfig:
    # Option A
    train_start_year: int = 2015
    train_end_year: int = 2023
    validation_years: list[int] = field(default_factory=lambda: [2024])
    test_years: list[int] = field(default_factory=lambda: [2025])

    # If False, skip separate validation holdout and rely only on rolling CV
    use_validation_holdout: bool = True


@dataclass
class CVConfig:
    enabled: bool = True
    strategy: str = "rolling_year"

    # With train years 2015-2023 this yields:
    # train 2015-2018 -> val 2019
    # train 2015-2019 -> val 2020
    # ...
    min_train_years: int = 4


@dataclass
class RuntimeConfig:
    random_seed: int = 42
    use_gpu: bool = False
    n_jobs: int = -1

    # Tuning can be very expensive on full data.
    # Keep None for full data, or use e.g. 0.10 for 10%.
    sample_fraction_for_tuning: float | None = 0.10

    # LSTM is much heavier; use a smaller fraction if you enable it for tuning
    sample_fraction_for_lstm_tuning: float | None = 0.03

    # For final fit, usually use full training data
    sample_fraction_for_final_train: float | None = None


@dataclass
class ModelConfig:
    run_xgb: bool = True
    run_lstm: bool = False

    tune_xgb: bool = True
    tune_lstm: bool = False

    # XGBoost feature set
    xgb_feature_set_name: str = "xgb_full"

    # LSTM feature set
    lstm_variant_name: str = "context_full"


@dataclass
class XGBSearchConfig:
    # keep modest at first
    param_grid: list[dict] = field(default_factory=lambda: [
        {
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 350,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
        },
        {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_lambda": 5.0,
        },
    ])


@dataclass
class LSTMSearchConfig:
    # heavier, keep smaller
    param_grid: list[dict] = field(default_factory=lambda: [
        {
            "units1": 32,
            "units2": 16,
            "dropout": 0.2,
            "batch_size": 256,
            "epochs": 10,
            "learning_rate": 1e-3,
        },
        {
            "units1": 64,
            "units2": 32,
            "dropout": 0.2,
            "batch_size": 256,
            "epochs": 10,
            "learning_rate": 1e-3,
        },
        {
            "units1": 64,
            "units2": 32,
            "dropout": 0.3,
            "batch_size": 256,
            "epochs": 12,
            "learning_rate": 1e-3,
        },
    ])


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    xgb_search: XGBSearchConfig = field(default_factory=XGBSearchConfig)
    lstm_search: LSTMSearchConfig = field(default_factory=LSTMSearchConfig)


CONFIG = PipelineConfig()
Path(CONFIG.data.output_dir).mkdir(parents=True, exist_ok=True)