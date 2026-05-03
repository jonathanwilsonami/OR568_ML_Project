from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # Automatically resolve project root relative to this file
    project_root: Path = Path(__file__).resolve().parents[1]

    # Directory containing flights_canonical_YYYY.parquet
    canonical_dir: Path = None

    # File naming pattern
    file_pattern: str = "flights_canonical_{year}.parquet"

    # Where to save results / artifacts
    output_dir: Path = None

    def __post_init__(self):
        self.canonical_dir = self.project_root / "data_pipeline" / "data" / "features"
        self.output_dir = self.project_root / "ml_pipeline" / "model_outputs"


@dataclass
class SplitConfig:
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

    # Avoid -1 on huge XGB fits; fewer threads usually lowers memory pressure
    n_jobs: int = 8

    # ---------------------------------------------------------------
    # EARLY-SAMPLE FRACTIONS  (applied before .collect() — the primary
    # OOM control knob).  These replace the old post-collect fractions
    # for train/val.  Set to None for the full partition.
    #
    # Rule of thumb for a 16 GB machine:
    #   train: 0.10  ->  ~5.5 M rows   (fast CV diagnostics)
    #   train: 0.35  ->  ~19 M rows    (final refit quality)
    #   val  : 0.50  ->  ~3.5 M rows
    #   test : None  ->  full ~6.9 M   (unbiased evaluation)
    # ---------------------------------------------------------------
    train_sample_fraction: float | None = 0.35   # applied at collect time
    val_sample_fraction: float | None = 0.50     # applied at collect time
    test_sample_fraction: float | None = None    # keep full test set

    # CV tuning uses a further in-memory subsample of the already-sampled
    # train partition (cheap — data is already in RAM at this point).
    sample_fraction_for_tuning: float | None = 0.30

    # LSTM is much heavier; use a smaller fraction if you enable it for tuning
    sample_fraction_for_lstm_tuning: float | None = 0.03

    # Final refit uses whatever is in final_train_df (already early-sampled).
    # Set to None to use all of final_train_df, or reduce further if needed.
    sample_fraction_for_final_train: float | None = None


@dataclass
class ModelConfig:
    run_xgb: bool = True
    run_lstm: bool = False

    tune_xgb: bool = False  
    tune_lstm: bool = False

    # XGBoost feature set
    xgb_feature_set_name: str = "xgb_full_aircraft"

    # Disable multi-model batch run
    xgb_feature_set_names: list[str] = field(default_factory=list)

    # Enable multi-model batch run
    # xgb_feature_set_names: list[str] = field(default_factory=lambda: [
    #     "xgb_schedule",
    #     "xgb_context",
    #     "xgb_2hop_propagation",
    #     "xgb_full",
    #     "xgb_full_aircraft",
    # ])

    # LSTM feature set
    lstm_variant_name: str = "context_full"


@dataclass
class XGBSearchConfig:
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
class AircraftConfig:
    enabled: bool = True
    cache_path: Path = None

    def __post_init__(self):
        if self.cache_path is None:
            project_root = Path(__file__).resolve().parents[1]
            self.cache_path = project_root / "data_pipeline" / "data" / "faa_registry" / "ReleasableAircraft.zip"


@dataclass
class ArtifactConfig:
    enabled: bool = True
    runs_dirname: str = "runs"
    save_predictions: bool = True
    save_feature_importance: bool = True
    version_prefix: str = "v"


@dataclass
class VisualizationConfig:
    enabled: bool = True

    make_summary_table: bool = True
    make_cv_table: bool = True
    make_metric_bar_charts: bool = True
    make_cv_metric_bar_charts: bool = True
    make_feature_importance_chart: bool = True
    make_roc_curve: bool = True
    make_actual_vs_predicted: bool = True

    top_n_features: int = 20
    scatter_sample_n: int = 5000
    figure_dpi: int = 150
    image_format: str = "png"

    make_side_by_side_metric_charts: bool = True
    make_overlay_metric_charts: bool = True


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    xgb_search: XGBSearchConfig = field(default_factory=XGBSearchConfig)
    lstm_search: LSTMSearchConfig = field(default_factory=LSTMSearchConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    visualizations: VisualizationConfig = field(default_factory=VisualizationConfig)
    aircraft: AircraftConfig = field(default_factory=AircraftConfig)


CONFIG = PipelineConfig()
CONFIG.data.output_dir.mkdir(parents=True, exist_ok=True)