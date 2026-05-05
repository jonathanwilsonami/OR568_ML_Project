from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    canonical_dir: Path = None
    file_pattern: str = "flights_canonical_{year}.parquet"
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

    # True = final model trains on 2015-2023 + 2024, evaluates on 2025
    use_validation_holdout: bool = True


@dataclass
class CVConfig:
    # ---------------------------------------------------------------
    # CV DISABLED for the final refit run.
    # Set enabled=True to restore rolling CV diagnostics.
    # ---------------------------------------------------------------
    enabled: bool = False
    strategy: str = "rolling_year"
    min_train_years: int = 4


@dataclass
class RuntimeConfig:
    random_seed: int = 42
    use_gpu: bool = False
    n_jobs: int = 8

    # Streaming architecture — no sampling needed.
    # Peak RAM = one year at a time (~1-2 GB).
    train_sample_fraction: float | None = None
    val_sample_fraction: float | None = None
    test_sample_fraction: float | None = None
    sample_fraction_for_tuning: float | None = None
    sample_fraction_for_lstm_tuning: float | None = 0.03
    sample_fraction_for_final_train: float | None = None


@dataclass
class ModelConfig:
    run_xgb: bool = True
    run_lstm: bool = False

    tune_xgb: bool = False
    tune_lstm: bool = False

    # Single feature set for the final refit run
    xgb_feature_set_name: str = "xgb_full_aircraft"

    # Empty list = use xgb_feature_set_name above (single model)
    xgb_feature_set_names: list[str] = field(default_factory=list)

    # Enable multi-model batch run
    # xgb_feature_set_names: list[str] = field(default_factory=lambda: [
    #     "xgb_schedule",
    #     "xgb_context",
    #     "xgb_2hop_propagation",
    #     "xgb_full",
    #     "xgb_full_aircraft",
    # ])

    lstm_variant_name: str = "context_full"


@dataclass
class XGBSearchConfig:
    # ---------------------------------------------------------------
    # BEST PARAMS from prior tuned run.
    # Only param_grid[0] is used since tune_xgb=False and cv=False.
    # ---------------------------------------------------------------
    param_grid: list[dict] = field(default_factory=lambda: [
        {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
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
    ])


@dataclass
class AircraftConfig:
    enabled: bool = True
    cache_path: Path = None

    def __post_init__(self):
        if self.cache_path is None:
            project_root = Path(__file__).resolve().parents[1]
            self.cache_path = (
                project_root / "data_pipeline" / "data"
                / "faa_registry" / "ReleasableAircraft.zip"
            )


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