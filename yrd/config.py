from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class YRDExperimentConfig:
    root_dir: Path = Path(".")
    dataset_path: Path = Path("data/dataset_yrd.nc")
    station_path: Path = Path("data/stations_yrd.csv")
    sample_mode: str = "windowed"
    history_hours: int = 24
    horizons: tuple[int, ...] = (1, 24)
    target_variables: tuple[str, str] = ("O3", "PM2.5")
    meteorology_variables: tuple[str, ...] = (
        "t2m",
        "d2m",
        "sp",
        "tp",
        "blh",
        "msdwswrf",
        "u100",
        "v100",
    )
    train_end: pd.Timestamp = pd.Timestamp("2021-12-31 23:00:00")
    val_end: pd.Timestamp = pd.Timestamp("2022-12-31 23:00:00")
    test_end: pd.Timestamp = pd.Timestamp("2023-12-31 23:00:00")
    model_name: str = "resmlp"
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    norm_type: str = "layernorm"
    activation: str = "silu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 3
    max_epochs: int = 100
    early_stopping_patience: int = 10
    seed: int = 0
    box_size: float = math.sqrt(12.0)
    causal_graph_box_size_by_variable: dict[str, float] | None = None
    causal_graph_nonnegative_variables: tuple[str, ...] = ()
    smoke_station_count: int = 4
    smoke_samples_per_split: int = 48

    @property
    def input_variables(self) -> tuple[str, ...]:
        return self.target_variables + self.meteorology_variables

    @property
    def cache_dir(self) -> Path:
        return self.root_dir / "exp" / "cache" / "yrd_coupling"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "fig" / "yrd_shanghai" / "artifacts"
