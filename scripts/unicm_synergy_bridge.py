"""Lightweight interaction-only residual adapter for frozen UniCM mode forecasts."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


MODE_NAMES = (
    "ENSO",
    "NPMM",
    "SPMM",
    "IOB",
    "IOD",
    "SIOD",
    "TNA",
    "nino12",
    "nino3",
    "nino4",
    "WWV",
)

# Keep the geographically related Niño modes together in all reported outputs.
DEFAULT_MODULE = (0, 7, 8, 9, 4)


class SynergyBridge(nn.Module):
    """Correct selected UniCM mode forecasts using pairwise history interactions.

    The adapter has no single-mode or linear-calibration path. Each feature is a
    Hadamard product between low-rank representations of two source histories.
    The final projection is zero-initialized, so attaching an untrained bridge is
    exactly equivalent to the frozen checkpoint.
    """

    def __init__(
        self,
        *,
        history_len: int = 12,
        prediction_len: int = 24,
        source_indices: Sequence[int] = DEFAULT_MODULE,
        target_indices: Sequence[int] = DEFAULT_MODULE,
        rank: int = 8,
        hidden_size: int = 64,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if len(source_indices) < 2:
            raise ValueError("SynergyBridge requires at least two source modes.")
        if len(set(source_indices)) != len(source_indices):
            raise ValueError("source_indices must be unique.")
        if len(set(target_indices)) != len(target_indices):
            raise ValueError("target_indices must be unique.")

        self.history_len = int(history_len)
        self.prediction_len = int(prediction_len)
        self.source_indices = tuple(int(i) for i in source_indices)
        self.target_indices = tuple(int(i) for i in target_indices)
        self.pairs = tuple(combinations(range(len(self.source_indices)), 2))

        self.temporal_projection = nn.Linear(self.history_len, int(rank), bias=False)
        interaction_size = len(self.pairs) * int(rank)
        self.interaction_norm = nn.LayerNorm(interaction_size)
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_size, int(hidden_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.output_projection = nn.Linear(
            int(hidden_size),
            len(self.target_indices) * self.prediction_len,
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def interaction_features(self, standardized_history: torch.Tensor) -> torch.Tensor:
        """Return explicit pairwise features from ``[batch, mode, history]`` input."""

        if standardized_history.ndim != 3:
            raise ValueError(
                "standardized_history must have shape [batch, mode, history]."
            )
        if standardized_history.shape[-1] != self.history_len:
            raise ValueError(
                f"Expected history length {self.history_len}, "
                f"got {standardized_history.shape[-1]}."
            )

        index = torch.as_tensor(
            self.source_indices,
            device=standardized_history.device,
            dtype=torch.long,
        )
        selected = standardized_history.index_select(1, index)
        embedded = self.temporal_projection(selected)
        pairwise = torch.stack(
            [embedded[:, left] * embedded[:, right] for left, right in self.pairs],
            dim=1,
        )
        return pairwise.flatten(start_dim=1)

    def correction(self, standardized_history: torch.Tensor) -> torch.Tensor:
        features = self.interaction_features(standardized_history)
        hidden = self.interaction_mlp(self.interaction_norm(features))
        return self.output_projection(hidden).reshape(
            standardized_history.shape[0],
            len(self.target_indices),
            self.prediction_len,
        )

    def forward(
        self,
        base_forecast: torch.Tensor,
        standardized_history: torch.Tensor,
    ) -> torch.Tensor:
        """Add the learned correction to selected targets only."""

        if base_forecast.ndim != 3:
            raise ValueError("base_forecast must have shape [batch, mode, lead].")
        if base_forecast.shape[-1] != self.prediction_len:
            raise ValueError(
                f"Expected {self.prediction_len} leads, got {base_forecast.shape[-1]}."
            )

        target_index = torch.as_tensor(
            self.target_indices,
            device=base_forecast.device,
            dtype=torch.long,
        )
        target_map = F.one_hot(
            target_index,
            num_classes=base_forecast.shape[1],
        ).to(dtype=base_forecast.dtype)
        full_correction = torch.matmul(
            self.correction(standardized_history).transpose(1, 2),
            target_map,
        ).transpose(1, 2)
        return base_forecast + full_correction
