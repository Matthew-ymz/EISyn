from __future__ import annotations

import torch


def _build_activation(name: str) -> torch.nn.Module:
    normalized = name.lower()
    if normalized == "relu":
        return torch.nn.ReLU()
    if normalized == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_norm(name: str, hidden_dim: int) -> torch.nn.Module:
    normalized = name.lower()
    if normalized == "layernorm":
        return torch.nn.LayerNorm(hidden_dim)
    if normalized == "rmsnorm":
        rmsnorm = getattr(torch.nn, "RMSNorm", None)
        if rmsnorm is None:
            return torch.nn.LayerNorm(hidden_dim)
        return rmsnorm(hidden_dim)
    raise ValueError(f"Unsupported norm type: {name}")


class ResidualMLPBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int, *, dropout: float, norm_type: str, activation: str) -> None:
        super().__init__()
        self.norm = _build_norm(norm_type, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation = _build_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class PersistenceBaseline(torch.nn.Module):
    def __init__(self, target_dim: int, horizons: tuple[int, ...]) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.horizons = tuple(horizons)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        if x.ndim == 4:
            latest_snapshot = x[:, -1]
        elif x.ndim == 3:
            latest_snapshot = x
        else:
            raise ValueError(f"PersistenceBaseline expects 3D or 4D input, got shape {tuple(x.shape)}.")
        batch_size, n_stations, n_features = latest_snapshot.shape
        if self.target_dim % n_stations == 0:
            target_features = self.target_dim // n_stations
            if target_features <= n_features:
                last = latest_snapshot[:, :, :target_features].reshape(batch_size, -1)
            else:
                last = latest_snapshot.reshape(batch_size, -1)[:, : self.target_dim]
        else:
            last = latest_snapshot.reshape(batch_size, -1)[:, : self.target_dim]
        return {horizon: last for horizon in self.horizons}


class SingleStationMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        history_hours: int,
        target_dim: int,
        hidden_dim: int,
        horizons: tuple[int, ...],
    ) -> None:
        super().__init__()
        input_dim = n_features * history_hours
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.heads = torch.nn.ModuleDict(
            {str(horizon): torch.nn.Linear(hidden_dim, target_dim) for horizon in horizons}
        )

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        hidden = self.trunk(x.reshape(x.shape[0], -1))
        return {int(horizon): head(hidden) for horizon, head in self.heads.items()}


class JointStationMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        n_stations: int,
        n_features: int,
        history_hours: int,
        target_dim: int,
        hidden_dim: int,
        horizons: tuple[int, ...],
        model_name: str = "baseline",
        num_layers: int = 2,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        input_dim = n_stations * n_features * history_hours
        self.model_name = model_name
        self.model_kwargs = {
            "n_stations": n_stations,
            "n_features": n_features,
            "history_hours": history_hours,
            "target_dim": target_dim,
            "hidden_dim": hidden_dim,
            "horizons": tuple(horizons),
            "model_name": model_name,
            "num_layers": num_layers,
            "dropout": dropout,
            "norm_type": norm_type,
            "activation": activation,
        }

        if model_name == "baseline":
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
            )
        elif model_name == "resmlp":
            layers: list[torch.nn.Module] = [
                torch.nn.Linear(input_dim, hidden_dim),
                _build_activation(activation),
            ]
            block_count = max(1, num_layers)
            for _ in range(block_count):
                layers.append(
                    ResidualMLPBlock(
                        hidden_dim,
                        dropout=dropout,
                        norm_type=norm_type,
                        activation=activation,
                    )
                )
            layers.append(_build_norm(norm_type, hidden_dim))
            self.trunk = torch.nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        self.heads = torch.nn.ModuleDict(
            {str(horizon): torch.nn.Linear(hidden_dim, target_dim) for horizon in horizons}
        )

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        hidden = self.trunk(x.reshape(x.shape[0], -1))
        return {int(horizon): head(hidden) for horizon, head in self.heads.items()}
