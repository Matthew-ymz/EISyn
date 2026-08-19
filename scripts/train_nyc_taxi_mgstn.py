#!/usr/bin/env python3
"""Full MGSTN reproduction for the paper-compatible NYC Taxi task."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "nyc_taxi_mgstn_2023" / "nyc_taxi_mgstn_hourly.npz"
RUN_DIR = ROOT / "results" / "nyc_taxi_mgstn"
LOG_DIR = ROOT / "docs" / "log" / "nyc_taxi_mgstn"
PROGRESS = LOG_DIR / "live_progress.json"
PAPER_TARGETS = {
    "inflow_mae": 7.921,
    "inflow_rmse": 13.181,
    "outflow_mae": 8.215,
    "outflow_rmse": 15.407,
}


@dataclass(frozen=True)
class Config:
    hidden_dim: int = 64
    transformer_layers: int = 2
    graph_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    max_epochs: int = 100
    patience: int = 10


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_adjacency(adjacency: np.ndarray) -> torch.Tensor:
    adjacency = adjacency.astype(np.float32)
    degree = adjacency.sum(axis=1)
    inv = np.zeros_like(degree)
    inv[degree > 0] = degree[degree > 0] ** -0.5
    return torch.from_numpy(inv[:, None] * adjacency * inv[None, :])


class TaxiWindows(Dataset):
    def __init__(
        self,
        flow_z: np.ndarray,
        attributes_z: np.ndarray,
        targets: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.flow = torch.from_numpy(flow_z.astype(np.float32))
        self.attributes = torch.from_numpy(attributes_z.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        t = int(self.indices[item])
        recent_idx = np.arange(t - 7, t)
        daily_idx = t - 24 * np.arange(5, 0, -1)
        weekly_idx = t - 168 * np.arange(7, 0, -1)
        return {
            "recent": self.flow[recent_idx].permute(1, 2, 0),
            "daily": self.flow[daily_idx].permute(1, 2, 0),
            "weekly": self.flow[weekly_idx].permute(1, 2, 0),
            "recent_attr": self.attributes[recent_idx],
            "daily_attr": self.attributes[daily_idx],
            "weekly_attr": self.attributes[weekly_idx],
            "target_attr": self.attributes[t],
            "target": self.targets[t],
        }


class ResidualGCN(nn.Module):
    def __init__(self, hidden: int, layers: int, adjacency: torch.Tensor, dropout: float) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, norm in zip(self.layers, self.norms):
            message = torch.einsum("ij,bjd->bid", self.adjacency, x)
            x = norm(x + self.dropout(torch.nn.functional.gelu(linear(message))))
        return x


class TypedRGCN(nn.Module):
    def __init__(
        self,
        hidden: int,
        layers: int,
        semantic_adjacency: np.ndarray,
        node_types: np.ndarray,
        dropout: float,
    ) -> None:
        super().__init__()
        n_relations = int(node_types.max()) + 1
        relation_adjacencies = []
        for relation in range(n_relations):
            mask = (node_types[None, :] == relation).astype(np.float32)
            relation_adjacencies.append(normalize_adjacency(semantic_adjacency * mask).numpy())
        self.register_buffer("relations", torch.from_numpy(np.stack(relation_adjacencies)))
        self.weights = nn.ParameterList(
            nn.Parameter(torch.empty(n_relations, hidden, hidden)) for _ in range(layers)
        )
        self.self_linears = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for weight, self_linear, norm in zip(self.weights, self.self_linears, self.norms):
            aggregated = torch.einsum("rij,bjd->brid", self.relations, x)
            message = torch.einsum("brid,rdh->bih", aggregated, weight)
            x = norm(x + self.dropout(torch.nn.functional.gelu(message + self_linear(x))))
        return x


class DeStationaryEncoderLayer(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.output = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        score = torch.einsum("blhd,bshd->bhls", q, k) / math.sqrt(self.head_dim)
        score = score * tau[:, None, None, None] + delta[:, None, None, :]
        attention = torch.softmax(score, dim=-1)
        attended = torch.einsum("bhls,bshd->blhd", attention, v).reshape(batch, length, hidden)
        x = self.norm1(x + self.dropout(self.output(attended)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class NonStationaryTemporal(nn.Module):
    def __init__(self, length: int, hidden: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, hidden)
        self.position = nn.Parameter(torch.empty(1, length, hidden))
        nn.init.normal_(self.position, std=0.02)
        self.tau = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, 1), nn.Softplus())
        self.delta = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, length))
        self.layers = nn.ModuleList(
            DeStationaryEncoderLayer(hidden, heads, dropout) for _ in range(layers)
        )
        self.statistics = nn.Linear(4, hidden)
        self.output = nn.Linear(2 * hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,F,L]; NST is shared across nodes.
        batch, nodes, features, length = x.shape
        series = x.permute(0, 1, 3, 2).reshape(batch * nodes, length, features)
        mean = series.mean(dim=1)
        std = series.std(dim=1, unbiased=False).clamp_min(1e-5)
        stats = torch.cat([mean, std], dim=-1)
        stationary = (series - mean[:, None, :]) / std[:, None, :]
        encoded = self.input_projection(stationary) + self.position
        tau = self.tau(stats).squeeze(-1) + 1e-3
        delta = self.delta(stats)
        for layer in self.layers:
            encoded = layer(encoded, tau, delta)
        pooled = encoded[:, -1] + encoded.mean(dim=1)
        representation = self.output(torch.cat([pooled, self.statistics(stats)], dim=-1))
        return representation.reshape(batch, nodes, -1)


class STNBranch(nn.Module):
    def __init__(
        self,
        length: int,
        attribute_dim: int,
        config: Config,
        distance_adjacency: torch.Tensor,
        semantic_adjacency: np.ndarray,
        node_types: np.ndarray,
        n_nodes: int,
    ) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.flow_projection = nn.Linear(2 * length, hidden)
        self.node_embedding = nn.Embedding(n_nodes, hidden)
        self.type_embedding = nn.Embedding(int(node_types.max()) + 1, hidden)
        self.register_buffer("node_index", torch.arange(n_nodes))
        self.register_buffer("node_types", torch.from_numpy(node_types.astype(np.int64)))
        self.gcn = ResidualGCN(hidden, config.graph_layers, distance_adjacency, config.dropout)
        self.rgcn = TypedRGCN(hidden, config.graph_layers, semantic_adjacency, node_types, config.dropout)
        self.spatial_fusion = nn.Linear(2 * hidden, hidden)
        self.temporal = NonStationaryTemporal(
            length, hidden, config.transformer_layers, config.attention_heads, config.dropout
        )
        self.attribute = nn.Sequential(nn.Linear(attribute_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * hidden, hidden),
        )

    def forward(self, flow: torch.Tensor, attributes: torch.Tensor) -> torch.Tensor:
        batch, nodes, _, _ = flow.shape
        projected = self.flow_projection(flow.reshape(batch, nodes, -1))
        distance_repr = self.gcn(projected)
        semantic_base = self.node_embedding(self.node_index) + self.type_embedding(self.node_types)
        semantic_repr = self.rgcn(semantic_base.unsqueeze(0).expand(batch, -1, -1))
        spatial = self.spatial_fusion(torch.cat([distance_repr, semantic_repr], dim=-1))
        temporal = self.temporal(flow)
        attribute = self.attribute(attributes.mean(dim=1)).unsqueeze(1).expand(-1, nodes, -1)
        return self.output(torch.cat([spatial, temporal, attribute], dim=-1))


class MGSTN(nn.Module):
    def __init__(self, data: dict, config: Config, attribute_dim: int) -> None:
        super().__init__()
        distance = normalize_adjacency(data["distance_adjacency"])
        common = dict(
            attribute_dim=attribute_dim,
            config=config,
            distance_adjacency=distance,
            semantic_adjacency=data["semantic_adjacency"],
            node_types=data["semantic_types"],
            n_nodes=len(data["zone_ids"]),
        )
        self.recent = STNBranch(length=7, **common)
        self.daily = STNBranch(length=5, **common)
        self.weekly = STNBranch(length=7, **common)
        hidden = config.hidden_dim
        self.granularity_fusion = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(2 * hidden, hidden)
        )
        self.target_attribute = nn.Sequential(nn.Linear(attribute_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        recent = self.recent(batch["recent"], batch["recent_attr"])
        daily = self.daily(batch["daily"], batch["daily_attr"])
        weekly = self.weekly(batch["weekly"], batch["weekly_attr"])
        fused = self.granularity_fusion(torch.cat([recent, daily, weekly], dim=-1))
        target_attr = self.target_attribute(batch["target_attr"]).unsqueeze(1).expand_as(fused)
        return self.output(torch.cat([fused, target_attr], dim=-1))


def load_data(smoke: bool) -> tuple[dict, dict, dict]:
    saved = np.load(DATA_PATH, allow_pickle=False)
    data = {key: saved[key] for key in saved.files if key != "metadata"}
    data["metadata"] = json.loads(str(saved["metadata"]))
    flow = data["flow"].astype(np.float32)
    attributes = np.concatenate([data["weather"], data["date_features"]], axis=1).astype(np.float32)
    all_indices = np.arange(7 * 168, len(flow))
    if smoke:
        all_indices = all_indices[: min(1024, len(all_indices))]
    train_n = int(0.70 * len(all_indices))
    valid_n = int(0.20 * len(all_indices))
    splits = {
        "train": all_indices[:train_n],
        "valid": all_indices[train_n : train_n + valid_n],
        "test": all_indices[train_n + valid_n :],
    }
    train_times = splits["train"]
    flow_mean = flow[train_times].mean(axis=0)
    flow_std = flow[train_times].std(axis=0)
    flow_std[flow_std < 1e-4] = 1.0
    flow_z = (flow - flow_mean) / flow_std
    attribute_mean = attributes[train_times].mean(axis=0)
    attribute_std = attributes[train_times].std(axis=0)
    attribute_std[attribute_std < 1e-4] = 1.0
    attributes_z = (attributes - attribute_mean) / attribute_std
    normalization = {
        "flow_mean": flow_mean,
        "flow_std": flow_std,
        "attribute_mean": attribute_mean,
        "attribute_std": attribute_std,
    }
    prepared = {"flow_z": flow_z, "attributes_z": attributes_z, "targets": flow_z}
    return data, prepared, {"indices": splits, "normalization": normalization}


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    flow_mean: np.ndarray,
    flow_std: np.ndarray,
) -> dict:
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        batch = move(batch, device)
        predictions.append(model(batch).cpu().numpy())
        targets.append(batch["target"].cpu().numpy())
    prediction_z = np.concatenate(predictions)
    target_z = np.concatenate(targets)
    prediction = np.maximum(prediction_z * flow_std + flow_mean, 0.0)
    target = target_z * flow_std + flow_mean
    metrics = {}
    for channel, name in enumerate(["inflow", "outflow"]):
        error = prediction[:, :, channel] - target[:, :, channel]
        metrics[f"{name}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
    metrics["normalized_mse"] = float(np.mean((prediction_z - target_z) ** 2))
    return metrics


def train_seed(seed: int, config: Config, device: torch.device, smoke: bool, run_position: int, total_runs: int) -> dict:
    seed_everything(seed)
    data, prepared, split_info = load_data(smoke)
    datasets = {
        name: TaxiWindows(prepared["flow_z"], prepared["attributes_z"], prepared["targets"], indices)
        for name, indices in split_info["indices"].items()
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=config.batch_size, shuffle=True, num_workers=0),
        "valid": DataLoader(datasets["valid"], batch_size=config.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(datasets["test"], batch_size=config.batch_size, shuffle=False, num_workers=0),
    }
    model = MGSTN(data, config, prepared["attributes_z"].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    best_loss = math.inf
    best_epoch = 0
    patience_left = config.patience
    history = []
    run_name = (
        f"{'smoke_' if smoke else ''}h{config.hidden_dim}_tl{config.transformer_layers}_"
        f"gl{config.graph_layers}_seed_{seed}"
    )
    checkpoint = RUN_DIR / f"{run_name}.pt"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = criterion(prediction, batch["target"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(
            model,
            loaders["valid"],
            device,
            split_info["normalization"]["flow_mean"],
            split_info["normalization"]["flow_std"],
        )
        val_loss = validation["normalized_mse"]
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), **validation})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            patience_left = config.patience
            torch.save({"model": model.state_dict(), "config": asdict(config), "seed": seed}, checkpoint)
        else:
            patience_left -= 1
        atomic_json(
            PROGRESS,
            {
                "phase": "train",
                "current": (run_position - 1) * config.max_epochs + epoch,
                "total": total_runs * config.max_epochs,
                "unit": "epoch_budget",
                "elapsed_seconds": time.monotonic() - started,
                "metrics": {
                    "seed": seed,
                    "epoch": epoch,
                    "train_mse": float(np.mean(losses)),
                    "val_mse": val_loss,
                    "best_epoch": best_epoch,
                },
                "updated_at": time.time(),
            },
        )
        print(
            f"seed={seed} epoch={epoch:03d} train={np.mean(losses):.5f} "
            f"val={val_loss:.5f} best={best_loss:.5f}",
            flush=True,
        )
        if patience_left <= 0:
            break

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    test = evaluate(
        model,
        loaders["test"],
        device,
        split_info["normalization"]["flow_mean"],
        split_info["normalization"]["flow_std"],
    )
    result = {
        "run_name": run_name,
        "seed": seed,
        "status": "completed",
        "config": asdict(config),
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_loss,
        "test": test,
        "history": history,
        "data_metadata": data["metadata"],
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_json(RUN_DIR / f"{run_name}.json", result)
    return result


def summarize(results: list[dict], config: Config, smoke: bool) -> dict:
    metric_names = ["inflow_mae", "inflow_rmse", "outflow_mae", "outflow_rmse"]
    metrics = {}
    for name in metric_names:
        values = np.asarray([result["test"][name] for result in results])
        metrics[name] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1 if len(values) > 1 else 0)),
            "paper": PAPER_TARGETS[name],
            "relative_difference": float(values.mean() / PAPER_TARGETS[name] - 1.0),
            "within_5_percent": bool(abs(values.mean() / PAPER_TARGETS[name] - 1.0) <= 0.05),
        }
    summary = {
        "status": "completed",
        "smoke": smoke,
        "config": asdict(config),
        "runs": results,
        "summary": metrics,
        "acceptance_all_within_5_percent": all(value["within_5_percent"] for value in metrics.values()),
    }
    atomic_json(RUN_DIR / ("smoke_summary.json" if smoke else "summary.json"), summary)
    return summary


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"run scripts/prepare_nyc_taxi_mgstn.py first: {DATA_PATH}")
    config = Config(
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        graph_layers=args.graph_layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    device = choose_device(args.device)
    results = []
    for position, seed in enumerate(args.seeds, start=1):
        run_name = (
            f"{'smoke_' if args.smoke else ''}h{config.hidden_dim}_tl{config.transformer_layers}_"
            f"gl{config.graph_layers}_seed_{seed}"
        )
        result_path = RUN_DIR / f"{run_name}.json"
        if args.resume and result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("config") == asdict(config) and existing.get("status") == "completed":
                print(f"resume: reuse {result_path}", flush=True)
                results.append(existing)
                continue
        results.append(train_seed(seed, config, device, args.smoke, position, len(args.seeds)))
    summary = summarize(results, config, args.smoke)
    atomic_json(
        PROGRESS,
        {
            "phase": "complete",
            "current": len(args.seeds),
            "total": len(args.seeds),
            "unit": "seed",
            "metrics": {key: value["mean"] for key, value in summary["summary"].items()},
            "updated_at": time.time(),
        },
    )
    print(json.dumps(summary["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
