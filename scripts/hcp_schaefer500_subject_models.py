"""Shared and personalized Schaefer-500 delta-dynamics predictors.

All public helpers keep subjects as separate arrays.  The only allowed adaptation
for a held-out subject is fitting a low-dimensional adapter on its calibration
transitions; no function in this module concatenates separate subject timelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge


@dataclass(frozen=True)
class SubjectFold:
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]


def make_subject_folds(subjects: Sequence[str], *, n_splits: int, seed: int) -> tuple[SubjectFold, ...]:
    """Create deterministic subject-only folds, with every subject held out once."""
    ordered = tuple(sorted(str(subject) for subject in subjects))
    if len(ordered) < n_splits or len(ordered) % n_splits:
        raise ValueError("subjects must divide evenly into n_splits.")
    shuffled = np.asarray(ordered, dtype=object)
    np.random.default_rng(seed).shuffle(shuffled)
    folds = []
    for held_out in np.split(shuffled, n_splits):
        test_subjects = tuple(sorted(str(value) for value in held_out.tolist()))
        train_subjects = tuple(subject for subject in ordered if subject not in set(test_subjects))
        folds.append(SubjectFold(train_subjects=train_subjects, test_subjects=test_subjects))
    return tuple(folds)


@dataclass(frozen=True)
class SubjectNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, series: np.ndarray, *, calibration_end: int) -> "SubjectNormalizer":
        values = np.asarray(series, dtype=float)
        if values.ndim != 2 or not 1 < calibration_end <= len(values):
            raise ValueError("calibration_end must select a non-empty prefix of a [time, parcel] series.")
        calibration = values[:calibration_end]
        mean = calibration.mean(axis=0, keepdims=True)
        scale = calibration.std(axis=0, ddof=1, keepdims=True)
        return cls(mean=mean, scale=np.where(scale > 1.0e-12, scale, 1.0))

    def transform(self, series: np.ndarray) -> np.ndarray:
        return (np.asarray(series, dtype=float) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


def make_history_samples(
    series: np.ndarray, *, order: int, start: int = 0, end: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make x_t,...,x_{t-order+1} rows without crossing ``[start, end)``."""
    values = np.asarray(series, dtype=float)
    if values.ndim != 2 or order < 1:
        raise ValueError("series must be [time, parcel] and order must be positive.")
    final = len(values) if end is None else int(end)
    first_target = int(start) + int(order)
    if start < 0 or final > len(values) or final <= first_target:
        raise ValueError("interval is too short for requested history order.")
    target_indices = np.arange(first_target, final)
    history = np.concatenate(
        [values[target_indices - 1 - lag] for lag in range(order)], axis=1
    )
    return history, values[target_indices], target_indices


def recursive_rollout(
    history: np.ndarray, *, horizon: int, predict_delta: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Recursively forecast standardized states from newest-first history."""
    state_history = np.asarray(history, dtype=float).copy()
    if state_history.ndim != 2 or horizon < 1:
        raise ValueError("history must be [order, parcel] and horizon must be positive.")
    result = []
    for _ in range(int(horizon)):
        next_state = state_history[0] + np.asarray(predict_delta(state_history), dtype=float)
        result.append(next_state)
        state_history = np.vstack((next_state[None, :], state_history[:-1]))
    return np.asarray(result)


def fit_adapter_ridge(features: np.ndarray, basis: np.ndarray, truth: np.ndarray, *, ridge: float) -> np.ndarray:
    """Fit a subject adapter from calibration residuals only.

    ``basis[n, parcel, rank]`` maps the rank-vector adapter to a delta residual.
    ``features`` is accepted to make the calibration alignment explicit and is
    validated rather than used as an additional source of information.
    """
    design_features = np.asarray(features)
    values = np.asarray(basis, dtype=float)
    targets = np.asarray(truth, dtype=float)
    if values.ndim != 3 or targets.shape != values.shape[:2] or len(design_features) != len(values):
        raise ValueError("features, basis, and truth must share their calibration-row dimension.")
    design = values.reshape(-1, values.shape[-1])
    response = targets.reshape(-1)
    gram = design.T @ design + float(ridge) * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ response)


class DeltaRidge:
    """A single shared or individual multi-output Ridge delta predictor."""

    def __init__(self, *, order: int, alpha: float) -> None:
        self.order = int(order)
        self.alpha = float(alpha)
        self.model: Ridge | None = None

    def fit(self, sequences: Sequence[np.ndarray], *, calibration_end: int) -> "DeltaRidge":
        histories, deltas = [], []
        for sequence in sequences:
            history, next_state, _ = make_history_samples(sequence, order=self.order, end=calibration_end)
            histories.append(history)
            deltas.append(next_state - history[:, : next_state.shape[1]])
        self.model = Ridge(alpha=self.alpha, fit_intercept=True).fit(np.concatenate(histories), np.concatenate(deltas))
        return self

    def predict_delta(self, history: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DeltaRidge must be fit before prediction.")
        row = np.asarray(history, dtype=float).reshape(1, -1)
        return np.asarray(row @ self.model.coef_.T + self.model.intercept_, dtype=float).reshape(-1)


class HierarchicalLowRankDeltaVAR:
    """Shared Ridge dynamics plus a calibration-fitted low-rank subject adapter."""

    def __init__(self, *, order: int, alpha: float, rank: int, adapter_ridge: float) -> None:
        self.order = int(order)
        self.alpha = float(alpha)
        self.rank = int(rank)
        self.adapter_ridge = float(adapter_ridge)
        self.shared: DeltaRidge | None = None
        self.components: np.ndarray | None = None  # [rank, feature, parcel]

    def fit(self, sequences: Mapping[str, np.ndarray], *, calibration_end: int) -> "HierarchicalLowRankDeltaVAR":
        self.shared = DeltaRidge(order=self.order, alpha=self.alpha).fit(
            list(sequences.values()), calibration_end=calibration_end
        )
        if self.shared.model is None:
            raise RuntimeError("shared Ridge fit failed.")
        base = np.asarray(self.shared.model.coef_, dtype=float).T
        # The shared transition itself supplies U and V.  A held-out subject
        # only estimates the rank-wise scaling z_i from its calibration rows,
        # avoiding a second dense 500-by-500 fit for every subject.
        left, _, right = np.linalg.svd(base, full_matrices=False)
        effective_rank = min(self.rank, right.shape[0])
        self.components = np.einsum(
            "fr,ro->rfo", left[:, :effective_rank], right[:effective_rank], optimize=True
        )
        return self

    def _basis(self, features: np.ndarray) -> np.ndarray:
        if self.components is None:
            raise RuntimeError("HierarchicalLowRankDeltaVAR must be fit before adaptation.")
        return np.einsum("nf,rfo->nor", np.asarray(features, dtype=float), self.components, optimize=True)

    def fit_adapter(self, sequence: np.ndarray, *, calibration_end: int) -> np.ndarray:
        if self.shared is None:
            raise RuntimeError("fit the shared model before adapting a subject.")
        history, next_state, _ = make_history_samples(sequence, order=self.order, end=calibration_end)
        target = next_state - history[:, : next_state.shape[1]]
        shared = np.asarray(self.shared.model.predict(history), dtype=float)
        return fit_adapter_ridge(history, self._basis(history), target - shared, ridge=self.adapter_ridge)

    def predict_delta(self, history: np.ndarray, adapter: np.ndarray) -> np.ndarray:
        if self.shared is None or self.shared.model is None:
            raise RuntimeError("HierarchicalLowRankDeltaVAR must be fit before prediction.")
        row = np.asarray(history, dtype=float).reshape(1, -1)
        shared = np.asarray(row @ self.shared.model.coef_.T + self.shared.model.intercept_, dtype=float).reshape(-1)
        adjustment = np.einsum("opr,r->op", self._basis(row), np.asarray(adapter, dtype=float))[0]
        return shared + adjustment


class NeuralDeltaModel:
    """Shared MLP or factorized TCN with an optional FiLM-style subject adapter."""

    def __init__(
        self,
        *,
        kind: str,
        order: int,
        width: int,
        adapter_dim: int,
        learning_rate: float,
        epochs: int,
        seed: int,
        personalized: bool,
        device: str | None = None,
    ) -> None:
        if kind not in {"mlp", "tcn"}:
            raise ValueError("kind must be 'mlp' or 'tcn'.")
        self.kind, self.order, self.width = kind, int(order), int(width)
        self.adapter_dim, self.learning_rate, self.epochs = int(adapter_dim), float(learning_rate), int(epochs)
        self.seed, self.personalized, self.device_name = int(seed), bool(personalized), device
        self.net = None
        self.n_parcels: int | None = None

    def _device(self):
        import torch

        if self.device_name is not None:
            return torch.device(self.device_name)
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def _build_network(self, n_parcels: int):
        import torch

        adapter_dim = self.adapter_dim if self.personalized else 0

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_layer = torch.nn.Linear(n_parcels * self_order, self_width)
                self.hidden = torch.nn.Linear(self_width, self_width)
                self.out = torch.nn.Linear(self_width, n_parcels)
                self.skip = torch.nn.Linear(n_parcels * self_order, n_parcels)
                self.film = torch.nn.Linear(adapter_dim, 2 * self_width) if adapter_dim else None

            def forward(self, history, adapter):
                flat = history.reshape(len(history), -1)
                hidden = torch.nn.functional.silu(self.in_layer(flat))
                hidden = torch.nn.functional.silu(self.hidden(hidden))
                if self.film is not None:
                    gamma, beta = self.film(adapter).chunk(2, dim=-1)
                    hidden = hidden * (1.0 + gamma) + beta
                return self.skip(flat) + self.out(hidden)

        class TCN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.project = torch.nn.Linear(n_parcels, self_width)
                self.convs = torch.nn.ModuleList(
                    [torch.nn.Conv1d(self_width, self_width, kernel_size=3, dilation=d, padding=d) for d in (1, 2, 4)]
                )
                self.out = torch.nn.Linear(self_width, n_parcels)
                self.film = torch.nn.Linear(adapter_dim, 2 * self_width) if adapter_dim else None

            def forward(self, history, adapter):
                values = self.project(history.flip(dims=(1,))).transpose(1, 2)
                for conv in self.convs:
                    values = torch.nn.functional.silu(conv(values)[..., : values.shape[-1]] + values)
                hidden = values[..., -1]
                if self.film is not None:
                    gamma, beta = self.film(adapter).chunk(2, dim=-1)
                    hidden = hidden * (1.0 + gamma) + beta
                return self.out(hidden)

        self_order, self_width = self.order, self.width
        return MLP() if self.kind == "mlp" else TCN()

    def fit(self, sequences: Mapping[str, np.ndarray], *, calibration_end: int) -> "NeuralDeltaModel":
        import torch

        torch.manual_seed(self.seed)
        device = self._device()
        names = tuple(sorted(sequences))
        self.n_parcels = int(np.asarray(sequences[names[0]]).shape[1])
        self.net = self._build_network(self.n_parcels).to(device)
        adapters = torch.nn.Embedding(len(names), self.adapter_dim).to(device) if self.personalized else None
        parameters = list(self.net.parameters()) + ([] if adapters is None else list(adapters.parameters()))
        optimizer = torch.optim.AdamW(parameters, lr=self.learning_rate, weight_decay=1.0e-4)
        tensors = {name: torch.tensor(np.asarray(sequences[name][:calibration_end], dtype=np.float32).tolist(), device=device) for name in names}
        generator = torch.Generator(device=device).manual_seed(self.seed + 101)
        batch_size = 32
        for _ in range(self.epochs):
            subject_ids = torch.randint(len(names), (batch_size,), generator=generator, device=device)
            rows = torch.randint(self.order, calibration_end, (batch_size,), generator=generator, device=device)
            histories, target = [], []
            for subject_id, row in zip(subject_ids.tolist(), rows.tolist()):
                values = tensors[names[subject_id]]
                histories.append(torch.stack([values[row - 1 - lag] for lag in range(self.order)]))
                target.append(values[row] - values[row - 1])
            history = torch.stack(histories)
            delta = torch.stack(target)
            conditioning = adapters(subject_ids) if adapters is not None else torch.zeros((batch_size, 0), device=device)
            optimizer.zero_grad()
            loss = torch.mean((self.net(history, conditioning) - delta) ** 2)
            loss.backward()
            optimizer.step()
        self.net.eval()
        return self

    def fit_adapter(self, sequence: np.ndarray, *, calibration_end: int, steps: int = 20) -> np.ndarray:
        import torch

        if self.net is None or self.n_parcels is None:
            raise RuntimeError("fit the shared neural model before adaptation.")
        if not self.personalized:
            return np.zeros(0, dtype=float)
        device = self._device()
        values = torch.tensor(np.asarray(sequence[:calibration_end], dtype=np.float32).tolist(), device=device)
        adapter = torch.zeros((1, self.adapter_dim), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([adapter], lr=3.0e-2)
        for step in range(int(steps)):
            rows = torch.arange(self.order, calibration_end, device=device)[step % (calibration_end - self.order) :: 8]
            histories = torch.stack([torch.stack([values[int(row) - 1 - lag] for lag in range(self.order)]) for row in rows])
            truth = values[rows] - values[rows - 1]
            optimizer.zero_grad()
            loss = torch.mean((self.net(histories, adapter.expand(len(rows), -1)) - truth) ** 2)
            loss.backward()
            optimizer.step()
        return np.asarray(adapter.detach().cpu().tolist()[0], dtype=float)

    def predict_delta(self, history: np.ndarray, adapter: np.ndarray | None = None) -> np.ndarray:
        import torch

        if self.net is None:
            raise RuntimeError("NeuralDeltaModel must be fit before prediction.")
        device = self._device()
        values = torch.tensor(np.asarray(history, dtype=np.float32).tolist(), device=device).unsqueeze(0)
        condition = np.zeros(0, dtype=np.float32) if adapter is None else np.asarray(adapter, dtype=np.float32)
        condition_tensor = torch.tensor(condition.tolist(), device=device).reshape(1, -1)
        with torch.no_grad():
            result = self.net(values, condition_tensor)
        return np.asarray(result.detach().cpu().tolist()[0], dtype=float)
