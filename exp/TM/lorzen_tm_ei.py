from __future__ import annotations

import argparse
import json
import os
import warnings
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "tmp" / "matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from yrd import clip_nonnegative_ei, estimate_mutual_information_transport_map, lift_transport_source_features


DEFAULT_SOURCE_NAME = "yt.csv"
DEFAULT_TARGET_NAME = "yt+1.csv"


def simulate_lorenz96_time_series(
    *,
    node_count: int = 8,
    steps: int = 1200,
    dt: float = 0.01,
    forcing: float = 8.0,
    seed: int = 0,
    initial_noise_std: float = 0.01,
) -> np.ndarray:
    """Simulate a Lorenz-96 trajectory with RK4 integration."""

    if node_count < 4:
        raise ValueError("node_count must be at least 4.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    rng = np.random.default_rng(seed)
    state = np.full(int(node_count), float(forcing), dtype=float)
    state += float(initial_noise_std) * rng.normal(size=int(node_count))
    trajectory = np.empty((int(steps) + 1, int(node_count)), dtype=float)
    trajectory[0] = state
    for step in range(int(steps)):
        state = _rk4_lorenz96_step(state, dt=float(dt), forcing=float(forcing))
        trajectory[step + 1] = state
    return trajectory


def run_lorzen_mlp_tm_ei_experiment(
    *,
    output_dir: str | Path,
    node_count: int = 8,
    steps: int = 1400,
    burn_in: int = 200,
    dt: float = 0.01,
    forcing: float = 8.0,
    lag: int = 3,
    train_sample_count: int | None = 1000,
    intervention_sample_count: int = 2000,
    box_width: float = 1.5,
    top_k: int = 4,
    seed: int = 0,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    max_iter: int = 1000,
) -> dict[str, Any]:
    """Simulate Lorenz-96, fit an MLP transition model, then estimate TM-EI."""

    trajectory = simulate_lorenz96_time_series(
        node_count=node_count,
        steps=steps,
        dt=dt,
        forcing=forcing,
        seed=seed,
    )
    if burn_in < 0 or burn_in >= trajectory.shape[0] - lag:
        raise ValueError("burn_in must leave at least one supervised sample.")
    if lag < 1:
        raise ValueError("lag must be positive.")
    if box_width <= 0.0:
        raise ValueError("box_width must be positive.")
    if intervention_sample_count < 4:
        raise ValueError("intervention_sample_count must be at least 4.")

    labels = [f"M{i}" for i in range(1, int(node_count) + 1)]
    source_values = trajectory[burn_in:-lag]
    target_values = trajectory[burn_in + lag :]
    if train_sample_count is not None:
        source_values = source_values[: int(train_sample_count)]
        target_values = target_values[: int(train_sample_count)]
    if source_values.shape[0] < 8:
        raise ValueError("not enough supervised samples for MLP training.")

    x_train, x_test, y_train, y_test = train_test_split(
        source_values,
        target_values,
        test_size=0.25,
        random_state=seed,
        shuffle=True,
    )
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=int(max_iter),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=seed,
    )
    model = make_pipeline(StandardScaler(), mlp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    rng = np.random.default_rng(seed + 70000)
    center = source_values.mean(axis=0)
    half_width = float(box_width) / 2.0
    intervention_source = rng.uniform(
        low=center - half_width,
        high=center + half_width,
        size=(int(intervention_sample_count), int(node_count)),
    )
    intervention_target = np.asarray(model.predict(intervention_source), dtype=float)
    source_df = pd.DataFrame(intervention_source, columns=labels)
    target_df = pd.DataFrame(intervention_target, columns=labels)
    ei_matrix, edge_table = compute_pairwise_tm_ei_matrix(source_df, target_df)
    groundtruth = lorenz96_groundtruth_adjacency(int(node_count))
    predicted_graph = threshold_top_k_per_target(ei_matrix, top_k=top_k)
    metrics = evaluate_against_groundtruth(ei_matrix, predicted_graph, groundtruth, top_k=top_k)
    metrics.update(
        {
            "estimator_mode": "mlp_intervention",
            "target_mode": "next",
            "lag": int(lag),
            "node_count": int(node_count),
            "steps": int(steps),
            "burn_in": int(burn_in),
            "dt": float(dt),
            "forcing": float(forcing),
            "train_sample_count": int(source_values.shape[0]),
            "intervention_sample_count": int(intervention_sample_count),
            "box_width": float(box_width),
            "seed": int(seed),
            "train_r2": float(r2_score(y_train, train_prediction)),
            "test_r2": float(r2_score(y_test, test_prediction)),
            "mlp_iterations": int(mlp.n_iter_),
        }
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matrix_path = output_path / "lorzen_tm_ei_matrix.csv"
    edge_path = output_path / "lorzen_tm_ei_edges.csv"
    graph_path = output_path / "lorzen_tm_ei_topk_graph.csv"
    truth_path = output_path / "lorzen_lorenz96_groundtruth.csv"
    summary_path = output_path / "lorzen_tm_ei_summary.json"
    heatmap_path = output_path / "lorzen_tm_ei_heatmap.png"
    history_path = output_path / "lorzen_mlp_training_history.csv"
    causal_graph_path = output_path / "lorzen_ei_causal_graph.png"

    ei_matrix.to_csv(matrix_path)
    edge_table.to_csv(edge_path, index=False)
    predicted_graph.to_csv(graph_path)
    groundtruth.to_csv(truth_path)
    pd.DataFrame({"iteration": np.arange(1, len(mlp.loss_curve_) + 1), "loss": mlp.loss_curve_}).to_csv(history_path, index=False)
    plot_lorzen_tm_ei_heatmap(ei_matrix, predicted_graph, groundtruth, output_path=heatmap_path, metrics=metrics)
    reference_metrics = plot_reference_style_ei_causal_graph(
        ei_matrix,
        groundtruth,
        output_path=causal_graph_path,
        top_k=top_k,
    )
    metrics["reference_style_accuracy"] = float(reference_metrics["accuracy"])
    metrics["reference_style_f1"] = float(reference_metrics["f1"])
    metrics["reference_style_auc"] = float(reference_metrics["auc"])

    summary = {
        "metrics": metrics,
        "artifacts": {
            "ei_matrix": str(matrix_path),
            "edge_table": str(edge_path),
            "topk_graph": str(graph_path),
            "groundtruth": str(truth_path),
            "heatmap": str(heatmap_path),
            "causal_graph": str(causal_graph_path),
            "training_history": str(history_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "trajectory": trajectory,
        "source": source_df,
        "target": target_df,
        "model": model,
        "ei_matrix": ei_matrix,
        "edge_table": edge_table,
        "predicted_graph": predicted_graph,
        "groundtruth": groundtruth,
        "metrics": metrics,
        "summary_path": summary_path,
        "heatmap_path": heatmap_path,
        "causal_graph_path": causal_graph_path,
        "training_history_path": history_path,
    }


def load_lorzen_transition_csvs(input_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load headerless Lorenz transition samples from a directory or zip archive."""

    path = Path(input_path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            source_member = _find_archive_member(archive, DEFAULT_SOURCE_NAME)
            target_member = _find_archive_member(archive, DEFAULT_TARGET_NAME)
            with archive.open(source_member) as source_fp:
                source = pd.read_csv(source_fp, header=None)
            with archive.open(target_member) as target_fp:
                target = pd.read_csv(target_fp, header=None)
    else:
        source = pd.read_csv(path / DEFAULT_SOURCE_NAME, header=None)
        target = pd.read_csv(path / DEFAULT_TARGET_NAME, header=None)

    if source.shape != target.shape:
        raise ValueError(f"yt and yt+1 must have matching shapes, got {source.shape} and {target.shape}.")
    if source.shape[1] < 2:
        raise ValueError("Lorenz transition data must contain at least two variables.")

    labels = [f"M{i}" for i in range(1, source.shape[1] + 1)]
    source.columns = labels
    target.columns = labels
    return source.astype(float), target.astype(float)


def prepare_lagged_transition_samples(
    one_step_source: pd.DataFrame,
    one_step_target: pd.DataFrame,
    *,
    lag: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct a trajectory and return ``M(t)`` and ``M(t+lag)`` samples."""

    if lag < 1:
        raise ValueError("lag must be positive.")
    if one_step_source.shape != one_step_target.shape:
        raise ValueError("one-step source and target must have matching shapes.")
    if lag > len(one_step_source):
        raise ValueError("lag cannot exceed the reconstructed trajectory length minus one.")

    trajectory = pd.concat(
        [one_step_source, one_step_target.iloc[[-1]]],
        axis=0,
        ignore_index=True,
    )
    source = trajectory.iloc[:-lag].reset_index(drop=True)
    target = trajectory.iloc[lag:].reset_index(drop=True)
    return source, target


def lorenz96_groundtruth_adjacency(node_count: int) -> pd.DataFrame:
    """Return source-to-target Lorenz-96 adjacency with rows as targets."""

    if node_count < 4:
        raise ValueError("Lorenz-96 groundtruth requires at least four nodes.")
    adjacency = np.zeros((node_count, node_count), dtype=int)
    for target_index in range(node_count):
        for source_index in (
            target_index,
            (target_index - 1) % node_count,
            (target_index - 2) % node_count,
            (target_index + 1) % node_count,
        ):
            adjacency[target_index, source_index] = 1
    labels = [f"M{i}" for i in range(1, node_count + 1)]
    return pd.DataFrame(adjacency, index=labels, columns=labels)


def compute_pairwise_tm_ei_matrix(source: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate pairwise source-to-target EI with the transport-map backend."""

    labels = list(source.columns)
    rows: list[dict[str, Any]] = []
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    for target_label in labels:
        target_values = target[[target_label]].to_numpy(dtype=float)
        for source_label in labels:
            source_values = source[[source_label]].to_numpy(dtype=float)
            lifted_source = lift_transport_source_features(source_values)
            summary = estimate_mutual_information_transport_map(lifted_source, target_values)
            ei = clip_nonnegative_ei(float(summary["mi_hat"]))
            matrix.loc[target_label, source_label] = ei
            rows.append(
                {
                    "target": target_label,
                    "source": source_label,
                    "tm_ei": ei,
                    "raw_mi_hat": float(summary["mi_hat"]),
                    "bias_correction": float(summary["bias_correction"]),
                    "backend": str(summary["backend"]),
                }
            )
    return matrix, pd.DataFrame(rows)


def compute_surrogate_intervention_tm_ei_matrix(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    box_width: float,
    sample_count: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a quadratic transition surrogate and estimate EI under uniform interventions."""

    if box_width <= 0.0:
        raise ValueError("box_width must be positive.")
    if sample_count < 4:
        raise ValueError("sample_count must be at least 4.")

    labels = list(source.columns)
    source_values = source.to_numpy(dtype=float)
    target_values = target.to_numpy(dtype=float)
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=1e-6),
    )
    model.fit(source_values, target_values)

    rng = np.random.default_rng(seed)
    center = source_values.mean(axis=0)
    half_width = float(box_width) / 2.0
    intervention_source = rng.uniform(
        low=center - half_width,
        high=center + half_width,
        size=(int(sample_count), source_values.shape[1]),
    )
    intervention_target = np.asarray(model.predict(intervention_source), dtype=float)
    intervention_source_df = pd.DataFrame(intervention_source, columns=labels)
    intervention_target_df = pd.DataFrame(intervention_target, columns=labels)
    return compute_pairwise_tm_ei_matrix(intervention_source_df, intervention_target_df)


def threshold_top_k_per_target(ei_matrix: pd.DataFrame, *, top_k: int = 4) -> pd.DataFrame:
    """Convert EI scores into a directed graph by keeping top-k sources per target row."""

    if top_k < 1:
        raise ValueError("top_k must be positive.")
    graph = pd.DataFrame(0, index=ei_matrix.index, columns=ei_matrix.columns, dtype=int)
    for target_label, row in ei_matrix.iterrows():
        selected = row.sort_values(ascending=False).head(min(top_k, row.shape[0])).index
        graph.loc[target_label, selected] = 1
    return graph


def build_reference_style_causal_strength(
    ei_matrix: pd.DataFrame,
    *,
    top_k: int = 4,
    floor: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a nonnegative raw-EI heatmap with non-top-k cells suppressed."""

    if top_k < 1:
        raise ValueError("top_k must be positive.")
    values = ei_matrix.astype(float)
    graph = threshold_top_k_per_target(values, top_k=top_k)
    strength = values.clip(lower=0.0).where(graph.astype(bool), float(floor))
    return strength, graph


def plot_reference_style_ei_causal_graph(
    ei_matrix: pd.DataFrame,
    groundtruth: pd.DataFrame,
    *,
    output_path: Path,
    top_k: int = 4,
    floor: float = 0.0,
    vmax: float | None = None,
) -> dict[str, float | int]:
    """Render a single-panel EI causal graph close to the provided reference style."""

    strength, graph = build_reference_style_causal_strength(ei_matrix, top_k=top_k, floor=floor)
    metrics = evaluate_against_groundtruth(ei_matrix, graph, groundtruth, top_k=top_k)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finite_strength = strength.to_numpy(dtype=float)
    if vmax is None:
        vmax = float(np.nanmax(finite_strength)) if finite_strength.size else 1.0
    vmax = max(float(vmax), float(floor) + 1e-12)

    fig, ax = plt.subplots(figsize=(5.8, 5.4), constrained_layout=True)
    image = ax.imshow(strength.to_numpy(dtype=float), cmap="YlOrRd", vmin=float(floor), vmax=float(vmax), aspect="equal")
    ax.set_title(
        "EI Causal Graph\n"
        f"thr=row top-{int(top_k)}  Acc={metrics['accuracy']:.3f}\n"
        f"F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Source macro variable", fontsize=10)
    ax.set_ylabel("Target macro variable", fontsize=10)
    ax.set_xticks(np.arange(strength.shape[1]), labels=strength.columns)
    ax.set_yticks(np.arange(strength.shape[0]), labels=strength.index)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xticks(np.arange(-0.5, strength.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, strength.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.92)
    colorbar.set_label("EI causal strength", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return metrics


def evaluate_against_groundtruth(
    ei_matrix: pd.DataFrame,
    predicted_graph: pd.DataFrame,
    groundtruth: pd.DataFrame,
    *,
    top_k: int,
) -> dict[str, float | int]:
    truth = groundtruth.to_numpy(dtype=int).ravel()
    prediction = predicted_graph.to_numpy(dtype=int).ravel()
    scores = ei_matrix.to_numpy(dtype=float).ravel()
    metrics: dict[str, float | int] = {
        "top_k": int(top_k),
        "accuracy": float(accuracy_score(truth, prediction)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(truth, scores))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def plot_lorzen_tm_ei_heatmap(
    ei_matrix: pd.DataFrame,
    predicted_graph: pd.DataFrame,
    groundtruth: pd.DataFrame,
    *,
    output_path: Path,
    metrics: dict[str, float | int],
) -> Path:
    """Save the TM-EI heatmap and graph comparison figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2), constrained_layout=True)
    panels = [
        (ei_matrix, "TM effective information", "viridis", "nats"),
        (predicted_graph, f"TM graph, top-{metrics['top_k']} per target", "Blues", "edge"),
        (groundtruth, "Groundtruth Lorenz-96", "Blues", "edge"),
    ]
    for ax, (frame, title, cmap, colorbar_label) in zip(axes, panels):
        image = ax.imshow(frame.to_numpy(dtype=float), cmap=cmap, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("Source macro variable")
        ax.set_ylabel("Target macro variable")
        ax.set_xticks(np.arange(frame.shape[1]), labels=frame.columns)
        ax.set_yticks(np.arange(frame.shape[0]), labels=frame.index)
        for row_index in range(frame.shape[0]):
            for col_index in range(frame.shape[1]):
                value = frame.iloc[row_index, col_index]
                label = f"{value:.2f}" if frame is ei_matrix else str(int(value))
                ax.text(col_index, row_index, label, ha="center", va="center", fontsize=7, color=_cell_text_color(float(value), frame))
        fig.colorbar(image, ax=ax, shrink=0.82, label=colorbar_label)
    fig.suptitle(
        "Lorzen transport-map EI reproduction"
        f"  Acc={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}  AUC={metrics['auc']:.3f}",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_lorzen_tm_ei_experiment(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    top_k: int = 4,
    target_mode: str = "next",
    estimator_mode: str = "observed",
    box_width: float = 1.0,
    sample_count: int | None = None,
    seed: int = 0,
    lag: int = 1,
) -> dict[str, Any]:
    one_step_source, one_step_target = load_lorzen_transition_csvs(input_path)
    source, target = prepare_lagged_transition_samples(one_step_source, one_step_target, lag=lag)
    target_for_ei = _resolve_target_mode(source, target, target_mode=target_mode)
    source_for_ei, target_for_ei = _select_sample_rows(source, target_for_ei, sample_count=sample_count)
    if estimator_mode == "observed":
        ei_matrix, edge_table = compute_pairwise_tm_ei_matrix(source_for_ei, target_for_ei)
    elif estimator_mode == "surrogate_intervention":
        resolved_sample_count = int(sample_count or len(source))
        ei_matrix, edge_table = compute_surrogate_intervention_tm_ei_matrix(
            source_for_ei,
            target_for_ei,
            box_width=box_width,
            sample_count=resolved_sample_count,
            seed=seed,
        )
    else:
        raise ValueError("estimator_mode must be 'observed' or 'surrogate_intervention'.")
    groundtruth = lorenz96_groundtruth_adjacency(one_step_source.shape[1])
    predicted_graph = threshold_top_k_per_target(ei_matrix, top_k=top_k)
    metrics = evaluate_against_groundtruth(ei_matrix, predicted_graph, groundtruth, top_k=top_k)
    metrics.update(
        {
            "target_mode": target_mode,
            "estimator_mode": estimator_mode,
            "box_width": float(box_width),
            "sample_count": int(len(source_for_ei) if estimator_mode == "observed" else sample_count or len(source)),
            "seed": int(seed),
            "lag": int(lag),
        }
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matrix_path = output_path / "lorzen_tm_ei_matrix.csv"
    edge_path = output_path / "lorzen_tm_ei_edges.csv"
    graph_path = output_path / "lorzen_tm_ei_topk_graph.csv"
    truth_path = output_path / "lorzen_lorenz96_groundtruth.csv"
    summary_path = output_path / "lorzen_tm_ei_summary.json"
    heatmap_path = output_path / "lorzen_tm_ei_heatmap.png"

    ei_matrix.to_csv(matrix_path)
    edge_table.to_csv(edge_path, index=False)
    predicted_graph.to_csv(graph_path)
    groundtruth.to_csv(truth_path)
    plot_lorzen_tm_ei_heatmap(ei_matrix, predicted_graph, groundtruth, output_path=heatmap_path, metrics=metrics)

    summary = {
        "input_path": str(Path(input_path)),
        "sample_count": int(one_step_source.shape[0]),
        "effective_sample_count": int(metrics["sample_count"]),
        "lag": int(lag),
        "node_count": int(one_step_source.shape[1]),
        "metrics": metrics,
        "artifacts": {
            "ei_matrix": str(matrix_path),
            "edge_table": str(edge_path),
            "topk_graph": str(graph_path),
            "groundtruth": str(truth_path),
            "heatmap": str(heatmap_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "source": source_for_ei,
        "target": target_for_ei,
        "ei_matrix": ei_matrix,
        "edge_table": edge_table,
        "predicted_graph": predicted_graph,
        "groundtruth": groundtruth,
        "metrics": metrics,
        "summary_path": summary_path,
        "heatmap_path": heatmap_path,
    }


def _find_archive_member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [name for name in archive.namelist() if Path(name).name == basename and not name.endswith("/")]
    if not matches:
        raise FileNotFoundError(f"Could not find {basename!r} in {archive.filename!r}.")
    return matches[0]


def _lorenz96_rhs(state: np.ndarray, *, forcing: float) -> np.ndarray:
    return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + float(forcing)


def _rk4_lorenz96_step(state: np.ndarray, *, dt: float, forcing: float) -> np.ndarray:
    k1 = _lorenz96_rhs(state, forcing=forcing)
    k2 = _lorenz96_rhs(state + 0.5 * dt * k1, forcing=forcing)
    k3 = _lorenz96_rhs(state + 0.5 * dt * k2, forcing=forcing)
    k4 = _lorenz96_rhs(state + dt * k3, forcing=forcing)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _resolve_target_mode(source: pd.DataFrame, target: pd.DataFrame, *, target_mode: str) -> pd.DataFrame:
    if target_mode == "next":
        return target
    if target_mode == "delta":
        return target - source
    raise ValueError("target_mode must be 'next' or 'delta'.")


def _select_sample_rows(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    sample_count: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_count is None or int(sample_count) >= len(source):
        return source, target
    if int(sample_count) < 4:
        raise ValueError("sample_count must be at least 4.")
    indices = np.linspace(0, len(source) - 1, int(sample_count), dtype=int)
    return source.iloc[indices].reset_index(drop=True), target.iloc[indices].reset_index(drop=True)


def _cell_text_color(value: float, frame: pd.DataFrame) -> str:
    if frame.to_numpy(dtype=float).max(initial=0.0) <= 1.0 and frame.to_numpy(dtype=float).min(initial=0.0) >= 0.0:
        return "white" if value >= 0.5 else "#1f2937"
    values = frame.to_numpy(dtype=float)
    threshold = float(np.nanmin(values) + 0.6 * (np.nanmax(values) - np.nanmin(values)))
    return "white" if value >= threshold else "#1f2937"


def _parse_hidden_layer_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--hidden-layer-sizes must contain positive integers, e.g. 128,64.")
    return sizes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce Lorzen TM-EI causal graph experiment.")
    parser.add_argument("input_path", type=Path, nargs="?", help="Directory or zip containing yt.csv and yt+1.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("exp/TM/lorzen/results"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--target-mode", choices=("next", "delta"), default="next")
    parser.add_argument("--estimator-mode", choices=("observed", "surrogate_intervention"), default="observed")
    parser.add_argument("--box-width", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--simulate-lorenz96-mlp", action="store_true")
    parser.add_argument("--node-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--forcing", type=float, default=8.0)
    parser.add_argument("--train-sample-count", type=int, default=1000)
    parser.add_argument("--hidden-layer-sizes", type=str, default="128,64")
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args(argv)

    if args.simulate_lorenz96_mlp:
        result = run_lorzen_mlp_tm_ei_experiment(
            output_dir=args.output_dir,
            node_count=args.node_count,
            steps=args.steps,
            burn_in=args.burn_in,
            dt=args.dt,
            forcing=args.forcing,
            lag=args.lag,
            train_sample_count=args.train_sample_count,
            intervention_sample_count=int(args.sample_count or 2000),
            box_width=args.box_width,
            top_k=args.top_k,
            seed=args.seed,
            hidden_layer_sizes=_parse_hidden_layer_sizes(args.hidden_layer_sizes),
            max_iter=args.max_iter,
        )
    else:
        if args.input_path is None:
            parser.error("input_path is required unless --simulate-lorenz96-mlp is used.")
        result = run_lorzen_tm_ei_experiment(
            args.input_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
            target_mode=args.target_mode,
            estimator_mode=args.estimator_mode,
            box_width=args.box_width,
            sample_count=args.sample_count,
            seed=args.seed,
            lag=args.lag,
        )
    print(json.dumps({"metrics": result["metrics"], "summary_path": str(result["summary_path"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
