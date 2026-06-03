from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from yrd import clip_nonnegative_ei, estimate_mutual_information_transport_map, lift_transport_source_features


DEFAULT_SOURCE_NAME = "yt.csv"
DEFAULT_TARGET_NAME = "yt+1.csv"


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
) -> dict[str, Any]:
    source, target = load_lorzen_transition_csvs(input_path)
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
    groundtruth = lorenz96_groundtruth_adjacency(source.shape[1])
    predicted_graph = threshold_top_k_per_target(ei_matrix, top_k=top_k)
    metrics = evaluate_against_groundtruth(ei_matrix, predicted_graph, groundtruth, top_k=top_k)
    metrics.update(
        {
            "target_mode": target_mode,
            "estimator_mode": estimator_mode,
            "box_width": float(box_width),
            "sample_count": int(len(source_for_ei) if estimator_mode == "observed" else sample_count or len(source)),
            "seed": int(seed),
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
        "sample_count": int(source.shape[0]),
        "effective_sample_count": int(metrics["sample_count"]),
        "node_count": int(source.shape[1]),
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
        "source": source,
        "target": target,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce Lorzen TM-EI causal graph experiment.")
    parser.add_argument("input_path", type=Path, help="Directory or zip containing yt.csv and yt+1.csv.")
    parser.add_argument("--output-dir", type=Path, default=Path("exp/TM/lorzen/results"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--target-mode", choices=("next", "delta"), default="next")
    parser.add_argument("--estimator-mode", choices=("observed", "surrogate_intervention"), default="observed")
    parser.add_argument("--box-width", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / "tmp" / "matplotlib").resolve()))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    result = run_lorzen_tm_ei_experiment(
        args.input_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
        target_mode=args.target_mode,
        estimator_mode=args.estimator_mode,
        box_width=args.box_width,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    print(json.dumps({"metrics": result["metrics"], "summary_path": str(result["summary_path"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
