#!/usr/bin/env python3
"""Cache subject-wise Yeo7 network-PC1 time series and plot selected subjects."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hcp_schaefer500_yeo7_pca_mlp_comparison import DEFAULT_DATA, DEFAULT_LABELS, default_data_key, default_yeo7_labels, load_hcp_series, load_yeo7_groups


DEFAULT_DATA_ROOT = DEFAULT_DATA.parents[1]
DEFAULT_CACHE_DIR = ROOT / "results" / "hcp_schaefer500_yeo7_pc1" / "cache"
DEFAULT_FIGURE = ROOT / "results" / "hcp_schaefer500_yeo7_pc1" / "subject_time_series"
COLORS = ("#4C78A8", "#72B7B2", "#F2CF5B", "#E45756", "#B279A2", "#59A14F", "#9D755D")


def reduce_subject(series: np.ndarray, groups: Mapping[str, Sequence[int]], *, fit_end: int | None = None) -> dict[str, np.ndarray]:
    values = np.asarray(series, dtype=float)
    stop = len(values) if fit_end is None else int(fit_end)
    if stop < 3 or stop > len(values):
        raise ValueError("fit_end must contain at least three rows and lie within series.")
    network_names = list(groups)
    scores = np.empty((values.shape[0], len(network_names)), dtype=float)
    explained = np.empty(len(network_names), dtype=float)
    component_weights = np.zeros(values.shape[1], dtype=float)
    parcel_network_index = np.empty(values.shape[1], dtype=int)
    for network_index, name in enumerate(network_names):
        indices = np.asarray(groups[name], dtype=int)
        model = PCA(n_components=1, svd_solver="full").fit(values[:stop, indices])
        weights = np.asarray(model.components_[0], dtype=float)
        projected = model.transform(values[:, indices])[:, 0]
        if float(weights.sum()) < 0.0:
            weights, projected = -weights, -projected
        scores[:, network_index] = projected
        explained[network_index] = float(model.explained_variance_ratio_[0])
        component_weights[indices] = weights
        parcel_network_index[indices] = network_index
    return {
        "scores": scores,
        "network_names": np.asarray(network_names),
        "explained_variance_ratio": explained,
        "components": component_weights,
        "parcel_network_index": parcel_network_index,
    }


def cache_subject(cache_dir: Path, *, subject: str, series: np.ndarray, groups: Mapping[str, Sequence[int]], fit_end: int | None = None) -> Path:
    payload = reduce_subject(series, groups, fit_end=fit_end)
    destination = Path(cache_dir) / f"{subject}_yeo7_pc1.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, subject=np.asarray(subject), input_shape=np.asarray(series.shape), **payload)
    return destination


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def plot_subject_time_series(cache_paths: Sequence[Path], destination: Path) -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8})
    payloads = [load_cache(path) for path in cache_paths]
    names = [str(value) for value in payloads[0]["network_names"].tolist()]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.4), sharex=True, sharey=True, constrained_layout=True)
    handles = []
    for panel_index, (axis, payload) in enumerate(zip(axes.flat, payloads)):
        scores = np.asarray(payload["scores"], dtype=float)
        zscores = (scores - scores.mean(axis=0, keepdims=True)) / np.where(scores.std(axis=0, ddof=1, keepdims=True) > 1e-12, scores.std(axis=0, ddof=1, keepdims=True), 1.0)
        for index, name in enumerate(names):
            line, = axis.plot(np.arange(1, len(scores) + 1), zscores[:, index], color=COLORS[index], linewidth=0.65, alpha=0.9, label=name)
            if panel_index == 0:
                handles.append(line)
        axis.axhline(0.0, color="#777777", linewidth=0.5, zorder=0)
        axis.set_title(str(payload["subject"].item()), fontsize=9, fontweight="bold")
        axis.set_xlim(1, len(scores))
        axis.set_ylim(-4.2, 4.2)
        axis.tick_params(direction="out", length=3, width=0.7)
    for axis in axes[:, 0]:
        axis.set_ylabel("PC1 score (z)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time point")
    fig.legend(handles=handles, labels=names, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=7, frameon=False, handlelength=1.5, columnspacing=1.1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".png", {"dpi": 300}), (".svg", {}), (".pdf", {})):
        fig.savefig(destination.with_suffix(suffix), bbox_inches="tight", **kwargs)


def run(data_root: Path, labels: Path, cache_dir: Path, figure: Path, plot_subjects: Sequence[str], *, parcel_count: int = 500, data_key: str | None = None, development_end: int = 900) -> dict[str, object]:
    count = int(parcel_count)
    key = data_key or default_data_key(count)
    groups = load_yeo7_groups(labels, expected_parcels=count)
    files = sorted(Path(data_root).glob("sub-*/*.mat"))
    if not files:
        raise FileNotFoundError(f"No HCP MAT files found below {data_root}.")
    cache_paths = []
    rows = []
    for path in files:
        subject = path.parent.name
        series = load_hcp_series(path, parcel_count=count, data_key=key)
        cache_path = cache_subject(cache_dir, subject=subject, series=series, groups=groups, fit_end=development_end)
        cache_paths.append(cache_path)
        payload = load_cache(cache_path)
        rows.append({"subject": subject, "cache": str(cache_path), "mean_pc1_explained_variance": float(payload["explained_variance_ratio"].mean())})
    paths_by_subject = {path.stem.removesuffix("_yeo7_pc1"): path for path in cache_paths}
    selected = list(plot_subjects)
    if not selected:
        selected = sorted(paths_by_subject)[:4]
    if len(selected) != 4 or any(subject not in paths_by_subject for subject in selected):
        raise ValueError("plot_subjects must name exactly four cached subjects.")
    plot_subject_time_series([paths_by_subject[subject] for subject in selected], figure)
    summary = {"n_subjects": len(rows), "parcel_count": count, "data_key": key, "labels": str(labels), "development_end": int(development_end), "network_order": list(groups), "network_sizes": {name: len(indices) for name, indices in groups.items()}, "plot_subjects": selected, "rows": rows}
    figure.parent.joinpath("cache_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--parcel-count", type=int, choices=(500, 1000), default=500)
    parser.add_argument("--data-key", default="", help="MAT variable name; defaults to Schaefer<parcel-count>.")
    parser.add_argument("--development-end", type=int, default=900)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--plot-subjects", default="sub-100206,sub-100307,sub-100408,sub-100610")
    args = parser.parse_args(argv)
    subjects = [part.strip() for part in str(args.plot_subjects).split(",") if part.strip()]
    labels = args.labels or default_yeo7_labels(args.parcel_count)
    print(json.dumps(run(args.data_root, labels, args.cache_dir, args.figure, subjects, parcel_count=args.parcel_count, data_key=args.data_key or None, development_end=args.development_end), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
