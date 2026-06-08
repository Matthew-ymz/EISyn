#!/usr/bin/env python3
"""Reproduce the SURD paper's synergistic-collider example for target Q1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yrd import estimate_specific_mutual_information_transport_map

ATOM_ORDER = ("R123", "R12", "R13", "R23", "U1", "U2", "U3", "S12", "S13", "S23", "S123")
SOURCE_SUBSETS = {
    "1": (0,),
    "2": (1,),
    "3": (2,),
    "12": (0, 1),
    "13": (0, 2),
    "23": (1, 2),
    "123": (0, 1, 2),
}

DEFAULT_RESULT_DIR = ROOT / "results" / "surd_original_synergistic_collider"
DEFAULT_FIGURE_DIR = ROOT / "fig" / "surd_original_synergistic_collider"


@dataclass(frozen=True)
class SurdColliderConfig:
    n_samples: int = 2_000_000
    burn_in: int = 10_000
    bins: int = 100
    seed: int = 0
    q1_noise: float = 0.001
    q23_noise: float = 0.1
    transport_degree: int = 3
    target_anchors: int = 256
    conditional_samples: int = 128


@dataclass(frozen=True)
class SurdResult:
    atoms: dict[str, float]
    normalized_atoms: dict[str, float]
    mutual_information: float
    target_entropy: float
    leak: float
    normalized_leak: float
    specific_atoms_by_target_bin: list[dict[str, float]]


def simulate_synergistic_collider(config: SurdColliderConfig) -> pd.DataFrame:
    """Simulate the paper's Q2,Q3 synergistic collider into Q1."""

    rng = np.random.default_rng(int(config.seed))
    total = int(config.n_samples + config.burn_in + 1)
    data = np.zeros((total, 3), dtype=float)
    for t in range(total - 1):
        q1, q2, q3 = data[t]
        data[t + 1, 0] = np.sin(q2 * q3) + float(config.q1_noise) * rng.normal()
        data[t + 1, 1] = 0.5 * q2 + float(config.q23_noise) * rng.normal()
        data[t + 1, 2] = 0.5 * q3 + float(config.q23_noise) * rng.normal()
    return pd.DataFrame(data[config.burn_in :], columns=["Q1", "Q2", "Q3"]).reset_index(drop=True)


def _bin_columns(values: np.ndarray, bins: int | None) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer) and bins is None:
        _, inverse = np.unique(array, axis=0, return_inverse=True)
        if array.ndim == 1:
            return inverse.astype(int)
        return np.column_stack(
            [np.unique(array[:, col], return_inverse=True)[1] for col in range(array.shape[1])]
        ).astype(int)

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    binned = np.zeros_like(array, dtype=int)
    for col in range(array.shape[1]):
        values_col = array[:, col].astype(float)
        if bins is None:
            unique = np.unique(values_col)
            mapping = {value: idx for idx, value in enumerate(unique)}
            binned[:, col] = [mapping[value] for value in values_col]
            continue
        low = float(np.min(values_col))
        high = float(np.max(values_col))
        if high <= low:
            binned[:, col] = 0
            continue
        edges = np.linspace(low, high, int(bins) + 1)
        binned[:, col] = np.clip(np.digitize(values_col, edges[1:-1], right=False), 0, int(bins) - 1)
    return binned[:, 0] if values.ndim == 1 else binned


def _joint_codes(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(matrix, dtype=int)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    _, inverse = np.unique(array, axis=0, return_inverse=True)
    return inverse.astype(int), int(inverse.max()) + 1


def _specific_mi_for_subset(source_codes: np.ndarray, target_codes: np.ndarray, n_target: int) -> np.ndarray:
    source = np.asarray(source_codes, dtype=int).reshape(-1)
    target = np.asarray(target_codes, dtype=int).reshape(-1)
    n_source = int(source.max()) + 1
    counts = np.zeros((n_target, n_source), dtype=float)
    np.add.at(counts, (target, source), 1.0)
    total = float(counts.sum())
    target_counts = counts.sum(axis=1)
    source_counts = counts.sum(axis=0)
    specific = np.zeros(n_target, dtype=float)
    for y in range(n_target):
        if target_counts[y] <= 0:
            continue
        source_given_y = counts[y] / target_counts[y]
        valid = counts[y] > 0
        p_y_given_source = counts[y, valid] / source_counts[valid]
        p_y = target_counts[y] / total
        specific[y] = float(np.sum(source_given_y[valid] * np.log2(p_y_given_source / p_y)))
    return specific


def _assign_specific_atoms(specific_values: dict[str, float]) -> dict[str, float]:
    atoms = {name: 0.0 for name in ATOM_ORDER}

    singles = sorted(
        [("1", specific_values["1"]), ("2", specific_values["2"]), ("3", specific_values["3"])],
        key=lambda item: (item[1], item[0]),
    )
    previous = 0.0
    for rank, (label, value) in enumerate(singles):
        increment = max(0.0, float(value) - previous)
        remaining = "".join(sorted(item[0] for item in singles[rank:]))
        if rank < len(singles) - 1:
            atoms[f"R{remaining}"] += increment
        else:
            atoms[f"U{label}"] += increment
        previous = float(value)

    for order, labels in ((2, ("12", "13", "23")), (3, ("123",))):
        lower_labels = tuple(label for label in SOURCE_SUBSETS if len(label) == order - 1)
        lower_max = max(float(specific_values[label]) for label in lower_labels)
        ordered = sorted([(label, float(specific_values[label])) for label in labels], key=lambda item: (item[1], item[0]))
        previous = 0.0
        for label, value in ordered:
            if previous >= lower_max:
                increment = value - previous
            elif value > lower_max > previous:
                increment = value - lower_max
            else:
                increment = 0.0
            atoms[f"S{label}"] += max(0.0, float(increment))
            previous = value
    return atoms


def decompose_surd_3source(
    sources: np.ndarray,
    target: np.ndarray,
    *,
    bins: int | None = None,
) -> SurdResult:
    """Compute the original SURD 11 atoms for three sources and one target."""

    source_bins = _bin_columns(np.asarray(sources), bins)
    target_bins = _bin_columns(np.asarray(target).reshape(-1), bins)
    if source_bins.ndim != 2 or source_bins.shape[1] != 3:
        raise ValueError("sources must contain exactly three columns.")
    if source_bins.shape[0] != len(target_bins):
        raise ValueError("sources and target must share the sample axis.")

    target_codes, n_target = _joint_codes(target_bins)
    subset_specific: dict[str, np.ndarray] = {}
    for label, cols in SOURCE_SUBSETS.items():
        codes, _ = _joint_codes(source_bins[:, cols])
        subset_specific[label] = _specific_mi_for_subset(codes, target_codes, n_target)

    target_counts = np.bincount(target_codes, minlength=n_target).astype(float)
    target_prob = target_counts / float(target_counts.sum())
    atoms = {name: 0.0 for name in ATOM_ORDER}
    specific_rows: list[dict[str, float]] = []
    for y, weight in enumerate(target_prob):
        values = {label: float(series[y]) for label, series in subset_specific.items()}
        specific_atoms = _assign_specific_atoms(values)
        specific_rows.append({"target_bin": float(y), "p_target": float(weight), **specific_atoms})
        for name, value in specific_atoms.items():
            atoms[name] += float(weight) * float(value)

    mutual_information = float(np.sum(target_prob * subset_specific["123"]))
    positive = target_prob[target_prob > 0.0]
    target_entropy = float(-np.sum(positive * np.log2(positive)))
    leak = max(0.0, target_entropy - mutual_information)
    denominator = mutual_information if mutual_information > 1.0e-12 else 1.0
    normalized_atoms = {name: float(value / denominator) for name, value in atoms.items()}
    normalized_leak = float(leak / target_entropy) if target_entropy > 1.0e-12 else 0.0
    return SurdResult(
        atoms={name: float(value) for name, value in atoms.items()},
        normalized_atoms=normalized_atoms,
        mutual_information=mutual_information,
        target_entropy=target_entropy,
        leak=leak,
        normalized_leak=normalized_leak,
        specific_atoms_by_target_bin=specific_rows,
    )


def decompose_surd_3source_transport_map(
    sources: np.ndarray,
    target: np.ndarray,
    *,
    degree: int = 3,
    target_anchors: int = 256,
    conditional_samples: int = 128,
    seed: int = 0,
) -> SurdResult:
    """Compute the original SURD atoms using transport-map specific MI."""

    source_array = np.asarray(sources, dtype=float)
    target_array = np.asarray(target, dtype=float).reshape(-1, 1)
    if source_array.ndim != 2 or source_array.shape[1] != 3:
        raise ValueError("sources must contain exactly three columns.")
    if source_array.shape[0] != target_array.shape[0]:
        raise ValueError("sources and target must share the sample axis.")
    rng = np.random.default_rng(seed)
    anchor_count = min(int(target_anchors), len(target_array))
    anchor_indices = rng.choice(len(target_array), size=anchor_count, replace=False)
    anchors = target_array[anchor_indices]
    subset_specific: dict[str, np.ndarray] = {}
    for offset, (label, cols) in enumerate(SOURCE_SUBSETS.items()):
        summary = estimate_specific_mutual_information_transport_map(
            source_array[:, cols],
            target_array,
            target_anchors=anchors,
            degree=degree,
            conditional_samples=conditional_samples,
            seed=seed + offset,
        )
        subset_specific[label] = np.maximum(0.0, np.asarray(summary["specific_mi"], dtype=float))

    atoms = {name: 0.0 for name in ATOM_ORDER}
    specific_rows: list[dict[str, float]] = []
    for anchor_index in range(anchor_count):
        values = {label: float(series[anchor_index]) for label, series in subset_specific.items()}
        for label in ("12", "13", "23"):
            values[label] = max(values[label], *(values[item] for item in label))
        values["123"] = max(values["123"], values["12"], values["13"], values["23"])
        specific_atoms = _assign_specific_atoms(values)
        specific_rows.append(
            {"target_bin": float(anchor_index), "p_target": float(1.0 / anchor_count), **specific_atoms}
        )
        for name, value in specific_atoms.items():
            atoms[name] += float(value / anchor_count)
    mutual_information = float(sum(atoms.values()))
    target_entropy = float("nan")
    denominator = mutual_information if mutual_information > 1.0e-12 else 1.0
    return SurdResult(
        atoms={name: float(value) for name, value in atoms.items()},
        normalized_atoms={name: float(value / denominator) for name, value in atoms.items()},
        mutual_information=mutual_information,
        target_entropy=target_entropy,
        leak=float("nan"),
        normalized_leak=float("nan"),
        specific_atoms_by_target_bin=specific_rows,
    )


def decompose_surd_2source_transport_map(
    left: np.ndarray,
    right: np.ndarray,
    target: np.ndarray,
    *,
    degree: int = 3,
    target_anchors: int = 256,
    conditional_samples: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Compute bivariate SURD R/U/S atoms with original specific-MI increments."""

    left_array = np.asarray(left, dtype=float).reshape(-1, 1)
    right_array = np.asarray(right, dtype=float).reshape(-1, 1)
    target_array = np.asarray(target, dtype=float).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    anchor_count = min(int(target_anchors), len(target_array))
    anchors = target_array[rng.choice(len(target_array), size=anchor_count, replace=False)]
    specific = {}
    for offset, (label, source) in enumerate(
        (("x", left_array), ("y", right_array), ("xy", np.column_stack([left_array, right_array])))
    ):
        summary = estimate_specific_mutual_information_transport_map(
            source,
            target_array,
            target_anchors=anchors,
            degree=degree,
            conditional_samples=conditional_samples,
            seed=seed + offset,
        )
        specific[label] = np.maximum(0.0, np.asarray(summary["specific_mi"], dtype=float))
    specific["xy"] = np.maximum(specific["xy"], np.maximum(specific["x"], specific["y"]))
    redundancy = np.minimum(specific["x"], specific["y"])
    unique_x = specific["x"] - redundancy
    unique_y = specific["y"] - redundancy
    synergy = specific["xy"] - np.maximum(specific["x"], specific["y"])
    joint = float(np.mean(specific["xy"]))
    return {
        "redundancy": float(np.mean(redundancy)),
        "unique_x": float(np.mean(unique_x)),
        "unique_y": float(np.mean(unique_y)),
        "synergy": float(np.mean(synergy)),
        "joint_ei": joint,
        "synergy_fraction": float(np.mean(synergy) / joint) if joint > 1.0e-12 else 0.0,
        "density_backend": f"polynomial_triangular_transport_map_degree_{degree}",
    }


def _plot_q1(result: SurdResult, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "R": "#607d8b",
        "U": "#e57373",
        "S": "#fdb462",
    }
    values = [result.normalized_atoms[name] for name in ATOM_ORDER]
    bar_colors = [
        colors["R"] if name.startswith("R") else colors["U"] if name.startswith("U") else colors["S"]
        for name in ATOM_ORDER
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.2), constrained_layout=True)
    ax.bar(np.arange(len(ATOM_ORDER)), values, color=bar_colors, edgecolor="black", linewidth=0.7)
    ax.set_xticks(np.arange(len(ATOM_ORDER)), ATOM_ORDER, rotation=60, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(r"$\Delta I_{(\cdot)\to 1} / I(Q_1^+; Q)$")
    ax.set_title(r"SURD reproduction for target $Q_1^+$")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_reproduction(
    config: SurdColliderConfig,
    *,
    result_dir: Path = DEFAULT_RESULT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
) -> dict[str, object]:
    series = simulate_synergistic_collider(config)
    q1 = decompose_surd_3source_transport_map(
        series[["Q1", "Q2", "Q3"]].to_numpy(dtype=float)[:-1],
        series["Q1"].to_numpy(dtype=float)[1:],
        degree=config.transport_degree,
        target_anchors=config.target_anchors,
        conditional_samples=config.conditional_samples,
        seed=config.seed,
    )
    figure_path = figure_dir / "surd_q1_synergistic_collider_11_atoms.png"
    _plot_q1(q1, figure_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_dir / "summary_q1.json"
    payload = {
        "config": asdict(config),
        "q1": {
            "atoms": q1.atoms,
            "normalized_atoms": q1.normalized_atoms,
            "mutual_information": q1.mutual_information,
            "target_entropy": q1.target_entropy,
            "leak": q1.leak,
            "normalized_leak": q1.normalized_leak,
            "top_atom": max(q1.normalized_atoms, key=q1.normalized_atoms.get),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"summary_path": str(summary_path), "figure_path": str(figure_path), "q1": payload["q1"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    config = SurdColliderConfig()
    if args.smoke:
        config = replace(
            config,
            n_samples=20_000,
            burn_in=1_000,
            bins=40,
            target_anchors=96,
            conditional_samples=64,
        )
    if args.samples is not None:
        config = replace(config, n_samples=int(args.samples))
    if args.bins is not None:
        config = replace(config, bins=int(args.bins))
    output = run_reproduction(config, result_dir=args.result_dir, figure_dir=args.figure_dir)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
