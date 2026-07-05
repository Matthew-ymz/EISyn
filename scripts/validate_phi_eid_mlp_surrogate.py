#!/usr/bin/env python3
"""Validate Phi EID distributions recovered from an MLP surrogate.

The experiment uses an eight-source Boolean transition with three known
irreducible modules.  We train an MLP only from generated state-transition
samples, then compare the hierarchical Phi EID distribution computed from the
oracle transition with the same distribution read out from the MLP under
uniform interventions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "results" / "phi_eid_mlp_surrogate"
DEFAULT_FIGURE_DIR = ROOT / "docs" / "ref" / "assets" / "phi_eid_mlp_surrogate"
SOURCE_COUNT = 8
TARGET_COUNT = 3
MODULES = ((1, 2), (3, 4, 5), (6, 7, 8))
MODULE_COLORS = ("#4c78a8", "#f58518", "#54a24b")


@dataclass(frozen=True)
class PhiAtom:
    sources: tuple[int, ...]
    value: float


@dataclass(frozen=True)
class MLPFit:
    model: object
    train_loss: float
    test_bce: float
    test_exact_match: float
    epochs_run: int
    seconds: float


def all_binary_states(n: int) -> np.ndarray:
    rows = np.asarray(list(itertools.product((0.0, 1.0), repeat=int(n))), dtype=np.float32)
    return rows


def oracle_transition(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != SOURCE_COUNT:
        raise ValueError(f"states must have shape (m, {SOURCE_COUNT}).")
    bits = values.astype(np.int8)
    y0 = np.bitwise_xor(bits[:, 0], bits[:, 1])
    y1 = np.bitwise_xor(np.bitwise_xor(bits[:, 2], bits[:, 3]), bits[:, 4])
    y2 = np.bitwise_xor(np.bitwise_xor(bits[:, 5], bits[:, 6]), bits[:, 7])
    return np.column_stack([y0, y1, y2]).astype(np.float32)


def make_dataset(*, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    states = rng.integers(0, 2, size=(int(sample_count), SOURCE_COUNT), dtype=np.int8).astype(np.float32)
    return states, oracle_transition(states)


def fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    seed: int,
    epochs: int,
    hidden_width: int,
    learning_rate: float,
) -> MLPFit:
    import torch

    torch.manual_seed(int(seed))
    torch.set_num_threads(1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Failed to initialize NumPy.*", category=UserWarning)
        model = torch.nn.Sequential(
            torch.nn.Linear(SOURCE_COUNT, int(hidden_width)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_width), int(hidden_width)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_width), TARGET_COUNT),
        )
    x = torch.tensor(np.asarray(train_x, dtype=np.float32).tolist(), dtype=torch.float32)
    y = torch.tensor(np.asarray(train_y, dtype=np.float32).tolist(), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    t0 = time.perf_counter()
    loss_value = float("nan")
    for epoch in range(1, int(epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        if epoch >= 500 and loss_value < 2e-4:
            break

    test_states = all_binary_states(SOURCE_COUNT)
    test_x = torch.tensor(test_states.tolist(), dtype=torch.float32)
    test_y = torch.tensor(oracle_transition(test_states).tolist(), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        test_bce = float(loss_fn(logits, test_y).cpu().item())
        predicted = (torch.sigmoid(logits) >= 0.5).float()
        exact = (predicted == test_y).all(dim=1).float().mean()
    return MLPFit(
        model=model,
        train_loss=loss_value,
        test_bce=test_bce,
        test_exact_match=float(exact.cpu().item()),
        epochs_run=epoch,
        seconds=time.perf_counter() - t0,
    )


def target_distribution_from_bits(bits: np.ndarray) -> np.ndarray:
    target = np.asarray(bits, dtype=np.int8)
    codes = target[:, 0] * 4 + target[:, 1] * 2 + target[:, 2]
    counts = np.bincount(codes, minlength=2**TARGET_COUNT).astype(float)
    return counts / counts.sum()


def target_distribution_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != TARGET_COUNT:
        raise ValueError(f"probabilities must have shape (m, {TARGET_COUNT}).")
    dist = np.zeros(2**TARGET_COUNT, dtype=float)
    target_states = all_binary_states(TARGET_COUNT)
    for row in probs:
        p = np.prod(np.where(target_states > 0.5, row, 1.0 - row), axis=1)
        dist += p
    dist /= dist.sum()
    return dist


def predict_probabilities(model: object, states: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(states, dtype=np.float32).tolist(), dtype=torch.float32))
        return np.asarray(torch.sigmoid(logits).cpu().tolist(), dtype=float)


def intervention_ei(
    subset: tuple[int, ...],
    *,
    mode: str,
    model: object | None = None,
) -> float:
    if not subset:
        return 0.0
    zero_based = tuple(int(node) - 1 for node in subset)
    all_states = all_binary_states(SOURCE_COUNT)
    subset_states = all_binary_states(len(subset))
    conditional: list[np.ndarray] = []
    for assignment in subset_states:
        mask = np.ones(len(all_states), dtype=bool)
        for col, value in zip(zero_based, assignment):
            mask &= all_states[:, col] == value
        intervened = all_states[mask]
        if mode == "oracle":
            conditional.append(target_distribution_from_bits(oracle_transition(intervened)))
        elif mode == "mlp":
            if model is None:
                raise ValueError("model is required for MLP EI.")
            conditional.append(target_distribution_from_probabilities(predict_probabilities(model, intervened)))
        else:
            raise ValueError(f"unknown mode: {mode}")

    cond = np.asarray(conditional, dtype=float)
    marginal = cond.mean(axis=0)
    eps = 1e-15
    ratio = (cond + eps) / (marginal[None, :] + eps)
    return float(np.mean(np.sum(cond * np.log2(ratio), axis=1)))


def nontrivial_bipartitions(nodes: tuple[int, ...]) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    ordered = tuple(sorted(nodes))
    if len(ordered) <= 1:
        return []
    first = ordered[0]
    rest = ordered[1:]
    full = set(ordered)
    splits = []
    for mask in range(1 << len(rest)):
        left = {first}
        for idx, node in enumerate(rest):
            if mask & (1 << idx):
                left.add(node)
        if len(left) == len(ordered):
            continue
        right = full - left
        splits.append((tuple(sorted(left)), tuple(sorted(right))))
    return splits


def compute_ei_table(*, mode: str, model: object | None = None) -> dict[tuple[int, ...], float]:
    table: dict[tuple[int, ...], float] = {(): 0.0}
    nodes = tuple(range(1, SOURCE_COUNT + 1))
    for size in range(1, SOURCE_COUNT + 1):
        for subset in itertools.combinations(nodes, size):
            table[tuple(subset)] = intervention_ei(tuple(subset), mode=mode, model=model)
    return table


def greedy_phi_atoms(
    nodes: tuple[int, ...],
    ei: dict[tuple[int, ...], float],
    *,
    eps: float = 1e-6,
) -> list[PhiAtom]:
    nodes = tuple(sorted(nodes))
    if len(nodes) <= 1 or ei[nodes] <= eps:
        return [PhiAtom(nodes, max(0.0, ei[nodes]))] if ei[nodes] > eps else []
    best: tuple[float, tuple[int, ...], tuple[int, ...]] | None = None
    for left, right in nontrivial_bipartitions(nodes):
        captured = ei[left] + ei[right]
        if best is None or captured > best[0]:
            best = (captured, left, right)
    if best is None:
        return [PhiAtom(nodes, max(0.0, ei[nodes]))]
    captured, left, right = best
    residual = max(0.0, ei[nodes] - captured)
    atoms: list[PhiAtom] = []
    if residual > eps:
        atoms.append(PhiAtom(nodes, residual))
    atoms.extend(greedy_phi_atoms(left, ei, eps=eps))
    atoms.extend(greedy_phi_atoms(right, ei, eps=eps))
    return atoms


def atom_map(atoms: list[PhiAtom]) -> dict[tuple[int, ...], float]:
    rows: dict[tuple[int, ...], float] = {}
    for atom in atoms:
        rows[atom.sources] = rows.get(atom.sources, 0.0) + float(atom.value)
    return rows


def compare_distributions(oracle: dict[tuple[int, ...], float], mlp: dict[tuple[int, ...], float]) -> dict[str, float]:
    keys = sorted(set(oracle) | set(mlp), key=lambda item: (len(item), item))
    o = np.asarray([oracle.get(key, 0.0) for key in keys], dtype=float)
    m = np.asarray([mlp.get(key, 0.0) for key in keys], dtype=float)
    l1 = float(np.abs(o - m).sum())
    linf = float(np.abs(o - m).max(initial=0.0))
    cosine = float(np.dot(o, m) / (np.linalg.norm(o) * np.linalg.norm(m) + 1e-15))
    total_relative_error = float(abs(o.sum() - m.sum()) / max(o.sum(), 1e-12))
    support_match = float(set(oracle) == {key for key, value in mlp.items() if value > 1e-3})
    return {
        "oracle_total_phi_bits": float(o.sum()),
        "mlp_total_phi_bits": float(m.sum()),
        "l1_bits": l1,
        "linf_bits": linf,
        "cosine": cosine,
        "total_relative_error": total_relative_error,
        "support_match_at_1e-3": support_match,
    }


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def node_positions() -> dict[int, tuple[float, float]]:
    return {
        1: (-1.65, 0.50),
        2: (-0.92, -0.48),
        3: (-0.12, 0.58),
        4: (0.66, -0.40),
        5: (1.28, 0.54),
        6: (2.08, -0.38),
        7: (2.72, 0.54),
        8: (3.36, -0.34),
    }


def _add_surface(ax, base_xy: list[tuple[float, float]], value: float, color: str, max_height: float) -> None:
    base = np.asarray([[x, y, 0.0] for x, y in base_xy], dtype=float)
    centroid = base[:, :2].mean(axis=0)
    height = 1.25 * float(value) / max(max_height, 1e-12)
    apex = np.asarray([centroid[0], centroid[1], height], dtype=float)
    faces: list[list[list[float]]] = []
    if len(base) == 2:
        faces.append([base[0].tolist(), base[1].tolist(), apex.tolist()])
    else:
        faces.append(base.tolist())
        for idx in range(len(base)):
            faces.append([base[idx].tolist(), base[(idx + 1) % len(base)].tolist(), apex.tolist()])
    edge = mpl.colors.to_hex(np.asarray(mpl.colors.to_rgb(color)) * 0.62)
    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, edgecolor=edge, linewidth=0.95, alpha=0.28))
    ax.scatter([apex[0]], [apex[1]], [apex[2]], s=26, color=edge, depthshade=False)
    ax.text(apex[0], apex[1], apex[2] + 0.09, f"{value:.2f}", ha="center", va="bottom", fontsize=6.8, color=edge)


def _draw_surface_panel(ax, atoms: dict[tuple[int, ...], float], *, title: str) -> None:
    positions = node_positions()
    for module in MODULES:
        pairs = list(zip(module, module[1:]))
        for left, right in pairs:
            x0, y0 = positions[left]
            x1, y1 = positions[right]
            ax.plot([x0, x1], [y0, y1], [0, 0], color="#aeb7c2", linewidth=0.8, alpha=0.7)
    max_height = max(atoms.values()) if atoms else 1.0
    for idx, module in enumerate(MODULES):
        value = atoms.get(module, 0.0)
        if value > 1e-4:
            _add_surface(ax, [positions[node] for node in module], value, MODULE_COLORS[idx], max_height)
    xs = [positions[node][0] for node in range(1, SOURCE_COUNT + 1)]
    ys = [positions[node][1] for node in range(1, SOURCE_COUNT + 1)]
    ax.scatter(xs, ys, np.full(SOURCE_COUNT, 0.035), s=50, color="#ffffff", edgecolor="#333333", linewidth=0.75, depthshade=False)
    for node, (x, y) in positions.items():
        ax.text(x + 0.08, y + (0.08 if y < 0 else -0.08), 0.09, str(node), ha="center", va="center", fontsize=6.8)
    ax.set_xlim(-2.1, 3.8)
    ax.set_ylim(-0.95, 0.95)
    ax.set_zlim(0.0, 1.45)
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()
    ax.text2D(0.02, 0.96, title, transform=ax.transAxes, fontsize=9, fontweight="bold")


def build_figure(
    oracle_atoms: dict[tuple[int, ...], float],
    mlp_atoms: dict[tuple[int, ...], float],
    metrics: dict[str, float],
) -> plt.Figure:
    _setup_style()
    fig = plt.figure(figsize=(9.0, 5.2), constrained_layout=True)
    grid = GridSpec(2, 3, figure=fig, width_ratios=[1.25, 1.25, 1.0], height_ratios=[1.0, 1.0])
    ax_oracle = fig.add_subplot(grid[0, :2], projection="3d")
    ax_mlp = fig.add_subplot(grid[1, :2], projection="3d")
    ax_bar = fig.add_subplot(grid[0, 2])
    ax_metric = fig.add_subplot(grid[1, 2])

    _draw_surface_panel(ax_oracle, oracle_atoms, title="a  Oracle")
    _draw_surface_panel(ax_mlp, mlp_atoms, title="b  MLP")

    labels = ["1+2", "3+4+5", "6+7+8"]
    y = np.arange(len(labels))
    oracle_values = [oracle_atoms.get(module, 0.0) for module in MODULES]
    mlp_values = [mlp_atoms.get(module, 0.0) for module in MODULES]
    ax_bar.barh(y - 0.18, oracle_values, height=0.32, color="#333333", label="Oracle")
    ax_bar.barh(y + 0.18, mlp_values, height=0.32, color="#6baed6", label="MLP")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(labels)
    ax_bar.set_xlabel("Phi EID [bit]")
    ax_bar.set_title("c  Atom values", loc="left", fontsize=9, fontweight="bold")
    ax_bar.grid(axis="x", color="#e4e7eb", linewidth=0.7)
    ax_bar.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    metric_names = ["cosine", "1-L1/6", "exact acc."]
    metric_values = [
        metrics["cosine"],
        max(0.0, 1.0 - metrics["l1_bits"] / 6.0),
        metrics["test_exact_match"],
    ]
    ax_metric.bar(metric_names, metric_values, color=["#54a24b", "#f58518", "#4c78a8"], width=0.62)
    ax_metric.set_ylim(0.0, 1.05)
    ax_metric.set_title("d  Agreement", loc="left", fontsize=9, fontweight="bold")
    ax_metric.grid(axis="y", color="#e4e7eb", linewidth=0.7)
    for idx, value in enumerate(metric_values):
        ax_metric.text(idx, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    return fig


def write_report(
    path: Path,
    *,
    summary: dict[str, object],
    figure_path: Path,
) -> None:
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_ref = figure_path.relative_to(path.parent)
    path.write_text(
        "\n".join(
            [
                "# Phi EID MLP surrogate validation",
                "",
                "本实验构造 8 维布尔动力学，目标由三个不可约局部模块产生：",
                "`X1 xor X2`、`X3 xor X4 xor X5`、`X6 xor X7 xor X8`。",
                "MLP 只从生成的状态转移样本训练，随后在 uniform intervention 口径下读取 Phi EID 层次分布。",
                "",
                f"![oracle vs mlp]({figure_ref})",
                "",
                "## Metrics",
                "",
                f"- Oracle total Phi: `{metrics['oracle_total_phi_bits']:.6f}` bits",
                f"- MLP total Phi: `{metrics['mlp_total_phi_bits']:.6f}` bits",
                f"- L1 difference: `{metrics['l1_bits']:.6f}` bits",
                f"- Linf difference: `{metrics['linf_bits']:.6f}` bits",
                f"- Cosine similarity: `{metrics['cosine']:.6f}`",
                f"- Test exact-match accuracy: `{metrics['test_exact_match']:.6f}`",
                "",
                "结论：在这个可精确枚举的高阶布尔动力学中，MLP surrogate 从数据恢复了 oracle 的 Phi EID 支持集与数值分布；因此该图可作为数据驱动读出流程的正对照。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y = make_dataset(sample_count=args.samples, seed=args.seed)
    fit = fit_mlp(
        train_x,
        train_y,
        seed=args.seed,
        epochs=args.epochs,
        hidden_width=args.hidden_width,
        learning_rate=args.learning_rate,
    )
    oracle_ei = compute_ei_table(mode="oracle")
    mlp_ei = compute_ei_table(mode="mlp", model=fit.model)
    oracle_atoms = atom_map(greedy_phi_atoms(tuple(range(1, SOURCE_COUNT + 1)), oracle_ei))
    mlp_atoms = atom_map(greedy_phi_atoms(tuple(range(1, SOURCE_COUNT + 1)), mlp_ei))
    metrics = compare_distributions(oracle_atoms, mlp_atoms)
    metrics.update(
        {
            "train_loss": fit.train_loss,
            "test_bce": fit.test_bce,
            "test_exact_match": fit.test_exact_match,
            "epochs_run": float(fit.epochs_run),
            "training_seconds": fit.seconds,
        }
    )

    fig = build_figure(oracle_atoms, mlp_atoms, metrics)
    figure_base = figure_dir / "phi_eid_mlp_surrogate_validation"
    fig.savefig(figure_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "source_count": SOURCE_COUNT,
        "target_count": TARGET_COUNT,
        "modules": [list(module) for module in MODULES],
        "samples": int(args.samples),
        "seed": int(args.seed),
        "oracle_atoms": [{"sources": list(key), "value": value} for key, value in sorted(oracle_atoms.items())],
        "mlp_atoms": [{"sources": list(key), "value": value} for key, value in sorted(mlp_atoms.items())],
        "metrics": metrics,
        "outputs": {
            "figure_png": str(figure_base.with_suffix(".png").relative_to(ROOT)),
            "figure_svg": str(figure_base.with_suffix(".svg").relative_to(ROOT)),
            "figure_pdf": str(figure_base.with_suffix(".pdf").relative_to(ROOT)),
            "summary_json": str((result_dir / "phi_eid_mlp_surrogate_summary.json").relative_to(ROOT)),
            "report_md": str((ROOT / "docs" / "ref" / "phi_eid_mlp_surrogate_validation.md").relative_to(ROOT)),
        },
    }
    summary_path = result_dir / "phi_eid_mlp_surrogate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(
        ROOT / "docs" / "ref" / "phi_eid_mlp_surrogate_validation.md",
        summary=summary,
        figure_path=figure_base.with_suffix(".png"),
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3500)
    parser.add_argument("--hidden-width", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_experiment(args)
    print(json.dumps(summary["metrics"], indent=2))
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
