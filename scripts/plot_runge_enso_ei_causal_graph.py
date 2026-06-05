from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_runge_pairwise_mlp_ei import (
    PairwiseMlpEiConfig,
    _frame_content_hash,
    _model_config_hash,
    build_lagged_dataset,
    load_component_scores,
    predict_mlp,
    sample_max_entropy_features,
    split_temporal_arrays,
    train_or_load_mlp,
)


RESULT_DIR = ROOT / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects"
PAIRWISE_DIR = ROOT / "results" / "runge" / "pairwise_mlp_tm_ei"
FIG_DIR = ROOT / "fig" / "runge" / "pairwise_mlp_tm_ei_path_effects"
OUT_PATH = FIG_DIR / "enso_ei_causal_graph.png"
LAGGED_OUT_PATH = FIG_DIR / "enso_ei_causal_graph_lagged.png"
LAG_DIAGNOSTIC_PATH = ROOT / "results" / "runge" / "pairwise_mlp_tm_ei_path_effects" / "enso_lag_diagnostics.csv"


NODE_POS = {
    9: (0.50, 0.82),
    16: (0.77, 0.76),
    4: (0.23, 0.78),
    59: (0.08, 0.53),
    33: (0.17, 0.42),
    0: (0.35, 0.54),
    1: (0.66, 0.54),
    18: (0.08, 0.24),
    19: (0.20, 0.12),
    11: (0.30, 0.15),
    57: (0.45, 0.24),
    54: (0.55, 0.15),
    17: (0.72, 0.22),
    36: (0.88, 0.32),
    46: (0.93, 0.58),
    50: (0.84, 0.12),
}

NODE_GROUP = {
    0: "enso",
    1: "enso",
    33: "diagnostic",
    9: "driver",
    16: "driver",
    4: "driver",
    59: "driver",
}


def node_label(index: int) -> str:
    if index == 0:
        return "No.0\nENSO gateway"
    if index == 1:
        return "No.1\nENSO mediator"
    return f"No.{index}"


def config_from_manifest() -> PairwiseMlpEiConfig:
    manifest = pd.read_json(PAIRWISE_DIR / "manifest.json", typ="series")
    raw = dict(manifest["config"])
    fields = PairwiseMlpEiConfig.__dataclass_fields__
    kwargs = {key: raw[key] for key in fields if key in raw}
    kwargs["component_scores"] = Path(kwargs.get("component_scores", "results/runge/2015_gateways/component_weekly_scores.csv"))
    kwargs["output_dir"] = Path(kwargs.get("output_dir", "."))
    kwargs["ei_estimator"] = "tm"
    kwargs["gateway_mode"] = "pairwise"
    kwargs["source_mode"] = "latest"
    return PairwiseMlpEiConfig(**kwargs)


def lag_resolved_tm_ei(pairs: set[tuple[int, int]]) -> pd.DataFrame:
    from exp.TM.transport_map_density import estimate_mutual_information_transport_map

    config = config_from_manifest()
    frame = load_component_scores(ROOT / config.component_scores)
    names = list(frame.columns)
    n_components = len(names)
    lag = int(config.lag)
    features, targets = build_lagged_dataset(frame, lag=lag, horizon=int(config.horizon))
    splits = split_temporal_arrays(
        features,
        targets,
        train_fraction=float(config.train_fraction),
        val_fraction=float(config.val_fraction),
    )
    data_hash = _frame_content_hash(frame)
    config_hash = _model_config_hash(config, n_components=n_components, n_rows=len(frame), data_hash=data_hash)
    model, scalers, _, _ = train_or_load_mlp(
        splits,
        config,
        ROOT / "results" / "runge" / "pairwise_mlp_ei" / "mlp_transition.pt",
        config_hash=config_hash,
    )
    intervention_features = sample_max_entropy_features(
        splits["train"][0],
        n_components=n_components,
        lag=lag,
        samples=int(config.intervention_samples),
        low_q=float(config.quantile_low),
        high_q=float(config.quantile_high),
        seed=int(config.seed),
    )
    predictions = predict_mlp(model, scalers, intervention_features)
    pcmci = pd.read_csv(ROOT / "results" / "runge" / "2015_gateways" / "causal_edges.csv")
    rows = []
    for source, target in sorted(pairs):
        target_state = predictions[:, [target]]
        lag_scores = []
        for lag_idx in range(lag):
            lag_weeks = lag - lag_idx
            col = lag_idx * n_components + source
            summary = estimate_mutual_information_transport_map(intervention_features[:, [col]], target_state)
            lag_scores.append((lag_weeks, max(0.0, float(summary["mi_hat"]))))
        best_lag, best_ei = max(lag_scores, key=lambda item: item[1])
        pcmci_rows = pcmci[(pcmci["source"] == source) & (pcmci["target"] == target)].sort_values("lag")
        rows.append(
            {
                "source_index": source,
                "target_index": target,
                "peid_best_lag_weeks": int(best_lag),
                "peid_best_ei": float(best_ei),
                **{f"peid_lag_{lag_weeks}_ei": float(score) for lag_weeks, score in lag_scores},
                "pcmci_lags_weeks": ",".join(str(int(value)) for value in pcmci_rows["lag"].to_list()),
                "pcmci_coefficients": ",".join(f"{float(value):.4g}" for value in pcmci_rows["coefficient"].to_list()),
            }
        )
    return pd.DataFrame(rows)


def draw_arrow(
    ax: plt.Axes,
    source: int,
    target: int,
    weight: float,
    max_weight: float,
    *,
    color: str,
    style: str = "solid",
    rad: float = 0.0,
    label: str | None = None,
    alpha: float = 0.9,
) -> None:
    x1, y1 = NODE_POS[source]
    x2, y2 = NODE_POS[target]
    width = 0.8 + 4.2 * (weight / max_weight)
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12 + 22 * (weight / max_weight),
        linewidth=width,
        linestyle=style,
        color=color,
        alpha=alpha,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=26,
        shrinkB=26,
        zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        ax.text(
            mx,
            my + 0.018,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            color="#344054",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "#d0d5dd", "lw": 0.4},
            zorder=5,
        )


def lag_label(lags: pd.DataFrame, source: int, target: int, fallback_weight: float) -> str:
    match = lags[(lags["source_index"] == source) & (lags["target_index"] == target)]
    if match.empty:
        return f"{fallback_weight:.3f}"
    row = match.iloc[0]
    pcmci = f"; P {row.pcmci_lags_weeks}w" if str(row.pcmci_lags_weeks) else ""
    return f"L{int(row.peid_best_lag_weeks)} {float(row.peid_best_ei):.3f}{pcmci}"


def main() -> None:
    direct = pd.read_csv(RESULT_DIR / "direct_effects.csv")
    pairwise = pd.read_csv(PAIRWISE_DIR / "pairwise_ei_edges.csv")

    direct_edges = direct[
        (direct["source_index"].isin(NODE_POS))
        & (direct["target_index"].isin(NODE_POS))
        & (direct["source_index"] != direct["target_index"])
        & (direct["source_index"].isin([0, 1]) | direct["target_index"].isin([0, 1]))
    ].copy()

    cross = pairwise[
        (pairwise["source_index"].isin([0, 1]))
        & (pairwise["target_index"].isin([0, 1]))
        & (pairwise["source_index"] != pairwise["target_index"])
    ].copy()
    diagnostic_pairs = {(1, 0), (0, 33), (1, 33)}
    diagnostic = pairwise[
        pairwise.apply(lambda row: (int(row["source_index"]), int(row["target_index"])) in diagnostic_pairs, axis=1)
    ].copy()

    lag_pairs = {
        (int(row.source_index), int(row.target_index))
        for row in direct_edges.itertuples(index=False)
    } | {
        (int(row.source_index), int(row.target_index))
        for row in cross.itertuples(index=False)
    } | diagnostic_pairs
    lag_diagnostics = lag_resolved_tm_ei(lag_pairs)
    LAG_DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    lag_diagnostics.to_csv(LAG_DIAGNOSTIC_PATH, index=False)

    max_weight = float(max(direct_edges["direct_effect"].max(), cross["ei"].max(), diagnostic["ei"].max()))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "figure.dpi": 180,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main retained direct EI edges around No.0 and No.1.
    for row in direct_edges.sort_values("direct_effect", ascending=True).itertuples(index=False):
        s = int(row.source_index)
        t = int(row.target_index)
        weight = float(row.direct_effect)
        if s in (0, 1):
            color = "#2563eb" if s == 0 else "#c2410c"
            rad = 0.05 if s == 0 else -0.05
        elif t in (0, 1):
            color = "#475467"
            rad = -0.04 if t == 0 else 0.04
        else:
            color = "#98a2b3"
            rad = 0.0
        label = lag_label(lag_diagnostics, s, t, weight) if weight >= 0.03 or {s, t} & {0, 1, 33} else None
        draw_arrow(ax, s, t, weight, max_weight, color=color, rad=rad, label=label)

    # The pairwise No.0/No.1 relation exists but is too weak to survive top-5
    # sparsification; draw it explicitly as a dotted diagnostic edge.
    for row in cross.itertuples(index=False):
        s = int(row.source_index)
        t = int(row.target_index)
        weight = float(row.ei)
        draw_arrow(
            ax,
            s,
            t,
            weight,
            max_weight,
            color="#7c3aed",
            style=(0, (1.4, 2.2)),
            rad=0.28 if s == 0 else -0.28,
            label="weak " + lag_label(lag_diagnostics, s, t, weight),
            alpha=0.72,
        )

    for row in diagnostic.itertuples(index=False):
        s = int(row.source_index)
        t = int(row.target_index)
        if (s, t) in {(0, 1), (1, 0)}:
            continue
        weight = max(float(row.ei), 0.002)
        draw_arrow(
            ax,
            s,
            t,
            weight,
            max_weight,
            color="#0f766e",
            style=(0, (3.0, 2.5)),
            rad=-0.22 if (s, t) == (1, 33) else 0.16,
            label="diag " + lag_label(lag_diagnostics, s, t, float(row.ei)),
            alpha=0.78,
        )

    for index, (x, y) in NODE_POS.items():
        group = NODE_GROUP.get(index, "readout")
        if group == "enso":
            face = "#fef3c7" if index == 0 else "#fee2e2"
            edge = "#b45309" if index == 0 else "#b91c1c"
            size = 0.070
            lw = 2.0
        elif group == "diagnostic":
            face = "#f1f5f9"
            edge = "#334155"
            size = 0.052
            lw = 1.4
        elif group == "driver":
            face = "#eef2ff"
            edge = "#475467"
            size = 0.050
            lw = 1.2
        else:
            face = "#f8fafc"
            edge = "#98a2b3"
            size = 0.046
            lw = 1.0
        circle = plt.Circle((x, y), size, facecolor=face, edgecolor=edge, linewidth=lw, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, node_label(index), ha="center", va="center", fontsize=8.2, color="#101828", zorder=4)

    ax.text(
        0.03,
        0.965,
        "MLP-TM-EI causal readout around ENSO components",
        ha="left",
        va="top",
        fontsize=13,
        weight="bold",
        color="#101828",
    )
    ax.text(
        0.03,
        0.925,
        "Labels show PEID max lag over the four input weeks; P marks matching PCMCI lag(s). Green dashed edges diagnose the paper's No.1/No.0/No.33 path.",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#475467",
    )
    ax.text(0.22, 0.04, "Blue: No.0 outgoing EI", color="#2563eb", fontsize=8.2, ha="center")
    ax.text(0.50, 0.04, "Orange: No.1 outgoing EI", color="#c2410c", fontsize=8.2, ha="center")
    ax.text(0.78, 0.04, "Gray: incoming EI to ENSO nodes", color="#475467", fontsize=8.2, ha="center")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=240, bbox_inches="tight")
    fig.savefig(LAGGED_OUT_PATH, dpi=240, bbox_inches="tight")
    print(LAGGED_OUT_PATH)
    print(LAG_DIAGNOSTIC_PATH)


if __name__ == "__main__":
    main()
