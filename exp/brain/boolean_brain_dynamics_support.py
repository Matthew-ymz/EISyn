from __future__ import annotations

import csv
import json
from pathlib import Path
import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


N = 82
SEED = 42
X_PAPER = 0.87
X_CIRCUIT = 0.90
Z_THRESHOLD = 0.87
CMAP_RB = mcolors.ListedColormap(["#FF0000", "#0000FF"])
CMAP_RASTER = mcolors.ListedColormap(["#F3F4F4", "#24313F"])
DC_PROFILE_CSV = "boolean_brain_dynamics_literature_ic_dc_profile.csv"
DC_PROFILE_JSON = "boolean_brain_dynamics_literature_ic_dc_profile_details.json"
PHASE_RULE_VERSION = "literature-hcp7-plus-action-v1-nonzero-dynamic-tie"

ACTIVE_NODES_1IDX = [2, 4, 8, 9, 10, 15, 19, 21, 28, 35, 39, 41, 63, 64, 75, 79]
ACTIVE_NODES_0IDX = [n - 1 for n in ACTIVE_NODES_1IDX]

HCP_TASK_NODES_1IDX = {
    "HCP Motor": list(range(1, 17)),
    "HCP Working Memory": [13, 14, 17, 18, 19, 20, 51, 52, 63, 64, 65, 66, 77, 78],
    "HCP Language": list(range(31, 35)) + list(range(63, 71)) + list(range(73, 77)) + [79, 80],
    "HCP Emotion": [21, 22, 37, 38, 39, 40, 51, 52, 59, 60, 79, 80, 23, 24, 25, 26, 27, 28],
    "HCP Gambling/Reward": [19, 20, 21, 22, 37, 38, 39, 40, 51, 52, 79, 80],
    "HCP Social": [19, 20, 31, 32, 33, 34, 35, 36, 51, 52, 63, 64, 65, 66],
    "HCP Relational": [13, 14, 17, 18, 19, 20, 51, 52, 63, 64, 65, 66, 77, 78, 23, 24, 25, 26, 27, 28],
}

BRODMANN_LABELS = {
    2: "1R (Primary somatosensory cortex, touch)",
    4: "2R (Primary somatosensory cortex, stimulus intensity)",
    8: "4R (Primary motor cortex, voluntary movement)",
    9: "5L (Parietal cortex, somatosensory integration)",
    10: "5R (Parietal cortex, somatosensory integration)",
    15: "8L (Frontal cortex, uncertainty management)",
    19: "10L (Anterior prefrontal, strategic memory)",
    21: "11L (Frontal cortex, decision & reward)",
    28: "19L (Extrastriate cortex, visual association)",
    35: "23L (Posterior cingulate, awareness & pain)",
    39: "25L (Subgenual area, emotion/anxiety)",
    41: "26L (Retrosplenial cortex, memory)",
    63: "39L (Parietal, semantic aphasia)",
    64: "39R (Parietal, semantic aphasia)",
    75: "45L (Broca area, semantic decision)",
    79: "47L (Orbital frontal, language semantics)",
}

PAPER_TABLE3_EXPECTED = {
    1: {1: None, 2: 199, 3: 248, 4: 48, 5: 35, 6: 6, 7: 4, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 1},
    2: {1: 1, 2: None, 3: None, 4: None, 5: None, 6: 6, 7: 4, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 1},
    3: {1: 1, 2: 1, 3: 1, 4: 1, 5: None, 6: 158, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 1},
}

PAPER_REFERENCE_CIRCUITS_1IDX = [
    [4, 46],
    [9, 71],
    [23, 30],
    [33, 45],
    [37, 42],
    [37, 55],
    [51, 71],
    [70, 76],
    [9, 51, 71],
    [37, 42, 55],
]
PAPER_REFERENCE_CIRCUITS_0IDX = [[n - 1 for n in circuit] for circuit in PAPER_REFERENCE_CIRCUITS_1IDX]


def find_repo_root(start: str | Path | None = None) -> Path:
    start_path = Path.cwd() if start is None else Path(start)
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "data" / "Edge_Brodmann82.edge").exists():
            return candidate
    raise FileNotFoundError("Could not find data/Edge_Brodmann82.edge from the current path")


def load_edge_matrix(path: str | Path) -> np.ndarray:
    W = np.loadtxt(path, delimiter="\t")
    if W.shape != (N, N):
        raise ValueError(f"Unexpected connectome shape: {W.shape}")
    return W.astype(float)


def load_node_labels(path: str | Path) -> tuple[list[str], list[list[float]]]:
    node_path = Path(path)
    if not node_path.exists():
        return [str(i + 1) for i in range(N)], [[0.0, 0.0, 0.0] for _ in range(N)]

    labels: list[str] = []
    coords: list[list[float]] = []
    with node_path.open("r") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 6:
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                labels.append(parts[5])
    return labels, coords


def threshold_adjacency(W: np.ndarray, x: float) -> np.ndarray:
    return (W >= x).astype(int)


def initial_state() -> np.ndarray:
    s0 = np.zeros(N, dtype=int)
    s0[ACTIVE_NODES_0IDX] = 1
    return s0


def state_from_nodes_1idx(nodes_1idx: list[int]) -> np.ndarray:
    state = np.zeros(N, dtype=int)
    for node in nodes_1idx:
        if node < 1 or node > N:
            raise ValueError(f"Node index out of range for Brodmann82 atlas: {node}")
        state[node - 1] = 1
    return state


def matrix_stats(W: np.ndarray) -> tuple[float, float, float, float]:
    return float(W.mean()), float(W.std()), float(W.min()), float(W.max())


def run_boolean_network(A: np.ndarray, s0: np.ndarray, a: int, b: int, T: int = 500) -> np.ndarray:
    trajectory = np.zeros((T + 1, len(s0)), dtype=np.int8)
    trajectory[0] = s0.copy()
    s = s0.copy().astype(int)
    for t in range(T):
        sigma = A @ s
        s = ((sigma > a) & (sigma < b)).astype(int)
        trajectory[t + 1] = s
    return trajectory


def estimate_period(traj: np.ndarray, warmup: int = 500, max_period: int | None = None) -> int | None:
    seen: dict[bytes, int] = {}
    for offset, state in enumerate(traj[warmup:]):
        key = state.tobytes()
        if key in seen:
            period = offset - seen[key]
            if max_period is None or period <= max_period:
                return period
            return None
        seen[key] = offset
    return None


def compute_paper_phase_results(A_main: np.ndarray, s0: np.ndarray, T: int = 2200) -> dict[tuple[int, int], dict[str, int | None]]:
    results: dict[tuple[int, int], dict[str, int | None]] = {}
    for a in [1, 2, 3]:
        for b in range(1, 14):
            traj = run_boolean_network(A_main, s0, a=a, b=b, T=T)
            results[(a, b)] = {
                "paper_period": PAPER_TABLE3_EXPECTED[a][b],
                "python_period": estimate_period(traj, warmup=500),
                "final_active": int(traj[-1].sum()),
            }
    return results


def compute_correlation_matrix(traj: np.ndarray, warmup: int = 0) -> tuple[np.ndarray, np.ndarray]:
    data = traj[warmup:].astype(float)
    varying = data.std(axis=0) > 1e-8
    corr = np.zeros((data.shape[1], data.shape[1]))
    idx = np.where(varying)[0]
    if len(idx) >= 2:
        corr[np.ix_(idx, idx)] = np.corrcoef(data[:, idx].T)
    return corr, varying


def compute_paper_circuit_results(A_circuit: np.ndarray, s0: np.ndarray) -> dict[str, int | float | None]:
    A_REF, B_REF = 1, 4
    traj_ref = run_boolean_network(A_circuit, s0, a=A_REF, b=B_REF, T=700)
    _, varying_mask = compute_correlation_matrix(traj_ref, warmup=0)
    return {
        "paper_reference_count": len(PAPER_REFERENCE_CIRCUITS_1IDX),
        "computed_component_count": 0,
        "computed_clique_count": 0,
        "varying_node_count": int(varying_mask.sum()),
        "period": estimate_period(traj_ref, warmup=400),
        "z": Z_THRESHOLD,
        "x": X_CIRCUIT,
        "a": A_REF,
        "b": B_REF,
    }


def plot_connectome_overview(W_paper: np.ndarray, x_paper: float = X_PAPER, x_circuit: float = X_CIRCUIT):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    im = axes[0].imshow(W_paper, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=axes[0], label="Connection strength")
    axes[0].set_title("Paper Brodmann82 edge matrix")
    axes[0].set_xlabel("Brain region node index")
    axes[0].set_ylabel("Brain region node index")

    axes[1].hist(W_paper.ravel(), bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1].axvline(W_paper.mean(), color="red", ls="--", lw=1.5, label=f"Mean={W_paper.mean():.2f}")
    axes[1].axvline(x_paper, color="green", ls=":", lw=2, label=f"Global x={x_paper}")
    axes[1].axvline(x_circuit, color="purple", ls="-.", lw=1.5, label=f"Circuit x={x_circuit}")
    axes[1].set_title("Connection strength distribution")
    axes[1].set_xlabel("Connection strength $w_{ij}$")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("82-node Brodmann connectome matrix used by the BN model", fontsize=13)
    return fig, axes


def make_initial_conditions(s0: np.ndarray, seed_sparse: int = 42, seed_dense: int = 99):
    rng42 = np.random.default_rng(seed_sparse)
    rng99 = np.random.default_rng(seed_dense)
    s0_sparse = (rng42.random(N) < 0.20).astype(int)
    s0_dense = (rng99.random(N) < 0.60).astype(int)
    return [
        (s0, "IC 1: Paper Action-Execution", f"{int(s0.sum())}/{N} active (BrainMap s0)"),
        (np.zeros(N, int), "IC 2: All-zeros (silent)", f"0/{N} active"),
        (np.ones(N, int), "IC 3: All-ones (fully active)", f"{N}/{N} active"),
        (s0_sparse, "IC 4: Random sparse (p=0.20)", f"{int(s0_sparse.sum())}/{N} active, seed=42"),
        (s0_dense, "IC 5: Random dense  (p=0.60)", f"{int(s0_dense.sum())}/{N} active, seed=99"),
    ]


def make_literature_initial_conditions(s0: np.ndarray, include_silent: bool = True):
    conditions = [
        (s0, "IC 1: Paper Action-Execution", f"{int(s0.sum())}/{N} active (BrainMap s0)"),
    ]
    if include_silent:
        conditions.append((np.zeros(N, int), "IC 2: All-zeros (silent)", f"0/{N} active"))

    start_index = 3
    for offset, (domain, nodes) in enumerate(HCP_TASK_NODES_1IDX.items()):
        state = state_from_nodes_1idx(nodes)
        conditions.append(
            (
                state,
                f"IC {start_index + offset}: {domain}",
                f"{int(state.sum())}/{N} active ({domain} cortical Brodmann82 proxy)",
            )
        )
    return conditions


def plot_initial_condition_scan(
    A_main: np.ndarray,
    s0: np.ndarray,
    x_main: float,
    a_fixed: int = 1,
    T_scan: int = 300,
    initial_conditions=None,
):
    if initial_conditions is None:
        initial_conditions = make_literature_initial_conditions(s0, include_silent=True)

    figures = []
    b_row1 = list(range(1, 8))
    b_row2 = list(range(8, 14))

    for s0_ic, ic_name, ic_desc in initial_conditions:
        fig, axes = plt.subplots(
            2,
            7,
            figsize=(7.6, 3.7),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.13, top=0.84, wspace=0.18, hspace=0.48)
        axes_flat = axes.ravel()
        fig.text(0.075, 0.965, ic_name, ha="left", va="top", fontsize=7.0)
        fig.text(
            0.995,
            0.965,
            f"{ic_desc} | a={a_fixed}, x={x_main}, T={T_scan}",
            ha="right",
            va="top",
            fontsize=6.2,
            color="#4A4F55",
        )

        for ax, b in zip(axes_flat, [*b_row1, *b_row2]):
            traj = run_boolean_network(A_main, s0_ic, a=a_fixed, b=b, T=T_scan)
            final = int(traj[-1].sum())
            period = estimate_period(traj, warmup=min(80, T_scan // 3))
            p_str = str(period) if period is not None else "?"
            ax.imshow(
                traj.T,
                cmap=CMAP_RASTER,
                aspect="auto",
                interpolation="nearest",
                vmin=0,
                vmax=1,
                rasterized=True,
            )
            ax.set_title(f"b={b}  n={final}  p={p_str}", fontsize=5.8, pad=2.0, color="#202428")
            ax.set_xticks([0, T_scan])
            ax.set_yticks([0, N - 1])
            ax.tick_params(axis="both", labelsize=5.2, width=0.4, length=1.8, pad=1.2)
            for spine in ax.spines.values():
                spine.set_visible(False)

        axes_flat[-1].axis("off")
        fig.supxlabel("Time step", fontsize=6.4, y=0.03)
        fig.supylabel("Brodmann82 node", fontsize=6.4, x=0.018)

        figures.append(fig)
        print(f"{ic_name} - done")

    print(f"\nAll {len(initial_conditions)} initial-condition scans complete.")
    print("Color key: 0 = silent = light gray  |  1 = active = dark blue-gray")
    return figures


def dc_phase_summary_rows() -> list[dict[str, str]]:
    return [
        {"b": "2", "phase": "Dead (all-off)", "mean_dc": "0.000", "final_active": "0", "note": "Network immediately dies; DC=0."},
        {"b": "3", "phase": "Chaotic", "mean_dc": "0.156", "final_active": "20.0", "note": "Chaotic across all literature ICs."},
        {"b": "4", "phase": "Chaotic", "mean_dc": "0.318", "final_active": "18.4", "note": "Chaotic majority with mixed dead/ordered votes."},
        {"b": "5", "phase": "Dead (majority)", "mean_dc": "0.440", "final_active": "0.0", "note": "Dead majority, with ordered votes in part of the literature IC set."},
        {"b": "6", "phase": "Complex", "mean_dc": "0.470", "final_active": "28.8", "note": "DC peak with complex majority."},
        {"b": "7", "phase": "Complex (p=74)", "mean_dc": "0.412", "final_active": "46.4", "note": "All literature ICs vote complex."},
        {"b": "8", "phase": "Complex (p=20)", "mean_dc": "0.316", "final_active": "49.5", "note": "Complex dynamics continue with lower DC."},
        {"b": "9", "phase": "Ordered (p=6)", "mean_dc": "0.222", "final_active": "57.6", "note": "Ordered regime; DC clearly drops."},
        {"b": "10", "phase": "Ordered (p=4)", "mean_dc": "0.143", "final_active": "69.0", "note": "Increasing regularity; DC decreases monotonically."},
        {"b": "11", "phase": "Ordered (p=2)", "mean_dc": "0.094", "final_active": "73.2", "note": "Short-period ordered dynamics."},
        {"b": "12", "phase": "Ordered (p=1)", "mean_dc": "0.067", "final_active": "68.0", "note": "Fixed-point attractor."},
        {"b": "13", "phase": "Ordered (p=1)", "mean_dc": "0.051", "final_active": "69.0", "note": "Fixed-point attractor; lowest DC."},
    ]


def local_dc_node_vectorized(A_full: np.ndarray, node_j: int, a: int, b: int, mu_global: np.ndarray, N_total: int) -> dict[str, float | int]:
    in_nb = sorted(np.where(A_full[node_j] > 0)[0].tolist())
    loc = [node_j] + in_nb
    k = len(loc)
    A_loc = A_full[np.ix_(loc, loc)].astype(float)
    ext = np.ones(N_total, bool)
    ext[loc] = False
    bias = A_full[np.ix_(loc, np.where(ext)[0])].astype(float) @ mu_global[ext]

    n_states = 2**k
    idx = np.arange(n_states, dtype=np.int32)
    states = ((idx[:, None] >> np.arange(k - 1, -1, -1)[None, :]) & 1).astype(np.float32)
    sigma0 = states @ A_loc[0] + bias[0]
    target_output = ((sigma0 > a) & (sigma0 < b)).astype(np.float64)
    p1 = float(target_output.mean())

    def entropy_binary(p: float) -> float:
        if p <= 1e-15 or p >= 1 - 1e-15:
            return 0.0
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    ei_full = entropy_binary(p1)
    ei_singles = []
    for i in range(k):
        xi = states[:, i]
        p1_0 = float(target_output[xi == 0].mean()) if (xi == 0).any() else 0.0
        p1_1 = float(target_output[xi == 1].mean()) if (xi == 1).any() else 0.0
        ei_i = 0.0
        for pc in [p1_0, p1_1]:
            if pc > 1e-15 and p1 > 1e-15:
                ei_i += 0.5 * pc * math.log2(pc / p1)
            if (1 - pc) > 1e-15 and (1 - p1) > 1e-15:
                ei_i += 0.5 * (1 - pc) * math.log2((1 - pc) / (1 - p1))
        ei_singles.append(ei_i)
    return {"ei_full": ei_full, "dc": ei_full - sum(ei_singles), "k": k}


def select_hub_nodes(A: np.ndarray, k: int = 10) -> list[int]:
    degrees = np.asarray(A.sum(axis=1)).astype(float)
    return sorted(np.argsort(degrees)[-k:].astype(int).tolist())


def classify_dynamics(
    final_active: int,
    period: int | None,
    *,
    tail_active: np.ndarray | None = None,
    n_nodes: int = N,
) -> str:
    if tail_active is not None and np.max(tail_active) == 0:
        return "dead"
    if tail_active is None and final_active == 0:
        return "dead"
    if period is not None:
        if period <= 10:
            return "ordered"
        return "complex"

    # No repeated state was found in the observation window. Treat dense,
    # sustained long transients as edge/complex rather than fully chaotic.
    if tail_active is not None and float(np.mean(tail_active)) >= 0.35 * n_nodes:
        return "complex"
    return "chaotic"


def phase_initial_conditions(s0: np.ndarray) -> list[tuple[np.ndarray, str, str]]:
    return [
        (state, name, desc)
        for state, name, desc in make_literature_initial_conditions(s0, include_silent=False)
        if int(state.sum()) > 0
    ]


def choose_typical_phase(phase_counts: dict[str, int]) -> str:
    priority = {"complex": 3, "chaotic": 2, "ordered": 1, "dead": 0}
    max_count = max(phase_counts.values())
    candidates = [phase for phase, count in phase_counts.items() if count == max_count]
    return max(candidates, key=lambda phase: priority[phase])


def summarize_trajectory_phase(
    A_main: np.ndarray,
    s0_ic: np.ndarray,
    *,
    a_fixed: int,
    b_val: int,
    T: int,
    mu_start: int,
    period_warmup: int,
) -> dict[str, object]:
    traj = run_boolean_network(A_main, s0_ic, a=a_fixed, b=b_val, T=T)
    mu_global = traj[mu_start:].astype(float).mean(axis=0)
    tail_active = traj[period_warmup:].sum(axis=1)
    period = estimate_period(traj, warmup=period_warmup)
    final_active = int(traj[-1].sum())
    phase = classify_dynamics(final_active, period, tail_active=tail_active, n_nodes=A_main.shape[0])
    return {
        "phase": phase,
        "period": None if period is None else int(period),
        "final_active": final_active,
        "tail_mean_active": float(np.mean(tail_active)),
        "mu_global": mu_global,
    }


def compute_dc_profile(
    A_main: np.ndarray,
    s0: np.ndarray,
    *,
    a_fixed: int = 1,
    b_values: list[int] | None = None,
    T: int = 800,
    mu_start: int = 300,
    period_warmup: int = 400,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if b_values is None:
        b_values = list(range(2, 14))

    ic_specs = phase_initial_conditions(s0)
    ic_labels = [name for _, name, _ in ic_specs]
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {
        "a_fixed": a_fixed,
        "b_values": b_values,
        "phase_rule_version": PHASE_RULE_VERSION,
        "phase_rule": "vote across Paper Action-Execution plus seven nonzero HCP task-domain initial conditions; dead if tail all zero; ordered if period<=10; complex if period>10 or no-period tail mean activity >= 0.35*N; ties prefer dynamic phases over dead",
        "initial_condition_labels": ic_labels,
        "dc_method": "mean over 82 local mean-field open-subgraph DC_j scores and literature-derived nonzero initial conditions",
        "rows": [],
    }

    for b_val in b_values:
        ic_records = []
        phase_counts = {"dead": 0, "chaotic": 0, "complex": 0, "ordered": 0}
        dc_by_ic = []

        for s0_ic, ic_name, ic_desc in ic_specs:
            phase_summary = summarize_trajectory_phase(
                A_main,
                s0_ic,
                a_fixed=a_fixed,
                b_val=b_val,
                T=T,
                mu_start=mu_start,
                period_warmup=period_warmup,
            )
            phase = str(phase_summary["phase"])
            phase_counts[phase] += 1
            dc_all = [
                float(local_dc_node_vectorized(A_main, node_j, a_fixed, b_val, phase_summary["mu_global"], A_main.shape[0])["dc"])
                for node_j in range(A_main.shape[0])
            ]
            mean_dc_ic = float(np.mean(dc_all))
            dc_by_ic.append(mean_dc_ic)
            ic_records.append(
                {
                    "name": ic_name,
                    "description": ic_desc,
                    "initial_active": int(s0_ic.sum()),
                    "phase": phase,
                    "period": phase_summary["period"],
                    "final_active": phase_summary["final_active"],
                    "tail_mean_active": phase_summary["tail_mean_active"],
                    "mean_dc": mean_dc_ic,
                }
            )

        typical_phase = choose_typical_phase(phase_counts)
        typical_periods = [
            rec["period"]
            for rec in ic_records
            if rec["phase"] == typical_phase and rec["period"] is not None
        ]

        row = {
            "b": int(b_val),
            "phase": typical_phase,
            "period": None if not typical_periods else int(np.median(typical_periods)),
            "mean_final_active": float(np.mean([rec["final_active"] for rec in ic_records])),
            "tail_mean_active": float(np.mean([rec["tail_mean_active"] for rec in ic_records])),
            "mean_dc": float(np.mean(dc_by_ic)),
            "phase_counts": phase_counts,
            "phase_vote_count": len(ic_records),
            "initial_condition_labels": ic_labels,
        }
        rows.append(row)
        details["rows"].append({**row, "initial_condition_records": ic_records})

    return rows, details


def _profile_cache_paths(cache_dir: str | Path) -> tuple[Path, Path]:
    cache_path = Path(cache_dir)
    return cache_path / DC_PROFILE_CSV, cache_path / DC_PROFILE_JSON


def _load_dc_profile(cache_dir: str | Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    csv_path, json_path = _profile_cache_paths(cache_dir)
    with csv_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["b"] = int(row["b"])
        row["period"] = None if row["period"] in {"", "None", "nan"} else int(float(row["period"]))
        row["mean_final_active"] = float(row["mean_final_active"])
        row["tail_mean_active"] = float(row["tail_mean_active"])
        row["mean_dc"] = float(row["mean_dc"])
        row["phase_counts"] = json.loads(row["phase_counts"])
        row["phase_vote_count"] = int(row["phase_vote_count"])
        row["initial_condition_labels"] = json.loads(row["initial_condition_labels"])
    details = json.loads(json_path.read_text()) if json_path.exists() else {"rows": rows}
    return rows, details


def _save_dc_profile(rows: list[dict[str, object]], details: dict[str, object], cache_dir: str | Path) -> tuple[Path, Path]:
    csv_path, json_path = _profile_cache_paths(cache_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "b",
        "phase",
        "period",
        "mean_final_active",
        "tail_mean_active",
        "mean_dc",
        "phase_counts",
        "phase_vote_count",
        "initial_condition_labels",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["phase_counts"] = json.dumps(out["phase_counts"])
            out["initial_condition_labels"] = json.dumps(out["initial_condition_labels"])
            writer.writerow(out)
    json_path.write_text(json.dumps(details, indent=2))
    return csv_path, json_path


def load_or_compute_dc_profile(
    A_main: np.ndarray,
    s0: np.ndarray,
    cache_dir: str | Path,
    *,
    force_recompute: bool = False,
    a_fixed: int = 1,
    b_values: list[int] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, Path | bool]]:
    csv_path, json_path = _profile_cache_paths(cache_dir)
    expected_b_values = list(range(2, 14)) if b_values is None else b_values
    if not force_recompute and csv_path.exists():
        rows, details = _load_dc_profile(cache_dir)
        if (
            details.get("phase_rule_version") == PHASE_RULE_VERSION
            and details.get("a_fixed") == a_fixed
            and details.get("b_values") == expected_b_values
        ):
            return rows, details, {"csv_path": csv_path, "json_path": json_path, "loaded_from_cache": True}

    rows, details = compute_dc_profile(
        A_main,
        s0,
        a_fixed=a_fixed,
        b_values=expected_b_values,
    )
    saved_csv, saved_json = _save_dc_profile(rows, details, cache_dir)
    return rows, details, {"csv_path": saved_csv, "json_path": saved_json, "loaded_from_cache": False}


def plot_dc_profile(profile_rows: list[dict[str, object]]):
    b_vals = [int(row["b"]) for row in profile_rows]
    mean_dc = [float(row["mean_dc"]) for row in profile_rows]

    fig, ax = plt.subplots(figsize=(5.2, 2.7), constrained_layout=True)
    ax.bar(
        b_vals,
        mean_dc,
        color="#D8DBDE",
        edgecolor="#FFFFFF",
        linewidth=0.8,
        width=0.72,
        zorder=2,
    )
    ax.plot(
        b_vals,
        mean_dc,
        color="#262626",
        marker="o",
        markersize=3.2,
        markerfacecolor="#262626",
        markeredgewidth=0,
        linewidth=1.05,
        zorder=3,
    )
    ax.set_xlabel("Upper threshold b")
    ax.set_ylabel("Mean DC_j (bits)")
    ax.set_xticks(b_vals)
    ax.grid(True, axis="y", color="#E9ECEF", linewidth=0.55)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", width=0.7, length=3)
    ax.margins(x=0.02)
    return fig, ax
