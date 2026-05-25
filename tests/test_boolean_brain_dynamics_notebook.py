import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exp.brain import boolean_brain_dynamics_support as brain_bn


def execute_notebook(notebook_path: Path) -> dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}_cell_{index}", "exec"), namespace, namespace)
        if "plt" in namespace:
            namespace["plt"].close("all")
    return namespace


def test_boolean_brain_notebook_exposes_paper_aligned_reproduction():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "brain" / "boolean_brain_dynamics.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert "W_paper" in notebook_text
    assert "A_paper" in notebook_text
    assert "X_PAPER = 0.87" in notebook_text
    assert "X_CIRCUIT = 0.90" in notebook_text
    assert "A_main = A_paper" in notebook_text
    assert "A_circuit = threshold_adjacency(W_paper, X_CIRCUIT)" in notebook_text
    assert "PAPER_TABLE3_EXPECTED" in notebook_text
    assert "PAPER_REFERENCE_CIRCUITS_1IDX" in notebook_text
    assert "paper_phase_results" in notebook_text
    assert "paper_circuit_results" in notebook_text
    assert "dc_profile_df" in notebook_text
    assert "mean_dc" in notebook_text
    assert "phase_counts" in notebook_text
    assert "typical phase" in notebook_text
    assert "boolean_brain_dynamics_literature_ic_dc_profile.csv" in notebook_text
    assert "A_EXPERIMENT = brain_bn.A_EXPERIMENT" in notebook_text
    assert "A_EXPERIMENT = 0" not in notebook_text
    assert "a_fixed=A_EXPERIMENT" in notebook_text
    assert "Phi^EID" not in notebook_text
    assert "phi_eid" not in notebook_text
    assert "hub mean-field open subgraph" not in notebook_text


def test_boolean_brain_notebook_removes_calibrated_branch():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "brain" / "boolean_brain_dynamics.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert "W_sym" not in notebook_text
    assert "X_EDGE" not in notebook_text
    assert "校准" not in notebook_text
    assert "calibrated" not in notebook_text.lower()
    assert "对称化" not in notebook_text


def test_boolean_brain_notebook_runs_paper_parameters_without_extra_processing():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "brain" / "boolean_brain_dynamics.ipynb"
    namespace = execute_notebook(notebook_path)

    phase = namespace["paper_phase_results"]
    circuits = namespace["paper_circuit_results"]

    assert namespace["X_MAIN"] == 0.87
    assert namespace["X_CIRCUIT"] == 0.90
    assert namespace["A_EXPERIMENT"] == brain_bn.A_EXPERIMENT
    assert namespace["A_MAIN_LABEL"] == "paper raw x=0.87"
    assert namespace["A_CIRCUIT_LABEL"] == "paper raw x=0.90"
    assert phase[(1, 4)]["paper_period"] == 48
    assert circuits["paper_reference_count"] == 10
    assert circuits["x"] == 0.90
    assert circuits["a"] == 1
    assert circuits["b"] == 4
    assert circuits["z"] == 0.87


def test_boolean_brain_notebook_computes_dc_profile():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "brain" / "boolean_brain_dynamics.ipynb"
    namespace = execute_notebook(notebook_path)

    profile = namespace["dc_profile_df"]
    assert list(profile["b"]) == list(range(2, 14))
    assert np.isfinite(profile["mean_dc"].to_numpy(dtype=float)).all()
    assert "phase_counts" in profile.columns
    assert "loaded_from_cache" in namespace["dc_cache_info"]
    assert namespace["dc_profile_details"]["a_fixed"] == brain_bn.A_EXPERIMENT


def test_boolean_brain_notebook_adds_node_dc_topology_analysis():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "brain" / "boolean_brain_dynamics.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert "node_dc_topology_df" in notebook_text
    assert "dc_topology_corr_df" in notebook_text
    assert "plot_node_dc_network" in notebook_text
    assert "boolean_brain_node_dc_topology_b6.csv" in notebook_text
    assert "boolean_brain_node_dc_network_b6.png" in notebook_text
    assert "Spearman" in notebook_text


def test_boolean_brain_support_defines_literature_initial_conditions():
    s0 = brain_bn.initial_state()

    initial_conditions = brain_bn.make_literature_initial_conditions(s0, include_silent=True)
    labels = [label for _, label, _ in initial_conditions]
    active_counts = [int(state.sum()) for state, _, _ in initial_conditions]

    assert labels == [
        "IC 1: Paper Action-Execution",
        "IC 2: All-zeros (silent)",
        "IC 3: HCP Motor",
        "IC 4: HCP Working Memory",
        "IC 5: HCP Language",
        "IC 6: HCP Emotion",
        "IC 7: HCP Gambling/Reward",
        "IC 8: HCP Social",
        "IC 9: HCP Relational",
    ]
    assert active_counts == [16, 0, 16, 14, 18, 18, 12, 14, 20]
    assert np.array_equal(initial_conditions[0][0], s0)
    assert brain_bn.DC_PROFILE_CSV == "boolean_brain_dynamics_literature_ic_dc_profile.csv"


def test_boolean_brain_phase_initial_conditions_use_action_plus_hcp_domains():
    s0 = brain_bn.initial_state()

    labels = [label for _, label, _ in brain_bn.phase_initial_conditions(s0)]

    assert labels == [
        "IC 1: Paper Action-Execution",
        "IC 3: HCP Motor",
        "IC 4: HCP Working Memory",
        "IC 5: HCP Language",
        "IC 6: HCP Emotion",
        "IC 7: HCP Gambling/Reward",
        "IC 8: HCP Social",
        "IC 9: HCP Relational",
    ]


def test_boolean_brain_dc_profile_plot_is_neutral_and_title_free():
    rows = [
        {"b": 2, "mean_dc": 0.0, "phase": "dead"},
        {"b": 3, "mean_dc": 0.15, "phase": "chaotic"},
        {"b": 6, "mean_dc": 0.47, "phase": "complex"},
        {"b": 9, "mean_dc": 0.22, "phase": "ordered"},
    ]

    fig, ax = brain_bn.plot_dc_profile(rows)
    bar_colors = {patch.get_facecolor() for patch in ax.patches}

    assert fig._suptitle is None
    assert ax.get_title() == ""
    assert ax.get_legend() is None
    assert len(bar_colors) == 1
    assert ax.get_xlabel() == "Upper threshold b"
    assert ax.get_ylabel() == "Mean DC_j (bits)"


def test_boolean_brain_node_topology_table_uses_adjacency_direction():
    A = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=int,
    )
    W = A.astype(float)
    dc_values = np.array([0.4, 0.1, 0.3, 0.2])

    table = brain_bn.build_node_dc_topology_table(
        A,
        W,
        dc_values,
        node_labels=["n1", "n2", "n3", "n4"],
        node_coords=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    )

    node1 = table.loc[table["node"] == 1].iloc[0]
    node2 = table.loc[table["node"] == 2].iloc[0]
    node3 = table.loc[table["node"] == 3].iloc[0]

    assert list(table["node"]) == [1, 2, 3, 4]
    assert node1["label"] == "n1"
    assert node1["dc"] == 0.4
    assert node1["in_degree"] == 0
    assert node1["out_degree"] == 2
    assert node2["in_degree"] == 2
    assert node2["out_degree"] == 0
    assert node3["in_degree"] == 1
    assert node3["out_degree"] == 2
    assert {"betweenness", "pagerank", "clustering", "weighted_in_strength"}.issubset(table.columns)


def test_boolean_brain_dc_topology_correlations_are_spearman_ranked():
    table = pd.DataFrame(
        {
            "node": [1, 2, 3, 4, 5],
            "dc": [0.1, 0.2, 0.4, 0.8, 1.6],
            "in_degree": [1, 2, 3, 4, 5],
            "out_degree": [5, 4, 3, 2, 1],
            "constant": [1, 1, 1, 1, 1],
        }
    )

    corr = brain_bn.compute_dc_topology_correlations(table, metric_cols=["in_degree", "out_degree", "constant"])

    assert list(corr["metric"])[:2] == ["in_degree", "out_degree"]
    assert corr.loc[corr["metric"] == "in_degree", "spearman_r"].iloc[0] == 1.0
    assert corr.loc[corr["metric"] == "out_degree", "spearman_r"].iloc[0] == -1.0
    assert "constant" not in set(corr["metric"])
    assert (corr["n"] == 5).all()


def test_boolean_brain_node_dc_network_plot_has_external_colorbar_and_no_title():
    A = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=int,
    )
    table = brain_bn.build_node_dc_topology_table(A, A.astype(float), np.array([0.4, 0.1, 0.3, 0.2]))

    fig, ax = brain_bn.plot_node_dc_network(A, table, colorbar_label="DC_j")

    assert fig._suptitle is None
    assert ax.get_title() == ""
    assert not ax.axison
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "DC_j"


def test_boolean_brain_node_topology_table_allows_self_loops():
    A = np.array([[1, 0], [1, 0]], dtype=int)

    table = brain_bn.build_node_dc_topology_table(A, A.astype(float), np.array([0.2, 0.3]))

    assert list(table["in_degree"]) == [1, 1]
    assert list(table["out_degree"]) == [2, 0]
    assert np.isfinite(table[["betweenness", "pagerank", "clustering", "core_number"]].to_numpy(dtype=float)).all()


def test_boolean_brain_initial_condition_scan_plot_is_compact_and_neutral():
    A = np.zeros((brain_bn.N, brain_bn.N), dtype=int)
    s0 = brain_bn.initial_state()

    figures = brain_bn.plot_initial_condition_scan(
        A,
        s0,
        x_main=0.87,
        a_fixed=2,
        T_scan=8,
        initial_conditions=[(s0, "IC 1: Paper Action-Execution", "16/82 active")],
    )
    fig = figures[0]
    width, height = fig.get_size_inches()
    images = [ax.images[0] for ax in fig.axes if ax.images]
    title_text = " ".join(ax.get_title() for ax in fig.axes)
    figure_text = " ".join(text.get_text() for text in fig.texts)

    assert fig._suptitle is None
    assert width <= 8.0
    assert height <= 4.0
    assert len(images) == 13
    assert "#FF0000" not in images[0].get_cmap().colors
    assert "#0000FF" not in images[0].get_cmap().colors
    assert "final:" not in title_text
    assert "period:" not in title_text
    assert "a=2" in figure_text


def test_boolean_brain_dc_cache_depends_on_a_fixed(tmp_path, monkeypatch):
    csv_path = tmp_path / brain_bn.DC_PROFILE_CSV
    json_path = tmp_path / brain_bn.DC_PROFILE_JSON
    csv_path.write_text(
        "b,phase,period,mean_final_active,tail_mean_active,mean_dc,phase_counts,phase_vote_count,initial_condition_labels\n"
        '2,dead,1,0,0,0,"{""dead"": 8, ""chaotic"": 0, ""complex"": 0, ""ordered"": 0}",8,"[""old""]"\n'
    )
    json_path.write_text(
        json.dumps(
            {
                "phase_rule_version": brain_bn.PHASE_RULE_VERSION,
                "a_fixed": 1,
                "b_values": [2],
                "rows": [],
            }
        )
    )

    def fake_compute_dc_profile(A_main, s0, *, a_fixed=1, b_values=None):
        return (
            [
                {
                    "b": 2,
                    "phase": "dead",
                    "period": 1,
                    "mean_final_active": 0.0,
                    "tail_mean_active": 0.0,
                    "mean_dc": 0.0,
                    "phase_counts": {"dead": 1, "chaotic": 0, "complex": 0, "ordered": 0},
                    "phase_vote_count": 1,
                    "initial_condition_labels": ["new"],
                }
            ],
            {
                "phase_rule_version": brain_bn.PHASE_RULE_VERSION,
                "a_fixed": a_fixed,
                "b_values": b_values,
                "initial_condition_labels": ["new"],
                "rows": [],
            },
        )

    monkeypatch.setattr(brain_bn, "compute_dc_profile", fake_compute_dc_profile)

    _, details, cache_info = brain_bn.load_or_compute_dc_profile(
        np.zeros((brain_bn.N, brain_bn.N), dtype=int),
        brain_bn.initial_state(),
        tmp_path,
        a_fixed=2,
        b_values=[2],
    )

    assert cache_info["loaded_from_cache"] is False
    assert details["a_fixed"] == 2


def test_boolean_brain_profile_uses_typical_phase_across_initial_conditions():
    notebook_path = Path(__file__).resolve().parents[1] / "exp" / "brain" / "boolean_brain_dynamics.ipynb"
    namespace = execute_notebook(notebook_path)

    profile = namespace["dc_profile_df"]
    row_b6 = profile.loc[profile["b"] == 6].iloc[0]
    row_b5 = profile.loc[profile["b"] == 5].iloc[0]
    peak = profile.loc[profile["mean_dc"].idxmax()]
    expected_labels = [
        "IC 1: Paper Action-Execution",
        "IC 3: HCP Motor",
        "IC 4: HCP Working Memory",
        "IC 5: HCP Language",
        "IC 6: HCP Emotion",
        "IC 7: HCP Gambling/Reward",
        "IC 8: HCP Social",
        "IC 9: HCP Relational",
    ]

    assert row_b5["initial_condition_labels"] == expected_labels
    assert row_b5["phase_vote_count"] == len(expected_labels)
    assert row_b5["phase"] == "dead"
    assert row_b5["phase_counts"]["dead"] > row_b5["phase_counts"]["ordered"]
    assert row_b6["phase"] == "complex"
    assert row_b6["phase_counts"]["complex"] > row_b6["phase_counts"]["ordered"]
    assert peak["b"] == 6
    assert peak["phase"] == "complex"
