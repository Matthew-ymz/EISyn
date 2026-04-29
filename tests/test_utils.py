import json
import re
import unittest
import subprocess
import sys
import math
from unittest.mock import patch
from pathlib import Path

import numpy as np
import utils as utils_module

from utils import (
    build_deterministic_boolean_tpm,
    build_marshall_example1_tpm,
    build_probabilistic_boolean_tpm,
    coarse_grain_binary_or_tpm,
    discrete_integrated_information,
    dimension_averaged_causal_emergence_for_dynamics,
    dimension_averaged_effective_information_for_dynamics,
    enumerate_surjective_binary_mappings,
    find_discrete_complexes,
    effective_information_from_tpm,
    enumerate_partitions_fixed_blocks,
    enumerate_pair_partitions,
    enumerate_binary_states,
    gaussian_conditional_covariance,
    gaussian_mutual_information,
    jacobian_uniform_box_total_synergy_for_dynamics,
    joint_ei_decomposition,
    linear_gaussian_ei_decomposition,
    linear_gaussian_mediano_metrics,
    linear_gaussian_uniform_box_effective_information,
    linear_gaussian_uniform_box_ei_decomposition,
    render_causal_graph_svg,
    render_coarse_graining_comparison_svg,
    render_matrix_heatmap_svg,
    render_downward_causation_comparison_svg,
    render_topology_mechanism_svg,
    coarse_graining_subset_synergy_compactness,
    planted_module_pack_score,
    search_binary_or_fixed_macro_dim_coarse_grainings,
    search_binary_or_pair_coarse_grainings,
    search_marshall_example1_all_pair_partitions,
    solve_stationary_covariance,
    source_subset_tpm,
    subset_synergy_scores,
    system_ei_decomposition,
    target_ei_decomposition,
    search_marshall_example1_macro_mappings,
)


def build_rosas_downward_causation_tpm(n_nodes: int = 3) -> np.ndarray:
    """Rosas et al. (2020) Example 2: one future node equals the system XOR."""

    states = enumerate_binary_states(n_nodes)
    n_states = len(states)
    tpm = np.zeros((n_states, n_states), dtype=float)

    for row_index, state in enumerate(states):
        parity = int(state.sum() % 2)
        for col_index, next_state in enumerate(states):
            if int(next_state[0]) == parity:
                tpm[row_index, col_index] = 1.0 / (2 ** (n_nodes - 1))

    return tpm


def build_rosas_causal_decoupling_tpm(n_nodes: int = 3) -> np.ndarray:
    """Rosas et al. (2020) Example 1: parity persists without single-target effects."""

    states = enumerate_binary_states(n_nodes)
    n_states = len(states)
    tpm = np.zeros((n_states, n_states), dtype=float)
    parities = np.array([int(state.sum() % 2) for state in states], dtype=int)

    for row_index, parity in enumerate(parities):
        matching = np.flatnonzero(parities == parity)
        tpm[row_index, matching] = 1.0 / len(matching)

    return tpm


def build_mixed_downward_causation_tpm(n_nodes: int = 3) -> np.ndarray:
    """A 3-node mixed rule where both downward-causation components are positive."""

    states = enumerate_binary_states(n_nodes)
    n_states = len(states)
    tpm = np.zeros((n_states, n_states), dtype=float)

    for row_index, state in enumerate(states):
        x1, x2, x3 = (int(value) for value in state)
        target = int((x1 and x2) ^ x3)
        for col_index, next_state in enumerate(states):
            if int(next_state[0]) == target:
                tpm[row_index, col_index] = 1.0 / (2 ** (n_nodes - 1))

    return tpm


def build_two_source_gate_tpm(gate: str) -> np.ndarray:
    """Build a 3-node TPM whose third future node is a two-source gate."""

    states = enumerate_binary_states(3)
    index_by_state = {tuple(state.tolist()): idx for idx, state in enumerate(states)}
    tpm = np.zeros((len(states), len(states)), dtype=float)

    for row_index, state in enumerate(states):
        x0, x1, _ = (int(value) for value in state)
        if gate == "copy":
            target = x0
        elif gate == "and":
            target = int(x0 and x1)
        elif gate == "xor":
            target = int((x0 + x1) % 2)
        else:
            raise ValueError(f"Unsupported gate: {gate}")

        for next_x0 in (0, 1):
            for next_x1 in (0, 1):
                next_state = (next_x0, next_x1, target)
                tpm[row_index, index_by_state[next_state]] += 0.25

    return tpm


def build_rq3_boolean_network_tpm() -> np.ndarray:
    """Handcrafted 6-node Boolean network for the RQ3 causal-emergence notebook."""

    def update_rule(state: tuple[int, ...]) -> tuple[int, ...]:
        a1, a2, b1, b2, c1, c2 = (int(value) for value in state)
        alpha_next = int(b1 or b2)
        beta_next = int(c1 or c2)
        gamma_next = int((a1 or a2) ^ (b1 or b2))
        return (
            alpha_next,
            alpha_next,
            beta_next,
            beta_next,
            gamma_next,
            gamma_next,
        )

    return build_deterministic_boolean_tpm(6, update_rule)


def build_deterministic_boolean_tpm(
    n_nodes: int,
    update_fn,
) -> np.ndarray:
    """Build a deterministic TPM from a Boolean update function."""

    states = enumerate_binary_states(n_nodes)
    index_by_state = {tuple(state.tolist()): idx for idx, state in enumerate(states)}
    tpm = np.zeros((len(states), len(states)), dtype=float)
    for row_index, state in enumerate(states):
        next_state = tuple(int(bit) for bit in update_fn(tuple(int(v) for v in state)))
        tpm[row_index, index_by_state[next_state]] = 1.0
    return tpm


def load_notebook_namespace_until(
    notebook_path: Path,
    *,
    stop_when: str,
) -> dict[str, object]:
    """Execute notebook code cells until the cell containing `stop_when`."""

    notebook = json.loads(notebook_path.read_text())
    namespace: dict[str, object] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{notebook_path.name}_cell_{index}", "exec"), namespace, namespace)
        if stop_when in source:
            break
    return namespace


class EffectiveInformationFromTpmTests(unittest.TestCase):
    def test_dimension_averaged_effective_information_for_numpy_linear_dynamics_matches_formula(
        self,
    ) -> None:
        coupling = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float)
        covariance = np.diag([0.25, 1.0])
        box_half_width = 4.0

        measured = dimension_averaged_effective_information_for_dynamics(
            lambda x: coupling @ x,
            state_dim=2,
            output_covariance=covariance,
            intervention_bound=box_half_width,
            n_mc_samples=512,
            seed=0,
        )
        expected = (
            math.log(2.0 * box_half_width)
            + 0.5 * math.log(np.linalg.det(coupling))
            - 0.5 * math.log(2.0 * math.pi * math.e)
            - 0.25 * math.log(np.linalg.det(covariance))
        )

        self.assertAlmostEqual(measured, expected, places=3)

    def test_dimension_averaged_effective_information_requires_explicit_covariance(self) -> None:
        with self.assertRaises(ValueError):
            dimension_averaged_effective_information_for_dynamics(
                lambda x: x,
                state_dim=1,
                output_covariance=None,
                intervention_bound=1.0,
                n_mc_samples=32,
                seed=0,
            )

    def test_dimension_averaged_effective_information_supports_torch_modules(self) -> None:
        snippet = """
import json
import numpy as np
import torch
from utils import dimension_averaged_effective_information_for_dynamics

module = torch.nn.Sequential(
    torch.nn.Linear(1, 8),
    torch.nn.Tanh(),
    torch.nn.Linear(8, 1),
).double()
with torch.no_grad():
    module[0].weight[:] = torch.tensor([[0.6], [-0.4], [0.9], [1.1], [-0.7], [0.3], [0.2], [-1.0]], dtype=torch.float64)
    module[0].bias[:] = torch.tensor([0.1, -0.2, 0.3, -0.1, 0.05, 0.2, -0.3, 0.15], dtype=torch.float64)
    module[2].weight[:] = torch.tensor([[0.4, -0.3, 0.2, 0.1, -0.25, 0.35, -0.15, 0.5]], dtype=torch.float64)
    module[2].bias[:] = torch.tensor([0.05], dtype=torch.float64)

covariance = np.array([[0.3]], dtype=float)
torch_value = dimension_averaged_effective_information_for_dynamics(
    module,
    state_dim=1,
    output_covariance=covariance,
    intervention_bound=1.5,
    n_mc_samples=256,
    seed=0,
)
numpy_value = dimension_averaged_effective_information_for_dynamics(
    lambda x: module(torch.tensor(x, dtype=torch.float64)).detach().cpu().numpy(),
    state_dim=1,
    output_covariance=covariance,
    intervention_bound=1.5,
    n_mc_samples=256,
    seed=0,
)
print(json.dumps({"torch_value": torch_value, "numpy_value": numpy_value}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            self.skipTest(f"torch runtime unavailable: {completed.stderr.strip()}")

        payload = json.loads(completed.stdout.strip())
        self.assertAlmostEqual(payload["torch_value"], payload["numpy_value"], places=3)

    def test_dimension_averaged_effective_information_handles_known_nonlinear_dynamics_function(
        self,
    ) -> None:
        covariance = np.array([[0.5]], dtype=float)

        measured = dimension_averaged_effective_information_for_dynamics(
            lambda x: np.array([x[0] ** 3], dtype=float),
            state_dim=1,
            output_covariance=covariance,
            intervention_bound=np.array([[1.0, 2.0]], dtype=float),
            n_mc_samples=4096,
            seed=0,
        )

        expected = (
            math.log(1.0)
            + math.log(3.0)
            + 2.0 * (2.0 * math.log(2.0) - 1.0)
            - 0.5 * math.log(2.0 * math.pi * math.e)
            - 0.5 * math.log(0.5)
        )
        self.assertAlmostEqual(measured, expected, places=2)

    def test_dimension_averaged_causal_emergence_compares_two_dynamics(self) -> None:
        macro = lambda x: np.array([1.5 * x[0]], dtype=float)
        micro = lambda x: np.array([0.75 * x[0]], dtype=float)

        value = dimension_averaged_causal_emergence_for_dynamics(
            macro_dynamics=macro,
            micro_dynamics=micro,
            macro_dim=1,
            micro_dim=1,
            macro_output_covariance=np.array([[0.2]], dtype=float),
            micro_output_covariance=np.array([[0.8]], dtype=float),
            intervention_bound=2.0,
            n_mc_samples=256,
            seed=0,
        )

        expected = (
            math.log(1.5 / 0.75)
            - 0.5 * math.log(0.2)
            + 0.5 * math.log(0.8)
        )
        self.assertAlmostEqual(value, expected, places=3)

    def test_jacobian_uniform_box_total_synergy_matches_manual_bilinear_monte_carlo_formula(
        self,
    ) -> None:
        covariance = np.diag([0.05, 1.0])
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
        n_mc_samples = 512
        seed = 0
        atol = 1e-12

        measured = jacobian_uniform_box_total_synergy_for_dynamics(
            lambda x: np.array([x[0] * x[1], 0.0], dtype=float),
            state_dim=2,
            source_partition=[[0], [1]],
            target_indices=[0],
            output_covariance=covariance,
            intervention_bound=bounds,
            n_mc_samples=n_mc_samples,
            seed=seed,
            atol=atol,
        )

        rng = np.random.default_rng(seed)
        samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_mc_samples, 2))
        x0 = samples[:, 0]
        x1 = samples[:, 1]
        intervention_variance = ((bounds[:, 1] - bounds[:, 0]) ** 2) / 12.0
        sigma_joint = covariance[0, 0]
        sigma_left = covariance[0, 0] + intervention_variance[1] * (x0**2)
        sigma_right = covariance[0, 0] + intervention_variance[0] * (x1**2)

        metric_joint = np.clip((x0**2 + x1**2) / sigma_joint, atol, None)
        metric_left = np.clip((x1**2) / sigma_left, atol, None)
        metric_right = np.clip((x0**2) / sigma_right, atol, None)
        expected = 0.5 * float(np.mean(np.log(metric_joint) - np.log(metric_left) - np.log(metric_right)))

        self.assertAlmostEqual(measured["syn_total"], expected, places=10)

    def test_jacobian_uniform_box_total_synergy_uses_pointwise_effective_noise_inside_log(
        self,
    ) -> None:
        covariance = np.diag([0.05, 1.0])
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
        n_mc_samples = 512
        seed = 0
        atol = 1e-12

        measured = jacobian_uniform_box_total_synergy_for_dynamics(
            lambda x: np.array([x[0] * x[1], 0.0], dtype=float),
            state_dim=2,
            source_partition=[[0], [1]],
            target_indices=[0],
            output_covariance=covariance,
            intervention_bound=bounds,
            n_mc_samples=n_mc_samples,
            seed=seed,
            atol=atol,
        )

        rng = np.random.default_rng(seed)
        samples = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_mc_samples, 2))
        x0 = samples[:, 0]
        x1 = samples[:, 1]
        intervention_variance = ((bounds[:, 1] - bounds[:, 0]) ** 2) / 12.0
        sigma_joint = covariance[0, 0]
        sigma_left_pointwise = covariance[0, 0] + intervention_variance[1] * (x0**2)
        sigma_right_pointwise = covariance[0, 0] + intervention_variance[0] * (x1**2)

        metric_joint = np.clip((x0**2 + x1**2) / sigma_joint, atol, None)
        metric_left_pointwise = np.clip((x1**2) / sigma_left_pointwise, atol, None)
        metric_right_pointwise = np.clip((x0**2) / sigma_right_pointwise, atol, None)
        expected_pointwise = 0.5 * float(
            np.mean(
                np.log(metric_joint)
                - np.log(metric_left_pointwise)
                - np.log(metric_right_pointwise)
            )
        )

        sigma_left_averaged = covariance[0, 0] + intervention_variance[1] * float(np.mean(x0**2))
        sigma_right_averaged = covariance[0, 0] + intervention_variance[0] * float(np.mean(x1**2))
        metric_left_averaged = np.clip((x1**2) / sigma_left_averaged, atol, None)
        metric_right_averaged = np.clip((x0**2) / sigma_right_averaged, atol, None)
        expected_averaged = 0.5 * float(
            np.mean(
                np.log(metric_joint)
                - np.log(metric_left_averaged)
                - np.log(metric_right_averaged)
            )
        )

        self.assertAlmostEqual(measured["syn_total"], expected_pointwise, places=10)
        self.assertGreater(abs(expected_pointwise - expected_averaged), 1e-3)

    def test_jacobian_uniform_box_total_synergy_tracks_larger_explicit_output_noise(self) -> None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=float)

        def collider_mean(x: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    np.sin(x[1] * x[2]),
                    0.5 * x[1],
                    0.5 * x[2],
                ],
                dtype=float,
            )

        low_noise = jacobian_uniform_box_total_synergy_for_dynamics(
            collider_mean,
            state_dim=3,
            source_partition=[[1], [2]],
            target_indices=[0],
            output_covariance=np.diag([1e-6, 0.01, 0.01]),
            intervention_bound=bounds,
            n_mc_samples=1024,
            seed=1,
        )
        high_noise = jacobian_uniform_box_total_synergy_for_dynamics(
            collider_mean,
            state_dim=3,
            source_partition=[[1], [2]],
            target_indices=[0],
            output_covariance=np.diag([5e-2, 0.01, 0.01]),
            intervention_bound=bounds,
            n_mc_samples=1024,
            seed=1,
        )

        self.assertGreater(
            high_noise["whole_effective_noise"][0, 0],
            low_noise["whole_effective_noise"][0, 0],
        )

    def test_jacobian_uniform_box_total_synergy_is_zero_when_one_source_is_irrelevant(self) -> None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=float)

        def single_source_mean(x: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    0.8 * x[1],
                    0.5 * x[1],
                    0.5 * x[2],
                ],
                dtype=float,
            )

        measured = jacobian_uniform_box_total_synergy_for_dynamics(
            single_source_mean,
            state_dim=3,
            source_partition=[[1], [2]],
            target_indices=[0],
            output_covariance=np.diag([1e-6, 0.01, 0.01]),
            intervention_bound=bounds,
            n_mc_samples=1024,
            seed=3,
        )

        self.assertAlmostEqual(measured["syn_total"], 0.0, places=8)

    def test_two_independent_couples_decompose_into_two_main_complexes(self) -> None:
        def update(state: tuple[int, ...]) -> tuple[int, ...]:
            x0, x1, x2, x3 = state
            return (x1, x0, x3, x2)

        tpm = build_deterministic_boolean_tpm(4, update)
        current_state = (1, 0, 1, 0)

        whole = discrete_integrated_information(
            tpm,
            n_nodes=4,
            subset_indices=[0, 1, 2, 3],
            current_state=current_state,
        )
        complexes = find_discrete_complexes(
            tpm,
            n_nodes=4,
            current_state=current_state,
        )

        self.assertAlmostEqual(whole["phi"], 0.0, places=8)
        self.assertEqual(
            [tuple(item["subset_indices"]) for item in complexes if item["is_main_complex"]],
            [(0, 1), (2, 3)],
        )

    def test_copy_outputs_do_not_join_the_main_complex(self) -> None:
        def update(state: tuple[int, ...]) -> tuple[int, ...]:
            a, b, c, d, e = state
            return (
                int(b and c),
                int(a and c),
                int(a and b),
                a,
                b,
            )

        tpm = build_deterministic_boolean_tpm(5, update)
        current_state = (0, 0, 0, 0, 0)

        complexes = find_discrete_complexes(
            tpm,
            n_nodes=5,
            current_state=current_state,
        )

        self.assertEqual(tuple(complexes[0]["subset_indices"]), (0, 1, 2))
        self.assertTrue(complexes[0]["is_main_complex"])
        self.assertGreater(complexes[0]["phi"], 0.0)
        self.assertNotIn((0, 1, 2, 3, 4), [tuple(item["subset_indices"]) for item in complexes])

    def test_notebook_import_preamble_works_from_exp_directory(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "mediano_discrete_benchmark.ipynb"

        snippet = f"""
import json
from pathlib import Path
nb = json.loads(Path({str(notebook_path)!r}).read_text())
code = ''.join(nb['cells'][1]['source'])
exec(compile(code, 'notebook_preamble', 'exec'), {{}}, {{}})
"""
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=project_root / "exp",
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\\n{completed.stdout}\\nstderr:\\n{completed.stderr}",
        )

    def test_linear_gaussian_notebook_import_preamble_works_from_exp_directory(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "linear_gaussian_benchmark.ipynb"

        snippet = f"""
import json
from pathlib import Path
nb = json.loads(Path({str(notebook_path)!r}).read_text())
code = ''.join(nb['cells'][1]['source'])
exec(compile(code, 'linear_gaussian_notebook_preamble', 'exec'), {{}}, {{}})
"""
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=project_root / "exp",
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\\n{completed.stdout}\\nstderr:\\n{completed.stderr}",
        )

    def test_experiment6_notebook_import_preamble_works_from_exp_directory(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "experiment6_downward_causation.ipynb"

        snippet = f"""
import json
from pathlib import Path
nb = json.loads(Path({str(notebook_path)!r}).read_text())
code = ''.join(nb['cells'][1]['source'])
exec(compile(code, 'experiment6_notebook_preamble', 'exec'), {{}}, {{}})
"""
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=project_root / "exp",
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\\n{completed.stdout}\\nstderr:\\n{completed.stderr}",
        )

    def test_transport_map_notebook_import_preamble_works_from_exp_directory(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "tm_nonlinear.ipynb"

        snippet = f"""
import json
from pathlib import Path
nb = json.loads(Path({str(notebook_path)!r}).read_text())
code = ''.join(nb['cells'][1]['source'])
exec(compile(code, 'transport_map_notebook_preamble', 'exec'), {{}}, {{}})
"""
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=project_root / "exp",
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\\n{completed.stdout}\\nstderr:\\n{completed.stderr}",
        )

    def test_transport_map_notebook_has_compact_shape_and_reuses_module_helpers(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (project_root / "exp" / "tm_nonlinear.ipynb").read_text()
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertLessEqual(len(notebook["cells"]), 8)
        self.assertIn("summarize_two_source_synergy_transport_map", code_text)
        self.assertIn("run_alpha_sweep_tm", code_text)
        self.assertIn("draw_alpha_case_factor_graph", code_text)
        self.assertNotIn("class AffineTransportMapDensityEstimator", code_text)
        self.assertNotIn("def estimate_mutual_information_transport_map", code_text)
        self.assertNotIn("def simulate_synergistic_collider", code_text)
        self.assertNotIn("jacobian_uniform_box_total_synergy_for_dynamics", code_text)

    def test_transport_map_notebook_contains_chinese_method_and_result_sections(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (project_root / "exp" / "tm_nonlinear.ipynb").read_text()
        )

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        for phrase in [
            "transport map",
            "已知动力学",
            "alpha",
            "alpha = 0",
            "Syn / EI",
            "单源",
            "因子图",
        ]:
            self.assertIn(phrase, markdown_text)

        for symbol in [
            "simulate_alpha_case_intervention",
            "run_alpha_sweep_tm",
            "plot_alpha_sweep_tm",
            "joint input required",
        ]:
            self.assertIn(symbol, code_text)

    def test_transport_map_notebook_excludes_removed_density_and_collider_sections(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (project_root / "exp" / "tm_nonlinear.ipynb").read_text()
        )

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        for phrase in [
            "密度层准确性",
            "维度鲁棒性",
            "样本量鲁棒性",
            "计算时间",
            "collider",
        ]:
            self.assertNotIn(phrase, markdown_text.lower())

        for symbol in [
            "fit_kde_density_estimator",
            "fit_knn_density_estimator",
            "run_density_family_benchmark",
            "plot_density_accuracy_summary",
            "plot_density_robustness_scan",
            "plot_density_timing_summary",
            "simulate_synergistic_collider",
            "plot_synergistic_collider_overview",
            "plot_synergistic_information_bars",
            "estimate_nis_target_metrics",
            "compare_uniform_intervention_nis_vs_transport",
            "plot_uniform_intervention_nis_vs_transport",
        ]:
            self.assertNotIn(symbol, code_text)

    def test_yrd_shanghai_notebook_adds_transport_map_comparison_section_and_helpers(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads((project_root / "exp" / "yrd_shanghai.ipynb").read_text())

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        for phrase in [
            "transport map",
            "概率密度",
            "对比实验",
        ]:
            self.assertIn(phrase, markdown_text)

        for symbol in [
            "project_source_groups_to_transport_features",
            "summarize_transport_map_group_decomposition",
            "transport_map_overview_df",
        ]:
            self.assertIn(symbol, code_text)

    def test_yrd_shanghai_tm_causal_graph_notebook_contains_tm_graph_workflow(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads((project_root / "exp" / "yrd_shanghai_tm_graph.ipynb").read_text())

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        for phrase in [
            "transport map",
            "O3 -> O3",
            "O3 + PM2.5 -> O3",
            "概率密度",
        ]:
            self.assertIn(phrase, markdown_text)

        for symbol in [
            'CAUSAL_GRAPH_METHOD = "tm"',
            "compute_station_causal_graph_results",
            "build_transport_map_global_causal_summary",
        ]:
            self.assertIn(symbol, code_text)

    def test_yrd_transport_map_group_summary_returns_finite_scores(self) -> None:
        from yrd.transport_map import summarize_transport_map_group_decomposition

        rng = np.random.default_rng(0)
        x = rng.normal(size=(96, 4))
        y = np.stack(
            [
                0.8 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2],
                -0.3 * x[:, 1] + 0.7 * x[:, 3],
            ],
            axis=1,
        )
        summary = summarize_transport_map_group_decomposition(
            x,
            y,
            source_groups={
                "group_a": [0, 1],
                "group_b": [2],
                "group_c": [3],
            },
        )

        self.assertTrue(np.isfinite(summary["ei_tm"]))
        self.assertTrue(np.isfinite(summary["syn_tm"]))
        self.assertEqual(set(summary["group_ei_tm"]), {"group_a", "group_b", "group_c"})
        for value in summary["group_ei_tm"].values():
            self.assertTrue(np.isfinite(value))

    def test_yrd_transport_map_group_summary_flattens_high_rank_source_tensor(self) -> None:
        from yrd.transport_map import summarize_transport_map_group_decomposition

        rng = np.random.default_rng(1)
        source_tensor = rng.normal(size=(48, 2, 3, 4))
        flat = source_tensor.reshape(source_tensor.shape[0], -1)
        target = np.stack(
            [
                0.6 * flat[:, 0] - 0.4 * flat[:, 7] + 0.2 * flat[:, 17],
                -0.5 * flat[:, 5] + 0.3 * flat[:, 19],
            ],
            axis=1,
        )

        summary = summarize_transport_map_group_decomposition(
            source_tensor,
            target,
            source_groups={
                "first_block": [0, 1, 2, 3],
                "mixed_block": [5, 7, 11, 17, 19],
            },
        )

        self.assertTrue(np.isfinite(summary["ei_tm"]))
        self.assertTrue(np.isfinite(summary["syn_tm"]))
        self.assertEqual(summary["feature_metadata"]["first_block"]["original_dim"], 4)
        self.assertEqual(summary["feature_metadata"]["mixed_block"]["original_dim"], 5)

    def test_experiment4_notebook_uses_chinese_markdown_explanations(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (project_root / "exp" / "experiment4_boolean_motif_causal_graphs.ipynb").read_text()
        )

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )

        self.assertIn("主示例网络", markdown_text)
        self.assertIn("结果说明", markdown_text)
        self.assertIn("布尔", markdown_text)

    def test_experiment4_notebook_no_longer_embeds_old_svg_subtitles(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_text = (
            project_root / "exp" / "experiment4_boolean_motif_causal_graphs.ipynb"
        ).read_text()

        self.assertNotIn("COPY edge plus AND/XOR motifs in the update rules", notebook_text)
        self.assertNotIn("Only pairwise EI edges above the display threshold are shown", notebook_text)

    def test_experiment4_notebook_adds_weight_table_sections(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (project_root / "exp" / "experiment4_boolean_motif_causal_graphs.ipynb").read_text()
        )

        markdown_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        self.assertIn("边权重汇总表", markdown_text)
        self.assertIn("Pairwise EI 边权重", markdown_text)
        self.assertIn("协同超边权重", markdown_text)
        self.assertIn("图里不再直接写权重数字", markdown_text)
        self.assertIn("pairwise_weight_rows", code_text)
        self.assertIn("hyperedge_weight_rows", code_text)

    def test_experiment4_main_example_truth_graph_keeps_only_copy_as_plain_edge(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "experiment4_boolean_motif_causal_graphs.ipynb"

        snippet = """
import json
from pathlib import Path
nb = json.loads(Path(%r).read_text())
code_cells = []
for cell in nb['cells']:
    if cell.get('cell_type') != 'code':
        continue
    source = ''.join(cell.get('source', []))
    code_cells.append(source)
    if 'def build_main_example()' in source:
        break
namespace = {}
for idx, source in enumerate(code_cells):
    exec(compile(source, f'experiment4_truth_cell_{idx}', 'exec'), namespace, namespace)
_, node_specs, gt_edges, gt_hyperedges = namespace['build_main_example']()
assert gt_edges == [(0, 2, 'COPY')], gt_edges
assert node_specs[4]['alpha'] == 0.0
assert node_specs[4]['gamma'] == 4.5
assert node_specs[4]['parity_sources'] == [0, 1]
truth_hyperedges = {
    (tuple(edge['sources']), int(edge['target']), edge['label'])
    for edge in gt_hyperedges
}
assert ((0, 1), 3, 'AND') in truth_hyperedges, truth_hyperedges
assert ((0, 1), 4, 'XOR') in truth_hyperedges, truth_hyperedges
assert len(truth_hyperedges) == 2, truth_hyperedges
""" % str(notebook_path)
        completed = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=project_root / "exp",
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\\n{completed.stdout}\\nstderr:\\n{completed.stderr}",
        )

    def test_discrete_notebook_contains_mini_network_supplement_section(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "mediano_discrete_benchmark.ipynb"
        notebook = json.loads(notebook_path.read_text())

        markdown_text = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

        self.assertIn("极小网络补充实验", markdown_text)
        self.assertIn("Copy-2", markdown_text)
        self.assertIn("Coop-2", markdown_text)
        self.assertIn("Parity-2", markdown_text)
        self.assertNotIn("Parity-3", markdown_text)
        self.assertNotIn("Parity-Control-3", markdown_text)

    def test_discrete_notebook_mini_network_results_show_stronger_parity_lift(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "mediano_discrete_benchmark.ipynb"
        notebook = json.loads(notebook_path.read_text())

        code_cells = []
        found_results_cell = False
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            code_cells.append(source)
            if "mini_results = {}" in source:
                found_results_cell = True
                break

        self.assertTrue(found_results_cell, "Notebook is missing the mini_results cell.")

        namespace: dict[str, object] = {}
        for idx, source in enumerate(code_cells):
            exec(compile(source, f"mini_notebook_cell_{idx}", "exec"), namespace, namespace)

        mini_results = namespace["mini_results"]
        self.assertGreater(
            mini_results["Coop-2"]["syn_high"],
            mini_results["Copy-2"]["syn_high"],
        )
        self.assertGreater(
            mini_results["Parity-2"]["syn_high"],
            mini_results["Coop-2"]["syn_high"],
        )
        self.assertGreater(
            mini_results["Coop-2"]["rho_high"],
            mini_results["Copy-2"]["rho_high"],
        )
        self.assertGreater(
            mini_results["Parity-2"]["rho_high"],
            mini_results["Coop-2"]["rho_high"],
        )

    def test_discrete_notebook_documents_coop_and_parity_math(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "mediano_discrete_benchmark.ipynb"
        notebook = json.loads(notebook_path.read_text())

        markdown_text = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )

        self.assertIn("p(x_j^{t+1}=1 \\mid x_t) = \\sigma", markdown_text)
        self.assertIn("x_i^t x_k^t - \\frac{1}{4}", markdown_text)
        self.assertIn("\\left(\\sum_{i \\in P_j} x_i^t\\right) \\bmod 2", markdown_text)
        self.assertIn("\\ell_1^{\\mathrm{copy}}", markdown_text)
        self.assertIn("\\ell_1^{\\mathrm{coop}}", markdown_text)
        self.assertIn("\\ell_1^{\\mathrm{parity}}", markdown_text)

    def test_discrete_notebook_two_node_parity_check_has_positive_lift(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "mediano_discrete_benchmark.ipynb"
        notebook = json.loads(notebook_path.read_text())

        code_cells = []
        found_results_cell = False
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            code_cells.append(source)
            if "mini_results = {}" in source:
                found_results_cell = True
                break

        self.assertTrue(found_results_cell, "Notebook is missing the mini_results cell.")

        namespace: dict[str, object] = {}
        for idx, source in enumerate(code_cells):
            exec(compile(source, f"parity2_notebook_cell_{idx}", "exec"), namespace, namespace)

        mini_results = namespace["mini_results"]
        self.assertGreater(
            mini_results["Parity-2"]["syn_high"],
            mini_results["Copy-2"]["syn_high"],
        )
        self.assertGreater(
            mini_results["Parity-2"]["rho_high"],
            mini_results["Copy-2"]["rho_high"],
        )

    def test_linear_gaussian_notebook_two_node_examples_share_same_noise_covariance(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "linear_gaussian_benchmark.ipynb"
        notebook = json.loads(notebook_path.read_text())

        code_cells = []
        found_results_cell = False
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            code_cells.append(source)
            if "mini_results = {}" in source:
                found_results_cell = True
                break

        self.assertTrue(found_results_cell, "Notebook is missing the mini_results cell.")

        namespace: dict[str, object] = {}
        for idx, source in enumerate(code_cells):
            exec(compile(source, f"linear_gaussian_mini_cell_{idx}", "exec"), namespace, namespace)

        mini_results = namespace["mini_results"]
        self.assertEqual(set(mini_results.keys()), {"Diag", "Cross"})
        np.testing.assert_allclose(
            np.diag(mini_results["Diag"]["coupling"]),
            np.diag(mini_results["Cross"]["coupling"]),
            atol=1e-9,
        )
        self.assertAlmostEqual(mini_results["Cross"]["coupling"][0, 1], 0.8, places=9)
        self.assertAlmostEqual(mini_results["Cross"]["coupling"][1, 0], 0.0, places=9)
        np.testing.assert_allclose(
            mini_results["Diag"]["noise_covariance"],
            mini_results["Cross"]["noise_covariance"],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            mini_results["Diag"]["noise_covariance"],
            (0.2**2) * np.eye(2),
            atol=1e-9,
        )
        self.assertAlmostEqual(mini_results["Diag"]["syn_high"], 0.0, places=9)
        self.assertGreater(mini_results["Cross"]["syn_high"], 0.25)
        self.assertAlmostEqual(
            mini_results["Diag"]["ei_full"],
            mini_results["Cross"]["ei_full"],
            places=9,
        )

    def test_linear_gaussian_notebook_two_node_outputs_focus_on_ei_and_synergy(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (project_root / "exp" / "linear_gaussian_benchmark.ipynb").read_text()
        )

        table_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "mini-two-node-table")
        panel_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "mini-two-node-panels")

        table_source = "".join(table_cell["source"])
        panel_source = "".join(panel_cell["source"])

        self.assertIn("总 EI (EI_full)", table_source)
        self.assertIn("最高阶协同 (Syn_high)", table_source)
        self.assertNotIn("UnBudget", table_source)
        self.assertNotIn("rho_high", table_source)

        self.assertNotIn("mechanism noise structure", panel_source)
        self.assertNotIn("Syn_high={syn_high:.3f}", panel_source)

        output_mimes = {
            mime
            for output in panel_cell.get("outputs", [])
            for mime in output.get("data", {}).keys()
        }
        self.assertIn("image/svg+xml", output_mimes)
        self.assertNotIn("text/html", output_mimes)

    def test_linear_gaussian_two_node_results_are_integrated_into_research_doc(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        doc_text = (project_root / "doc" / "研究框架.md").read_text()

        self.assertIn("#### 4.2.1 二维最小线性高斯对照", doc_text)
        self.assertIn(
            "../fig/mediano_linear_gaussian_benchmark/two-node-low-high-mechanism-01.svg",
            doc_text,
        )
        self.assertIn("| 系统 | A 结构 | 总 EI", doc_text)
        self.assertIn("| Diag |", doc_text)
        self.assertIn("| Cross |", doc_text)

    def test_linear_gaussian_notebook_fixed_a_noise_scan_shows_synergy_growth(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_path = project_root / "exp" / "linear_gaussian_benchmark.ipynb"
        notebook = json.loads(notebook_path.read_text())

        code_cells = []
        found_results_cell = False
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            code_cells.append(source)
            if "noise_scan_results = {}" in source:
                found_results_cell = True
                break

        self.assertTrue(found_results_cell, "Notebook is missing the fixed-A noise scan results cell.")

        namespace: dict[str, object] = {}
        for idx, source in enumerate(code_cells):
            exec(compile(source, f"linear_gaussian_noise_scan_cell_{idx}", "exec"), namespace, namespace)

        noise_scan_results = namespace["noise_scan_results"]
        self.assertEqual(set(noise_scan_results.keys()), {"Diag", "Cross"})

        for name in ["Diag", "Cross"]:
            result = noise_scan_results[name]
            self.assertEqual(result["coupling"].shape, (2, 2))
            self.assertEqual(len(result["series"]), 5)
            for point in result["series"]:
                covariance = point["noise_covariance"]
                self.assertAlmostEqual(covariance[0, 0], 0.04, places=9)
                self.assertAlmostEqual(covariance[1, 1], 0.04, places=9)
                self.assertAlmostEqual(covariance[0, 1], covariance[1, 0], places=9)

        low_by_rho = {round(point["rho"], 1): point for point in noise_scan_results["Diag"]["series"]}
        high_by_rho = {round(point["rho"], 1): point for point in noise_scan_results["Cross"]["series"]}

        self.assertAlmostEqual(low_by_rho[0.0]["syn_high"], 0.0, places=9)
        self.assertGreater(low_by_rho[0.8]["syn_high"], low_by_rho[0.0]["syn_high"])
        self.assertAlmostEqual(low_by_rho[-0.8]["syn_high"], low_by_rho[0.8]["syn_high"], places=9)

        self.assertGreater(high_by_rho[0.0]["syn_high"], 0.25)
        self.assertGreater(high_by_rho[-0.8]["syn_high"], high_by_rho[0.0]["syn_high"])
        self.assertLess(high_by_rho[0.8]["syn_high"], high_by_rho[0.0]["syn_high"])
        self.assertAlmostEqual(
            low_by_rho[0.0]["ei_full"],
            high_by_rho[0.0]["ei_full"],
            places=9,
        )

    def test_linear_gaussian_notebook_noise_scan_uses_single_combined_curve_figure(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (project_root / "exp" / "linear_gaussian_benchmark.ipynb").read_text()
        )

        curve_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "fixed-a-noise-scan-curves")
        curve_source = "".join(curve_cell["source"])

        self.assertIn("render_combined_noise_scan_curve_svg", curve_source)
        self.assertIn("stroke-dasharray", curve_source)
        self.assertIn("总 EI / 最高阶协同 随噪声相关性变化", curve_source)
        self.assertNotIn("render_noise_scan_curves_svg", curve_source)
        self.assertNotIn("总 EI (EI_full) 随噪声相关性变化", curve_source)
        self.assertNotIn("最高阶协同 (Syn_high) 随噪声相关性变化", curve_source)

    def test_enumerate_binary_states_returns_lexicographic_order(self) -> None:
        states = enumerate_binary_states(2)
        expected = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        np.testing.assert_array_equal(states, expected)

    def test_identity_tpm_has_one_bit_of_ei(self) -> None:
        tpm = np.array([[1.0, 0.0], [0.0, 1.0]])

        ei = effective_information_from_tpm(tpm)

        self.assertAlmostEqual(ei, 1.0, places=9)

    def test_uniform_output_tpm_has_zero_ei(self) -> None:
        tpm = np.array([[0.5, 0.5], [0.5, 0.5]])

        ei = effective_information_from_tpm(tpm)

        self.assertAlmostEqual(ei, 0.0, places=9)

    def test_source_subset_tpm_averages_over_complement_uniformly(self) -> None:
        system_tpm = np.eye(4)

        subset_tpm = source_subset_tpm(system_tpm, n_nodes=2, source_indices=[0])
        expected = np.array(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
            ]
        )

        np.testing.assert_allclose(subset_tpm, expected)

    def test_joint_ei_decomposition_matches_identity_on_two_nodes(self) -> None:
        system_tpm = np.eye(4)

        summary = joint_ei_decomposition(system_tpm, n_nodes=2)

        self.assertAlmostEqual(summary["ei_full"], 2.0, places=9)
        self.assertEqual(len(summary["ei_singles"]), 2)
        self.assertAlmostEqual(summary["ei_singles"][0], 1.0, places=9)
        self.assertAlmostEqual(summary["ei_singles"][1], 1.0, places=9)
        self.assertAlmostEqual(
            summary["syn_high"],
            summary["ei_full"] - sum(summary["ei_singles"]),
            places=9,
        )
        self.assertAlmostEqual(summary["syn_high"], 0.0, places=9)

    def test_xor_example_has_unit_effective_information_from_whole_to_target(self) -> None:
        discrete_effective_information = getattr(
            utils_module,
            "discrete_effective_information",
            None,
        )
        self.assertIsNotNone(discrete_effective_information)

        tpm = build_rosas_downward_causation_tpm(n_nodes=3)
        ei_full = discrete_effective_information(
            tpm,
            n_nodes=3,
            source_indices=[0, 1, 2],
            target_indices=[0],
        )
        ei_singles = [
            discrete_effective_information(
                tpm,
                n_nodes=3,
                source_indices=[source],
                target_indices=[0],
            )
            for source in range(3)
        ]

        self.assertAlmostEqual(ei_full, 1.0, places=9)
        for ei_single in ei_singles:
            self.assertAlmostEqual(ei_single, 0.0, places=9)

    def test_xor_example_has_positive_downward_causation_only_on_driven_target(self) -> None:
        discrete_downward_causation = getattr(
            utils_module,
            "discrete_downward_causation",
            None,
        )
        discrete_downward_causation_all_targets = getattr(
            utils_module,
            "discrete_downward_causation_all_targets",
            None,
        )
        self.assertIsNotNone(discrete_downward_causation)
        self.assertIsNotNone(discrete_downward_causation_all_targets)

        tpm = build_rosas_downward_causation_tpm(n_nodes=3)
        dc_target_0 = discrete_downward_causation(tpm, n_nodes=3, target_index=0)
        dc_target_1 = discrete_downward_causation(tpm, n_nodes=3, target_index=1)
        dc_target_2 = discrete_downward_causation(tpm, n_nodes=3, target_index=2)
        summary = discrete_downward_causation_all_targets(tpm, n_nodes=3)

        self.assertAlmostEqual(dc_target_0["ei_full"], 1.0, places=9)
        self.assertAlmostEqual(dc_target_0["dc"], 1.0, places=9)
        self.assertAlmostEqual(dc_target_1["dc"], 0.0, places=9)
        self.assertAlmostEqual(dc_target_2["dc"], 0.0, places=9)
        self.assertEqual(len(summary), 3)
        self.assertAlmostEqual(summary[0]["dc"], 1.0, places=9)
        self.assertAlmostEqual(summary[1]["dc"], 0.0, places=9)
        self.assertAlmostEqual(summary[2]["dc"], 0.0, places=9)

    def test_causal_decoupling_control_has_zero_downward_causation_for_single_targets(self) -> None:
        discrete_downward_causation_all_targets = getattr(
            utils_module,
            "discrete_downward_causation_all_targets",
            None,
        )
        self.assertIsNotNone(discrete_downward_causation_all_targets)

        tpm = build_rosas_causal_decoupling_tpm(n_nodes=3)
        summary = discrete_downward_causation_all_targets(tpm, n_nodes=3)

        self.assertEqual(len(summary), 3)
        for row in summary:
            self.assertAlmostEqual(row["ei_full"], 0.0, places=9)
            self.assertAlmostEqual(row["dc"], 0.0, places=9)

    def test_mixed_example_activates_both_downward_causation_components(self) -> None:
        discrete_downward_causation = getattr(
            utils_module,
            "discrete_downward_causation",
            None,
        )
        discrete_downward_causation_all_targets = getattr(
            utils_module,
            "discrete_downward_causation_all_targets",
            None,
        )
        discrete_effective_information = getattr(
            utils_module,
            "discrete_effective_information",
            None,
        )
        self.assertIsNotNone(discrete_downward_causation)
        self.assertIsNotNone(discrete_downward_causation_all_targets)
        self.assertIsNotNone(discrete_effective_information)

        tpm = build_mixed_downward_causation_tpm(n_nodes=3)
        dc_target_0 = discrete_downward_causation(tpm, n_nodes=3, target_index=0)
        summary = discrete_downward_causation_all_targets(tpm, n_nodes=3)
        environment_ei = discrete_effective_information(
            tpm,
            n_nodes=3,
            source_indices=[1, 2],
            target_indices=[0],
        )

        self.assertAlmostEqual(dc_target_0["ei_full"], 1.0, places=9)
        self.assertAlmostEqual(dc_target_0["ei_singles"][0], 0.0, places=9)
        self.assertAlmostEqual(dc_target_0["ei_singles"][1], 0.0, places=9)
        self.assertAlmostEqual(dc_target_0["ei_singles"][2], 0.18872187554086717, places=9)
        self.assertAlmostEqual(environment_ei, 0.5, places=9)
        self.assertAlmostEqual(dc_target_0["joint_term"], 0.5, places=9)
        self.assertAlmostEqual(dc_target_0["environment_synergy"], 0.31127812445913283, places=9)
        self.assertAlmostEqual(dc_target_0["dc"], 0.8112781244591328, places=9)
        self.assertAlmostEqual(summary[1]["dc"], 0.0, places=9)
        self.assertAlmostEqual(summary[2]["dc"], 0.0, places=9)

    def test_target_subset_tpm_preserves_row_stochasticity_for_single_target(self) -> None:
        target_subset_tpm = getattr(utils_module, "target_subset_tpm", None)
        self.assertIsNotNone(target_subset_tpm)

        tpm = build_two_source_gate_tpm("xor")
        coarse = target_subset_tpm(tpm, n_nodes=3, target_indices=[2])

        self.assertEqual(coarse.shape, (8, 2))
        np.testing.assert_allclose(coarse.sum(axis=1), np.ones(8))
        np.testing.assert_allclose(coarse[0], np.array([1.0, 0.0]))
        np.testing.assert_allclose(coarse[-1], np.array([1.0, 0.0]))

    def test_target_specific_discrete_ei_distinguishes_copy_and_xor(self) -> None:
        discrete_effective_information = getattr(
            utils_module,
            "discrete_effective_information",
            None,
        )
        self.assertIsNotNone(discrete_effective_information)

        copy_tpm = build_two_source_gate_tpm("copy")
        xor_tpm = build_two_source_gate_tpm("xor")

        copy_ei = discrete_effective_information(
            copy_tpm,
            n_nodes=3,
            source_indices=[0],
            target_indices=[2],
        )
        xor_single = discrete_effective_information(
            xor_tpm,
            n_nodes=3,
            source_indices=[0],
            target_indices=[2],
        )
        xor_joint = discrete_effective_information(
            xor_tpm,
            n_nodes=3,
            source_indices=[0, 1],
            target_indices=[2],
        )

        self.assertAlmostEqual(copy_ei, 1.0, places=9)
        self.assertAlmostEqual(xor_single, 0.0, places=9)
        self.assertAlmostEqual(xor_joint, 1.0, places=9)

    def test_discrete_synergy_matches_copy_and_logic_gates(self) -> None:
        discrete_synergy = getattr(utils_module, "discrete_synergy", None)
        self.assertIsNotNone(discrete_synergy)

        copy_tpm = build_two_source_gate_tpm("copy")
        and_tpm = build_two_source_gate_tpm("and")
        xor_tpm = build_two_source_gate_tpm("xor")

        copy_syn = discrete_synergy(
            copy_tpm,
            n_nodes=3,
            source_indices=[0, 1],
            target_indices=[2],
        )
        and_syn = discrete_synergy(
            and_tpm,
            n_nodes=3,
            source_indices=[0, 1],
            target_indices=[2],
        )
        xor_syn = discrete_synergy(
            xor_tpm,
            n_nodes=3,
            source_indices=[0, 1],
            target_indices=[2],
        )

        self.assertAlmostEqual(copy_syn, 0.0, places=9)
        self.assertGreater(and_syn, 0.15)
        self.assertAlmostEqual(xor_syn, 1.0, places=9)
        self.assertGreater(xor_syn, and_syn)

    def test_discrete_causal_graph_summary_returns_pairwise_edges_and_hyperedges(self) -> None:
        discrete_causal_graph = getattr(utils_module, "discrete_causal_graph", None)
        self.assertIsNotNone(discrete_causal_graph)

        summary = discrete_causal_graph(
            build_two_source_gate_tpm("and"),
            n_nodes=3,
        )

        self.assertIn("pairwise_ei", summary)
        self.assertIn("hyperedges", summary)
        self.assertEqual(summary["pairwise_ei"].shape, (3, 3))
        self.assertTrue(any(edge["target"] == 2 for edge in summary["hyperedges"]))
        and_target_edges = [edge for edge in summary["hyperedges"] if edge["target"] == 2]
        self.assertTrue(any(tuple(edge["sources"]) == (0, 1) for edge in and_target_edges))

    def test_sample_tpm_rollout_and_observed_mi_capture_copy_relation(self) -> None:
        sample_tpm_rollout = getattr(utils_module, "sample_tpm_rollout", None)
        observed_mutual_information_graph = getattr(
            utils_module,
            "observed_mutual_information_graph",
            None,
        )
        self.assertIsNotNone(sample_tpm_rollout)
        self.assertIsNotNone(observed_mutual_information_graph)

        tpm = build_two_source_gate_tpm("copy")
        past_states, future_states = sample_tpm_rollout(
            tpm,
            n_nodes=3,
            n_steps=2000,
            burn_in=50,
            seed=7,
        )
        mi_matrix = observed_mutual_information_graph(
            tpm,
            n_nodes=3,
            n_steps=2000,
            burn_in=50,
            seed=7,
        )

        self.assertEqual(past_states.shape, (2000, 3))
        self.assertEqual(future_states.shape, (2000, 3))
        self.assertEqual(mi_matrix.shape, (3, 3))
        self.assertTrue(np.all(mi_matrix >= -1e-12))
        self.assertGreater(mi_matrix[0, 2], 0.6)

    def test_causal_graph_svg_renderers_include_motif_and_summary_labels(self) -> None:
        render_ground_truth_causal_graph_svg = getattr(
            utils_module,
            "render_ground_truth_causal_graph_svg",
            None,
        )
        render_causal_graph_svg = getattr(utils_module, "render_causal_graph_svg", None)
        render_recovery_summary_svg = getattr(
            utils_module,
            "render_recovery_summary_svg",
            None,
        )
        self.assertIsNotNone(render_ground_truth_causal_graph_svg)
        self.assertIsNotNone(render_causal_graph_svg)
        self.assertIsNotNone(render_recovery_summary_svg)

        ground_truth_svg = render_ground_truth_causal_graph_svg(
            "Ground truth",
            n_nodes=3,
            directed_edges=[(0, 2, "COPY")],
            hyperedges=[{"sources": (0, 1), "target": 2, "label": "XOR"}],
            node_labels=["x0", "x1", "x2"],
        )
        causal_svg = render_causal_graph_svg(
            "Pairwise EI",
            pairwise_matrix=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            hyperedges=[{"sources": (0, 1), "target": 2, "value": 1.0, "label": "XOR"}],
            node_labels=["x0", "x1", "x2"],
        )
        summary_svg = render_recovery_summary_svg(
            "Recovery summary",
            rows=[
                {"motif": "COPY", "edge_recall": 1.0, "hyperedge_f1": 0.0},
                {"motif": "XOR", "edge_recall": 0.0, "hyperedge_f1": 1.0},
            ],
        )

        self.assertIn("Ground truth", ground_truth_svg)
        self.assertIn("COPY", ground_truth_svg)
        self.assertIn("XOR", ground_truth_svg)
        self.assertIn("Legend", ground_truth_svg)
        self.assertIn("markerUnits='userSpaceOnUse'", ground_truth_svg)
        self.assertIn("Pairwise EI", causal_svg)
        self.assertIn("x0(t)", causal_svg)
        self.assertIn("x2(t+1)", causal_svg)
        self.assertNotIn("1.00", causal_svg)
        self.assertIn("stroke-width='3.60'", causal_svg)
        self.assertIn("r='11.0'", causal_svg)
        self.assertIn("Recovery summary", summary_svg)
        self.assertIn("edge_recall", summary_svg)
        self.assertIn("hyperedge_f1", summary_svg)

    def test_causal_graph_svg_offsets_dense_edge_labels(self) -> None:
        render_causal_graph_svg = getattr(utils_module, "render_causal_graph_svg", None)
        self.assertIsNotNone(render_causal_graph_svg)

        pairwise_matrix = np.zeros((5, 5), dtype=float)
        pairwise_matrix[0, 2] = 0.81
        pairwise_matrix[0, 3] = 0.26
        pairwise_matrix[1, 3] = 0.26
        pairwise_matrix[0, 4] = 0.25
        pairwise_matrix[1, 4] = 0.25

        svg = render_causal_graph_svg(
            "Dense labels",
            pairwise_matrix=pairwise_matrix,
            hyperedges=[
                {"sources": (0, 1), "target": 3, "value": 0.17, "label": "AND"},
                {"sources": (0, 1), "target": 4, "value": 0.91, "label": "XOR"},
            ],
            node_labels=["x0", "x1", "x2", "x3", "x4"],
            edge_threshold=0.08,
            hyperedge_threshold=0.08,
            show_edge_values=True,
        )

        self.assertNotRegex(svg, r">0\.\d{2}<")
        self.assertNotIn("AND |", svg)
        self.assertNotIn("XOR |", svg)
        self.assertIn("stroke-dasharray='5,3'", svg)

    def test_causal_graph_svg_scales_hyperedge_circle_radius(self) -> None:
        render_causal_graph_svg = getattr(utils_module, "render_causal_graph_svg", None)
        self.assertIsNotNone(render_causal_graph_svg)

        svg = render_causal_graph_svg(
            "Hyperedge radii",
            pairwise_matrix=np.zeros((4, 4), dtype=float),
            hyperedges=[
                {"sources": (0, 1), "target": 2, "value": 0.20, "label": "AND"},
                {"sources": (1, 2), "target": 3, "value": 0.90, "label": "XOR"},
            ],
            node_labels=["x0", "x1", "x2", "x3"],
            hyperedge_threshold=0.05,
            show_edge_values=False,
        )

        radii = [float(value) for value in re.findall(r"<circle cx='[^']+' cy='[^']+' r='([^']+)' fill='white' stroke='[^']+' stroke-width='2'/>", svg)]
        self.assertEqual(len(radii), 2)
        self.assertGreater(max(radii), min(radii))

    def test_ground_truth_legend_is_shifted_farther_right(self) -> None:
        render_ground_truth_causal_graph_svg = getattr(
            utils_module,
            "render_ground_truth_causal_graph_svg",
            None,
        )
        self.assertIsNotNone(render_ground_truth_causal_graph_svg)

        svg = render_ground_truth_causal_graph_svg(
            "Ground truth",
            n_nodes=5,
            directed_edges=[(0, 2, "COPY")],
            hyperedges=[
                {"sources": (0, 1), "target": 3, "label": "AND"},
                {"sources": (0, 1), "target": 4, "label": "XOR"},
            ],
            node_labels=["x0", "x1", "x2", "x3", "x4"],
        )

        legend_match = re.search(r"<text x='([^']+)' y='52\.0' font-size='11' font-weight='700' fill='#333'>Legend</text>", svg)
        self.assertIsNotNone(legend_match)
        self.assertGreater(float(legend_match.group(1)), 440.0)

    def test_ground_truth_hyperedge_junction_is_shifted_left_of_default_graph(self) -> None:
        render_ground_truth_causal_graph_svg = getattr(
            utils_module,
            "render_ground_truth_causal_graph_svg",
            None,
        )
        render_causal_graph_svg = getattr(utils_module, "render_causal_graph_svg", None)
        self.assertIsNotNone(render_ground_truth_causal_graph_svg)
        self.assertIsNotNone(render_causal_graph_svg)

        ground_truth_svg = render_ground_truth_causal_graph_svg(
            "Ground truth",
            n_nodes=5,
            directed_edges=[(0, 2, "COPY")],
            hyperedges=[
                {"sources": (0, 1), "target": 3, "label": "AND"},
                {"sources": (0, 1), "target": 4, "label": "XOR"},
            ],
            node_labels=["x0", "x1", "x2", "x3", "x4"],
        )
        default_svg = render_causal_graph_svg(
            "Hypergraph",
            pairwise_matrix=np.zeros((5, 5), dtype=float),
            hyperedges=[
                {"sources": (0, 1), "target": 3, "value": 1.0, "label": "AND"},
                {"sources": (0, 1), "target": 4, "value": 1.0, "label": "XOR"},
            ],
            node_labels=["x0", "x1", "x2", "x3", "x4"],
        )

        gt_match = re.search(
            r"<circle cx='([^']+)' cy='123\.3' r='[^']+' fill='white' stroke='#e17c05' stroke-width='2'/>",
            ground_truth_svg,
        )
        default_match = re.search(
            r"<circle cx='([^']+)' cy='123\.3' r='[^']+' fill='white' stroke='#e17c05' stroke-width='2'/>",
            default_svg,
        )
        self.assertIsNotNone(gt_match)
        self.assertIsNotNone(default_match)
        self.assertLess(float(gt_match.group(1)), float(default_match.group(1)))

    def test_probabilistic_boolean_tpm_is_row_stochastic(self) -> None:
        adjacency = np.zeros((2, 2))
        node_specs = [{"bias": 0.0}, {"bias": 0.0}]

        tpm = build_probabilistic_boolean_tpm(adjacency, node_specs)

        self.assertEqual(tpm.shape, (4, 4))
        np.testing.assert_allclose(tpm.sum(axis=1), np.ones(4))
        np.testing.assert_allclose(tpm, np.full((4, 4), 0.25))

    def test_topology_mechanism_svg_marks_coop_and_parity_nodes(self) -> None:
        adjacency = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        node_specs = [
            {"bias": 0.0, "alpha": 0.2, "beta": 0.0, "gamma": 0.0},
            {
                "bias": 0.0,
                "alpha": 0.5,
                "beta": 0.8,
                "gamma": 0.0,
                "coop_pairs": [(0, 2)],
            },
            {
                "bias": 0.0,
                "alpha": 0.6,
                "beta": 0.2,
                "gamma": 0.9,
                "parity_sources": [0, 1],
            },
        ]

        svg = render_topology_mechanism_svg(
            "T",
            "toy network",
            adjacency,
            node_specs,
        )

        self.assertIn("toy network", svg)
        self.assertIn("gamma=0.90", svg)
        self.assertIn("coop", svg)
        self.assertIn("parity", svg)
        self.assertIn("stroke-dasharray", svg)
        self.assertIn("marker-end", svg)

    def test_topology_mechanism_svg_hides_plain_edges_replaced_by_mechanism_sources(self) -> None:
        adjacency = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        node_specs = [
            {"bias": 0.0, "alpha": 0.2, "beta": 0.0, "gamma": 0.0},
            {"bias": 0.0, "alpha": 0.2, "beta": 0.0, "gamma": 0.0},
            {
                "bias": 0.0,
                "alpha": 0.6,
                "beta": 0.8,
                "gamma": 0.9,
                "coop_pairs": [(0, 1)],
                "parity_sources": [0, 1],
            },
        ]

        svg = render_topology_mechanism_svg(
            "T",
            "toy network",
            adjacency,
            node_specs,
        )

        self.assertEqual(svg.count("stroke='#a8b3bf'"), 1)
        self.assertIn("stroke='#e17c05'", svg)
        self.assertIn("stroke='#7b4ab5'", svg)

    def test_probabilistic_boolean_tpm_supports_high_order_coop_sources(self) -> None:
        adjacency = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        node_specs = [
            {"bias": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0},
            {"bias": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0},
            {"bias": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0},
            {
                "bias": 0.0,
                "alpha": 0.0,
                "beta": 2.0,
                "gamma": 0.0,
                "coop_sources": [0, 1, 2],
            },
        ]

        tpm = build_probabilistic_boolean_tpm(adjacency, node_specs)
        states = enumerate_binary_states(4)
        row_all_on = next(i for i, state in enumerate(states) if tuple(state.tolist()) == (1, 1, 1, 0))
        row_one_off = next(i for i, state in enumerate(states) if tuple(state.tolist()) == (1, 1, 0, 0))
        next_node_on = states[:, 3] == 1

        probability_all_on = float(tpm[row_all_on, next_node_on].sum())
        probability_one_off = float(tpm[row_one_off, next_node_on].sum())

        self.assertAlmostEqual(probability_all_on, 1.0 / (1.0 + np.exp(-1.75)), places=8)
        self.assertAlmostEqual(probability_one_off, 1.0 / (1.0 + np.exp(0.25)), places=8)
        self.assertGreater(probability_all_on, probability_one_off)

    def test_matrix_heatmap_svg_renders_labels_and_values(self) -> None:
        matrix = np.array([[0.5, -0.5], [0.0, 1.0]])

        svg = render_matrix_heatmap_svg(
            "A",
            matrix,
            subtitle="toy coupling",
            row_labels=["x1", "x2"],
            col_labels=["x1", "x2"],
        )

        self.assertIn("toy coupling", svg)
        self.assertIn(">0.50<", svg)
        self.assertIn(">-0.50<", svg)
        self.assertIn(">x1<", svg)
        self.assertIn(">x2<", svg)
        self.assertIn("rect", svg)

    def test_render_causal_graph_svg_respects_hyperedge_opacity(self) -> None:
        svg = render_causal_graph_svg(
            "Toy",
            pairwise_matrix=np.zeros((3, 3), dtype=float),
            hyperedges=[
                {"sources": (0, 1), "target": 2, "value": 0.8, "label": "XOR", "opacity": 0.35},
            ],
            node_labels=["a", "b", "c"],
            hyperedge_threshold=0.0,
            legend_items=[("XOR", "#7b4ab5")],
        )

        self.assertIn("opacity='0.35'", svg)
        self.assertIn("XOR", svg)

    def test_render_causal_graph_svg_uses_hyperedge_value_to_scale_radius_even_with_fixed_opacity(self) -> None:
        svg = render_causal_graph_svg(
            "Toy",
            pairwise_matrix=np.zeros((3, 3), dtype=float),
            hyperedges=[
                {"sources": (0, 1), "target": 2, "value": 0.2, "label": "syn", "opacity": 0.9},
                {"sources": (0, 2), "target": 1, "value": 1.0, "label": "syn", "opacity": 0.9},
            ],
            node_labels=["a", "b", "c"],
            hyperedge_threshold=0.0,
        )

        self.assertIn("r='6.2'", svg)
        self.assertIn("r='11.0'", svg)
        self.assertEqual(svg.count("opacity='0.90'"), 8)

    def test_render_coarse_graining_comparison_svg_shows_mapping_lines_and_macro_arrows(self) -> None:
        svg = render_coarse_graining_comparison_svg(
            "Toy comparison",
            micro_pairwise=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.4, 0.0, 0.0],
                    [0.4, 0.4, 0.0, 0.0],
                ],
                dtype=float,
            ),
            micro_labels=["a1", "a2", "b1", "b2"],
            macro_labels=["A", "B"],
            groups=((0, 1), (2, 3)),
            macro_pairwise=np.array([[0.0, 0.5], [0.0, 0.0]], dtype=float),
            macro_hyperedges=[{"sources": (0, 1), "target": 1, "value": 0.8, "label": "syn", "opacity": 0.9}],
            micro_hyperedges=[{"sources": (0, 1), "target": 2, "value": 0.4, "label": "syn", "opacity": 0.9}],
        )

        self.assertIn("Toy comparison", svg)
        self.assertIn("marker-end='url(#cg-arrow)'", svg)
        self.assertIn("A(t)", svg)
        self.assertNotIn("coarse-graining bridge", svg)
        self.assertNotIn("targets at t+1", svg)
        self.assertIn("a1", svg)
        self.assertIn("B(t+1)", svg)
        self.assertIn("stroke='#d9822b'", svg)
        self.assertIn("stroke='#2f7d63'", svg)
        self.assertIn("fill='#f7e1cf'", svg)
        self.assertIn("fill='#d9eee7'", svg)

    def test_render_coarse_graining_comparison_svg_aligns_time_slices_and_draws_both_coarse_grainings(self) -> None:
        svg = render_coarse_graining_comparison_svg(
            "Toy comparison",
            micro_pairwise=np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.4, 0.0, 0.0],
                    [0.4, 0.4, 0.0, 0.0],
                ],
                dtype=float,
            ),
            micro_labels=["a1", "a2", "b1", "b2"],
            macro_labels=["A", "B"],
            groups=((0, 1), (2, 3)),
            macro_pairwise=np.array([[0.0, 0.5], [0.0, 0.0]], dtype=float),
            macro_hyperedges=[{"sources": (0, 1), "target": 1, "value": 0.8, "label": "syn", "opacity": 0.9}],
            micro_hyperedges=[{"sources": (0, 1), "target": 2, "value": 0.4, "label": "syn", "opacity": 0.9}],
        )

        mapping_paths = re.findall(
            r"<path d='M (-?[0-9.]+),(-?[0-9.]+) Q (-?[0-9.]+),(-?[0-9.]+) (-?[0-9.]+),(-?[0-9.]+)' fill='none' stroke='(#[0-9a-f]{6})'",
            svg,
        )
        self.assertTrue(mapping_paths)
        colored_mapping_paths = [path for path in mapping_paths if path[-1] in {"#d9822b", "#2f7d63"}]
        self.assertEqual(len(colored_mapping_paths), 4)
        self.assertTrue(all(float(y2) > float(y1) for _, y1, _, _, _, y2, _ in colored_mapping_paths))
        self.assertTrue(any(abs(float(cx) - float(x1)) > 1e-6 for x1, _, cx, _, _, _, _ in colored_mapping_paths))
        self.assertGreaterEqual(len({float(x1) for x1, _, _, _, _, _, _ in colored_mapping_paths}), 2)

    def test_render_downward_causation_comparison_svg_contains_expected_metrics(self) -> None:
        renderer = getattr(utils_module, "render_downward_causation_comparison_svg", None)
        self.assertIsNotNone(renderer)

        svg = render_downward_causation_comparison_svg(
            decoupling_parity_ei=1.0,
            decoupling_single_ei=0.0,
            decoupling_dc_values=[0.0, 0.0, 0.0],
            downward_target_ei=1.0,
            downward_single_ei=0.0,
            downward_dc_values=[1.0, 0.0, 0.0],
        )

        self.assertIn("Causal decoupling", svg)
        self.assertIn("Downward causation", svg)
        self.assertIn("EI(full -&gt; parity_next) = 1.00", svg)
        self.assertIn("EI(full -&gt; x1_next) = 1.00", svg)
        self.assertIn("DC_1 = 1.00", svg)
        self.assertIn("DC_2 = 0.00", svg)
        self.assertIn("stroke-dasharray", svg)
        self.assertIn("XOR", svg)
        self.assertIn("x1(t+1)", svg)
        self.assertIn("x2(t+1)", svg)
        self.assertIn("x3(t+1)", svg)
        self.assertLess(svg.index("x1(t+1)"), svg.index("x2(t+1)"))
        self.assertLess(svg.index("x2(t+1)"), svg.index("x3(t+1)"))

    def test_render_mixed_downward_causation_svg_omits_panel_label_and_metric_box(self) -> None:
        renderer = getattr(utils_module, "render_mixed_downward_causation_svg", None)
        self.assertIsNotNone(renderer)

        svg = renderer(
            full_ei=1.0,
            environment_ei=0.5,
            x3_ei=0.18872187554086717,
            flexibility=0.5,
            environment_synergy=0.31127812445913283,
            dc_value=0.8112781244591328,
        )

        self.assertIn("Mixed downward causation", svg)
        self.assertIn("AND", svg)
        self.assertIn("XOR", svg)
        self.assertIn("x1(t+1)", svg)
        self.assertIn("x2(t+1)", svg)
        self.assertIn("x3(t+1)", svg)
        self.assertLess(svg.index("x1(t+1)"), svg.index("x2(t+1)"))
        self.assertLess(svg.index("x2(t+1)"), svg.index("x3(t+1)"))
        self.assertNotIn(">C</text>", svg)
        self.assertNotIn("EI(full -&gt; x1_next)", svg)
        self.assertNotIn("EI(env -&gt; x1_next)", svg)
        self.assertNotIn("EI(x3 -&gt; x1_next)", svg)
        self.assertNotIn("Flexibility =", svg)
        self.assertNotIn("Environment synergy =", svg)
        self.assertNotIn("DC_1 =", svg)

    def test_experiment6_notebook_documents_mixed_case_and_two_component_decomposition(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_text = (project_root / "exp" / "experiment6_downward_causation.ipynb").read_text()

        self.assertIn("(x1(t) AND x2(t)) XOR x3(t)", notebook_text)
        self.assertIn("flexibility", notebook_text)
        self.assertIn("environment synergy", notebook_text)
        self.assertIn("0.500", notebook_text)
        self.assertIn("0.311", notebook_text)

    def test_experiment6_notebook_renders_mixed_case_figure(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        notebook_text = (project_root / "exp" / "experiment6_downward_causation.ipynb").read_text()

        self.assertIn("render_mixed_downward_causation_svg", notebook_text)
        self.assertIn("mixed_downward_causation.svg", notebook_text)

    def test_framework_doc_mentions_mixed_downward_causation_case(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        framework_text = (project_root / "docs" / "研究框架.md").read_text()

        self.assertIn("flexibility", framework_text)
        self.assertIn("environment synergy", framework_text)
        self.assertIn("\\land x_t^{(2)}", framework_text)
        self.assertIn("\\oplus x_t^{(3)}", framework_text)

    def test_scalar_ar1_stationary_covariance_matches_closed_form(self) -> None:
        coupling = np.array([[0.4]])
        noise_cov = np.array([[1.0]])

        covariance = solve_stationary_covariance(coupling, noise_cov)

        self.assertAlmostEqual(covariance[0, 0], 1.0 / (1.0 - 0.4**2), places=9)

    def test_gaussian_conditional_covariance_matches_two_variable_example(self) -> None:
        covariance = np.array([[1.0, 0.5], [0.5, 1.0]])

        conditional = gaussian_conditional_covariance(
            covariance,
            target_indices=[0],
            given_indices=[1],
        )

        self.assertAlmostEqual(conditional[0, 0], 0.75, places=9)

    def test_gaussian_mutual_information_matches_closed_form(self) -> None:
        correlation = 0.5
        covariance = np.array([[1.0, correlation], [correlation, 1.0]])

        mutual_information = gaussian_mutual_information(
            covariance,
            source_indices=[0],
            target_indices=[1],
            log_base=np.e,
        )

        self.assertAlmostEqual(
            mutual_information,
            -0.5 * np.log(1.0 - correlation**2),
            places=9,
        )

    def test_linear_gaussian_ei_decomposition_has_zero_synergy_for_diagonal_system(self) -> None:
        coupling = np.diag([0.6, 0.4])
        noise_cov = np.diag([0.2, 0.3])

        summary = linear_gaussian_ei_decomposition(
            coupling,
            noise_cov,
            intervention_scale=1.0,
            log_base=np.e,
        )

        self.assertAlmostEqual(summary["syn_high"], 0.0, places=9)
        self.assertAlmostEqual(
            summary["syn_high"],
            summary["ei_full"] - summary["un_budget"],
            places=9,
        )
        self.assertEqual(len(summary["ei_singles"]), 2)

    def test_linear_gaussian_mediano_metrics_vanish_for_uncoupled_system(self) -> None:
        coupling = np.diag([0.5, 0.25])
        noise_cov = np.diag([0.4, 0.6])

        summary = linear_gaussian_mediano_metrics(
            coupling,
            noise_cov,
            log_base=np.e,
        )

        self.assertAlmostEqual(summary["phi"], 0.0, places=9)
        self.assertAlmostEqual(summary["phi_tilde"], 0.0, places=9)
        self.assertAlmostEqual(summary["causal_density"], 0.0, places=9)
        expected_psi = min(
            -0.5 * np.log(1.0 - 0.5**2),
            -0.5 * np.log(1.0 - 0.25**2),
        )
        self.assertAlmostEqual(summary["psi"], expected_psi, places=9)
        self.assertIn("tdmi", summary)
        self.assertIn("mean_abs_corr", summary)
        self.assertGreaterEqual(summary["tdmi"], 0.0)
        self.assertAlmostEqual(summary["mean_abs_corr"], 0.0, places=9)

    def test_linear_gaussian_many_to_few_example_can_produce_large_synergy(self) -> None:
        n_nodes = 8
        coupling = np.zeros((n_nodes, n_nodes))
        coupling[0, :] = 0.6 / np.sqrt(n_nodes)
        coupling[1, :] = 0.6 / np.sqrt(n_nodes)
        noise_cov = (0.3**2) * np.eye(n_nodes)

        summary = linear_gaussian_ei_decomposition(
            coupling,
            noise_cov,
            intervention_scale=np.sqrt(12.0),
            log_base=np.e,
        )

        self.assertGreater(summary["syn_high"], 0.6)
        self.assertGreater(summary["rho_high"], 0.5)
        self.assertGreater(summary["ei_full"], summary["un_budget"])

    def test_uniform_box_linear_gaussian_matches_one_dimensional_closed_form(self) -> None:
        coupling = np.array([[2.0]])
        noise_cov = np.array([[0.25]])

        ei = linear_gaussian_uniform_box_effective_information(
            coupling,
            noise_cov,
            source_indices=[0],
            target_indices=[0],
            box_size=4.0,
            log_base=np.e,
        )

        expected = np.log((2.0 * 4.0) / (np.sqrt(2.0 * np.pi * np.e) * 0.5))
        self.assertAlmostEqual(ei, expected, places=9)

    def test_uniform_box_linear_gaussian_decomposition_has_zero_synergy_for_diagonal_system(self) -> None:
        coupling = np.diag([0.6, 0.4])
        noise_cov = np.diag([0.2, 0.3])

        summary = linear_gaussian_uniform_box_ei_decomposition(
            coupling,
            noise_cov,
            box_size=np.sqrt(12.0),
            log_base=np.e,
        )

        self.assertAlmostEqual(summary["syn_high"], 0.0, places=9)
        self.assertAlmostEqual(
            summary["syn_high"],
            summary["ei_full"] - summary["un_budget"],
            places=9,
        )
        self.assertEqual(len(summary["ei_singles"]), 2)

    def test_uniform_box_many_to_few_example_can_produce_large_synergy(self) -> None:
        n_nodes = 8
        coupling = np.zeros((n_nodes, n_nodes))
        coupling[0, :] = 0.6 / np.sqrt(n_nodes)
        coupling[1, :] = 0.6 / np.sqrt(n_nodes)
        noise_cov = (0.3**2) * np.eye(n_nodes)

        summary = linear_gaussian_uniform_box_ei_decomposition(
            coupling,
            noise_cov,
            box_size=np.sqrt(12.0),
            log_base=np.e,
        )

        self.assertGreater(summary["syn_high"], 8.0)
        self.assertGreater(summary["ei_full"], summary["un_budget"])

    def test_uniform_box_full_rank_two_node_overlap_can_still_produce_synergy(self) -> None:
        coupling = np.array([[0.8, 0.8], [0.2, 0.0]])
        noise_cov = (0.2**2) * np.eye(2)

        summary = linear_gaussian_uniform_box_ei_decomposition(
            coupling,
            noise_cov,
            box_size=3.5,
            log_base=np.e,
        )

        self.assertNotAlmostEqual(float(np.linalg.det(coupling)), 0.0, places=9)
        self.assertGreater(summary["syn_high"], 0.5)
        self.assertGreater(summary["rho_high"], 0.5)


class RQ3CausalEmergenceUtilityTests(unittest.TestCase):
    def test_build_deterministic_boolean_tpm_creates_row_stochastic_matrix(self) -> None:
        tpm = build_deterministic_boolean_tpm(2, lambda state: (state[1], state[0]))

        self.assertEqual(tpm.shape, (4, 4))
        np.testing.assert_allclose(tpm.sum(axis=1), np.ones(4))
        np.testing.assert_allclose(tpm.max(axis=1), np.ones(4))

    def test_enumerate_pair_partitions_lists_all_matchings(self) -> None:
        partitions = enumerate_pair_partitions(range(6))

        self.assertEqual(len(partitions), 15)
        self.assertIn(((0, 1), (2, 3), (4, 5)), partitions)

    def test_enumerate_partitions_fixed_blocks_lists_all_set_partitions(self) -> None:
        partitions = enumerate_partitions_fixed_blocks(range(4), 2)

        self.assertEqual(len(partitions), 7)
        self.assertIn(((0,), (1, 2, 3)), partitions)
        self.assertIn(((0, 1), (2, 3)), partitions)
        self.assertIn(((0, 1, 2), (3,)), partitions)

    def test_rq3_intended_partition_is_unique_ei_maximizer(self) -> None:
        tpm = build_rq3_boolean_network_tpm()
        results = search_binary_or_pair_coarse_grainings(tpm, n_nodes=6)

        self.assertGreater(len(results), 1)
        best = results[0]
        runner_up = results[1]

        self.assertEqual(best["groups"], ((0, 1), (2, 3), (4, 5)))
        self.assertAlmostEqual(best["ei"], 3.0, places=9)
        self.assertAlmostEqual(best["syn"], 0.0, places=9)
        self.assertGreater(best["ei"], runner_up["ei"])
        self.assertLessEqual(best["syn"], runner_up["syn"])

    def test_fixed_macro_dim_search_recovers_intended_rq3_partition(self) -> None:
        tpm = build_rq3_boolean_network_tpm()
        results = search_binary_or_fixed_macro_dim_coarse_grainings(
            tpm,
            n_nodes=6,
            n_macro=3,
        )

        self.assertEqual(len(results), 90)
        self.assertEqual(results[0]["groups"], ((0, 1), (2, 3), (4, 5)))
        self.assertAlmostEqual(results[0]["ei"], 3.0, places=9)
        self.assertAlmostEqual(results[0]["syn"], 0.0, places=9)
        self.assertIn("compact_fraction", results[0])
        self.assertGreaterEqual(float(results[0]["compact_fraction"]), 0.0)
        self.assertLessEqual(float(results[0]["compact_fraction"]), 1.0)
        self.assertAlmostEqual(
            float(results[0]["compact_intra"]) + float(results[0]["compact_inter"]),
            float(results[0]["compact_total"]),
            places=9,
        )

    def test_subset_synergy_scores_cover_all_nontrivial_subsets(self) -> None:
        tpm = build_deterministic_boolean_tpm(4, lambda state: state)
        scores = subset_synergy_scores(tpm, n_nodes=4)

        self.assertEqual(len(scores), 11)
        self.assertIn((0, 1), scores)
        self.assertIn((0, 1, 2), scores)
        self.assertIn((0, 1, 2, 3), scores)

    def test_subset_synergy_scores_use_matching_future_subset_as_target(self) -> None:
        tpm = build_deterministic_boolean_tpm(3, lambda state: state)
        recorded_targets: list[tuple[int, ...]] = []

        def fake_discrete_effective_information(*args, **kwargs):
            recorded_targets.append(tuple(int(index) for index in kwargs["target_indices"]))
            return 0.0

        with patch.object(utils_module, "discrete_effective_information", side_effect=fake_discrete_effective_information):
            subset_synergy_scores(tpm, n_nodes=3)

        self.assertIn((0, 1), recorded_targets)
        self.assertIn((0, 2), recorded_targets)
        self.assertIn((1, 2), recorded_targets)
        self.assertIn((0, 1, 2), recorded_targets)
        self.assertNotIn((0, 1, 2, 3), recorded_targets)
        self.assertNotIn((0, 1, 2), recorded_targets[:3])

    def test_coarse_graining_subset_synergy_compactness_splits_total_score(self) -> None:
        score = coarse_graining_subset_synergy_compactness(
            groups=((0, 1), (2, 3), (4,)),
            subset_scores={
                (0, 1): 0.8,
                (2, 3): 0.6,
                (0, 1, 2): 0.5,
            },
        )

        self.assertAlmostEqual(score["intra_score"], 1.4, places=9)
        self.assertAlmostEqual(score["inter_score"], 0.5, places=9)
        self.assertAlmostEqual(score["total_score"], 1.9, places=9)
        self.assertAlmostEqual(score["compact_fraction"], 1.4 / 1.9, places=9)
        self.assertEqual(score["selected_subsets"], ((0, 1), (2, 3)))

    def test_system_ei_decomposition_uses_joint_next_state_as_target(self) -> None:
        tpm = build_rq3_boolean_network_tpm()
        summary = system_ei_decomposition(tpm, n_nodes=6)
        ei_full = effective_information_from_tpm(tpm)

        self.assertAlmostEqual(summary["ei_full"], ei_full, places=9)
        self.assertEqual(len(summary["single_eis"]), 6)
        self.assertAlmostEqual(summary["syn"], 0.5661656266226018, places=9)
        self.assertAlmostEqual(summary["rho_syn"], summary["syn"] / ei_full, places=9)

    def test_rq3_intended_macro_has_single_synergistic_target(self) -> None:
        tpm = build_rq3_boolean_network_tpm()
        macro_tpm = coarse_grain_binary_or_tpm(
            tpm,
            n_nodes=6,
            groups=((0, 1), (2, 3), (4, 5)),
        )

        summary = target_ei_decomposition(macro_tpm, n_nodes=3)
        system_summary = system_ei_decomposition(macro_tpm, n_nodes=3)

        self.assertAlmostEqual(summary["ei_full"], 3.0, places=9)
        self.assertAlmostEqual(summary["syn_budget"], 1.0, places=9)
        self.assertAlmostEqual(summary["rho_syn"], 1.0 / 3.0, places=9)
        self.assertAlmostEqual(system_summary["syn"], 0.0, places=9)
        self.assertAlmostEqual(system_summary["rho_syn"], 0.0, places=9)
        self.assertEqual(len(summary["targets"]), 3)
        self.assertAlmostEqual(summary["targets"][0]["synergy"], 0.0, places=9)
        self.assertAlmostEqual(summary["targets"][1]["synergy"], 0.0, places=9)
        self.assertAlmostEqual(summary["targets"][2]["synergy"], 1.0, places=9)

    def test_target_ei_decomposition_uses_system_ei_in_rho_syn(self) -> None:
        tpm = build_rq3_boolean_network_tpm()

        summary = target_ei_decomposition(tpm, n_nodes=6)
        ei_full = effective_information_from_tpm(tpm)

        self.assertAlmostEqual(summary["ei_full"], ei_full, places=9)
        self.assertAlmostEqual(
            summary["rho_syn"],
            summary["syn_budget"] / ei_full,
            places=9,
        )

    def test_marshall_example1_search_includes_paper_mapping_and_ranks_by_ei(self) -> None:
        tpm = build_marshall_example1_tpm()
        mappings = enumerate_surjective_binary_mappings(2)
        results = search_marshall_example1_macro_mappings(tpm)

        self.assertEqual(len(mappings), 14)
        self.assertEqual(len(results), 14 * 14)

        paper_mapping = (0, 0, 0, 1)
        matching = [
            row
            for row in results
            if row["alpha_mapping"] == paper_mapping and row["beta_mapping"] == paper_mapping
        ]
        self.assertEqual(len(matching), 1)

        paper_result = matching[0]
        self.assertAlmostEqual(paper_result["macro_tpm"].shape[0], 4)
        self.assertAlmostEqual(paper_result["macro_tpm"].shape[1], 4)

        eis = [float(row["ei"]) for row in results]
        self.assertEqual(eis, sorted(eis, reverse=True))
        self.assertGreaterEqual(float(results[0]["ei"]), float(paper_result["ei"]))
        self.assertEqual(tuple(results[0]["groups"]), ((0, 1), (2, 3)))

    def test_marshall_example1_global_pair_partition_search_keeps_paper_partition_best(self) -> None:
        results = search_marshall_example1_all_pair_partitions()

        self.assertEqual(len(results), 3 * 14 * 14)

        best = results[0]
        self.assertEqual(best["groups"], ((0, 1), (2, 3)))
        self.assertEqual(best["alpha_mapping"], (0, 0, 0, 1))
        self.assertEqual(best["beta_mapping"], (0, 0, 0, 1))
        self.assertAlmostEqual(best["ei"], 1.2735731818301856, places=12)

        best_by_partition: dict[tuple[tuple[int, int], ...], float] = {}
        for row in results:
            groups = tuple(row["groups"])
            best_by_partition.setdefault(groups, float(row["ei"]))

        self.assertEqual(
            set(best_by_partition),
            {((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))},
        )
        self.assertGreater(best_by_partition[((0, 1), (2, 3))], best_by_partition[((0, 2), (1, 3))])
        self.assertGreater(best_by_partition[((0, 1), (2, 3))], best_by_partition[((0, 3), (1, 2))])


if __name__ == "__main__":
    unittest.main()
