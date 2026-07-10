import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_hcp_lausanne_phi_eid_pilot as pilot


class HcpLausannePhiEidPilotTests(unittest.TestCase):
    def test_freesurfer_mapping_has_83_non_unknown_rois(self) -> None:
        mapping = pilot.build_freesurfer_lausanne83_mapping()

        self.assertEqual(len(mapping), 83)
        self.assertNotIn("Unknown", set(mapping.values()))
        self.assertEqual(mapping[16], "Brain-Stem")
        self.assertEqual(mapping[1005], "ctx-lh-cuneus")
        self.assertEqual(mapping[2005], "ctx-rh-cuneus")

    def test_lagged_samples_make_one_step_transition(self) -> None:
        series = np.arange(5 * 3, dtype=float).reshape(5, 3)

        source, target = pilot.make_lagged_samples(series, tau=1)

        np.testing.assert_array_equal(source, series[:-1])
        np.testing.assert_array_equal(target, series[1:])

    def test_circular_shift_null_preserves_each_roi_values(self) -> None:
        series = np.arange(20, dtype=float).reshape(5, 4)

        shifted = pilot.circular_shift_null(series, seed=7)

        self.assertEqual(shifted.shape, series.shape)
        for column in range(series.shape[1]):
            self.assertEqual(sorted(shifted[:, column].tolist()), sorted(series[:, column].tolist()))
        self.assertFalse(np.array_equal(shifted, series))

    def test_load_cached_roi_timeseries_roundtrips_saved_npz(self) -> None:
        labels = ["roi-a", "roi-b"]
        series = np.arange(12, dtype=float).reshape(6, 2)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "roi_timeseries"
            pilot.save_roi_timeseries(
                cache_dir.parent,
                subject="subj",
                run="REST",
                series=series,
                labels=labels,
                metadata={"synthetic": False},
            )

            loaded, metadata = pilot.load_cached_roi_timeseries(
                cache_dir,
                subject="subj",
                run="REST",
                expected_labels=labels,
            )

        np.testing.assert_array_equal(loaded, series)
        self.assertEqual(metadata["synthetic"], False)

    def test_hcp_result_run_name_distinguishes_rest_and_task_fmri(self) -> None:
        self.assertEqual(pilot.hcp_result_run_name("REST1_LR"), "rfMRI_REST1_LR")
        self.assertEqual(pilot.hcp_result_run_name("rfMRI_REST1_RL"), "rfMRI_REST1_RL")
        self.assertEqual(pilot.hcp_result_run_name("WM_LR"), "tfMRI_WM_LR")
        self.assertEqual(pilot.hcp_result_run_name("tfMRI_WM_RL"), "tfMRI_WM_RL")

    def test_aws_command_prefers_executable_then_python_module(self) -> None:
        command = pilot.aws_cli_command(aws_executable="/bin/aws", python_executable="/bin/python")
        self.assertEqual(command, ["/bin/aws"])

        module_command = pilot.aws_cli_command(aws_executable=None, python_executable="/bin/python")
        self.assertEqual(module_command, ["/bin/python", "-m", "awscli"])

    def test_gaussian_phi_eid_distinguishes_additive_and_coupled_systems(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.normal(size=(500, 3))
        additive_target = source + 0.05 * rng.normal(size=(500, 3))
        coupled_target = np.column_stack(
            [
                source[:, 0] + 0.7 * source[:, 1],
                source[:, 1] + 0.7 * source[:, 2],
                source[:, 2] + 0.7 * source[:, 0],
            ]
        ) + 0.05 * rng.normal(size=(500, 3))

        additive = pilot.gaussian_singleton_source_phi(source, additive_target, ridge=1.0e-6)
        coupled = pilot.gaussian_singleton_source_phi(source, coupled_target, ridge=1.0e-6)

        self.assertLess(abs(float(additive["raw_phi"])), 0.35)
        self.assertGreater(float(coupled["raw_phi"]), float(additive["raw_phi"]) + 0.2)

    def test_fit_mlp_transition_returns_validation_metrics(self) -> None:
        rng = np.random.default_rng(21)
        source = rng.normal(size=(80, 4))
        target = np.tanh(source @ np.array(
            [
                [0.4, -0.1, 0.0, 0.2],
                [0.2, 0.3, -0.2, 0.0],
                [0.0, 0.1, 0.5, -0.3],
                [-0.1, 0.0, 0.2, 0.4],
            ]
        ))

        result = pilot.fit_mlp_transition(source, target, hidden_dim=8, epochs=2, seed=3)

        self.assertIn("metrics", result)
        self.assertTrue(np.isfinite(result["metrics"]["rmse"]))

    def test_greedy_module_atoms_sum_to_root_phi(self) -> None:
        ei_table = {
            ("A",): 0.2,
            ("B",): 0.3,
            ("C",): 0.1,
            ("A", "B"): 0.9,
            ("A", "C"): 0.4,
            ("B", "C"): 0.5,
            ("A", "B", "C"): 1.3,
        }

        atoms = pilot.greedy_phi_atoms(("A", "B", "C"), ei_table, eps=1.0e-12)
        atom_sum = sum(atom.value for atom in atoms)
        root_phi = pilot.subset_phi_raw(
            ("A", "B", "C"),
            ei_table,
            {"A": 0.2, "B": 0.3, "C": 0.1},
        )

        self.assertAlmostEqual(atom_sum, root_phi)


if __name__ == "__main__":
    unittest.main()
