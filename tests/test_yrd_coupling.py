import unittest
from unittest.mock import patch

import numpy as np
import torch

from yrd.coupling import (
    build_one_step_station_source_groups,
    build_one_step_station_pollutant_feature_groups,
    build_station_source_groups,
    compute_group_ei_summary,
    compute_group_synergy_summary,
    compute_station_level_ei_summary,
    compute_subset_nis_summary,
    compute_station_pollutant_pair_synergy_summary,
    compute_station_level_nis_summary,
    jacobian_for_target_subset,
    select_evenly_spaced_indices,
    summarize_global_station_single_pollutant_ei,
    summarize_global_station_pollutant_synergy,
    summarize_global_station_coupling,
    summarize_coupling_summaries,
)
from yrd.intervention_sampling import (
    collapse_support_cover_box_profile_to_global_max,
    compute_training_input_center,
    estimate_support_cover_box_profile,
    sample_uniform_box_inputs,
)


class YRDJacobianTests(unittest.TestCase):
    def test_jacobian_for_target_subset_returns_selected_rows(self) -> None:
        linear = torch.nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0, 0.0],
                        [0.0, 0.0, 3.0, 0.0],
                    ]
                )
            )
        x = torch.ones(4, requires_grad=True)
        jac = jacobian_for_target_subset(linear, x, target_indices=[0, 2])
        self.assertEqual(tuple(jac.shape), (2, 4))


class YRDNisSummaryTests(unittest.TestCase):
    def test_compute_training_input_center_matches_mean_over_train_samples(self) -> None:
        x_train = np.array(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[2.0, 3.0], [4.0, 5.0]],
            ],
            dtype=np.float32,
        )

        center = compute_training_input_center(x_train)

        np.testing.assert_allclose(center, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    def test_sample_uniform_box_inputs_accepts_per_feature_widths_and_nonnegative_clipping(self) -> None:
        center = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
        box_size = np.array([2.0, 4.0], dtype=np.float32)
        lower_bounds = np.array([0.75, -3.0], dtype=np.float32)

        sample_a = sample_uniform_box_inputs(
            center=center,
            box_size=box_size,
            sample_count=4,
            seed=7,
            lower_bounds=lower_bounds,
        )
        sample_b = sample_uniform_box_inputs(
            center=center,
            box_size=box_size,
            sample_count=4,
            seed=7,
            lower_bounds=lower_bounds,
        )

        np.testing.assert_allclose(sample_a, sample_b)
        self.assertEqual(sample_a.shape, (4, 2, 2))
        self.assertTrue(np.all(sample_a[..., 0] >= 0.75))
        self.assertTrue(np.all(sample_a[..., 0] <= np.array([[2.0, 1.5]], dtype=np.float32)))
        self.assertTrue(np.all(sample_a[..., 1] >= np.array([[-3.0, -2.5]], dtype=np.float32)))
        self.assertTrue(np.all(sample_a[..., 1] <= np.array([[1.0, 1.5]], dtype=np.float32)))

    def test_estimate_support_cover_box_profile_covers_train_support_and_nonnegative_bounds(self) -> None:
        x_train = np.array(
            [
                [[1.0, -1.0], [3.0, 0.0]],
                [[5.0, 1.0], [7.0, 2.0]],
            ],
            dtype=np.float32,
        )

        profile = estimate_support_cover_box_profile(
            x_train=x_train,
            input_variables=("O3", "PM2.5"),
            gamma=1.10,
            stats={
                "O3": {"mean": 10.0, "std": 2.0},
                "PM2.5": {"mean": 20.0, "std": 5.0},
            },
            nonnegative_variables=("O3",),
        )

        np.testing.assert_allclose(
            profile["box_size_by_feature"],
            np.array([[4.4, 2.2], [4.4, 2.2]], dtype=np.float32),
        )
        self.assertAlmostEqual(profile["cover_radius_by_variable"]["O3"], 2.0)
        self.assertAlmostEqual(profile["cover_radius_by_variable"]["PM2.5"], 1.0)
        np.testing.assert_array_less(profile["support_low_by_feature"] - 1e-6, profile["feature_min"])
        np.testing.assert_array_less(profile["feature_max"], profile["support_high_by_feature"] + 1e-6)
        np.testing.assert_allclose(profile["lower_bounds"][:, 0], np.array([-5.0, -5.0], dtype=np.float32))
        self.assertTrue(np.isneginf(profile["lower_bounds"][:, 1]).all())

    def test_collapse_support_cover_box_profile_to_global_max_uses_single_l(self) -> None:
        profile = estimate_support_cover_box_profile(
            x_train=np.array(
                [
                    [[1.0, -1.0], [3.0, 0.0]],
                    [[5.0, 1.0], [7.0, 2.0]],
                ],
                dtype=np.float32,
            ),
            input_variables=("O3", "PM2.5"),
            gamma=1.10,
            stats={
                "O3": {"mean": 10.0, "std": 2.0},
                "PM2.5": {"mean": 20.0, "std": 5.0},
            },
            nonnegative_variables=("O3",),
        )

        scalar_profile = collapse_support_cover_box_profile_to_global_max(profile)

        self.assertEqual(scalar_profile["box_mode"], "global_max")
        self.assertAlmostEqual(scalar_profile["global_box_size"], 4.4, places=6)
        self.assertAlmostEqual(
            scalar_profile["original_box_size_by_variable"]["O3"],
            4.4,
            places=6,
        )
        self.assertAlmostEqual(
            scalar_profile["original_box_size_by_variable"]["PM2.5"],
            2.2,
            places=6,
        )
        self.assertAlmostEqual(
            scalar_profile["box_size_by_variable"]["O3"],
            4.4,
            places=6,
        )
        self.assertAlmostEqual(
            scalar_profile["box_size_by_variable"]["PM2.5"],
            4.4,
            places=6,
        )
        np.testing.assert_allclose(
            scalar_profile["box_size_by_feature"],
            np.array([[4.4, 4.4], [4.4, 4.4]], dtype=np.float32),
        )

    def test_compute_subset_nis_summary_returns_named_metrics(self) -> None:
        jacobian = np.array([[1.0, 0.0], [0.0, 1.5]], dtype=float)
        sigma_eps = np.eye(2) * 0.1
        summary = compute_subset_nis_summary(
            jacobian=jacobian,
            sigma_eps=sigma_eps,
            source_groups={"src_a": [0], "src_b": [1]},
            target_indices=[0, 1],
        )
        self.assertIn("ei_nis", summary)
        self.assertIn("syn_nis", summary)
        self.assertIn("group_ei_nis", summary)

    def test_select_evenly_spaced_indices_spans_full_range(self) -> None:
        indices = select_evenly_spaced_indices(n_samples=10, sample_count=4)
        self.assertEqual(indices, [0, 3, 6, 9])

    def test_summarize_coupling_summaries_reports_center_and_spread(self) -> None:
        summary = summarize_coupling_summaries(
            [
                {
                    "ei_nis": 1.0,
                    "syn_nis": 2.0,
                    "group_ei_nis": {"local_o3_history": 3.0, "cross_station_pollutants": 4.0},
                },
                {
                    "ei_nis": 5.0,
                    "syn_nis": 6.0,
                    "group_ei_nis": {"local_o3_history": 7.0, "cross_station_pollutants": 8.0},
                },
            ]
        )

        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["ei_nis"]["mean"], 3.0)
        self.assertAlmostEqual(summary["syn_nis"]["median"], 4.0)
        self.assertAlmostEqual(summary["group_ei_nis"]["local_o3_history"]["std"], 2.0)

    def test_compute_subset_nis_summary_accepts_target_subset_with_full_covariance(self) -> None:
        jacobian = np.array([[1.0, 0.2], [0.1, 1.5]], dtype=float)
        sigma_eps = np.eye(4) * 0.1
        summary = compute_subset_nis_summary(
            jacobian=jacobian,
            sigma_eps=sigma_eps,
            source_groups={"src_a": [0], "src_b": [1]},
            target_indices=[1, 3],
        )

        self.assertIn("ei_nis", summary)
        self.assertIn("src_a", summary["group_ei_nis"])

    def test_build_station_source_groups_partitions_flattened_input_by_station(self) -> None:
        groups = build_station_source_groups(history_hours=2, n_stations=3, n_features=4)
        self.assertEqual(list(groups), ["0", "1", "2"])
        self.assertEqual(len(groups["0"]), 8)
        self.assertTrue(set(groups["0"]).isdisjoint(groups["1"]))

    def test_build_one_step_station_source_groups_assigns_n_features_per_station(self) -> None:
        groups = build_one_step_station_source_groups(
            n_stations=3,
            n_features=10,
            station_ids=["A", "B", "C"],
        )
        self.assertEqual(list(groups), ["A", "B", "C"])
        self.assertEqual(len(groups["A"]), 10)
        self.assertEqual(len(groups["B"]), 10)
        self.assertEqual(sorted(groups["A"]), list(range(10)))
        self.assertTrue(set(groups["A"]).isdisjoint(groups["B"]))

    def test_build_one_step_station_pollutant_feature_groups_selects_o3_and_pm25(self) -> None:
        groups = build_one_step_station_pollutant_feature_groups(
            n_stations=3,
            n_features=10,
            pollutant_feature_indices={"O3": 0, "PM2.5": 1},
            station_ids=["A", "B", "C"],
        )

        self.assertEqual(list(groups), ["A", "B", "C"])
        self.assertEqual(groups["A"]["O3"], [0])
        self.assertEqual(groups["A"]["PM2.5"], [1])
        self.assertEqual(groups["B"]["O3"], [10])
        self.assertEqual(groups["B"]["PM2.5"], [11])
        self.assertTrue(set(groups["A"]["O3"] + groups["A"]["PM2.5"]).isdisjoint(groups["B"]["O3"] + groups["B"]["PM2.5"]))

    def test_compute_station_level_nis_summary_accepts_one_step_station_groups(self) -> None:
        groups = build_one_step_station_source_groups(
            n_stations=3,
            n_features=2,
            station_ids=["A", "B", "C"],
        )
        summary = compute_station_level_nis_summary(
            jacobian=np.array([[1.0, 0.5, 0.2, 0.1, 0.8, 0.4]], dtype=float),
            sigma_eps=np.eye(1) * 0.1,
            station_source_groups=groups,
            target_indices=[0],
        )

        self.assertIn("pairwise_station_ei_nis", summary)
        self.assertEqual(set(summary["pairwise_station_ei_nis"]), {"A", "B", "C"})

    def test_compute_station_pollutant_pair_synergy_summary_reports_station_edges(self) -> None:
        pollutant_groups = build_one_step_station_pollutant_feature_groups(
            n_stations=2,
            n_features=2,
            pollutant_feature_indices={"O3": 0, "PM2.5": 1},
            station_ids=["A", "B"],
        )
        summary = compute_station_pollutant_pair_synergy_summary(
            jacobian=np.array([[1.13672165, 0.95479425, 0.0, 0.0]], dtype=float),
            sigma_eps=np.array([[0.91102859]], dtype=float),
            station_pollutant_feature_groups=pollutant_groups,
            target_indices=[0],
        )

        self.assertIn("station_pair_synergy_nis", summary)
        self.assertIn("joint_station_pair_ei_nis", summary)
        self.assertGreater(summary["station_pair_synergy_nis"]["A"], 0.0)
        self.assertAlmostEqual(summary["station_pair_synergy_nis"]["B"], 0.0, places=8)

    def test_summarize_global_station_coupling_returns_edges_and_binary_hyperedges(self) -> None:
        station_groups = build_station_source_groups(
            history_hours=1,
            n_stations=3,
            n_features=1,
            station_ids=["A", "B", "C"],
        )
        summary_ab = compute_station_level_nis_summary(
            jacobian=np.array([[1.0, 0.8, 0.2]], dtype=float),
            sigma_eps=np.eye(1) * 0.1,
            station_source_groups=station_groups,
            target_indices=[0],
        )
        summary_c = compute_station_level_nis_summary(
            jacobian=np.array([[0.1, 0.4, 1.1]], dtype=float),
            sigma_eps=np.eye(1) * 0.1,
            station_source_groups=station_groups,
            target_indices=[0],
        )
        aggregated = summarize_global_station_coupling(
            [
                {"target_station_id": "A", **summary_ab},
                {"target_station_id": "A", **summary_ab},
                {"target_station_id": "C", **summary_c},
            ],
            station_ids=["A", "B", "C"],
        )

        self.assertTrue(aggregated["pairwise_edges"])
        self.assertTrue(aggregated["binary_hyperedges"])
        self.assertIn("A", aggregated["per_target_station"])
        self.assertTrue(all("target_station_id" in row for row in aggregated["pairwise_edges"]))
        self.assertTrue(all("source_station_ids" in row for row in aggregated["binary_hyperedges"]))

    def test_summarize_global_station_pollutant_synergy_returns_directed_edges(self) -> None:
        aggregated = summarize_global_station_pollutant_synergy(
            [
                {
                    "target_station_id": "A",
                    "station_pair_synergy_nis": {"A": 0.6, "B": 0.2},
                    "joint_station_pair_ei_nis": {"A": 1.1, "B": 0.8},
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.3, "PM2.5": 0.2},
                        "B": {"O3": 0.4, "PM2.5": 0.2},
                    },
                },
                {
                    "target_station_id": "A",
                    "station_pair_synergy_nis": {"A": 0.4, "B": 0.1},
                    "joint_station_pair_ei_nis": {"A": 0.9, "B": 0.7},
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.2, "PM2.5": 0.3},
                        "B": {"O3": 0.3, "PM2.5": 0.3},
                    },
                },
                {
                    "target_station_id": "B",
                    "station_pair_synergy_nis": {"A": 0.5, "B": 0.0},
                    "joint_station_pair_ei_nis": {"A": 1.0, "B": 0.6},
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.3, "PM2.5": 0.2},
                        "B": {"O3": 0.3, "PM2.5": 0.3},
                    },
                },
            ],
            station_ids=["A", "B"],
        )

        self.assertTrue(aggregated["conditional_synergy_edges"])
        self.assertIn("A", aggregated["per_target_station"])
        self.assertTrue(all("source_station_id" in row for row in aggregated["conditional_synergy_edges"]))
        a_to_a = next(
            row for row in aggregated["conditional_synergy_edges"]
            if row["source_station_id"] == "A" and row["target_station_id"] == "A"
        )
        self.assertAlmostEqual(a_to_a["mean"], 0.5)

    def test_summarize_global_station_pollutant_synergy_reports_syn_to_joint_ei_ratio_edges(self) -> None:
        aggregated = summarize_global_station_pollutant_synergy(
            [
                {
                    "target_station_id": "A",
                    "station_pair_synergy_nis": {"A": 0.9, "B": 0.3},
                    "joint_station_pair_ei_nis": {"A": 0.5, "B": 0.25},
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.3, "PM2.5": 0.3},
                        "B": {"O3": 0.1, "PM2.5": 0.2},
                    },
                },
                {
                    "target_station_id": "A",
                    "station_pair_synergy_nis": {"A": 0.3, "B": 0.1},
                    "joint_station_pair_ei_nis": {"A": 1.5, "B": 0.75},
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.2, "PM2.5": 0.2},
                        "B": {"O3": 0.2, "PM2.5": 0.2},
                    },
                },
            ],
            station_ids=["A", "B"],
        )

        self.assertIn("conditional_synergy_ratio_edges", aggregated)
        ratio_row = next(
            row for row in aggregated["conditional_synergy_ratio_edges"]
            if row["source_station_id"] == "A" and row["target_station_id"] == "A"
        )
        self.assertAlmostEqual(ratio_row["mean"], 0.6)
        self.assertAlmostEqual(abs(ratio_row["mean"]), 0.6)
        self.assertIn("conditional_synergy_ratio_nis", aggregated["per_target_station"]["A"])
        self.assertAlmostEqual(
            aggregated["per_target_station"]["A"]["conditional_synergy_ratio_nis"]["A"]["mean"],
            0.6,
        )

    def test_summarize_global_station_single_pollutant_ei_returns_pm25_edges(self) -> None:
        aggregated = summarize_global_station_single_pollutant_ei(
            [
                {
                    "target_station_id": "A",
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.2, "PM2.5": 0.4},
                        "B": {"O3": 0.1, "PM2.5": 0.3},
                    },
                },
                {
                    "target_station_id": "A",
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.4, "PM2.5": 0.6},
                        "B": {"O3": 0.2, "PM2.5": 0.1},
                    },
                },
                {
                    "target_station_id": "B",
                    "single_pollutant_ei_nis": {
                        "A": {"O3": 0.5, "PM2.5": 0.7},
                        "B": {"O3": 0.2, "PM2.5": 0.2},
                    },
                },
            ],
            station_ids=["A", "B"],
            feature_name="PM2.5",
        )

        self.assertTrue(aggregated["pairwise_edges"])
        self.assertIn("A", aggregated["per_target_station"])
        self.assertIn("single_feature_ei_nis", aggregated["per_target_station"]["A"])
        self.assertAlmostEqual(
            aggregated["per_target_station"]["A"]["single_feature_ei_nis"]["A"]["mean"],
            0.5,
        )
        a_to_a = next(
            row for row in aggregated["pairwise_edges"]
            if row["source_station_id"] == "A" and row["target_station_id"] == "A"
        )
        self.assertAlmostEqual(a_to_a["mean"], 0.5)

    def test_compute_group_ei_summary_supports_tm_backend(self) -> None:
        source_samples = np.array(
            [
                [0.0, 0.0, 0.2],
                [0.5, 0.1, 0.0],
                [1.0, 0.2, -0.2],
                [1.5, 0.3, 0.1],
                [2.0, 0.4, 0.0],
                [2.5, 0.5, -0.1],
            ],
            dtype=float,
        )
        target_samples = np.array(
            [
                [0.2 * row[0] - 0.1 * row[1] + 0.3 * row[2]]
                for row in source_samples
            ],
            dtype=float,
        )

        summary = compute_group_ei_summary(
            method="tm",
            source_groups={"o3_block": [0, 1], "pm25_block": [2]},
            source_samples=source_samples,
            target_samples=target_samples,
        )

        self.assertEqual(summary["method"], "tm")
        self.assertIn("backend", summary)
        self.assertIn("group_ei", summary)
        self.assertEqual(set(summary["group_ei"]), {"o3_block", "pm25_block"})
        self.assertTrue(np.isfinite(summary["ei"]))
        self.assertTrue(all(np.isfinite(value) for value in summary["group_ei"].values()))

    def test_compute_group_synergy_summary_switches_between_nis_and_tm(self) -> None:
        tm_source_samples = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.1],
                [1.0, 0.2],
                [1.5, 0.3],
                [2.0, 0.4],
                [2.5, 0.5],
            ],
            dtype=float,
        )
        tm_target_samples = np.array(
            [[0.7 * row[0] - 0.2 * row[1]] for row in tm_source_samples],
            dtype=float,
        )

        tm_summary = compute_group_synergy_summary(
            method="tm",
            source_groups={"o3": [0], "pm25": [1]},
            source_samples=tm_source_samples,
            target_samples=tm_target_samples,
        )
        self.assertEqual(tm_summary["method"], "tm")
        self.assertTrue(np.isfinite(tm_summary["ei"]))
        self.assertTrue(np.isfinite(tm_summary["syn"]))

        nis_summary = compute_group_synergy_summary(
            method="nis",
            source_groups={"o3": [0], "pm25": [1]},
            jacobian=np.array([[1.13672165, 0.95479425]], dtype=float),
            sigma_eps=np.array([[0.91102859]], dtype=float),
            target_indices=[0],
        )
        self.assertEqual(nis_summary["method"], "nis")
        self.assertIn("group_ei", nis_summary)
        self.assertGreater(nis_summary["syn"], 0.0)

    def test_compute_group_synergy_summary_clips_negative_ei_before_syn_for_tm(self) -> None:
        with patch(
            "yrd.coupling.estimate_mutual_information_transport_map",
            side_effect=[
                {"mi_hat": -0.2, "backend": "tm"},
                {"mi_hat": 0.3, "backend": "tm"},
                {"mi_hat": -0.1, "backend": "tm"},
            ],
        ):
            summary = compute_group_synergy_summary(
                method="tm",
                source_groups={"o3": [0], "pm25": [1]},
                source_samples=np.zeros((4, 2), dtype=float),
                target_samples=np.zeros((4, 1), dtype=float),
            )

        self.assertEqual(summary["ei"], 0.0)
        self.assertEqual(summary["group_ei"]["o3"], 0.0)
        self.assertEqual(summary["group_ei"]["pm25"], 0.3)
        self.assertAlmostEqual(summary["syn"], -0.3)

    def test_compute_group_synergy_summary_clips_negative_ei_before_syn_for_nis(self) -> None:
        def fake_subset_ei(
            _jacobian: np.ndarray,
            _sigma_eps: np.ndarray,
            subset: list[int],
            *,
            box_size: float,
            atol: float,
        ) -> float:
            del box_size, atol
            key = tuple(subset)
            if key == (0,):
                return -0.2
            if key == (1,):
                return 0.3
            if key == (0, 1):
                return -0.1
            raise AssertionError(f"unexpected subset: {key}")

        with patch("yrd.coupling._subset_ei_nis", side_effect=fake_subset_ei):
            summary = compute_group_synergy_summary(
                method="nis",
                source_groups={"o3": [0], "pm25": [1]},
                jacobian=np.zeros((1, 2), dtype=float),
                sigma_eps=np.eye(1, dtype=float),
                target_indices=[0],
            )

        self.assertEqual(summary["ei"], 0.0)
        self.assertEqual(summary["group_ei"]["o3"], 0.0)
        self.assertEqual(summary["group_ei"]["pm25"], 0.3)
        self.assertAlmostEqual(summary["syn"], -0.3)

    def test_transport_map_source_lift_handles_one_and_two_dimensional_blocks(self) -> None:
        from yrd.transport_map import lift_transport_source_features

        one_dim = np.array([[0.0], [1.0], [2.0]], dtype=float)
        two_dim = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=float)

        lifted_one = lift_transport_source_features(one_dim)
        lifted_two = lift_transport_source_features(two_dim)

        self.assertEqual(lifted_one.shape, (3, 3))
        self.assertEqual(lifted_two.shape, (3, 5))
        self.assertTrue(np.allclose(lifted_two[:, 2], two_dim[:, 0] * two_dim[:, 1]))

    def test_transport_map_two_source_synergy_helper_returns_finite_scores(self) -> None:
        from yrd.transport_map import summarize_two_source_synergy_transport_map

        left = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]], dtype=float)
        right = np.array([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]], dtype=float)
        target = np.array([[np.sin(l[0] * r[0])] for l, r in zip(left, right)], dtype=float)

        summary = summarize_two_source_synergy_transport_map(left, right, target)

        self.assertTrue(np.isfinite(summary["left_ei"]))
        self.assertTrue(np.isfinite(summary["right_ei"]))
        self.assertTrue(np.isfinite(summary["joint_ei"]))
        self.assertTrue(np.isfinite(summary["syn"]))

    def test_tm_station_summaries_use_requested_o3_and_pm25_blocks(self) -> None:
        station_groups = {
            "A": [0],
            "B": [2],
        }
        pollutant_groups = {
            "A": {"O3": [0], "PM2.5": [1]},
            "B": {"O3": [2], "PM2.5": [3]},
        }
        source_samples = np.array(
            [
                [0.0, 0.0, 0.2, 0.1],
                [0.5, 0.1, 0.3, 0.2],
                [1.0, 0.2, 0.4, 0.3],
                [1.5, 0.3, 0.5, 0.4],
                [2.0, 0.4, 0.6, 0.5],
                [2.5, 0.5, 0.7, 0.6],
            ],
            dtype=float,
        )
        target_samples = np.array(
            [
                [0.8 * row[0] - 0.1 * row[1] + 0.2 * row[3]]
                for row in source_samples
            ],
            dtype=float,
        )

        pairwise_summary = compute_station_level_ei_summary(
            method="tm",
            source_samples=source_samples,
            target_samples=target_samples,
            station_source_groups=station_groups,
        )
        self.assertIn("pairwise_station_ei", pairwise_summary)
        self.assertEqual(set(pairwise_summary["pairwise_station_ei"]), {"A", "B"})
        self.assertTrue(all(np.isfinite(value) for value in pairwise_summary["pairwise_station_ei"].values()))

        pollutant_summary = compute_station_pollutant_pair_synergy_summary(
            method="tm",
            source_samples=source_samples,
            target_samples=target_samples,
            station_pollutant_feature_groups=pollutant_groups,
        )
        self.assertIn("joint_station_pair_ei", pollutant_summary)
        self.assertIn("single_pollutant_ei", pollutant_summary)
        self.assertIn("station_pair_synergy", pollutant_summary)
        self.assertEqual(set(pollutant_summary["joint_station_pair_ei"]), {"A", "B"})
        self.assertIn("O3", pollutant_summary["single_pollutant_ei"]["A"])
        self.assertIn("PM2.5", pollutant_summary["single_pollutant_ei"]["A"])

    def test_tm_pollutant_pair_synergy_clips_negative_ei_before_syn(self) -> None:
        pollutant_groups = {
            "A": {"O3": [0], "PM2.5": [1]},
        }
        with patch(
            "yrd.coupling.estimate_mutual_information_transport_map",
            side_effect=[
                {"mi_hat": -0.4, "backend": "tm"},
                {"mi_hat": 0.3, "backend": "tm"},
                {"mi_hat": 0.2, "backend": "tm"},
            ],
        ):
            summary = compute_station_pollutant_pair_synergy_summary(
                method="tm",
                source_samples=np.zeros((4, 2), dtype=float),
                target_samples=np.zeros((4, 1), dtype=float),
                station_pollutant_feature_groups=pollutant_groups,
            )

        self.assertEqual(summary["single_pollutant_ei"]["A"]["O3"], 0.0)
        self.assertEqual(summary["single_pollutant_ei"]["A"]["PM2.5"], 0.3)
        self.assertEqual(summary["joint_station_pair_ei"]["A"], 0.2)
        self.assertAlmostEqual(summary["station_pair_synergy"]["A"], -0.1)


if __name__ == "__main__":
    unittest.main()
