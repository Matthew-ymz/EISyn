"""Analytic quantities, native failures, donor identity and leakage checks."""
import numpy as np
import pytest
import torch

from exp.TM.transport_map_density import AffineTransportMapDensityEstimator
from scripts.nyc_taxi_information_priors import (
    GaussianOracle, SourceRepresentation, TOLERANCE_BITS, fit_intervention,
    nonnegative, observational_priors, penalty_weights, shapley,
)
from scripts.run_nyc_taxi_information_prior_calibration import (
    cached_arrays, donor_batch, fit_calibrator, make_design, split_indices, tune_method,
)
from scripts.train_nyc_taxi_mgstn import TaxiWindows


def test_linear_joint_readout_analytic_shapley():
    a = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [0, 0], [0, 0.]])
    covariance = np.block([[np.eye(6), a], [a.T, a.T @ a + np.eye(2)]])
    oracle = GaussianOracle(AffineTransportMapDensityEstimator(np.zeros(8), covariance, 1000), {}, "analytic")
    table = np.array([oracle.mi([j for m in range(3) if s & (1 << m) for j in (2*m, 2*m+1)], [6, 7]) for s in range(8)])
    assert table[7] == pytest.approx(np.log2(3))
    singleton = table[[1, 2, 4]]
    xi = table - ((np.arange(8)[:, None] >> np.arange(3)) & 1) @ singleton
    expected = np.log2(3) - 2 * np.log2(1.5)
    np.testing.assert_allclose(shapley(xi), [expected/2, expected/2, 0], atol=1e-12)
    np.testing.assert_allclose(shapley(table) - shapley(xi), singleton, atol=1e-12)


def test_independence_zero_mi_and_natural_copy_surd():
    oracle = GaussianOracle(AffineTransportMapDensityEstimator(np.zeros(8), np.eye(8), 1000), {}, "independent")
    assert oracle.mi(list(range(6)), [6, 7]) == 0
    rng = np.random.default_rng(9)
    base = rng.normal(size=(1000, 2))
    source = np.stack([base, base, base], axis=1)
    scores, native = observational_priors(source, base + rng.normal(size=base.shape), {}, "copy")
    # Duplicate observations convey redundancy, with only the declared ridge approximation.
    assert np.max(scores["surd"]) < 1e-6
    assert np.min(native["surd_synergy_atoms"]) >= -TOLERANCE_BITS
    assert native["observational_joint_covariance"][0, 2] > .8


def test_nonnegative_fails_and_signed_mapping_is_explicit():
    audit = {}
    np.testing.assert_array_equal(nonnegative([-.5e-8, .3], "syn", audit), [0, .3])
    assert audit["syn"]["numerical_zero_count"] == 1
    with pytest.raises(ValueError, match=r"minimum=-0.001.*threshold=-1e-08.*affected_count=1"):
        nonnegative([-.001, 1], "syn", audit)
    with pytest.raises(ValueError, match="affected_count=1"):
        penalty_weights(np.array([[-.1, 0, 1]]), "xi_shapley", 1)
    p, record = penalty_weights(np.array([[-.1, 0, .2]]), "phi_wms", 1)
    assert record["offsets"] == [.1]
    assert p.mean() == pytest.approx(1)
    assert p[0, 0] > p[0, 1] > p[0, 2]


def test_affine_covariance_is_coherent_and_source_blocks_independent():
    rng = np.random.default_rng(71)
    source = rng.normal(size=(800, 3, 2))
    pool = rng.normal(size=(1000, 3, 2))
    target = source.reshape(800, 6) @ rng.normal(size=(6, 2)) + rng.normal(size=(800, 2))
    scores, native, diagnostics = fit_intervention(source, target, pool, np.arange(560), np.arange(560, 800), {}, "fit")
    covariance, a = native["joint_covariance"], native["coefficients"]
    np.testing.assert_array_equal(covariance[:2, 2:6], 0)
    np.testing.assert_allclose(covariance[6:, 6:], a.T @ covariance[:6, :6] @ a + native["residual_covariance"])
    assert all(np.min(v) >= 0 for v in scores.values())
    assert diagnostics["shapley_identity_error_bit"] < 1e-12


def test_donors_preserve_actual_local_channel_windows():
    flow = np.arange(1600 * 3 * 2, dtype=np.float32).reshape(1600, 3, 2)
    attributes = np.zeros((1600, 4), dtype=np.float32)
    prepared = dict(flow_z=flow, attributes_z=attributes)
    center = TaxiWindows(flow, attributes, flow, np.array([1400]))[0]
    donors = np.array([[[1200, 1210, 1220], [1300, 1310, 1320], [1400, 1410, 1420]]])
    batch = donor_batch(prepared, donors, center, torch.device("cpu"))
    for m, name in enumerate(("recent", "daily", "weekly")):
        for z in range(3):
            expected = TaxiWindows(flow, attributes, flow, np.array([donors[0, z, m]]))[0][name][z]
            torch.testing.assert_close(batch[name][0, z], expected)


def test_representation_fits_pool_only_and_keeps_constant_channels():
    rng = np.random.default_rng(5)
    flow = rng.poisson(4, size=(1400, 2, 2)).astype(float)
    flow[:, 0, 1] = 0
    pool = np.arange(1176, 1250)
    first = SourceRepresentation.fit(flow, pool)
    changed = flow.copy()
    changed[1300:] += 1000
    second = SourceRepresentation.fit(changed, pool)
    np.testing.assert_array_equal(first.scales, second.scales)
    assert first.constant[0, :, 1].all()
    scores = first.transform(flow, np.arange(1250, 1300))
    np.testing.assert_array_equal(scores[:, 0, :, 1], 0)


def test_gamma_zero_and_uniform_scores_give_same_prediction_and_tie_rule():
    rng = np.random.default_rng(8)
    source, frozen = rng.normal(size=(45, 2, 3, 2)), rng.normal(size=(45, 2, 2))
    target = frozen + .3 * source[:, :, 0] + .1
    design, _, _ = make_design(source, frozen, 30)
    p, _ = penalty_weights(np.ones((2, 3)), "uniform", 0)
    baseline, _ = fit_calibrator(design, frozen, target, 30, p, .1)
    for raw in (rng.random((2, 3)), np.ones((2, 3)), np.zeros((2, 3))):
        weights, _ = penalty_weights(raw, "xi_shapley", 0)
        actual, _ = fit_calibrator(design, frozen, target, 30, weights, .1)
        np.testing.assert_array_equal(actual, baseline)
    best, trials, _, _ = tune_method(design, frozen, target, 30, np.ones((2, 3)),
                                     "xi_shapley", np.zeros((2, 2)), np.ones((2, 2)), {})
    assert len(trials) == 40 and best["gamma"] == 0
    splits = split_indices()
    assert len(splits["F"]) == 1008 and len(splits["V"]) == 508


def test_expensive_cache_requires_matching_provenance(tmp_path):
    path, origins = tmp_path / "forward.npz", {}
    first = cached_arrays(path, {"donor_hash": "a"}, lambda: {"prediction": np.array([1.])}, origins)
    second = cached_arrays(path, {"donor_hash": "a"}, lambda: pytest.fail("matching cache recomputed"), origins)
    np.testing.assert_array_equal(first["prediction"], second["prediction"])
    changed = cached_arrays(path, {"donor_hash": "b"}, lambda: {"prediction": np.array([2.])}, origins)
    np.testing.assert_array_equal(changed["prediction"], [2.])


def test_exploration_draws_are_nested_disjoint_and_holm_controls_three():
    from scripts.complete_nyc_taxi_information_prior_calibration import holm, shap_draws, shap_precision
    initial, doubled = shap_draws(split_indices())
    assert (len(initial[0]), len(initial[1])) == (32, 16)
    assert (len(doubled[0]), len(doubled[1])) == (64, 32)
    assert set(initial[0]) <= set(doubled[0]) and set(initial[1]) <= set(doubled[1])
    assert not set(doubled[0]) & set(doubled[1])
    assert max(doubled[0]) < 7492 and max(doubled[1]) < 7492
    np.testing.assert_allclose(holm([.02, .001, .3]), [.04, .003, .3])
    assert shap_precision(np.ones((66, 3)), np.ones((66, 3)))["converged"]
    initial = np.broadcast_to([.8, .1, .1], (66, 3)).copy()
    doubled = np.broadcast_to([.1, .8, .1], (66, 3)).copy()
    assert not shap_precision(initial, doubled)["converged"]


def test_time_bootstrap_uses_identical_actual_hour_counts_for_equal_methods():
    from scripts.complete_nyc_taxi_information_prior_calibration import bootstrap_comparisons
    rng = np.random.default_rng(99)
    target, prediction = rng.normal(size=(180, 2, 2)), rng.normal(size=(180, 2, 2))
    predictions = {m: prediction for m in ("xi_shapley", "uniform", "ei_shapley", "phi_r")}
    comparisons, samples, weights = bootstrap_comparisons(predictions, target, np.zeros((2, 2)), np.ones((2, 2)),
                                                         ["uniform", "ei_shapley", "phi_r"])
    np.testing.assert_array_equal(weights.sum(axis=1), 180)
    for method, row in comparisons.items():
        np.testing.assert_array_equal(samples[method], samples["xi_shapley"])
        assert row["delta_ci95"] == [0., 0.]
        assert row["descriptive_holm_adjusted_p"] == 1


def test_cached_shap_keeps_branch_worlds_and_matches_full_forward(tmp_path):
    from scripts.complete_nyc_taxi_information_prior_calibration import compute_shap_with_cache
    from scripts.run_nyc_taxi_information_prior_calibration import grouped_shap
    class Branch(torch.nn.Module):
        def forward(self, flow, attributes):
            local = flow.mean(dim=(2, 3)) + .05 * flow.mean(dim=(1, 2, 3))[:, None]
            return torch.stack([local, local**2], dim=-1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.recent, self.daily, self.weekly = Branch(), Branch(), Branch()
            self.granularity_fusion = torch.nn.Identity()
            self.target_attribute, self.output = torch.nn.Linear(4, 2), torch.nn.Linear(8, 2)
        def forward(self, batch):
            fused = torch.cat([getattr(self, name)(batch[name], batch[name+"_attr"])
                               for name in ("recent", "daily", "weekly")], dim=-1)
            target = self.target_attribute(batch["target_attr"])[:, None].expand(-1, 3, -1)
            return self.output(torch.cat([fused, target], dim=-1))
    rng = np.random.default_rng(31)
    torch.manual_seed(31)
    flow = rng.normal(size=(1600, 3, 2)).astype(np.float32)
    prepared = dict(flow_z=flow, targets=flow, attributes_z=rng.normal(size=(1600, 4)).astype(np.float32))
    model = Model().eval()
    fg, bg, device = np.array([1400, 1410]), np.array([1200, 1300]), torch.device("cpu")
    direct, audit = grouped_shap(model, prepared, fg, bg, [1], device, batch_size=2, verify=True)
    assert audit["branch_full_max_error_z"] < 1e-6
    path = tmp_path / "shap.npz"
    cached = compute_shap_with_cache(model, prepared, fg, bg, 1, device, 2, path, {"hash": "test"}, {})
    np.testing.assert_allclose(cached, direct[0], atol=1e-12)
    with np.load(path) as saved:
        assert saved["branch_recent_baseline_then_hidden"].shape == (6, 2)
        assert saved["target_representation"].shape == (2, 2)
