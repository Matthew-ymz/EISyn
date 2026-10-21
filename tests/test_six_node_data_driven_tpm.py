"""Numerical safeguards for data-only complete categorical channels."""
import numpy as np
import pytest
import torch

from scripts import compare_six_node_data_driven_tpm as m


def test_aggregated_cross_entropy_matches_endpoint_loss_and_gradient():
    rng = np.random.default_rng(80)
    initial, future = rng.integers(0, 64, (2, 137))
    logits = torch.tensor(rng.normal(size=(64, 64)), dtype=torch.float64, requires_grad=True)
    counts = torch.tensor(m.transition_counts(initial, future), dtype=torch.float64)
    aggregated = m.count_cross_entropy(logits, counts)
    samplewise = torch.nn.functional.cross_entropy(logits[initial], torch.tensor(future))
    assert torch.allclose(aggregated, samplewise, atol=1e-14, rtol=0)
    gradient_a, = torch.autograd.grad(aggregated, logits, retain_graph=True)
    gradient_b, = torch.autograd.grad(samplewise, logits)
    assert torch.allclose(gradient_a, gradient_b, atol=1e-14, rtol=0)


def test_split_preserves_pairs_and_has_no_overlap():
    rng = np.random.default_rng(81)
    initial, future = rng.integers(0, 64, (2, 200))
    train, validation, indices = m.split_counts(initial, future, 0)
    assert set(indices['train_indices']).isdisjoint(indices['validation_indices'])
    assert np.array_equal(train + validation, m.transition_counts(initial, future))
    assert train.sum() == 160 and validation.sum() == 40


def test_direct_tpm_has_no_silent_zero_floor_or_unvisited_row_fallback():
    counts = np.eye(64, dtype=int)
    kernel = m.direct_tpm(counts)
    assert np.array_equal(kernel, counts)
    test = np.roll(counts, 1, axis=1)
    assert np.isinf(m.negative_log_likelihood(kernel, test))
    counts[0] = 0
    with pytest.raises(ValueError, match='unvisited'):
        m.direct_tpm(counts)
    smooth = m.direct_tpm(counts, 3)
    assert np.all(smooth > 0)
    assert np.allclose(smooth.sum(axis=1), 1, atol=1e-14)


def test_joint_output_preserves_conditional_dependence():
    # A joint response with only two possible outcomes cannot be represented by
    # six independent fair Bernoulli outputs. The categorical TPM retains it.
    counts = np.zeros((64, 64), dtype=int)
    counts[:, 0] = counts[:, 63] = 10
    kernel = m.direct_tpm(counts)
    assert np.all(kernel[:, [0, 63]] == .5)
    assert np.count_nonzero(kernel) == 128
    assert not np.isclose(kernel[0, 0], .5**6)


def test_mlp_extracts_complete_normalized_joint_tpm():
    torch.set_num_threads(1)
    counts = np.eye(64, dtype=int)*20 + 1
    config = m.MLPConfig(width=8, max_epochs=10, validation_interval=2, patience_checks=2)
    learned, selection = m.train_mlp(counts, seed=4, config=config, validation=counts)
    assert learned.shape == (64, 64)
    assert np.isfinite(learned).all() and np.all(learned > 0)
    assert np.allclose(learned.sum(1), 1, atol=1e-14)
    assert 1 <= selection['selected_epoch'] <= 10


def test_exact_ei_matches_identity_channel_without_synergy_projection():
    from scripts import search_six_node_mixed_dynamics as benchmark
    result = benchmark.spt(m.direct_tpm(np.eye(64)))
    for subset, ei in result['ei_bits'].items():
        assert abs(ei - len(subset)) < 1e-12
    assert abs(result['tree']['xi_bits']) < 1e-12
    assert result['syn_audit']['candidate_count'] == 301
    assert result['syn_audit']['violation_count'] == 0
    assert result['syn_audit']['clipping'] is False


def confirmation_fixture():
    from dataclasses import asdict
    rows = []
    for count in m.SAMPLE_BUDGETS:
        for seed in m.CONFIRMATION_SEEDS:
            methods = {}
            for name in ('empirical', 'smoothed', 'mlp', 'ctmc_prior', 'oracle'):
                success = name in ('ctmc_prior', 'oracle') or (name == 'mlp' and count == 48000)
                methods[name] = dict(
                    module_f1=1. if success else .5,
                    heldout_nll_nats=None if name == 'empirical' else 3.+.01*seed,
                    mean_absolute_subset_ei_error_bits=.01*seed, root_xi_bits=.04,
                    root_syn_bits=.001,
                    spt=dict(exact_original_two_groups=success,
                             groups=[[1, 2, 3], [4, 5, 6]] if success else [[1, 2, 4, 5, 6], [3]],
                             syn_audit=dict(minimum_syn_bits=.0001, small_negative_count=0, violation_count=0)))
            rows.append(dict(seed=seed, sample_count=count, protocol=m.CONFIRMATION_ID,
                             mlp=dict(config=asdict(m.MLPConfig())), final_refit_count=count, methods=methods))
    return rows


def test_two_seed_summary_uses_only_requested_replications_and_preserves_infinity():
    summary = m.summarize_confirmation(confirmation_fixture())
    assert summary['seeds'] == [1, 2]
    assert summary['pilot_seed_excluded'] == 0
    low, high = summary['conditions']
    assert low['methods']['mlp']['root_recovery_count'] == 0
    assert high['methods']['mlp']['root_recovery_count'] == 2
    assert high['methods']['mlp']['total'] == 2
    assert low['methods']['empirical']['excess_test_nll_nats']['infinite_count'] == 2
    assert high['paired_mlp_minus_ctmc']['module_f1']['values'] == [0., 0.]


def test_summary_rejects_extra_or_duplicate_seeds():
    rows = confirmation_fixture()
    with pytest.raises(ValueError, match='Missing, duplicated or unexpected'):
        m.summarize_confirmation(rows + [rows[0]])
    rows[0]['seed'] = 0
    with pytest.raises(ValueError, match='Missing, duplicated or unexpected'):
        m.summarize_confirmation(rows)


def test_summary_never_averages_only_finite_test_losses():
    rows = confirmation_fixture()
    rows[0]['methods']['smoothed']['heldout_nll_nats'] = None
    stats = m.summarize_confirmation(rows)['conditions'][0]['methods']['smoothed']['excess_test_nll_nats']
    assert stats['status'] == 'mixed_finite_and_infinite'
    assert stats['infinite_count'] == 1 and stats['finite_count'] == 1
    assert 'mean' not in stats


def test_cache_validation_supports_legacy_oracle_and_rejects_changed_results(tmp_path):
    import json
    from dataclasses import asdict
    from threadpoolctl import threadpool_limits
    from scripts import search_six_node_mixed_dynamics as benchmark
    configuration = dict(family='glauber_ctmc', beta=1., triad=1., tau=2., cross=.04)
    with threadpool_limits(limits=1):
        oracle = benchmark.spin_kernel(**configuration)
    initial, future, _ = benchmark.draw_joint(oracle, 1, 80)
    _, _, test_counts = benchmark.draw_joint(oracle, 20001, 80)
    kernel = np.full((64, 64), 1/64)
    train_indices, validation_indices = np.arange(64), np.arange(64, 80)
    methods = {}
    for name, value in [('mlp', kernel), ('oracle', oracle)]:
        methods[name] = dict(heldout_nll_nats=m.negative_log_likelihood(value, test_counts),
                             spt=dict(ei_bits={''.join(map(str, key)): ei
                                              for key, ei in benchmark.exact_ei(value).items()}))
    row = dict(protocol=m.CONFIRMATION_ID, seed=1, sample_count=80, configuration=configuration,
               mlp=dict(config=asdict(m.MLPConfig())), source_sha256=benchmark.digest(initial),
               future_sha256=benchmark.digest(future), split_sha256=benchmark.digest(train_indices), methods=methods)
    path = tmp_path/'cached.json'
    path.write_text(json.dumps(row))
    # Earlier archives contain fitted TPMs but not the oracle, which is rebuilt.
    np.savez(path.with_suffix('.npz'), initial=initial, future=future, train_indices=train_indices,
             validation_indices=validation_indices, test_counts=test_counts, mlp=kernel)
    assert m.checked_cached_confirmation(path, 1, 80, configuration) == row
    with pytest.raises(ValueError, match='Cached configuration'):
        m.checked_cached_confirmation(path, 1, 80, {**configuration, 'tau': 3.})
    row['methods']['mlp']['heldout_nll_nats'] += 1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='does not reproduce'):
        m.checked_cached_confirmation(path, 1, 80, configuration)
