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
