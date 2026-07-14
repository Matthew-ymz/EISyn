# Yeo7-PC1 RNN Comparison Implementation Plan

> **For agentic workers:** REQUIRED: Use `superpowers:executing-plans` to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether train-only Yeo7-PC1 RNN forecasters improve held-out chronological prediction over the tuned per-subject Δ-Ridge baseline, for both individual and shared 30-subject training.

**Architecture:** Add one self-contained experiment script. It will derive each subject's seven PC1 series only from the relevant training prefix, train a vanilla Elman RNN to predict the next-state delta from fixed-length histories, and evaluate the established 900:1200 held-out interval. Individual RNNs have subject-specific weights; the shared RNN pools fixed-length examples from all subjects while resetting state implicitly at every window. A paired subject-level analysis and a compact result figure provide the final comparison.

**Tech Stack:** Python 3.11, NumPy, SciPy, scikit-learn, PyTorch, Matplotlib, pytest.

---

## Chunk 1: Train-only data and model primitives

### Task 1: Define testable history and split behaviour

**Files:**
- Create: `tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py`
- Create: `scripts/run_hcp_schaefer500_yeo7_pc1_rnn_comparison.py`

- [ ] **Step 1: Write failing tests**

```python
def test_sequence_windows_keep_only_training_targets():
    windows, targets = make_sequence_samples(series, history=3, target_end=6)
    assert windows.shape == (3, 3, 2)
    assert targets.shape == (3, 2)

def test_pca_is_fitted_only_on_the_requested_training_prefix():
    reduced_a = reduce_train_only(raw, groups, train_end=6)
    reduced_b = reduce_train_only(raw_with_changed_future, groups, train_end=6)
    np.testing.assert_allclose(reduced_a[:6], reduced_b[:6])
```

- [ ] **Step 2: Run the focused test module and confirm RED**

Run: `/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py -q`

Expected: import/attribute failure because the new module does not exist.

- [ ] **Step 3: Implement minimal data helpers**

```python
def make_sequence_samples(series, history, target_end):
    # Return [n_samples, history, n_features] inputs and next-state targets;
    # targets are strictly before target_end.

def reduce_train_only(raw_series, groups, train_end):
    reducer = fit_yeo7_pc1(raw_series[:train_end], groups)
    return reducer.transform(raw_series)
```

- [ ] **Step 4: Re-run the focused tests and confirm GREEN**

Run: `/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py -q`

Expected: two tests pass.

## Chunk 2: Predictor and controlled comparison

### Task 2: Add a deterministic RNN predictor with exact output shape

**Files:**
- Modify: `scripts/run_hcp_schaefer500_yeo7_pc1_rnn_comparison.py`
- Modify: `tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py`

- [ ] **Step 1: Write failing test**

```python
def test_rnn_predictor_returns_one_delta_per_window():
    prediction = fit_rnn_predict(train_x, train_delta, eval_x, hidden=4, learning_rate=1e-2, epochs=3, seed=7)
    assert prediction.shape == (len(eval_x), train_delta.shape[1])
    assert np.isfinite(prediction).all()
```

- [ ] **Step 2: Run test and confirm RED**

Run: `/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py::test_rnn_predictor_returns_one_delta_per_window -q`

Expected: missing predictor failure.

- [ ] **Step 3: Implement the minimal vanilla RNN**

```python
net = torch.nn.RNN(input_size=n_features, hidden_size=hidden, batch_first=True, nonlinearity="tanh")
head = torch.nn.Linear(hidden, n_features)
# Train full-batch AdamW on standardized delta targets and return head(h_T).
```

- [ ] **Step 4: Run the focused test module and confirm GREEN**

Run: `/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py -q`

Expected: three tests pass.

### Task 3: Encode selection, testing and paired aggregation

**Files:**
- Modify: `scripts/run_hcp_schaefer500_yeo7_pc1_rnn_comparison.py`
- Modify: `tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py`

- [ ] **Step 1: Write failing tests**

```python
def test_subject_summary_pairs_all_three_models_by_subject():
    summary = summarize_subject_rows(rows)
    assert summary["shared_rnn_minus_ridge"]["n_subjects"] == 2
    assert summary["individual_rnn_minus_ridge"]["n_subjects"] == 2
```

- [ ] **Step 2: Run test and confirm RED**

Run: `/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py::test_subject_summary_pairs_all_three_models_by_subject -q`

Expected: missing aggregation failure.

- [ ] **Step 3: Implement fixed comparison protocol**

Use chronological validation folds `600, 700, 800`, test `900:1200`, train-only PCA/scaling, one-step standardized RMSE as primary metric, a fixed Δ-Ridge candidate grid matching the earlier experiment, a fixed RNN grid (`history ∈ {3,5,8}`, `hidden ∈ {8,16}`, `lr ∈ {1e-3,3e-3}`, `epochs=300`), three matched final seeds, and paired subject bootstrap confidence intervals.

- [ ] **Step 4: Re-run focused tests and confirm GREEN**

Run: `/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py -q`

Expected: four tests pass.

## Chunk 3: Run, plot, and report

### Task 4: Add smoke/full CLI, result figure, and report update

**Files:**
- Modify: `scripts/run_hcp_schaefer500_yeo7_pc1_rnn_comparison.py`
- Modify: `docs/reports/schaefer500_experiment.md`
- Create: `results/hcp_schaefer500_yeo7_pc1_rnn_comparison/summary.json`
- Create: `results/hcp_schaefer500_yeo7_pc1_rnn_comparison/heldout_subject_comparison.png`
- Create: `results/hcp_schaefer500_yeo7_pc1_rnn_comparison/report.md`

- [ ] **Step 1: Add smoke-mode test/assertion**

```python
def test_smoke_mode_uses_one_subject_and_one_candidate_per_model():
    config = build_config(smoke=True)
    assert config.subject_limit == 1
    assert config.rnn_candidates == 1
```

- [ ] **Step 2: Run test and confirm RED, then implement CLI**

Run: `/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py -q`

Expected: missing smoke configuration failure.

- [ ] **Step 3: Implement outputs**

Write JSON with the experiment contract, selected settings, seed-averaged per-subject metrics, paired RMSE deltas and bootstrap intervals. Render a two-panel Python/Matplotlib figure: held-out RMSE points by model and paired RNN-minus-Ridge subject deltas, with no overlapping legend. Update the existing report only after full results exist.

- [ ] **Step 4: Run smoke experiment**

Run: `/opt/anaconda3/envs/py311/bin/python scripts/run_hcp_schaefer500_yeo7_pc1_rnn_comparison.py --smoke --output-dir /tmp/yeo7-rnn-smoke`

Expected: finite metrics and one-subject JSON/PNG output.

- [ ] **Step 5: Run full experiment and inspect artifacts**

Run: `/opt/anaconda3/envs/py311/bin/python scripts/run_hcp_schaefer500_yeo7_pc1_rnn_comparison.py`

Expected: 30-subject paired results, complete report, and PNG/SVG/PDF figure.

- [ ] **Step 6: Verify**

Run: `/opt/anaconda3/envs/py311/bin/python -m pytest tests/test_hcp_schaefer500_yeo7_pc1_rnn_comparison.py -q && /opt/anaconda3/envs/py311/bin/python -m json.tool results/hcp_schaefer500_yeo7_pc1_rnn_comparison/summary.json >/dev/null`

Expected: all tests pass and JSON parsing succeeds.
