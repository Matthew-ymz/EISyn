# SIS Next-State PEID Alignment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible SIS experiment where a probabilistic MLP and the known stochastic dynamics produce PEID synergy values within 20% for two state-dependent mechanisms.

**Architecture:** Put stochastic finite-horizon simulation, probabilistic MLP training, PEID comparison, artifact generation, and CLI handling in a focused module. Reuse the model specification, RK4 integrator, PEID estimator, and plotting conventions from the classical benchmark. Keep the existing vector-field benchmark unchanged except for documentation links.

**Tech Stack:** Python, NumPy, pandas, PyTorch, matplotlib, pytest, repository transport-map estimator.

---

## Chunk 1: Transition And Dataset

### Task 1: Stochastic finite-horizon SIS transition

**Files:**
- Create: `scripts/sis_next_state_peid_alignment.py`
- Create: `tests/test_sis_next_state_peid_alignment.py`

- [ ] Write a failing test that checks `tau=1.0` resolves to 50 steps and fixed-seed stochastic integration is reproducible.
- [ ] Run `pytest tests/test_sis_next_state_peid_alignment.py -v` and verify the import or assertion fails.
- [ ] Implement `SisAlignmentConfig` and `simulate_sis_transition` using existing RK4, per-step noise `process_noise * sqrt(dt)`, and `[0,1]` clipping.
- [ ] Run the focused test and verify it passes.

### Task 2: Mixed training distribution

**Files:**
- Modify: `scripts/sis_next_state_peid_alignment.py`
- Modify: `tests/test_sis_next_state_peid_alignment.py`

- [ ] Write a failing test for equal natural/intervention sample counts, three stochastic replicates, correct target names, and finite bounded targets.
- [ ] Run the focused test and verify the expected failure.
- [ ] Implement noisy natural-trajectory generation and mixed training-pair construction.
- [ ] Run focused tests and verify they pass.

## Chunk 2: Probabilistic MLP And PEID

### Task 3: Conditional Gaussian MLP

**Files:**
- Modify: `scripts/sis_next_state_peid_alignment.py`
- Modify: `tests/test_sis_next_state_peid_alignment.py`

- [ ] Write a failing test that trains a smoke-size model and checks prediction mean/std shapes, positive std, and reproducible conditional samples.
- [ ] Run the focused test and verify failure.
- [ ] Implement standardized probabilistic MLP training with diagonal Gaussian negative log likelihood.
- [ ] Run focused tests and verify they pass.

### Task 4: Alignment calculation

**Files:**
- Modify: `scripts/sis_next_state_peid_alignment.py`
- Modify: `tests/test_sis_next_state_peid_alignment.py`

- [ ] Write a failing smoke test for relation-level Oracle/MLP synergy, relative errors, protocol metadata, and pass/fail evaluation.
- [ ] Run the focused test and verify failure.
- [ ] Implement matched Oracle and learned conditional sampling inside their predictor interfaces, then reuse `estimate_peid`.
- [ ] Run focused tests and verify they pass.

## Chunk 3: Artifacts And Full Verification

### Task 5: JSON, figure, and report

**Files:**
- Modify: `scripts/sis_next_state_peid_alignment.py`
- Modify: `tests/test_sis_next_state_peid_alignment.py`
- Modify: `docs/reports/classic_network_dynamics_benchmark.md`

- [ ] Write a failing artifact smoke test for JSON, PNG, and Markdown output.
- [ ] Run the test and verify failure.
- [ ] Implement artifact writers and an outside-right, non-overlapping legend.
- [ ] Run focused tests and verify they pass.

### Task 6: Full transport-map acceptance run

**Files:**
- Generate: `results/classic_network_dynamics_benchmark/sis_next_state_alignment.json`
- Generate: `fig/classic_network_dynamics_benchmark/sis_next_state_alignment.png`
- Modify: `docs/reports/classic_network_dynamics_benchmark.md`

- [ ] Run the full experiment with `tau=1.0`, process noise `0.05`, three stochastic training replicates, and transport-map PEID.
- [ ] Verify both target relation relative errors are at most `0.20`.
- [ ] Visually inspect the PNG for clipping and legend overlap.
- [ ] Run `pytest tests/test_sis_next_state_peid_alignment.py tests/test_classic_network_dynamics_benchmark.py -v`.
- [ ] Run `git diff --check` and inspect the final diff.
