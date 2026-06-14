# ODE Synergy Candidate Screening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a standalone four-candidate ODE synergy screening benchmark with matched broad one-step readouts, Oracle comparison, publication-style curves, and a concise Chinese report.

**Architecture:** Add one focused script that defines Euler-map candidates and reuses the existing broad one-step map sweep, fairness audit, MLP, SHAP, WMS, SURD, PEID, and plotting helpers. Keep all generated artifacts and report content separate from Part1.

**Tech Stack:** Python, NumPy, pandas, PyTorch, matplotlib, pytest, repository transport-map estimator.

---

## Chunk 1: Candidate Mechanisms And Protocol

### Task 1: Candidate builders

**Files:**
- Create: `scripts/ode_synergy_candidate_benchmark.py`
- Create: `tests/test_ode_synergy_candidate_benchmark.py`

- [ ] Write failing tests for the SIS, Lorenz, Rossler, and Kuramoto equations,
  registered relations, parameter grids, and structural zero mixed differences.
- [ ] Run the focused tests and confirm failure because the new module is
  missing.
- [ ] Implement the four builders as one-step Euler `MapSpec` objects.
- [ ] Run the focused tests and confirm pass.

### Task 2: Matched broad sweep

**Files:**
- Modify: `scripts/ode_synergy_candidate_benchmark.py`
- Modify: `tests/test_ode_synergy_candidate_benchmark.py`

- [ ] Write a failing smoke test requiring four methods, Oracle PEID, prediction
  diagnostics, shared readout/model digests, and a passing fairness audit.
- [ ] Run the focused smoke test and confirm failure.
- [ ] Implement the focused sweep wrapper using the existing broad one-step
  protocol.
- [ ] Run the focused smoke test and confirm pass.

## Chunk 2: Figure And Report

### Task 3: Combined candidate figure

**Files:**
- Modify: `scripts/ode_synergy_candidate_benchmark.py`
- Modify: `tests/test_ode_synergy_candidate_benchmark.py`

- [ ] Write a failing test requiring a `2x2` combined figure with one
  outside-axes shared legend.
- [ ] Implement the combined figure and persist its path in the summary JSON.
- [ ] Run the figure test and visually inspect the smoke figure.

### Task 4: Standalone screening report

**Files:**
- Modify: `scripts/ode_synergy_candidate_benchmark.py`
- Create: `docs/reports/ode_synergy_candidate_screening.md`
- Modify: `tests/test_ode_synergy_candidate_benchmark.py`

- [ ] Write a failing test requiring equations, protocol, candidate-specific
  interpretations, prediction/Oracle diagnostics, screening decisions, and
  the combined figure.
- [ ] Implement report generation from persisted result JSON.
- [ ] Run the report test and confirm pass.

## Chunk 3: Full Experiment And Verification

### Task 5: Run and verify

- [ ] Run the focused test module.
- [ ] Run the full three-seed four-candidate experiment.
- [ ] Inspect the combined and individual PNG figures for legend overlap,
  clipping, and unstable curves.
- [ ] Verify every saved fairness audit passes.
- [ ] Run `python -m py_compile scripts/ode_synergy_candidate_benchmark.py`.
- [ ] Run `git diff --check`.
