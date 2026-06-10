# Coupled Standard Map MLP+PEID Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a coupled-standard-map benchmark that conditionally compares trajectory-only and mixed-domain MLP training against Oracle PEID.

**Architecture:** Add one focused experiment module containing dynamics, dataset generation, periodic MLP fitting, matched Oracle/MLP PEID, gating, plotting, and reporting. Add a focused test module for equations, splitting, ground truth, gate logic, and smoke artifacts.

**Tech Stack:** Python, NumPy, PyTorch, matplotlib, pandas, transport-map mutual information, pytest.

---

## Chunk 1: Dynamics And Tests

### Task 1: Coupled-map equations and structural truth

**Files:**
- Create: `scripts/coupled_standard_map_peid.py`
- Create: `tests/test_coupled_standard_map_peid.py`

- [ ] Write failing tests for wrapping, impulse equations, analytic interaction \(J^2/2\), and structural null momentum sources.
- [ ] Run the focused tests and verify the expected import failure.
- [ ] Implement the configuration, wrapping, impulse, transition, and analytic-ground-truth helpers.
- [ ] Run the focused tests and verify they pass.

### Task 2: Trajectory split and periodic model

- [ ] Write failing tests for trajectory-disjoint splits and periodic feature invariance.
- [ ] Run the tests and verify the expected failures.
- [ ] Implement trajectory generation, split construction, periodic encoding, MLP fitting, and prediction metrics.
- [ ] Run the focused tests and verify they pass.

## Chunk 2: PEID And Conditional Experiment

### Task 3: Matched Oracle/MLP PEID

- [ ] Write failing tests for six source pairs per target, matched intervention digests, true-pair ranking, and momentum null-source reporting.
- [ ] Run the tests and verify the expected failures.
- [ ] Implement histogram and transport PEID evaluation on shared intervention samples.
- [ ] Run the focused tests and verify they pass.

### Task 4: Preregistered gate and mixed fallback

- [ ] Write failing tests that Mixed-MLP is skipped only when every gate passes.
- [ ] Run the tests and verify the expected failures.
- [ ] Implement gate evaluation, conditional Mixed-MLP training, caching, plotting, and Markdown reporting.
- [ ] Run the focused smoke test and verify artifact creation.

## Chunk 3: Full Run And Verification

### Task 5: Execute and interpret

- [ ] Run the full trajectory-only experiment.
- [ ] Let the preregistered gate decide whether Mixed-MLP runs.
- [ ] Visually inspect the PNG and verify no legend overlaps data.
- [ ] Run focused tests plus the existing related regression tests.
- [ ] Record exact prediction, PEID-ranking, true-edge error, and fallback results in the report.

