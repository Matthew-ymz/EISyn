# Coupled Henon Lorenz-Replacement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate a coupled Henon discrete-chaos synergy sweep against the existing Lorenz Part1 panel.

**Architecture:** Extend the existing discrete-iteration benchmark with one map builder and a focused sweep wrapper. Reuse the established natural-trajectory, MLP, four-method, aggregation, and plotting helpers; add Oracle and chaos diagnostics only to the focused Henon result.

**Tech Stack:** Python, NumPy, pandas, PyTorch, matplotlib, pytest.

---

## Chunk 1: Map And Focused Sweep

### Task 1: Coupled Henon map

**Files:**
- Modify: `scripts/discrete_iteration_dynamics_benchmark.py`
- Test: `tests/test_discrete_iteration_dynamics_benchmark.py`

- [ ] Write failing tests for the coupled Henon equations, structural relation, intervention domain, and bounded simulation.
- [ ] Run focused tests and confirm failure because the builder is missing.
- [ ] Implement `build_coupled_henon_spec`.
- [ ] Run focused tests and confirm pass.

### Task 2: Focused coupled Henon sweep

**Files:**
- Modify: `scripts/discrete_iteration_dynamics_benchmark.py`
- Test: `tests/test_discrete_iteration_dynamics_benchmark.py`

- [ ] Write a failing smoke test requiring four native readouts, Oracle PEID, chaos diagnostics, and exact displayed zero at `kappa=0`.
- [ ] Run the focused test and confirm failure because the runner is missing.
- [ ] Implement `run_coupled_henon_sweep`, CLI support, and a focused plot.
- [ ] Run focused tests and confirm pass.

## Chunk 2: Evaluation And Reporting

### Task 3: Lorenz replacement comparison

**Files:**
- Modify: `scripts/discrete_iteration_dynamics_benchmark.py`
- Test: `tests/test_discrete_iteration_dynamics_benchmark.py`
- Create: `docs/reports/coupled_henon_lorenz_replacement.md`

- [ ] Write a failing test for comparison metrics and comparison-figure generation.
- [ ] Implement normalized trend, relative variability, Oracle agreement, and replacement recommendation.
- [ ] Run smoke evaluation and focused tests.
- [ ] Run the full four-seed sweep and generate the comparison report and figure.

## Chunk 3: Verification

### Task 4: Verify outputs

- [ ] Run `pytest tests/test_discrete_iteration_dynamics_benchmark.py -q`.
- [ ] Run `python -m py_compile scripts/discrete_iteration_dynamics_benchmark.py`.
- [ ] Visually inspect both generated figures for clipping and overlap.
- [ ] Run `git diff --check`.

