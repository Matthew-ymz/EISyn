# Hierarchical Multistable Control Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one continuous hierarchical multistable-network example that ranks intervention supports using pre-control signed hierarchical PEID atoms and independently tests whether those supports minimize released-basin switching cost.

**Architecture:** Add a focused experiment module under `exp/network_revival/` containing the smooth hierarchical latch dynamics, transport-map EI table, signed greedy atoms, randomized control search, metrics, caching, and plotting. Keep the existing pair-ignition experiment unchanged. Add a thin CLI and focused tests, then summarize only observed results in `docs/reports/control.md`.

**Tech Stack:** Python, NumPy, pandas, matplotlib, repository RK4 and polynomial triangular transport-map estimator, pytest.

---

## Chunk 1: Model and signed hierarchy

### Task 1: Continuous hierarchical latch

**Files:**
- Create: `exp/network_revival/hierarchical_multistable_control.py`
- Test: `tests/test_hierarchical_multistable_control.py`

- [ ] Write failing tests for two stable released basins, single-node failure, and a known joint-support switch.
- [ ] Run `pytest tests/test_hierarchical_multistable_control.py -q` and confirm failure because the module is absent.
- [ ] Implement vectorized RK4 dynamics for source, module, and global latch states.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Signed hierarchical atoms

**Files:**
- Modify: `exp/network_revival/hierarchical_multistable_control.py`
- Modify: `tests/test_hierarchical_multistable_control.py`

- [ ] Write failing tests showing a negative split residual is retained and no `max(0, ...)` clipping occurs.
- [ ] Run the focused test and confirm the signed-atom assertion fails.
- [ ] Implement subset EI estimation with the degree-3 transport-map backend and signed residual selection.
- [ ] Re-run the focused tests and confirm they pass.

## Chunk 2: Independent intervention search

### Task 3: Randomized cost search and ranking audit

**Files:**
- Modify: `exp/network_revival/hierarchical_multistable_control.py`
- Modify: `tests/test_hierarchical_multistable_control.py`

- [ ] Write failing deterministic-seed tests for support enumeration, released-basin success, empirical minimum cost, and regret.
- [ ] Run the focused tests and confirm expected failures.
- [ ] Implement paired Sobol-style/quasi-random amplitude-duration search using NumPy RNG, with no basin outcomes entering the Phi score.
- [ ] Re-run focused tests and confirm they pass.

### Task 4: CLI, cache, and figure

**Files:**
- Create: `scripts/run_hierarchical_multistable_control.py`
- Modify: `exp/network_revival/hierarchical_multistable_control.py`
- Modify: `tests/test_hierarchical_multistable_control.py`

- [ ] Write a failing CLI smoke test using a tiny configuration.
- [ ] Implement JSON/NPZ outputs and a Python-only three-panel PNG/SVG/PDF figure with outside legends.
- [ ] Run the smoke test, visually inspect the PNG, and confirm no legend or label overlaps data.

## Chunk 3: Example run and report

### Task 5: Run the example and document only measured findings

**Files:**
- Modify: `docs/reports/control.md`
- Create under results: `results/hierarchical_multistable_control/summary.json`
- Create under figures: `fig/part3_hierarchical_multistable_control.{png,svg,pdf}`

- [ ] Run a smoke configuration first, then the default one-instance experiment.
- [ ] Inspect rankings, signed atoms, successful supports, empirical minimum costs, and regret.
- [ ] Add a concise section to `control.md`, preserving existing equation numbering and explicitly distinguishing empirical minimum from a proven global optimum.
- [ ] Run the focused test, relevant regression tests, Markdown checks, and inspect the final figure.

