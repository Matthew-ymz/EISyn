# Six-System Natural-Trajectory Synergy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add natural-trajectory Wilson–Cowan and Coupled Rössler four-method sweeps and combine them with the existing four systems in a six-panel figure.

**Architecture:** Extend the existing classic-network benchmark module with parameterized model builders, a reusable multi-initial-condition trajectory-pool generator, and one shared natural-trajectory sweep runner. Reuse the current aggregation and plotting helpers, then expand the combined figure from three to six inputs.

**Tech Stack:** Python, NumPy, pandas, PyTorch, matplotlib, pytest.

---

## Chunk 1: Models And Natural Trajectory Pools

### Task 1: Parameterized model specs

**Files:**
- Modify: `scripts/classic_network_dynamics_benchmark.py`
- Test: `tests/test_classic_network_dynamics_benchmark.py`

- [ ] Add failing tests for parameterized Rössler coupling and Wilson–Cowan gain.
- [ ] Run focused tests and confirm missing-builder failures.
- [ ] Implement `build_coupled_rossler_spec` and `build_wilson_cowan_gain_spec`.
- [ ] Run focused tests and confirm pass.

### Task 2: Multi-initial-condition natural trajectory pool

**Files:**
- Modify: `scripts/classic_network_dynamics_benchmark.py`
- Test: `tests/test_classic_network_dynamics_benchmark.py`

- [ ] Add a failing test requiring deterministic pool shapes, distinct train/readout digests, and transient coverage.
- [ ] Run focused test and confirm missing-helper failure.
- [ ] Implement `simulate_natural_trajectory_pool`.
- [ ] Run focused test and confirm pass.

## Chunk 2: Natural-Trajectory Four-Method Sweeps

### Task 3: Wilson–Cowan and Rössler sweep runners

**Files:**
- Modify: `scripts/classic_network_dynamics_benchmark.py`
- Test: `tests/test_classic_network_dynamics_benchmark.py`

- [ ] Add failing smoke tests for both new sweep runners and their result metadata.
- [ ] Run focused tests and confirm missing-runner failures.
- [ ] Implement a shared natural-trajectory sweep core plus model-specific wrappers and plots.
- [ ] Add CLI flags for both sweeps.
- [ ] Run focused tests and confirm pass.

## Chunk 3: Six-Panel Figure And Report

### Task 4: Expand combined figure

**Files:**
- Modify: `scripts/classic_network_dynamics_benchmark.py`
- Test: `tests/test_classic_network_dynamics_benchmark.py`

- [ ] Replace the three-input figure test with a failing six-input, `2 x 3` figure test.
- [ ] Run the test and confirm it fails against the old API.
- [ ] Expand the combined figure runner, default output path, and CLI description.
- [ ] Run focused test and confirm pass.

### Task 5: Generate results and document interpretation

**Files:**
- Modify: `docs/reports/Part1.md`
- Generate: `results/classic_network_dynamics_benchmark/*.json`
- Generate: `fig/classic_network_dynamics_benchmark/*.png`
- Generate: `fig/part1_synergy_comparison/six_system_four_method_synergy_panels.png`

- [ ] Run full Wilson–Cowan and Rössler sweeps with seeds `0 1 2 3`.
- [ ] Generate the six-panel combined figure.
- [ ] Update `Part1.md` with protocol, results, and interpretations.
- [ ] Visually inspect the final figure for clipping and legend overlap.

## Chunk 4: Verification

### Task 6: Regression verification

**Files:**
- Verify: `scripts/classic_network_dynamics_benchmark.py`
- Verify: `tests/test_classic_network_dynamics_benchmark.py`

- [ ] Run `pytest tests/test_classic_network_dynamics_benchmark.py -q`.
- [ ] Run focused related tests if the full module exposes failures outside the changed behavior.
- [ ] Run `python -m py_compile scripts/classic_network_dynamics_benchmark.py`.
- [ ] Inspect `git diff --check` and the final changed-file list.

