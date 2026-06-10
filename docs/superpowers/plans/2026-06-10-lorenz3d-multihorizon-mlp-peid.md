# Lorenz-3D Multihorizon MLP+PEID Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a reproducible Lorenz-3D experiment runner for direct multihorizon MLP prediction, matched Oracle/MLP PEID evaluation, cached artifacts, figures, and a Markdown report.

**Architecture:** Add one focused script containing the Lorenz dynamics, trajectory dataset, direct PyTorch MLP, finite-resolution PEID evaluator, experiment orchestration, plotting, and CLI. Keep defaults faithful to the approved specification, while exposing a small smoke mode for automated tests and a full mode for expensive runs. Reuse the repository polynomial transport-map estimator.

**Tech Stack:** Python, NumPy, PyTorch, matplotlib, repository transport-map density estimator, pytest.

---

## Chunk 1: Dynamics And Natural-Trajectory Dataset

### Task 1: Lorenz finite-horizon transition

**Files:**
- Create: `scripts/lorenz3d_multihorizon_peid.py`
- Create: `tests/test_lorenz3d_multihorizon_peid.py`

- [x] Write failing tests for the Lorenz vector field, RK4 transition shape, and exact horizon-to-step validation.
- [x] Run `pytest tests/test_lorenz3d_multihorizon_peid.py -v` and verify failure because the module is absent.
- [x] Implement `LorenzConfig`, vectorized Lorenz field, RK4 stepping, and direct finite-horizon transition.
- [x] Run focused tests and verify they pass.

### Task 2: Leakage-free natural trajectory splits

**Files:**
- Modify: `scripts/lorenz3d_multihorizon_peid.py`
- Modify: `tests/test_lorenz3d_multihorizon_peid.py`

- [x] Write failing tests for deterministic 8/2/2 trajectory splits, horizon targets, and no trajectory overlap.
- [x] Run the focused test and verify the expected failure.
- [x] Implement initial-state banks, burn-in/record simulation, trajectory IDs, deterministic subsampling, and split construction.
- [x] Run focused tests and verify they pass.

## Chunk 2: Direct MLP And Prediction Metrics

### Task 3: Independent direct MLP models

**Files:**
- Modify: `scripts/lorenz3d_multihorizon_peid.py`
- Modify: `tests/test_lorenz3d_multihorizon_peid.py`

- [x] Write a failing test that trains two horizon-specific smoke models and verifies prediction shape, finite metrics, and distinct model horizon metadata.
- [x] Run the focused test and verify failure.
- [x] Implement standardized deterministic PyTorch MLP training with validation early stopping and constant/linear baselines.
- [x] Run focused tests and verify they pass.

## Chunk 3: Matched PEID Evaluation

### Task 4: Enumerate all source-pair target relations

**Files:**
- Modify: `scripts/lorenz3d_multihorizon_peid.py`
- Modify: `tests/test_lorenz3d_multihorizon_peid.py`

- [x] Write a failing test for all nine source-pair/target rows, matched intervention samples, finite EI fields, and preservation of signed synergy.
- [x] Run the focused test and verify failure.
- [x] Implement global intervention boxes, Oracle/MLP paired transitions, polynomial transport-map EI, and optional paired bootstrap.
- [x] Run focused tests and verify they pass.

### Task 5: Cache and representative-point summaries

**Files:**
- Modify: `scripts/lorenz3d_multihorizon_peid.py`
- Modify: `tests/test_lorenz3d_multihorizon_peid.py`

- [x] Write a failing test for deterministic cache keys and NPZ/JSON cache reuse.
- [x] Run the focused test and verify failure.
- [x] Implement cache metadata, prediction-grid summaries, representative-point PEID summaries, and conditional-wing eligibility.
- [x] Run focused tests and verify they pass.

## Chunk 4: Artifacts, CLI, And Verification

### Task 6: End-to-end smoke experiment

**Files:**
- Modify: `scripts/lorenz3d_multihorizon_peid.py`
- Modify: `tests/test_lorenz3d_multihorizon_peid.py`
- Create: `docs/reports/lorenz3d_multihorizon_mlp_peid.md`

- [x] Write failing tests for JSON/NPZ, PNG figures, Markdown report, and script-path CLI help.
- [x] Run focused tests and verify failure.
- [x] Implement smoke/full CLI modes, outside-right legends, PNG-only primary plots, and report generation.
- [x] Run focused tests and verify they pass.

### Task 7: Repository verification and main integration

**Files:**
- Generate: `results/lorenz3d_multihorizon_peid/`
- Generate: `fig/lorenz3d_multihorizon_peid/`

- [x] Run the smoke experiment from the CLI and inspect produced summary values.
- [x] Visually inspect every generated PNG for clipping and legend overlap.
- [x] Run `pytest tests/test_lorenz3d_multihorizon_peid.py tests/test_classic_network_dynamics_benchmark.py -v`.
- [x] Run `git diff --check` and inspect the final diff.
- [x] Commit only Lorenz experiment files and generated smoke artifacts to `main`; leave unrelated `.npm-cache/` and `package-lock.json` untouched.
