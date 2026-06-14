# Kuramoto WMS Nonnegativity Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Coupled Hénon Part1 panel with a classical active-rotator Kuramoto phase-locking experiment that uses the same source-target relation for every method and exposes negative observational WMS alongside nonnegative intervention PEID.

**Architecture:** Extend the existing Kuramoto coupling sweep in `classic_network_dynamics_benchmark.py` to register only `{theta_1, theta_2} -> dtheta_1`, add Oracle PEID, phase-locking and WMS-component diagnostics, and use this payload in the six-panel Part1 compositor. Keep natural-trajectory WMS/SURD/SHAP and independent-intervention MLP+PEID semantics explicit and auditable.

**Tech Stack:** Python, NumPy, pandas, matplotlib, PyTorch, pytest, repository transport-map estimators.

---

## Chunk 1: Experiment Contract and Tests

### Task 1: Lock the Kuramoto source-target contract

**Files:**
- Modify: `tests/test_classic_network_dynamics_benchmark.py`
- Modify: `scripts/classic_network_dynamics_benchmark.py`

- [x] Add a failing test asserting the Kuramoto spec has one hyperedge, `theta1+theta2->dtheta1`, active-rotator strength `A=0.2`, and detuning metadata `|Delta omega|=0.1`.
- [x] Run the focused test and confirm it fails for the missing contract.
- [x] Update the spec names/equations and registered relation minimally.
- [x] Re-run the focused test and confirm it passes.

### Task 2: Lock diagnostics and method consistency

**Files:**
- Modify: `tests/test_classic_network_dynamics_benchmark.py`
- Modify: `scripts/classic_network_dynamics_benchmark.py`

- [x] Add failing tests for phase-locking value, WMS component output, Oracle PEID, shared intervention states, and shared MLP digest.
- [x] Run focused tests and confirm the expected failures.
- [x] Implement diagnostics and payload metadata without changing unrelated benchmark paths.
- [x] Re-run focused tests and confirm they pass.

## Chunk 2: Figure Integration

### Task 3: Replace Hénon in the Part1 compositor

**Files:**
- Modify: `tests/test_classic_network_dynamics_benchmark.py`
- Modify: `scripts/classic_network_dynamics_benchmark.py`
- Modify: `scripts/discrete_iteration_dynamics_benchmark.py` only if its caller contract requires it

- [x] Add a failing compositor test expecting panels: Standard Map, Wilson-Cowan, Kuramoto, Cournot, Ikeda, Nicholson-Bailey.
- [x] Run the focused test and confirm it fails under the current Hénon contract.
- [x] Replace the Hénon input with the Kuramoto result and set every comparison y-axis to `Synergy / Interaction`.
- [x] Keep the legend outside the axes and preserve tight/constrained layout.
- [x] Re-run the focused test and confirm it passes.

### Task 4: Add the Kuramoto physical diagnostic figure

**Files:**
- Modify: `tests/test_classic_network_dynamics_benchmark.py`
- Modify: `scripts/classic_network_dynamics_benchmark.py`

- [x] Add a failing test that the generated two-panel figure contains PLV, detuning reference, WMS, MLP+PEID, Oracle PEID, and the required y-axis label.
- [x] Run it and confirm failure.
- [x] Implement the two-panel publication figure with outside-right legend and zero/detuning references.
- [x] Re-run the test and confirm it passes.

## Chunk 3: Results and Report

### Task 5: Run the full experiment and validate success criteria

**Files:**
- Regenerate: `results/classic_network_dynamics_benchmark/kuramoto_coupling_synergy_sweep.json`
- Regenerate: `fig/classic_network_dynamics_benchmark/kuramoto_coupling_synergy_sweep.png`
- Regenerate: `fig/part1_synergy_comparison/six_system_four_method_synergy_panels.png`

- [x] Run the full seven-coupling, three-seed active-rotator sweep.
- [x] Check that at least two strongly locked points have WMS uncertainty below zero and PEID uncertainty above zero.
- [x] Check MLP+PEID/Oracle agreement and prediction error.
- [x] Visually inspect both generated figures for labels, clipping, and legend overlap.

### Task 6: Update the Part1 narrative

**Files:**
- Modify: `docs/reports/Part1.md`

- [x] Replace the Coupled Hénon section and table entry with the Kuramoto model, target, detuning reference, protocols, diagnostics, and measured result.
- [x] State that WMS is a signed synergy-minus-redundancy quantity rather than a nonnegative atom.
- [x] Verify equations and cross-references remain consistent.

### Task 7: Final verification

**Files:**
- Verify all modified files.

- [x] Run focused tests for Kuramoto and Part1 composition.
- [x] Run the relevant benchmark test modules.
- [x] Run `git diff --check`.
- [x] Review the final diff without reverting unrelated user changes.
