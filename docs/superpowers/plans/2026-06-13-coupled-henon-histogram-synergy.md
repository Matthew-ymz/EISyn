# Coupled Henon Histogram Synergy Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use a shared six-bin histogram estimator for all information-theoretic Coupled Hénon Part1 readouts and record four-, six-, and eight-bin sensitivity diagnostics.

**Architecture:** Configure the existing Hénon sweep to use the benchmark's histogram PEID and discrete SURD paths while leaving SHAP continuous. Add a focused sensitivity helper that reuses each run's held-out states, observed targets, and fitted-model targets so bin-count effects are separated from data/model variation.

**Tech Stack:** Python, NumPy, pandas, pytest, matplotlib.

---

## Chunk 1: Estimator Contract

### Task 1: Specify Hénon histogram behavior

**Files:**
- Modify: `tests/test_discrete_iteration_dynamics_benchmark.py`
- Modify: `scripts/discrete_iteration_dynamics_benchmark.py`

- [ ] Add a failing smoke-sweep test requiring `estimator == "histogram"`,
  `histogram_bins == 6`, and sensitivity results for bins `4, 6, 8`.
- [ ] Run the focused test and confirm it fails on the missing contract.
- [ ] Configure the Hénon sweep's primary estimator and metadata.
- [ ] Implement sensitivity diagnostics using the same held-out run data.
- [ ] Run the focused tests and confirm they pass.

## Chunk 2: Results And Report

### Task 2: Regenerate and document the panel

**Files:**
- Modify: `results/discrete_iteration_dynamics_benchmark/coupled_henon_synergy_sweep.json`
- Modify: `fig/discrete_iteration_dynamics_benchmark/coupled_henon_synergy_sweep.png`
- Modify: `fig/part1_synergy_comparison/six_system_four_method_synergy_panels.png`
- Modify: `docs/reports/Part1.md`

- [ ] Run the full three-seed Hénon sweep.
- [ ] Regenerate the six-system Part1 figure.
- [ ] Update the Hénon report section and SURD fidelity explanation.
- [ ] Run focused and benchmark regression tests.
- [ ] Visually inspect the regenerated figures.

