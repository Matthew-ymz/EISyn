# Kuramoto Short-Horizon Phi Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paired N=64 whole-state Oracle \(\Phi^{EID}\) multi-horizon sweep that audits synchronization before interpreting a short-horizon criticality peak.

**Architecture:** Extend the classical network benchmark with a dedicated tau-sweep runner. It generates one phase-state support per seed, applies every `(coupling, tau)` transition to that identical support, stores raw-order tail diagnostics, and renders a compact robustness figure. The existing single-horizon runner remains unchanged.

**Tech Stack:** Python, NumPy, pandas, Matplotlib, pytest, existing transport-map Oracle estimator.

---

## Chunk 1: Paired runner and result contract

### Task 1: Specify and test the tau-sweep payload

**Files:**
- Modify: `tests/test_classic_network_dynamics_benchmark.py`
- Modify: `scripts/classic_network_dynamics_benchmark.py`

- [ ] **Step 1: Write failing smoke test**

Assert that a tau-sweep on two couplings and two horizons writes JSON and a
figure, reuses source-state digests across all conditions for a seed, and
stores `raw_order_q95`, `raw_order_q99`, and `raw_order_strong_fraction`.

- [ ] **Step 2: Run the test and verify it fails because the runner is absent**

Run: `pytest tests/test_classic_network_dynamics_benchmark.py -k tau_sweep -v`

- [ ] **Step 3: Add the minimal paired runner**

Add a public runner named
`run_large_kuramoto_oracle_nsource_whole_state_tau_sweep`. Reuse existing
integration, phase-feature, and N-source Oracle functions. Keep the existing
single-horizon runner untouched.

- [ ] **Step 4: Re-run the targeted test**

Run: `pytest tests/test_classic_network_dynamics_benchmark.py -k tau_sweep -v`

## Chunk 2: CLI, publication figure, and report

### Task 2: Expose and document the controlled comparison

**Files:**
- Modify: `scripts/classic_network_dynamics_benchmark.py`
- Modify: `docs/reports/Part1.md`

- [ ] **Step 1: Add a CLI flag and explicit output locations**

Expose the paired tau runner without changing the existing whole-state sweep
CLI behavior.

- [ ] **Step 2: Render a two-panel figure**

Panel a: \(\Phi^{EID}(K)\) curves by horizon. Panel b: raw synchronization
order curves by horizon with the `r=0.8` guard. Keep the legend outside the
axes and export a PNG.

- [ ] **Step 3: Add a concise Part1 note**

Describe the fixed-across-K horizon rule, guard criterion, and interpretation
rule. Insert measured values only after the full run.

## Chunk 3: Execution and verification

### Task 3: Produce and audit the experiment

**Files:**
- Create: `results/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_sweep_n64.json`
- Create: `fig/classic_network_dynamics_benchmark/large_kuramoto_oracle_nsource_whole_state_tau_sweep_n64.png`

- [ ] **Step 1: Run a smoke sweep**

Verify the result schema, paired digest invariant, and figure rendering.

- [ ] **Step 2: Run the full N=64 sweep**

Use the documented horizon grid and shared supports. Preserve the machine-
readable JSON result.

- [ ] **Step 3: Visually inspect the figure and run targeted regression tests**

Confirm legend placement, unclipped labels, synchronization guard metrics, and
test success before documenting the outcome.
