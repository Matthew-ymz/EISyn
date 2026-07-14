# DMF Multi-Horizon Kuramoto Alignment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a fixed (U(0,1)^{83}) DMF intervention produces Kuramoto-like high-coupling EI contraction at sufficiently long, paired prediction horizons.

**Architecture:** Implement a dedicated runner that draws each fixed intervention and `sI` background once per `(seed, G)` and evolves the same batch to all requested horizons. Store whole EI, singleton-sum EI, PhiEID, target spatial spread, and target variance retention in an NPZ cache. A companion plot will present horizon-conditioned curves on matched axes.

**Tech Stack:** Python, NumPy, Matplotlib, pytest.

---

### Task 1: Paired multi-horizon transition sampler

**Files:**
- Create: `scripts/run_dmf_fixed_uniform_multihorizon.py`
- Test: `tests/test_dmf_fixed_uniform_multihorizon.py`

- [ ] Write a failing test requiring a transition batch with fixed `U(0,1)` sources, shared `sI` backgrounds, all requested horizons, and zero clipping.
- [ ] Run the test and confirm the runner is unavailable.
- [ ] Implement the minimal paired source/target rollout and diagnostics: whole EI, singleton-sum EI, PhiEID, target spatial SD, and target variance retained.
- [ ] Re-run the test and confirm it passes.

### Task 2: Smoke and confirmatory batches

**Files:**
- Create: `results/dmf_fixed_uniform_multihorizon/`
- Create: `docs/log/dmf_fixed_uniform_multihorizon_report.md`

- [ ] Run the declared smoke grid: horizons `1,10,50,100,300`, one seed, 512 samples, all G values.
- [ ] Select only horizons with observable high-G contraction for full confirmation; do not choose based solely on Phi height.
- [ ] Run selected horizons at 8 seeds and 4096 samples in recoverable batches; aggregate only matched outputs.

### Task 3: Curve-alignment figure and audit

**Files:**
- Create: `scripts/plot_dmf_fixed_uniform_multihorizon.py`
- Create: `fig/dmf_fixed_uniform_multihorizon_alignment.{png,svg,pdf}`

- [ ] Plot whole EI, singleton-sum EI, PhiEID, and target spatial SD versus G for each confirmed horizon with outside legends.
- [ ] Visually inspect that legends do not obscure data.
- [ ] Report paired seed evidence, zero clipping, state-contraction evidence, and whether the DMF curves meet the predeclared Kuramoto-alignment conditions.
