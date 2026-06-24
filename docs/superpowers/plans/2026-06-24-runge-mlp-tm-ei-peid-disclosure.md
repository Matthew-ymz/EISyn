# Runge MLP-TM-EI + PEID Experiment Disclosure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a standalone Chinese Markdown document that fully discloses the Runge MLP-TM-EI + PEID experiment without exposing repository-internal paths in reader-facing prose.

**Architecture:** Reconstruct the experiment from source code, saved configurations, and result artifacts; maintain an evidence table while drafting; explicitly label facts that cannot be verified. The delivered document is organized by reader questions rather than program execution order.

**Tech Stack:** Markdown, Python source/config inspection, JSON/NPZ/NetCDF artifact inspection, repository validation commands.

---

## Chunk 1: Evidence reconstruction

### Task 1: Audit data provenance and preprocessing

**Files:**
- Read: `scripts/reproduce_runge2015_gateways.py`
- Read: `scripts/run_runge_pairwise_mlp_ei.py`
- Read: `scripts/run_runge_peid_hypergraph.py`
- Read: `data/ncep_reanalysis_slp/`
- Read: `results/runge_2015_reproduction/`

- [ ] **Step 1: Identify the original dataset provider, product, variable, units, temporal coverage, spatial coverage, and raw time resolution.**
- [ ] **Step 2: Trace weekly aggregation, anomaly construction, seasonal-cycle removal, detrending, missing-value handling, weighting, PCA, and Varimax rotation.**
- [ ] **Step 3: Record each item as confirmed, inferred, or unconfirmed, with direct evidence.**
- [ ] **Step 4: Cross-check saved array shapes and metadata against the implementation.**

### Task 2: Audit supervised learning and prediction evaluation

**Files:**
- Read: `scripts/run_runge_pairwise_mlp_ei.py`
- Read: `scripts/run_runge_rnn_forecast_comparison.py`
- Read: `results/runge_pairwise_mlp_ei/`

- [ ] **Step 1: Recover lag-window construction, chronological splits, scaler fitting scope, and sample counts.**
- [ ] **Step 2: Recover Ridge, MLP, and fusion definitions, optimization settings, random seeds, and checkpoint selection.**
- [ ] **Step 3: Verify RMSE, MAE, correlation, bootstrap confidence interval, and reported p-value from saved outputs.**
- [ ] **Step 4: Record leakage controls and unresolved reproducibility risks.**

### Task 3: Audit EI, PEID, and network metrics

**Files:**
- Read: `scripts/run_runge_pairwise_mlp_ei.py`
- Read: `scripts/run_runge_peid_hypergraph.py`
- Read: `scripts/plot_runge_linear_mlp_peid_map.py`
- Read: `results/runge_pairwise_mlp_ei/`
- Read: `results/runge_peid_hypergraph/`

- [ ] **Step 1: Recover the intervention/reference distributions, Monte Carlo sizes, target construction, and EI estimator.**
- [ ] **Step 2: Recover pairwise edge filtering, two-source joint EI, Möbius second-order increment, and significance rules.**
- [ ] **Step 3: Recover path scaling, path depth, ACE/ACS/AMCE, and Hyper metric definitions.**
- [ ] **Step 4: Verify key rankings and values used in `docs/reports/Part2.md`.**

## Chunk 2: Document production

### Task 4: Draft the standalone disclosure

**Files:**
- Create: `docs/reports/Part2_Runge_MLP_TM_EI_PEID_Experimental_Details.md`
- Read: `docs/reports/Part2.md`

- [ ] **Step 1: Write the experiment scope and interpretation boundary.**
- [ ] **Step 2: Write data provenance and the full preprocessing chain, including a direct seasonal-adjustment answer.**
- [ ] **Step 3: Write sample construction, split, scaling, training, evaluation, EI, PEID, and network-metric sections.**
- [ ] **Step 4: Add a compact parameter table, uncertainty table, limitations, and paper-appendix method summary.**
- [ ] **Step 5: Remove repository-internal filenames and paths from reader-facing prose.**

### Task 5: Validate evidence and Markdown

**Files:**
- Verify: `docs/reports/Part2_Runge_MLP_TM_EI_PEID_Experimental_Details.md`

- [ ] **Step 1: Check every numeric and procedural claim against code or saved artifacts.**
- [ ] **Step 2: Search the document for unsupported certainty and ensure all evidence gaps are explicit.**
- [ ] **Step 3: Check heading structure, tables, math delimiters, bold vector/matrix notation, and equation references.**
- [ ] **Step 4: Run `git diff --check` and a Markdown/path disclosure check.**
- [ ] **Step 5: Inspect the final diff and commit the completed disclosure.**
