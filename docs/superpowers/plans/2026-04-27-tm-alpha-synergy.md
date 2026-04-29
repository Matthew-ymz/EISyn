# TM Alpha Synergy Notebook Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `exp/tm_vs_nis.ipynb` into a tm-only notebook that uses an `alpha`-controlled known dynamics family to show `syn≈0` at `alpha=0` and increasing `syn` as `alpha` grows, while also exposing single-source and total `EI` trends.

**Architecture:** Keep the notebook small and self-contained. Update notebook-facing tests first, then replace the old `nis-vs-tm` case sweep with an `alpha` sweep that calls `summarize_two_source_synergy_transport_map`, aggregates repeated runs, and exports tm-only summary figures and CSV artifacts.

**Tech Stack:** Python, Jupyter notebook JSON, `numpy`, `pandas`, `matplotlib`, `unittest`, existing `yrd.transport_map` helpers.

---

## File Structure

**Create:**

- `docs/superpowers/specs/2026-04-27-tm-alpha-synergy-design.md`
- `docs/superpowers/plans/2026-04-27-tm-alpha-synergy.md`

**Modify:**

- `tests/test_utils.py`
- `exp/tm_vs_nis.ipynb`

## Chunk 1: Notebook Contract

### Task 1: Update notebook-facing tests to the new `alpha + tm` story

**Files:**

- Modify: `tests/test_utils.py`

- [ ] **Step 1: Write the failing expectations**

Require the notebook to mention `alpha`, known dynamics, and `TM`-only outputs, and remove expectations for `NIS`, `Strong/Moderate/Zero synergy`, and old helper names.

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m unittest tests.test_utils.TestNotebookScripts.test_transport_map_notebook_has_compact_shape_and_reuses_module_helpers tests.test_utils.TestNotebookScripts.test_transport_map_notebook_contains_chinese_method_and_result_sections tests.test_utils.TestNotebookScripts.test_transport_map_notebook_excludes_removed_density_and_collider_sections -v`

Expected: FAIL because `exp/tm_vs_nis.ipynb` still contains the old `nis-vs-tm` case design.

## Chunk 2: Notebook Refactor

### Task 2: Replace the old three-case comparison with an `alpha` sweep

**Files:**

- Modify: `exp/tm_vs_nis.ipynb`

- [ ] **Step 1: Rewrite markdown cells**

Describe the known-dynamics purpose, the `alpha`-controlled mechanism, and the expected structural behavior:

- `alpha = 0` gives `syn ≈ 0`
- larger `alpha` gives larger `syn`
- single-source `EI` falls while joint structure becomes more important

- [ ] **Step 2: Rewrite the experiment cell**

Implement:

- `ALPHA_VALUES`
- `simulate_alpha_case_intervention`
- `estimate_tm_alpha_metrics`
- `run_alpha_sweep_tm`
- `summarize_alpha_sweep_tm`

- [ ] **Step 3: Rewrite the plotting cell**

Export:

- one tm decomposition figure
- one share/ratio figure
- one tm summary CSV
- one manifest JSON

Use an outside-right legend and `bbox_inches="tight"` so legends do not overlap data.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run: `python -m unittest tests.test_utils.TestNotebookScripts.test_transport_map_notebook_import_preamble_works_from_exp_directory tests.test_utils.TestNotebookScripts.test_transport_map_notebook_has_compact_shape_and_reuses_module_helpers tests.test_utils.TestNotebookScripts.test_transport_map_notebook_contains_chinese_method_and_result_sections tests.test_utils.TestNotebookScripts.test_transport_map_notebook_excludes_removed_density_and_collider_sections -v`

Expected: PASS.

## Chunk 3: Notebook Execution Validation

### Task 3: Execute the notebook and inspect the exported summary

**Files:**

- Modify: `exp/tm_vs_nis.ipynb` if execution exposes issues

- [ ] **Step 1: Execute the notebook**

Run: `jupyter nbconvert --to notebook --execute exp/tm_vs_nis.ipynb --output /tmp/tm_vs_nis.executed.ipynb`

Expected: notebook executes successfully and writes tm-only outputs under `fig/transport_map_mutual_information/`.

- [ ] **Step 2: Inspect the summary CSV**

Confirm:

- `alpha = 0` row has `tm_syn_mean` close to `0`
- `tm_syn_mean` rises across the `alpha` grid
- `tm_single_q2_mean` falls as `alpha` grows

- [ ] **Step 3: Stop only after the notebook-level evidence matches the intended story**
