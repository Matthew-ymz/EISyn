# Sin Synergy Ignition Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a runnable sin-synergy ODE ignition workflow to the existing three-node network-revival notebook.

**Architecture:** Keep the implementation notebook-local because this is a small exploratory extension of an existing notebook. Reuse project result paths and matplotlib conventions; add a regression test that checks the notebook contains the new workflow.

**Tech Stack:** Jupyter notebook JSON, NumPy, pandas, matplotlib, pytest.

---

## Chunk 1: Notebook Workflow

### Task 1: Regression Test

**Files:**
- Modify: `tests/test_network_revival_effective_information.py`

- [ ] Add assertions to `test_three_node_synergy_notebook_contains_sensitivity_workflow` for `sin_synergy_ignition_response.csv`, `make_sin_synergy_ignition_model`, and `evaluate_sin_synergy_ignition`.
- [ ] Run the specific test and verify it fails before notebook edits.

### Task 2: Notebook Section

**Files:**
- Modify: `exp/network_revival/notebook_three_node_synergy.ipynb`

- [ ] Add markdown defining the continuous sin-synergy ODE.
- [ ] Add code cells defining the ODE model, RK4 ignition evaluator, cost-grid runner, cache writer, and outside-legend plots.
- [ ] Save outputs under `results/network_revival_three_node_synergy/`.

### Task 3: Verification

**Files:**
- Verify: `exp/network_revival/notebook_three_node_synergy.ipynb`
- Verify: `tests/test_network_revival_effective_information.py`

- [ ] Execute the notebook top-to-bottom or with a bounded timeout.
- [ ] Run `pytest tests/test_network_revival_effective_information.py::test_three_node_synergy_notebook_contains_sensitivity_workflow -q`.
- [ ] Inspect generated figure files enough to confirm legends are outside plotted data.
