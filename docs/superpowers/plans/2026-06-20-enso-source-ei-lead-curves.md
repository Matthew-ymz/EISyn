# ENSO Source EI Lead Curves Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ENSO source-EI ranking bars with readable lead-dependent curves for ENSO self-history and selected non-self sources.

**Architecture:** Extend the existing plotting script with one normalized source/seed/lead table derived from pair-level rows. Keep lead-averaged summary generation unchanged, and render the selected source curves in separate self and non-self panels to handle their different scales.

**Tech Stack:** Python, pandas, matplotlib, pytest

---

## Chunk 1: Data extraction and source selection

### Task 1: Add focused source-EI tests

**Files:**
- Create: `tests/test_plot_unicm_enso_ei_syn_insight.py`
- Modify: `scripts/plot_unicm_enso_ei_syn_insight.py`

- [x] Write tests proving that pair-level left/right EI values normalize to one row per target/source/seed/lead, inconsistent duplicate estimates raise an error, and selection returns self plus non-self Top-5 with required NPMM/TNA additions and no duplicates.
- [x] Run `pytest tests/test_plot_unicm_enso_ei_syn_insight.py -v` and confirm failure before implementation.
- [x] Implement `compute_source_ei_leads` and `select_source_ei_curves` with deterministic ordering and duplicate-consistency validation.
- [x] Re-run the focused tests and confirm they pass.

## Chunk 2: Figure and report integration

### Task 2: Replace ranking bars with lead curves

**Files:**
- Modify: `scripts/plot_unicm_enso_ei_syn_insight.py`
- Modify: `docs/reports/Part2.md`
- Regenerate: `fig/unicm_enso_source_ei_rankings.png`
- Regenerate: `fig/unicm_enso_source_ei_rankings.svg`

- [x] Change `plot_source_ei` to draw a self-EI panel and a non-self panel, using seed means and standard-deviation bands.
- [x] Put the non-self legend outside the right axes and preserve tight, unclipped export.
- [x] Pass pair-level rows through the main plotting flow and update Figure 3 caption/prose from ranking language to lead-curve language.
- [x] Run the full plotting script to regenerate PNG/SVG and report content.

### Task 3: Verify outputs

**Files:**
- Verify: `scripts/plot_unicm_enso_ei_syn_insight.py`
- Verify: `docs/reports/Part2.md`
- Verify: `fig/unicm_enso_source_ei_rankings.png`

- [x] Run the focused pytest file and `python -m py_compile scripts/plot_unicm_enso_ei_syn_insight.py`.
- [x] Run targeted whitespace checks on the Python, test, and Markdown sources; generated Matplotlib SVG path data retains exporter-produced trailing spaces.
- [x] Visually inspect the generated PNG for readable curves, uncertainty bands, labels, and a non-overlapping legend.
- [x] Confirm Figure 3 still resolves through its existing asset path and stale nearby bar/ranking wording is removed.
