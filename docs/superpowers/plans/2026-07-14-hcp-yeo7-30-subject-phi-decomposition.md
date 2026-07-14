# HCP Yeo7 30 Subject Phi Decomposition Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Yeo7 history-source PhiEID greedy decomposition and its observed/null comparison from five HCP subjects to all 30 available subjects.

**Architecture:** The decomposition runner will discover valid `sub-*/*.mat` inputs when no explicit subject list is supplied, matching the existing all-subject Phi/null runner. Explicit `--subjects` remains an override for smoke runs. The existing per-subject observed/null pipeline, aggregation, heatmap, and report formats are retained; only their cohort scope and hard-coded wording change.

**Tech Stack:** Python 3, NumPy, SciPy, Matplotlib, pytest, HCP Schaefer-500 MATLAB inputs.

---

## Chunk 1: Cohort discovery and regression tests

### Task 1: Encode default cohort behavior

**Files:**
- Modify: `tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py`
- Modify: `scripts/run_hcp_schaefer500_yeo7_module_phi_decomposition.py`

- [ ] **Step 1: Write the failing test**

```python
def test_discover_subjects_lists_sorted_mat_subject_directories(tmp_path):
    for subject in ("sub-100408", "sub-100206"):
        path = tmp_path / subject
        path.mkdir()
        (path / "run.mat").touch()

    assert decomposition.discover_subjects(tmp_path) == ("sub-100206", "sub-100408")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py -q`

Expected: FAIL because `discover_subjects` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def discover_subjects(data_root: Path) -> tuple[str, ...]:
    return tuple(path.parent.name for path in sorted(Path(data_root).glob("sub-*/*.mat")))
```

Set `subjects: Sequence[str] | None = None` in `run`; call `discover_subjects` only when it is `None`. Let the CLI default be an empty `--subjects` string, which passes `None`, while a comma-separated value remains an explicit override.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py -q`

Expected: PASS.

### Task 2: Make cohort-derived metadata and prose accurate

**Files:**
- Modify: `scripts/run_hcp_schaefer500_yeo7_module_phi_decomposition.py`
- Test: `tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py`

- [ ] **Step 1: Write the failing test**

```python
def test_report_uses_actual_subject_and_null_counts(tmp_path):
    summary = {"rows": [{"subject": "sub-a", "top_atoms": []}],
               "config": {"null_replicates": 20},
               "core_summary": [], "null_rank_comparison": [],
               "null_core_summary": []}
    decomposition.write_report(summary, tmp_path / "report.md")
    assert "1 名被试" in (tmp_path / "report.md").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py -q`

Expected: FAIL because report text hard-codes five subjects and 100 null models.

- [ ] **Step 3: Write minimal implementation**

Derive headings, null-cohort prose, and denominator values from `len(summary["rows"])` and `summary["config"]["null_replicates"]`. Change the config description from “five subjects” to generic matched cohorts. Ensure `config.subjects` records the resolved list.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py -q`

Expected: PASS.

## Chunk 2: Full recomputation and documentation

### Task 3: Run the 30-subject decomposition

**Files:**
- Generate: `results/hcp_schaefer500_yeo7_module_phi_decomposition/{summary.json,report.md,top_core_consistency.png,top_core_consistency.svg,top_core_consistency.pdf}`

- [ ] **Step 1: Execute the runner with defaults**

Run: `python scripts/run_hcp_schaefer500_yeo7_module_phi_decomposition.py`

Expected: 30 progress lines and a 30-row summary with 20 null replicates per subject.

- [ ] **Step 2: Verify the result schema and cohort count**

Run: `python -c 'import json; s=json.load(open("results/hcp_schaefer500_yeo7_module_phi_decomposition/summary.json")); assert len(s["rows"]) == 30; assert all(len(r["null_top_atoms"]) == 20 for r in s["rows"]); print(len(s["rows"]))'`

Expected: `30`.

- [ ] **Step 3: Visually inspect the generated heatmap**

Open `results/hcp_schaefer500_yeo7_module_phi_decomposition/top_core_consistency.png` and check all subject rows, annotations, and colorbar are readable and no legend overlaps data.

### Task 4: Update the brain report

**Files:**
- Modify: `docs/reports/brain.md`

- [ ] **Step 1: Replace the five-subject-only section**

Update section 2.3 heading/prose, its summary table, figure caption, and null-rank interpretation from the generated 30-subject JSON/report. Preserve the caveat that greedy atoms are descriptive and rank tests do not make a unique biological-atom claim.

- [ ] **Step 2: Verify cross-references and numeric consistency**

Run: `rg -n "5 名被试|4 / 5|3 / 5|100 个 null|五被试" docs/reports/brain.md results/hcp_schaefer500_yeo7_module_phi_decomposition/report.md`

Expected: no obsolete five-subject statements in the updated HCP module decomposition content.

### Task 5: Final verification

**Files:**
- Verify: `tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py`
- Verify: `docs/reports/brain.md`

- [ ] **Step 1: Run the focused tests**

Run: `pytest tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py -q`

Expected: PASS.

- [ ] **Step 2: Review changed files**

Run: `git diff --check && git diff -- scripts/run_hcp_schaefer500_yeo7_module_phi_decomposition.py tests/test_hcp_schaefer500_yeo7_module_phi_decomposition.py docs/reports/brain.md`

Expected: no whitespace errors; all changes scoped to the approved 30-subject expansion.
