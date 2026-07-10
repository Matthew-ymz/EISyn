# Runge Exhaustive Degree-3 TM Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaustively score all Runge two-source/one-target hyperedges with the existing degree-3 polynomial TM estimator using recoverable batched computation.

**Architecture:** Add a focused batch scorer that computes the same in-sample polynomial triangular TM MI from cached residual sufficient statistics. Wrap it in a horizon/target chunked CLI that reuses the existing rollout and emits NPZ caches plus compact rankings.

**Tech Stack:** Python, NumPy/SciPy, pandas, pytest, existing Runge rollout/TM modules.

---

## Chunk 1: Exact batched estimator

### Task 1: Specify numerical equivalence

**Files:**
- Create: `tests/test_runge_exhaustive_degree3_tm.py`
- Create: `scripts/run_runge_exhaustive_degree3_tm.py`

- [ ] Write failing tests comparing the wished-for batch API with `estimate_mutual_information_transport_map(..., degree=3)` on deterministic nonlinear and near-constant samples.
- [ ] Assert canonical `a<b`, `[target,a,b]`/`[a,b]` ordering, degree/exponent order, `ddof=1`, low-scale handling, ridge with unpenalized intercept, `min_scale`, raw MI, post-clip EI and delta2.
- [ ] Verify RED with `pytest tests/test_runge_exhaustive_degree3_tm.py -q`.
- [ ] Implement the minimal residual-scale batch estimator.
- [ ] Verify GREEN and require maximum absolute error `<=1e-8` for single, joint and delta2 values.

### Task 2: Enumerate and cache complete candidates

**Files:**
- Modify: `tests/test_runge_exhaustive_degree3_tm.py`
- Modify: `scripts/run_runge_exhaustive_degree3_tm.py`

- [ ] Write failing tests for target exclusion, exact candidate count, deterministic ordering and resume-safe NPZ chunks.
- [ ] Require chunk schema/version and fingerprints for sources, rollout, model/config/blend, H, target, seed, source mode, sample count, degree/ridge/min_scale and candidate ordering.
- [ ] Implement all `C(60,2) × 58 = 102660` relations per horizon, chunked by target.
- [ ] Store only reusable numeric arrays and metadata; avoid CSV for the full cache. Write same-directory temporary NPZ, close/fsync, then `os.replace`.
- [ ] Verify stale, corrupted, incomplete, non-finite or index-invalid chunks are recomputed rather than silently reused.

## Chunk 2: Controlled benchmark and full run

### Task 3: H=1 validation

**Files:**
- Update: `docs/log/live_status.md`
- Update: `docs/log/run_history.jsonl`
- Create: `docs/log/logs/runge_exhaustive_tm_h1.log`

- [x] Reconstruct the 4096 intervention/source samples deterministically, require an exact hash match to `rollout_predictions_H060_n4096.npy`, persist a source-sample baseline for future resume checks, and verify model caches/config hashes and the MLP/Ridge blend. The old run did not retain an independent source artifact, so this limitation is recorded explicitly.
- [ ] Run a small smoke benchmark and compare sampled candidates against the legacy estimator.
- [ ] Run all H=1 candidates if the `1e-8` gate passes.
- [ ] Record runtime, peak memory, exact count, finite-value audit, and overlap/rank movement relative to the old discrete top-1000 rerank.
- [ ] Write a machine-readable H=1 gate result; require every check to pass before Task 4 can start.

### Task 4: All report horizons

**Files:**
- Update: `docs/log/live_status.md`
- Update: `docs/log/run_history.jsonl`
- Update: `docs/log/tuning_report.md`
- Create: recoverable NPZ chunks under the existing Runge result tree

- [ ] Refuse to run horizons beyond H=1 unless the H=1 gate artifact is present, fingerprint-matched and fully passing.
- [ ] Run `H=1..10,15,20,30,40,50,60` with resume enabled.
- [ ] Merge per-horizon rankings and verify `102660` rows for every horizon.
- [ ] Report whether the former discrete top-1000 omitted any new global TM top candidates.
