# Air Causal Search Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable batch experiment pipeline that searches `shanghai / nanjing / hangzhou / beijing` across `1h / 3h / 6h / 12h / 24h`, ranks runs by `O3 + PM2.5 -> O3` synergy under accuracy/stability constraints, and exports `O3 -> O3`, `PM2.5 -> O3`, and synergy graphs for the final shortlist.

**Architecture:** Keep the existing predictor/model stack and current-state-to-future-state formulation, so `Shanghai 1h` remains a valid anchor baseline. Add a new `yrd.air_search` orchestration layer, a generic intervention-sampling helper for `TM` box estimation, and a new CLI that drives coarse `NIS` screening, refine `TM` evaluation, and report generation while writing durable state under `docs/log/air_tuning/`.

**Tech Stack:** Python, `numpy`, `pandas`, `xarray`, `torch`, existing `yrd` utilities, `unittest`, repo-local CLI scripts.

---

## File Structure

**Create:**

- `yrd/air_search.py`
  - Owns city-to-dataset routing, one-step multi-horizon bundle preparation, prediction caching, coarse/refine/report orchestration, leaderboard rows, and graph artifact manifests.
- `yrd/intervention_sampling.py`
  - Owns generic intervention-box estimation from training support, nonnegative lower bounds, uniform-box sampling, and box diagnostics used by both the new search flow and existing Shanghai helpers.
- `scripts/run_air_search.py`
  - Owns CLI parsing and stage dispatch for `coarse`, `refine`, and `report`.
- `tests/test_yrd_air_search.py`
  - Owns routing, bundle, logging, and CLI-level tests for the new search flow.

**Modify:**

- `yrd/data.py:184-247`
  - Generalize `build_one_step_samples` so it supports arbitrary future horizons from the current snapshot instead of hardcoding `(1,)`.
- `yrd/coupling.py:449-778`
  - Add generic single-pollutant pairwise aggregation helpers so `PM2.5 -> O3` can be exported alongside existing synergy summaries.
- `yrd/shanghai_notebook.py:196-346,1141-1193`
  - Reuse the new intervention-sampling helpers and preserve backward-compatible wrappers/imports for current notebook/tests.
- `tests/test_yrd_data.py:88-167`
  - Add one-step multi-horizon sample tests.
- `tests/test_yrd_coupling.py:45-84,185-317`
  - Add `L_v` diagnostics tests and `PM2.5 -> O3` aggregation tests.
- `tests/test_yrd_pipeline.py:37-48`
  - Add CLI help coverage for `scripts/run_air_search.py`.

**Reference Only:**

- `docs/superpowers/specs/2026-04-22-air-causal-search-design.md`
- `docs/log/air_tuning/*`
- `exp/cache/yrd_coupling/shanghai_one_step_o3_station_graph_tm_causal_graph`

## Chunk 1: Search Foundation

### Task 1: Generalize one-step samples to arbitrary future horizons

**Files:**

- Modify: `yrd/data.py:184-247`
- Modify: `tests/test_yrd_data.py:88-167`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_one_step_samples_supports_non_unit_horizon() -> None:
    cfg = YRDExperimentConfig(sample_mode="one_step", horizons=(3,))
    result = build_one_step_samples(ds, metadata, cfg)
    assert set(result["splits"]["train"]["targets"]) == {3}


def test_build_one_step_samples_uses_requested_horizon_for_split_assignment() -> None:
    cfg = YRDExperimentConfig(sample_mode="one_step", horizons=(6,))
    result = build_one_step_samples(ds, metadata, cfg)
    assert result["splits"]["train"]["X"].shape[0] == expected_train_count
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m unittest tests.test_yrd_data.YRDOneStepSampleTests -v`
Expected: FAIL because `build_one_step_samples` currently raises `ValueError("build_one_step_samples currently supports horizons=(1,) only.")`.

- [ ] **Step 3: Implement the minimal generalization in `yrd/data.py`**

```python
def build_one_step_samples(...):
    max_horizon = max(cfg.horizons)
    split_data = {
        split: {"X": [], "times": [], "targets": {h: [] for h in cfg.horizons}}
        for split in ("train", "val", "test")
    }
    for current_index in range(n_time - max_horizon):
        future_time = pd.Timestamp(times[current_index + max_horizon])
        ...
        for horizon in cfg.horizons:
            payload["targets"][horizon].append(
                target_values[current_index + horizon].reshape(-1)
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_yrd_data.YRDOneStepSampleTests -v`
Expected: PASS with new coverage for arbitrary one-step horizons.

- [ ] **Step 5: Commit**

```bash
git add yrd/data.py tests/test_yrd_data.py
git commit -m "feat: support arbitrary one-step forecast horizons"
```

### Task 2: Build a generic one-step air-search bundle and prediction cache

**Files:**

- Create: `yrd/air_search.py`
- Test: `tests/test_yrd_air_search.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_city_scope_routes_beijing_to_bthsa_files() -> None:
    scope = resolve_city_scope("beijing")
    assert scope.dataset_path == Path("data/dataset_bthsa.nc")
    assert scope.station_path == Path("data/stations_bthsa.csv")


def test_build_air_search_config_keeps_one_step_anchor_shape() -> None:
    cfg = build_air_search_config(Path("."), city_en="shanghai", horizon=3, test_mode=True)
    assert cfg.sample_mode == "one_step"
    assert cfg.history_hours == 1
    assert cfg.horizons == (3,)
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m unittest tests.test_yrd_air_search.AirSearchRoutingTests -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'yrd.air_search'`.

- [ ] **Step 3: Implement the bundle/prediction helpers**

```python
@dataclass(frozen=True)
class CityScope:
    city_en: str
    dataset_path: Path
    station_path: Path


def build_air_search_config(root_dir: Path, *, city_en: str, horizon: int, test_mode: bool) -> YRDExperimentConfig:
    scope = resolve_city_scope(city_en)
    return replace(
        YRDExperimentConfig(root_dir=root_dir, dataset_path=scope.dataset_path, station_path=scope.station_path),
        sample_mode="one_step",
        history_hours=1,
        horizons=(int(horizon),),
        ...
    )


def prepare_air_search_bundle(...):
    ds, metadata = load_dataset(cfg, smoke=use_smoke, city_en=city_en)
    sample_bundle = build_one_step_samples(ds, metadata, cfg, smoke=use_smoke)
    ...
```

- [ ] **Step 4: Add prediction caching and verify the tests pass**

Run: `python -m unittest tests.test_yrd_air_search.AirSearchRoutingTests tests.test_yrd_pipeline.YRDTrainingHistoryTests -v`
Expected: PASS for routing/config tests and no regression in one-step prediction cache assumptions.

- [ ] **Step 5: Commit**

```bash
git add yrd/air_search.py tests/test_yrd_air_search.py
git commit -m "feat: add generic air search bundle helpers"
```

## Chunk 2: TM Sampling and Coupling Metrics

### Task 3: Extract intervention sampling and support-cover `L_v` estimation

**Files:**

- Create: `yrd/intervention_sampling.py`
- Modify: `yrd/shanghai_notebook.py:196-346,1141-1193`
- Modify: `tests/test_yrd_coupling.py:45-84`

- [ ] **Step 1: Write the failing tests**

```python
def test_estimate_support_cover_box_profile_covers_train_range() -> None:
    profile = estimate_support_cover_box_profile(
        x_train=x_train,
        input_variables=("O3", "PM2.5"),
        gamma=1.10,
        nonnegative_variables=("O3",),
        stats={"O3": {"mean": 5.0, "std": 2.0}, "PM2.5": {"mean": 3.0, "std": 1.0}},
    )
    assert profile["box_size_by_variable"]["O3"] >= profile["cover_radius_by_variable"]["O3"]
    assert profile["lower_bounds"] is not None
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m unittest tests.test_yrd_coupling.YRDNisSummaryTests -v`
Expected: FAIL because `estimate_support_cover_box_profile` does not exist yet.

- [ ] **Step 3: Implement the generic helper and wire Shanghai wrappers to it**

```python
def estimate_support_cover_box_profile(...):
    center = compute_training_input_center(x_train)
    feature_min = x_train.min(axis=(0, 1))
    feature_max = x_train.max(axis=(0, 1))
    cover_radius = np.maximum(center.mean(axis=0) - feature_min, feature_max - center.mean(axis=0))
    widths = gamma * cover_radius
    return {
        "center": center,
        "box_size_by_variable": dict(zip(input_variables, widths.tolist())),
        "cover_radius_by_variable": ...,
        "lower_bounds": resolve_nonnegative_lower_bounds_by_feature(...),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_yrd_coupling.YRDNisSummaryTests tests.test_yrd_shanghai_notebook.YRDShanghaiNotebookTests -v`
Expected: PASS, including the existing Shanghai sampling tests through compatibility imports/wrappers.

- [ ] **Step 5: Commit**

```bash
git add yrd/intervention_sampling.py yrd/shanghai_notebook.py tests/test_yrd_coupling.py
git commit -m "feat: add support-cover TM sampling helpers"
```

### Task 4: Add `PM2.5 -> O3` pairwise aggregation on top of existing pollutant decomposition

**Files:**

- Modify: `yrd/coupling.py:449-778`
- Modify: `tests/test_yrd_coupling.py:185-317`
- Modify: `yrd/air_search.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_summarize_global_station_single_pollutant_ei_returns_pm25_edges() -> None:
    aggregated = summarize_global_station_single_pollutant_ei(
        sample_summaries=[
            {
                "target_station_id": "A",
                "single_pollutant_ei_nis": {"A": {"O3": 0.2, "PM2.5": 0.4}, "B": {"O3": 0.1, "PM2.5": 0.3}},
            }
        ],
        station_ids=["A", "B"],
        feature_name="PM2.5",
    )
    assert aggregated["pairwise_edges"]
    assert aggregated["per_target_station"]["A"]["single_feature_ei_nis"]["A"]["mean"] == 0.4
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m unittest tests.test_yrd_coupling.YRDNisSummaryTests.test_summarize_global_station_single_pollutant_ei_returns_pm25_edges -v`
Expected: FAIL because the new summarizer does not exist.

- [ ] **Step 3: Implement the minimal aggregation and expose it in `yrd.air_search`**

```python
def summarize_global_station_single_pollutant_ei(...):
    per_target_station = {}
    edge_rows = []
    for target_station_id in station_ids:
        ...
        feature_summary = {
            station_id: _summary_stats([
                float(dict(row.get("single_pollutant_ei", row.get("single_pollutant_ei_nis", {}))
                      .get(station_id, {})).get(feature_name, 0.0))
                for row in target_rows
            ])
            for station_id in station_ids
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_yrd_coupling -v`
Expected: PASS, including the new `PM2.5 -> O3` directed-edge summary.

- [ ] **Step 5: Commit**

```bash
git add yrd/coupling.py yrd/air_search.py tests/test_yrd_coupling.py
git commit -m "feat: add PM2.5 to O3 pairwise aggregation"
```

## Chunk 3: CLI, Logging, and Smoke Execution

### Task 5: Implement the coarse screening CLI and durable logging

**Files:**

- Create: `scripts/run_air_search.py`
- Modify: `yrd/air_search.py`
- Modify: `tests/test_yrd_air_search.py`
- Modify: `tests/test_yrd_pipeline.py:37-48`

- [ ] **Step 1: Write the failing tests**

```python
def test_run_air_search_cli_supports_help_output() -> None:
    result = subprocess.run([sys.executable, "scripts/run_air_search.py", "--help"], ...)
    assert result.returncode == 0
    assert "coarse" in result.stdout
    assert "refine" in result.stdout


def test_build_coarse_row_prefers_absolute_syn_and_tracks_negative_ratio() -> None:
    row = build_coarse_row(
        city_en="nanjing",
        horizon=6,
        o3_rmse=8.0,
        baseline_o3_rmse=10.0,
        syn_mean=0.42,
        syn_negative_ratio=0.08,
    )
    assert row["passes_accuracy_gate"] is True
    assert row["primary_syn_mean"] == 0.42
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m unittest tests.test_yrd_air_search.AirSearchCliTests tests.test_yrd_pipeline.YRDCliTests -v`
Expected: FAIL because `scripts/run_air_search.py` and coarse-row helpers do not exist.

- [ ] **Step 3: Implement the coarse stage**

```python
def run_coarse_stage(args: argparse.Namespace) -> dict[str, object]:
    ensure_air_tuning_state(...)
    for city_en in parse_csv(args.cities):
        for horizon in parse_int_csv(args.horizons):
            cfg = build_air_search_config(...)
            bundle = prepare_air_search_bundle(...)
            predictions = run_or_load_air_search_predictions(...)
            coarse_row = build_coarse_row(...)
            append_run_history(...)
    write_leaderboard(...)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_yrd_air_search tests.test_yrd_pipeline -v`
Expected: PASS, including CLI help and coarse-row/logging behavior.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_air_search.py yrd/air_search.py tests/test_yrd_air_search.py tests/test_yrd_pipeline.py
git commit -m "feat: add coarse air causal search cli"
```

### Task 6: Implement refine/report stages, graph exports, and smoke verification

**Files:**

- Modify: `yrd/air_search.py`
- Modify: `scripts/run_air_search.py`
- Modify: `tests/test_yrd_air_search.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_choose_tm_gamma_prefers_smallest_covering_gamma_with_stable_signs() -> None:
    winner = choose_tm_gamma(
        [
            {"gamma": 1.00, "syn_negative_ratio": 0.35},
            {"gamma": 1.10, "syn_negative_ratio": 0.08},
            {"gamma": 1.20, "syn_negative_ratio": 0.07},
        ]
    )
    assert winner["gamma"] == 1.10


def test_report_manifest_contains_three_required_graph_types() -> None:
    manifest = build_report_manifest(...)
    assert {"o3_pairwise", "pm25_to_o3_pairwise", "o3_pm25_synergy"} <= set(manifest["graphs"])
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `python -m unittest tests.test_yrd_air_search.AirSearchRefineTests -v`
Expected: FAIL because refine/report helpers do not exist.

- [ ] **Step 3: Implement refine/report behavior**

```python
def run_refine_stage(args: argparse.Namespace) -> dict[str, object]:
    shortlist = load_shortlist(...)
    for run in shortlist:
        for gamma in parse_float_csv(args.tm_gammas):
            profile = estimate_support_cover_box_profile(...)
            for seed in parse_int_csv(args.tm_seeds):
                for sample_count in parse_int_csv(args.tm_sample_counts):
                    tm_summary = compute_tm_summary(...)
        export_pairwise_graph(...)
        export_pm25_to_o3_graph(...)
        export_synergy_graph(...)


def run_report_stage(args: argparse.Namespace) -> dict[str, object]:
    summarize_refine_runs(...)
    write_final_report(...)
```

- [ ] **Step 4: Run unit tests plus a smoke batch**

Run: `python -m unittest tests.test_yrd_air_search tests.test_yrd_coupling tests.test_yrd_data tests.test_yrd_pipeline -v`
Expected: PASS.

Run: `env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/run_air_search.py --cities shanghai,nanjing --horizons 1,3 --stage coarse --smoke`
Expected: PASS; writes smoke caches/manifests and updates `docs/log/air_tuning/run_history.jsonl`.

Run: `env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/run_air_search.py --cities shanghai --horizons 3 --stage refine --smoke --top-k 1 --tm-sample-counts 64 --tm-seeds 0 --tm-gammas 1.0,1.1`
Expected: PASS; emits smoke graph artifacts and a refine summary without requiring GUI or network access.

- [ ] **Step 5: Commit**

```bash
git add yrd/air_search.py scripts/run_air_search.py tests/test_yrd_air_search.py
git commit -m "feat: add refine and report stages for air causal search"
```

## Notes for Execution

- Keep the first implementation path on `sample_mode="one_step"` so it stays comparable to the existing `Shanghai 1h` anchor.
- Treat the historical Shanghai `1h` cache as reference-only unless its metadata matches the new standardized run context exactly.
- Do not update `docs/研究框架.md` during implementation; only touch it after a real winning configuration exists and the exported figure assets are stable.
- Keep all final inserted figure assets as `png` or `pdf`, not `svg`, to match repo document rules.

Plan complete and saved to `docs/superpowers/plans/2026-04-22-air-causal-search.md`. Ready to execute?
