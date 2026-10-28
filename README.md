# EISyn

Research code for **Revealing hierarchies of synergistic causal interactions in
complex systems**. The repository supports Partial Effective Information
Decomposition (PEID), integrated effective information and Synergy Partition
Trees (SPT), with controlled dynamical benchmarks and applications to brain
models, climate prediction and urban mobility.

## Installation

Use **Python 3.11** and run commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

A Conda alternative is available through
[environment.yml](environment.yml). Install [requirements-dev.txt](requirements-dev.txt)
for tests and notebooks, and [requirements-experiments.txt](requirements-experiments.txt)
for additional application dependencies.

See the [installation guide](docs/INSTALL.md) for platform instructions, import
checks and external data/model requirements. Installing dependencies does not
download datasets, pretrained weights or cached results.

## Manuscript experiments

| Analysis | Entry points and documentation |
|---|---|
| Controlled dynamics and hierarchy recovery | [Figure 1 composition](scripts/compose_part1_figure1.py), [three-example comparison](scripts/compose_spt_three_examples.py), [reproduction index](docs/reports/README.md#controlled-dynamics) |
| Brain dynamics and information organization | [Brain figure composition](scripts/compose_dmf_brain_main_figure.py), [ROI attribution](scripts/plot_dmf_roi_order_attribution.py), [reproduction index](docs/reports/README.md#brain) |
| Climate prediction with UniCM | [UniCM analysis](scripts/unicm_peid_syn_analysis.py), [coalition attribution](scripts/analyze_unicm_11mode_shapley.py), [reproduction index](docs/reports/README.md#climate) |
| NYC Taxi prediction and temporal hierarchy | [Data preparation](scripts/prepare_nyc_taxi_mgstn.py), [MGSTN training](scripts/train_nyc_taxi_mgstn.py), [SPT analysis](scripts/analyze_nyc_taxi_mgstn_spt.py), [reproduction index](docs/reports/README.md#urban-mobility) |
| Supplementary analyses | Nonlinear-system comparisons, Kuramoto synchronization, HCP recordings, SLP climate modes and structural-connectivity comparisons; see the [report index](docs/reports/README.md) |

The reproduction index identifies existing computation and figure sources. Figure-composition scripts use their upstream results; running a
composition script alone does not recompute an experiment. Some linked data and
results are local and still need publication archives for a complete paper
release.

## Shared code

- [utils.py](utils.py): discrete information measures, transition matrices and
  benchmark utilities.
- [exp/TM/transport_map_density.py](exp/TM/transport_map_density.py): transport-map
  density and mutual-information estimators.
- [scripts/spt.py](scripts/spt.py): tree construction, split selection,
  nonnegativity auditing and closure checks.
- [density_benchmark.py](density_benchmark.py): density-estimation comparisons.
- [formal/peid_appendix/](formal/peid_appendix/README.md): scoped Lean verification
  of appendix identities and conditional arguments.

Use the manuscript Methods and Supplementary Information for intervention
settings, estimator approximations, units and numerical validity limits. The
[reproduction index](docs/reports/README.md) maps analyses to code. Preserve each
analysis's stated Syn tolerance and failure behavior.

## Checks

After installing the development dependencies:

```bash
python -m pytest -q tests/test_spt_core.py tests/test_greedy_hierarchy_known_dynamics.py
```

These small checks cover the common tree contract and a known discrete system.
They do not replace the full application-specific validation. See the
[installation guide](docs/INSTALL.md#check-the-installation) for import checks.

## Data and outputs

`data/` holds local input data and upstream model resources; `results/` and
`exp/cache/` hold experiment results and reusable caches. `fig/`,
`docs/reports/assets/` and `paper_assets/` hold figure exports. Many of these
paths are excluded from Git, so a clone does not include every local artifact.
Use the [reproduction index](docs/reports/README.md) and the selected scripts
for resource locations and required inputs.

## License and citation

Repository-authored code is available under the [MIT License](LICENSE).
Third-party software, pretrained weights and datasets retain their original
licenses and access conditions.

Use [CITATION.cff](CITATION.cff) for software citation metadata and identify the
commit or release used. Version-specific DOI metadata will be added when the
paper code is archived.
