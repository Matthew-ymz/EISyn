# Experiment and reproducibility index

This index maps manuscript analyses to their computational and plotting entry
points. Use the manuscript Methods and Supplementary Information for the
scientific definitions, intervention settings and stated limitations. The
application scripts and their saved configurations specify the implemented
parameters.

## Methods

- [Transport-map density and information estimators](../../exp/TM/transport_map_density.py)
- [Discrete information and benchmark utilities](../../utils.py)
- [Synergy Partition Tree construction and numerical audits](../../scripts/spt.py)
- [Formal-verification scope and instructions](../../formal/peid_appendix/README.md)

When reproducing a result, preserve its intervention distribution, source and
target definitions, prediction horizon, estimator settings and information
units. Report the declared Syn tolerance and the numerical-zero/violation
counts. A significant nonnegativity violation must remain visible as a failed
calculation.

## Controlled dynamics

- [Compose the main benchmark figure](../../scripts/compose_part1_figure1.py)
- [Compose the three hierarchy examples](../../scripts/compose_spt_three_examples.py)
- [Six-node dynamics and planted structures](../../scripts/search_six_node_mixed_dynamics.py)
- [Kuramoto topology definitions](../../scripts/kuramoto_six_topology_core.py)
- [Known discrete-dynamics validation](../../scripts/validate_greedy_hierarchy_known_dynamics.py)
- [Nonlinear-system comparisons and synchronization routines](../../scripts/classic_network_dynamics_benchmark.py)

Composition scripts read upstream results. Obtain the corresponding inputs and
run the calculation stages before assembling a figure. Observational and
interventional estimates have separate interpretation and settings.

## Brain

- [Continuous subject-curve calculations](../../scripts/run_dmf_continuous_subject_curves.py)
- [ROI order attribution](../../scripts/plot_dmf_roi_order_attribution.py)
- [Main brain figure](../../scripts/compose_dmf_brain_main_figure.py)
- [Length-matched HCP supplementary analysis](../../scripts/run_hcp_schaefer1000_length_matched_rest_57.py)

These analyses require the structural-connectivity or time-series resources
expected by the selected script. Local results are commonly stored below
`results/dmf_schaefer100/` and the corresponding HCP result directories.
Model data and access-controlled recordings require their own acquisition
steps.

## Climate

- [UniCM model and information-analysis interface](../../scripts/unicm_peid_syn_analysis.py)
- [Coalition attribution](../../scripts/analyze_unicm_11mode_shapley.py)
- [Information-prior forecast calibration](../../scripts/run_unicm_target_xi_shapley_prior.py)
- [Earth-system figure generation](../../scripts/plot_earth_system_main_figures.py)
- [Supplementary SLP gateway calculations](../../scripts/reproduce_runge2015_gateways.py)

UniCM analyses use upstream model source and checkpoints under
`data/UniCM-checkpoint/`. SLP analyses use the data preparation and configuration
specified by their scripts. Checkpoint versions, preprocessing, fitting/test
splits and cached arrays must match the result being reproduced.

## Urban mobility

- [NYC Taxi data preparation](../../scripts/prepare_nyc_taxi_mgstn.py)
- [MGSTN training and evaluation](../../scripts/train_nyc_taxi_mgstn.py)
- [Finite-amplitude information calculation](../../scripts/compute_nyc_taxi_mgstn_quadratic_tm_full.py)
- [SPT analysis](../../scripts/analyze_nyc_taxi_mgstn_spt.py)
- [Figure composition](../../scripts/plot_nyc_taxi_multiscale_spt_composite.py)

The default prepared data are stored under `data/nyc_taxi_mgstn_2023/`.
Model checkpoints and downstream analysis results are separate inputs. Regional
temporal grouping and the complete SPT use their respective configurations;
keep those settings explicit when comparing outputs.

## Environment and release artifacts

See the [installation guide](../INSTALL.md) for Python dependencies and small
checks. Many datasets, checkpoints, result caches and figure exports are
excluded from Git. A complete paper release still needs accessible, versioned
input artifacts and a frozen figure-to-code manifest. Installation and a cached
figure replay cover different parts of reproducibility.
