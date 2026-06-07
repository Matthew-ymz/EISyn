# Transformer PEID check against Runge 2015 Fig. 3

## Setup

- Paper: Runge et al. (2015), Fig. 3 teleconnection graph.
- Zotero item key: `TY99I538`.
- Neural model: Transformer h=1 candidate selected inside the current `TransformerHorizonSelector` system.
- Candidate: `stageE_objective_skip_h2_d128_nh8_l1_ff512_poollast_possinusoidal_objdirect_multihorizon_skipno_skip_lr0p0015_do0p05_wd0p0001_seed42_e2073f44`.
- Checkpoint: `results/runge_transformer_forecast_sweep/candidates/stageE_objective_skip_h2_d128_nh8_l1_ff512_poollast_possinusoidal_objdirect_multihorizon_skipno_skip_lr0p0015_do0p05_wd0p0001_seed42_e2073f44/rnn_transition_06ab0c675354a1c9.pt`.
- Prediction context: lag=2 weeks, direct heads=[1, 2, 4, 8], recursive h1 rollout evaluated for h=1,2,3.
- Interventions: 4096 independent max-entropy samples from train-feature 0.05-0.95 quantiles.
- Estimator: affine transport-map mutual information, clipped at zero for EI.
- Node labels: Fig. 3 paper labels No.0, No.1, No.33, No.53, and No.59 map to the same local indices under the repo calibration.

## Fig. 3 Edge Comparison

Support calls use rank among all 60 possible sources for the same target, horizon, and input-lag setting: top10=strong, top20=moderate.

| check | edge | expected lag | best lag | mode | EI | source rank | call |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| solid_mediator_chain_1_to_0_lag2 | No.1->No.0 | 2.0 | 2 | direct_head:latest_input | 0.0530792 | 2 | strong_top10 |
| solid_mediator_chain_0_to_33_lag1 | No.0->No.33 | 1.0 | 1 | direct_head:latest_input | 0.0311581 | 3 | strong_top10 |
| weaker_path_1_to_53_any_lag1_3 | No.1->No.53 | 1-3 | 2 | recursive_h1:latest_input | 0.0324745 | 2 | strong_top10 |
| weaker_path_53_to_33_any_lag1_3 | No.53->No.33 | 1-3 | 2 | recursive_h1:latest_input | 0.00023459 | 50 | weak_positive |
| total_effect_1_to_33_lag3 | No.1->No.33 | 3.0 | 3 | recursive_h1:latest_input | 0.126488 | 1 | strong_top10 |
| dashed_common_driver_59_to_1_any_lag1_3 | No.59->No.1 | 1-3 | 2 | recursive_h1:latest_input | 0.00213458 | 39 | weak_positive |
| dashed_common_driver_59_to_33_any_lag1_3 | No.59->No.33 | 1-3 | 1 | direct_head:latest_input | 0.00866402 | 15 | moderate_top20 |

## Interpretation

Solid/total checks: 4 strong top-10, 0 moderate top-20, 1 weak positive, 0 absent/zero.
The Transformer PEID/TM-EI readout should be interpreted as partial agreement only when the solid Fig. 3 links rank highly. Dashed No.59 rows are confounder diagnostics, not desired solid-path recovery.

## PEID Delta2 Notes

`selected_node_peid_delta2.csv` reports second-order Mobius interactions for selected source pairs. Positive delta2 means the joint source pair explains target variation beyond the sum of its two single-source EI terms under the same intervention batch.

## Artifacts

- `results/runge/transformer_peid_fig3_edges/fig3_edge_comparison.csv`
- `results/runge/transformer_peid_fig3_edges/lag_resolved_tm_ei_edges.csv`
- `results/runge/transformer_peid_fig3_edges/selected_node_peid_delta2.csv`
- `results/runge/transformer_peid_fig3_edges/top_sources_for_fig3_targets.csv`
