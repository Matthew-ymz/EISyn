# Runge component MLP-TM-EI path-effect gateway readout

This experiment keeps the MLP/TM-EI intervention readout, but computes Runge-style path effects on a sparsified EI causal graph.

## Run

- Components: 60
- Rows: 3337
- Lag: 4
- Horizon: 1
- EI estimator: tm
- Gateway mode: path_effect
- Graph sparsify: source_topk
- Graph top-k: 5
- Graph quantile: 0.95
- Path alpha: 0.8
- Path scale factor: 1
- Direct EI edges: 300
- Total-effect paths: 3540
- Mediated paths: 17400
- MLP cache reused: False
- Overall test RMSE: 0.713185
- Overall test corr: 0.461528

## Top gateways

| paper_component | ace | acs | direct_out_strength | direct_in_strength |
| --- | ---: | ---: | ---: | ---: |
| No.4 | 0.00521111 | 0.00150392 | 0.284866 | 0.0773915 |
| No.2 | 0.00503596 | 0.00394063 | 0.258897 | 0.207111 |
| No.0 | 0.0045057 | 0.00261591 | 0.229355 | 0.137451 |
| No.1 | 0.0044806 | 0.00173307 | 0.235623 | 0.091175 |
| No.3 | 0.00420316 | 0.0017963 | 0.219496 | 0.0951801 |
| No.9 | 0.00393288 | 0.00318431 | 0.212258 | 0.157702 |
| No.6 | 0.00387701 | 0.00153928 | 0.193864 | 0.0800985 |
| No.11 | 0.00384106 | 0.00327235 | 0.198245 | 0.175156 |
| No.22 | 0.00325219 | 0.0033044 | 0.164579 | 0.174048 |
| No.41 | 0.00290161 | 0.00103169 | 0.147104 | 0.0511529 |

## Top mediators

| paper_component | amce | mediated_fraction |
| --- | ---: | ---: |
| No.2 | 1.791e-05 | 0.0738727 |
| No.11 | 1.1036e-05 | 0.0455199 |
| No.0 | 1.01606e-05 | 0.0419092 |
| No.9 | 1.00153e-05 | 0.0413096 |
| No.22 | 9.13716e-06 | 0.0376877 |
| No.48 | 8.21485e-06 | 0.0338835 |
| No.1 | 7.03851e-06 | 0.0290315 |
| No.4 | 6.93475e-06 | 0.0286035 |
| No.3 | 6.8769e-06 | 0.0283649 |
| No.37 | 6.57468e-06 | 0.0271183 |

## Pairwise EI vs linear coefficient matrix

- Compared elements: 3600
- Off-diagonal elements: 3540
- Support match fraction: 0.213333
- Spearman(abs linear coefficient, EI): 0.485476
- Per-element comparison: `ei_linear_coefficient_comparison.csv`.
