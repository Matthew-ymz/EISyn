# Runge component MLP-TM-EI path-effect gateway readout

This experiment keeps the MLP/TM-EI intervention readout, but computes Runge-style path effects on a sparsified EI causal graph.

## Run

- Components: 60
- Rows: 3339
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
- Total-effect paths: 3481
- Mediated paths: 17105
- MLP cache reused: True
- Overall test RMSE: 0.714863
- Overall test corr: 0.450806

## Top gateways

| paper_component | ace | acs | direct_out_strength | direct_in_strength |
| --- | ---: | ---: | ---: | ---: |
| No.0 | 0.00496352 | 0.000480063 | 0.256552 | 0.0261672 |
| No.13 | 0.0040986 | 0.00258046 | 0.208253 | 0.144555 |
| No.18 | 0.00370668 | 0.00179863 | 0.196691 | 0.0967358 |
| No.7 | 0.00344251 | 0.00352354 | 0.185037 | 0.185373 |
| No.29 | 0.00334748 | 0.00243375 | 0.180194 | 0.127612 |
| No.24 | 0.003094 | 0.00145019 | 0.161276 | 0.080278 |
| No.15 | 0.00262423 | 0.000670541 | 0.144387 | 0.0372208 |
| No.12 | 0.0023969 | 0.000796534 | 0.127024 | 0.0425909 |
| No.6 | 0.00233771 | 0.0014595 | 0.129927 | 0.0741049 |
| No.1 | 0.00224323 | 0.00183739 | 0.119804 | 0.0984574 |

## Top mediators

| paper_component | amce | mediated_fraction |
| --- | ---: | ---: |
| No.7 | 1.09789e-05 | 0.0789931 |
| No.13 | 1.01959e-05 | 0.0733593 |
| No.29 | 7.35036e-06 | 0.0528858 |
| No.18 | 6.17258e-06 | 0.0444117 |
| No.43 | 5.99149e-06 | 0.0431087 |
| No.14 | 4.765e-06 | 0.0342841 |
| No.8 | 4.53834e-06 | 0.0326533 |
| No.24 | 3.83988e-06 | 0.0276279 |
| No.1 | 3.8033e-06 | 0.0273647 |
| No.56 | 3.2408e-06 | 0.0233175 |

## Pairwise EI vs linear coefficient matrix

- Compared elements: 3600
- Off-diagonal elements: 3540
- Support match fraction: 0.367778
- Spearman(abs linear coefficient, EI): 0.516349
- Per-element comparison: `ei_linear_coefficient_comparison.csv`.
