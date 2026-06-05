# Runge component pairwise MLP-EI gateway readout

This experiment trains a cached MLP transition model on Varimax/PCA component score dynamics and reads out pairwise effective information under independent maximum-entropy interventions.

## Scope

- Source unit: `latest` component state read from the lagged MLP input.
- Target unit: one next-horizon component.
- Synergy and mediator blocking are intentionally not computed in this run.

## Run

- Components: 60
- Rows: 3339
- Lag: 4
- Horizon: 1
- Intervention samples: 4096
- EI estimator: tm
- MLP cache reused: False
- Overall test RMSE: 0.714863
- Overall test corr: 0.450806

## Top gateways

| paper_component | gateway_ei | susceptibility_ei | self_memory_ei |
| --- | ---: | ---: | ---: |
| No.0 | 0.00810023 | 0.00285975 | 0.236519 |
| No.7 | 0.00671438 | 0.00391044 | 0.111823 |
| No.18 | 0.00656847 | 0.00457267 | 0.094801 |
| No.24 | 0.0060384 | 0.00288237 | 0.264589 |
| No.13 | 0.00571756 | 0.00380675 | 0.165827 |
| No.29 | 0.00545403 | 0.00345501 | 0.105472 |
| No.1 | 0.00478886 | 0.00312305 | 0.179041 |
| No.14 | 0.0047563 | 0.00348793 | 0.0652022 |
| No.15 | 0.00465483 | 0.00235996 | 0.311568 |
| No.16 | 0.00417825 | 0.00325525 | 0.200738 |

## Pairwise EI vs linear coefficient matrix

- Compared elements: 3600
- Off-diagonal elements: 3540
- Support match fraction: 0.367778
- Spearman(abs linear coefficient, EI): 0.516349
- Per-element comparison: `ei_linear_coefficient_comparison.csv`.
