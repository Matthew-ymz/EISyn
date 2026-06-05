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
- EI estimator: discrete
- MLP cache reused: True
- Overall test RMSE: 0.714863
- Overall test corr: 0.450806

## Top gateways

| paper_component | gateway_ei | susceptibility_ei | self_memory_ei |
| --- | ---: | ---: | ---: |
| No.0 | 0.019931 | 0.012215 | 0.326063 |
| No.7 | 0.0179124 | 0.0138652 | 0.157183 |
| No.18 | 0.0172884 | 0.0144778 | 0.138497 |
| No.24 | 0.0166708 | 0.0122699 | 0.363106 |
| No.13 | 0.0162601 | 0.013392 | 0.232251 |
| No.29 | 0.0159254 | 0.013673 | 0.155027 |
| No.14 | 0.0151837 | 0.0132611 | 0.0956469 |
| No.15 | 0.0148611 | 0.011742 | 0.435105 |
| No.1 | 0.0147895 | 0.0127419 | 0.252767 |
| No.32 | 0.0143565 | 0.0121663 | 0.230834 |
