# Coupled Standard Map Six-Method Comparison

![Six-method comparison](../../fig/coupled_standard_map_method_comparison/coupled_standard_map_six_method_comparison.png)

## Protocol

- coupling values: `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`
- seeds: `[0, 1, 2, 3]`
- trajectories per full run: `16`
- steps per trajectory: `2500`
- targets: impulses `I1` and `I2`; symmetric target readouts are averaged only in the main figure
- analytic cross and interaction strength: `J^2 / 2`

## Surrogate Quality

| J | min R2 | max NRMSE | max circular MAE | gate pass rate |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.9973 | 0.0518 | 0.0409 | 0.75 |
| 0.2 | 0.9976 | 0.0487 | 0.0408 | 1.00 |
| 0.4 | 0.9978 | 0.0472 | 0.0412 | 1.00 |
| 0.6 | 0.9979 | 0.0456 | 0.0410 | 1.00 |
| 0.8 | 0.9982 | 0.0429 | 0.0410 | 1.00 |
| 1.0 | 0.9983 | 0.0410 | 0.0411 | 1.00 |

## Spearman Trend Against J^2/2

| readout | rho |
| --- | ---: |
| wms | 1.0000 |
| shap_interaction | 1.0000 |
| surd_synergy | -0.3714 |
| pcmci_cross | 1.0000 |
| neural_granger_cross | 1.0000 |
| mlp_peid_synergy | 1.0000 |
| oracle_peid_synergy | 1.0000 |

## Ground-Truth Diagnostics

### J=0 absolute readout

| readout | mean absolute value |
| --- | ---: |
| wms | 0.035009 |
| shap_interaction | 0.002328 |
| surd_synergy | 0.834626 |
| pcmci_cross | 0.000992 |
| neural_granger_cross | 0.004653 |
| mlp_peid_synergy | 0.014995 |
| oracle_peid_synergy | 0.015518 |

### True source versus momentum null

| method | cross > null rate | mean margin |
| --- | ---: | ---: |
| shap | 1.000 | 0.188747 |
| pcmci | 0.950 | 0.016868 |
| neural_granger | 1.000 | 1.141374 |
| mlp_peid | 1.000 | 0.328284 |

- MLP+PEID true pair top rate: `1.000`
- MLP+PEID mean relative Oracle error: `0.0021`
- MLP+PEID maximum relative Oracle error: `0.0096`

## Observed Result

MLP+PEID tracks Oracle synergy with Spearman `rho=1.000`, identifies `q1+q2` as the strongest pair in `100.0%` of positive-coupling runs, and has maximum relative Oracle error `0.962%`.

Observational SURD does not track the analytic coupling trend in this periodic system (`rho=-0.371`) and has a large `J=0` synergy readout. This is retained as a method failure rather than removed by post-hoc tuning.

## Interpretation Boundary

The panels retain each method's native scale. WMS and SURD are observational distribution readouts; SHAP and Neural Granger describe fitted predictive use; PCMCI reports lagged conditional dependence; MLP+PEID evaluates the learned mechanism under independent interventions. Their absolute magnitudes are therefore not interchangeable.

PEID rows are considered surrogate-valid only where the preregistered MLP quality gate passes. Oracle and learned PEID use identical intervention states and matched noise draws.
