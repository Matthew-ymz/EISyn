# UniCM SHAP vs PEID: ENSO and IOD

本轮使用 frozen UniCM Modeformer 的 full-history prediction cache 训练 tree surrogate，
再用 TreeSHAP 估计 mode-level 单源归因，并用 exact group Shapley interaction 估计 11 个 mode group 的二阶交互。没有重训 UniCM，也没有重新执行 checkpoint forward。

SHAP/SHAP interaction 是 surrogate prediction attribution；PEID single-source EI 与 pair Syn 是干预分布下的信息分解。二者用于对照机制线索，不能直接等同。

## Surrogate quality

| Target | min R2 | mean R2 | low-quality fits (R2 < 0.95) |
|---|---:|---:|---:|
| IOD | 0.188545 | 0.479678 | 69 |
| ENSO | 0.078265 | 0.492681 | 65 |

注意：`134/144` 个 seed-target-lead surrogate 的 R2 低于 `0.95`，主要影响长 lead 的归因强度解释；
下表排名仍可作为 frozen UniCM prediction cache 的筛查读数，但不应被解释为高保真 surrogate 下的最终机制结论。

## Single-source SHAP vs PEID EI

### ENSO

| SHAP rank | Mode | mean \|SHAP\| | PEID source EI | PEID rank |
|---:|---|---:|---:|---:|
| 1 | nino | 0.081007 | 0.473612 | 1 |
| 2 | nino3 | 0.012875 | 0.015599 | 3 |
| 3 | nino12 | 0.011668 | 0.015768 | 2 |
| 4 | nino4 | 0.010553 | 0.011068 | 7 |
| 5 | NPMM | 0.010524 | 0.011671 | 6 |
| 6 | SPMM | 0.009881 | 0.012534 | 5 |
| 7 | IOD | 0.009574 | 0.013361 | 4 |
| 8 | TNA | 0.006360 | 0.005806 | 9 |

### IOD

| SHAP rank | Mode | mean \|SHAP\| | PEID source EI | PEID rank |
|---:|---|---:|---:|---:|
| 1 | IOD | 0.038729 | 0.320329 | 1 |
| 2 | nino4 | 0.010866 | 0.019538 | 4 |
| 3 | nino | 0.010834 | 0.020106 | 3 |
| 4 | nino3 | 0.010210 | 0.020596 | 2 |
| 5 | SIOD | 0.008231 | 0.017881 | 5 |
| 6 | nino12 | 0.006568 | 0.010729 | 7 |
| 7 | NPMM | 0.006486 | 0.011246 | 6 |
| 8 | SPMM | 0.005341 | 0.009445 | 8 |

## Second-order SHAP interaction vs PEID Syn

### ENSO

| SHAP int. rank | Pair | mean \|interaction\| | PEID Syn | PEID rank |
|---:|---|---:|---:|---:|
| 1 | nino + nino3 | 0.109765 | 0.005216 | 1 |
| 2 | nino + nino4 | 0.101482 | 0.005194 | 2 |
| 3 | nino + NPMM | 0.091425 | 0.002686 | 5 |
| 4 | nino + nino12 | 0.090774 | 0.002589 | 6 |
| 5 | nino + SPMM | 0.089758 | 0.004559 | 3 |
| 6 | nino + IOD | 0.087262 | 0.004278 | 4 |
| 7 | nino + TNA | 0.072390 | 0.001499 | 8 |
| 8 | nino + IOB | 0.068928 | 0.001179 | 10 |
| 9 | nino + WWV | 0.065640 | 0.001728 | 7 |
| 10 | nino + SIOD | 0.062909 | 0.001091 | 11 |

### IOD

| SHAP int. rank | Pair | mean \|interaction\| | PEID Syn | PEID rank |
|---:|---|---:|---:|---:|
| 1 | IOD + nino4 | 0.070597 | 0.005648 | 3 |
| 2 | nino + IOD | 0.064600 | 0.007147 | 2 |
| 3 | IOD + SIOD | 0.057813 | 0.012107 | 1 |
| 4 | IOD + nino3 | 0.052491 | 0.002660 | 9 |
| 5 | NPMM + IOD | 0.046635 | 0.005263 | 4 |
| 6 | SPMM + IOD | 0.045937 | 0.004950 | 5 |
| 7 | nino + nino4 | 0.045703 | 0.000819 | 13 |
| 8 | nino3 + nino4 | 0.044251 | 0.001389 | 11 |
| 9 | IOB + IOD | 0.043320 | 0.004779 | 6 |
| 10 | IOD + nino12 | 0.042688 | 0.002530 | 10 |

## Outputs

- output directory: `results/unicm_shap_mode_attribution`
- figures: `fig/unicm_enso_iod_shap_mode_ranking.*`, `fig/unicm_enso_iod_shap_pair_interactions.*`
