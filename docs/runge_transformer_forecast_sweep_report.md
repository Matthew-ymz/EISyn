# Runge N=60 Transformer 预测调参报告

## 结论

本轮在真实 Runge weekly 60 维 component score 上完成了 Transformer 预测管线扩展、分阶段调参、final seed rerun 和 GRU reference 复算。当前完成 `338` 个 Transformer 候选，无失败候选。

最终系统为 **TransformerHorizonSelector**：每个 horizon 只按 validation RMSE 选择一个 Transformer 候选，不用 test split 做选择。

最终 average RMSE：

| system | validation avg RMSE | test avg RMSE |
| --- | ---: | ---: |
| TransformerHorizonSelector | 0.756903 | 0.764835 |
| BestBaseline | - | 0.765801 |
| GRU reference | - | 0.765320 |

平均 RMSE 的 paired circular block bootstrap：

| comparison | RMSE improvement | 95% CI | p(improvement <= 0) |
| --- | ---: | --- | ---: |
| Transformer vs BestBaseline | 0.000966 | [0.000501, 0.001448] | 0.000 |
| Transformer vs GRU reference | 0.000485 | [-0.000072, 0.001016] | 0.042 |

结论应精确表述为：当前 validation-selected Transformer 系统在 average RMSE 上优于 BestBaseline，且 bootstrap 证据较强；相对 GRU reference 的 average RMSE 也更低，one-sided bootstrap p 值为 0.042，但 two-sided 95% CI 仍略跨 0，因此这是正向但边界较窄的提升证据。逐 horizon 不应一概声称显著提升，因为 `h=4` 当前劣于 baseline 和 GRU。

## Horizon-wise 选择

| horizon | selected history | stage | validation RMSE | test RMSE |
| ---: | ---: | --- | ---: | ---: |
| 1 | 2 | D_lr | 0.699570 | 0.708132 |
| 2 | 2 | D_lr | 0.766453 | 0.772234 |
| 4 | 2 | C_pool_pos | 0.780382 | 0.789916 |
| 8 | 4 | B_capacity | 0.781209 | 0.789059 |

## Final Test Metrics

| model | horizon | RMSE | MAE | corr |
| --- | ---: | ---: | ---: | ---: |
| TransformerHorizonSelector | 1 | 0.708132 | 0.564669 | 0.472449 |
| TransformerHorizonSelector | 2 | 0.772234 | 0.616215 | 0.276928 |
| TransformerHorizonSelector | 4 | 0.789916 | 0.630457 | 0.184216 |
| TransformerHorizonSelector | 8 | 0.789059 | 0.629817 | 0.200723 |
| BestBaseline | 1 | 0.709344 | 0.565392 | 0.472241 |
| BestBaseline | 2 | 0.774436 | 0.617950 | 0.275234 |
| BestBaseline | 4 | 0.788826 | 0.629504 | 0.191196 |
| BestBaseline | 8 | 0.790600 | 0.631214 | 0.187715 |

## Bootstrap vs BestBaseline

`block_size=26`, `reps=5000` for the average comparison artifact. Per-horizon values below use the aligned horizon-specific final forecasts:

| horizon | best baseline | RMSE improvement | 95% CI | p(improvement <= 0) |
| ---: | --- | ---: | --- | ---: |
| 1 | MLP | 0.001212 | [0.000682, 0.001732] | 0.000 |
| 2 | MLP | 0.002203 | [0.001319, 0.003159] | 0.000 |
| 4 | TunedRidge | -0.001091 | [-0.002209, 0.000098] | 0.967 |
| 8 | MLP | 0.001541 | [0.000398, 0.002649] | 0.003 |
| average | BestBaseline | 0.000966 | [0.000501, 0.001448] | 0.000 |

## Bootstrap vs GRU Reference

GRU reference 使用 prior report 的 selected config 复算：

- `history=2`
- `hidden_dim=128`
- `dropout=0.0`
- `weight_decay=1e-5`
- `seed=42`
- `rnn_objective=rollout_multistep`
- `rnn_linear_blend_grid_steps=101`

不同 history 会导致 test target index 略有不同，因此 Transformer vs GRU 使用 target index 对齐后比较。

| horizon | aligned rows | Transformer RMSE | GRU RMSE | improvement | 95% CI | p(improvement <= 0) |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 500 | 0.708132 | 0.709416 | 0.001284 | [0.000746, 0.001808] | 0.000 |
| 2 | 500 | 0.772234 | 0.774509 | 0.002276 | [0.001378, 0.003211] | 0.000 |
| 4 | 500 | 0.789916 | 0.787236 | -0.002680 | [-0.004018, -0.001398] | 1.000 |
| 8 | 499 | 0.789059 | 0.790118 | 0.001059 | [-0.000348, 0.002493] | 0.073 |
| average | - | 0.764835 | 0.765320 | 0.000485 | [-0.000072, 0.001016] | 0.042 |

## Final Seed Reruns

Stage F 已对 validation top hyperparameter spec 执行 seeds `[42,43,44,45,46]`。由于 leaderboard top 3 在归一化后对应同一个 hyperparameter spec，最终新增 `F_seed=5` 个候选。

## 产物

- Candidate leaderboard: `results/runge_transformer_forecast_sweep/leaderboard.csv`
- Ensemble leaderboard: `results/runge_transformer_forecast_sweep/ensemble_leaderboard.csv`
- Horizon selector: `results/runge_transformer_forecast_sweep/horizon_selector_selection.csv`
- Final test metrics: `results/runge_transformer_forecast_sweep/final_test_metrics.csv`
- Bootstrap vs baseline: `results/runge_transformer_forecast_sweep/final_prediction_significance.json`
- Average bootstrap vs baseline: `results/runge_transformer_forecast_sweep/transformer_vs_bestbaseline_average_significance.json`
- Bootstrap vs GRU: `results/runge_transformer_forecast_sweep/transformer_vs_gru_significance.json`
- GRU reference arrays: `results/runge_transformer_forecast_sweep/gru_reference/history_02_h128_do0_wd1em05_seed42/forecast_arrays.npz`
- Figures: `fig/runge/transformer_forecast_sweep/leaderboard_val_rmse.png`, `fig/runge/transformer_forecast_sweep/final_multistep_rmse.png`
