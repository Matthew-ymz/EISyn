# Coupled Henon 与 Lorenz-3D 替代性比较

![comparison](../../fig/part1_synergy_comparison/lorenz_vs_coupled_henon.png)

## 实验

耦合 Hénon 映射为

$$
x_{t+1}=(1-\kappa)(1-1.4x_t^2+y_t)+\kappa x_tz_t,\qquad y_{t+1}=0.3x_t,
$$
$$
z_{t+1}=(1-\kappa)(1-1.4z_t^2+w_t)+\kappa z_tx_t,\qquad w_{t+1}=0.3z_t.
$$

主读出固定为 `x+z->x_tau`。MLP 使用覆盖完整注册状态盒的宽初值一步真实映射样本，并采用独立 train/validation/test 池；网络配置只按验证集预测 NRMSE 选择，不读取 Oracle 或 PEID。WMS、SURD 和 SHAP 使用独立 held-out 自然轨迹；冻结模型后，MLP+PEID 与 Oracle+PEID 在全部参数点复用同一批独立干预状态。

## 判定指标

- Henon 正耦合 PEID 趋势 Spearman: `1.0000`
- Henon Oracle PEID 参数趋势 Spearman: `1.0000`
- Henon MLP--Oracle PEID Spearman: `1.0000`
- Henon 正耦合混沌占比: `1.0000`
- Henon 平均有界轨迹占比: `1.0000`
- Henon 相对跨 seed 波动: `0.0073`
- Lorenz 相对跨 seed 波动: `0.3807`
- Henon MLP MSE / mean-target baseline MSE: `0.000011`
- Henon 宽域测试加权预测 NRMSE: `0.003914`

## 逐点结果

| kappa | MLP+PEID mean | MLP+PEID std | Oracle+PEID | Lyapunov | bounded |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.000 | 0.000000 | 0.000000 | 0.000000 | 0.425114 | 1.0000 |
| 0.020 | 0.004009 | 0.000157 | 0.002523 | 0.375207 | 1.0000 |
| 0.040 | 0.011921 | 0.000294 | 0.008327 | 0.124730 | 1.0000 |
| 0.050 | 0.017606 | 0.000354 | 0.012919 | 0.303747 | 1.0000 |
| 0.060 | 0.024412 | 0.000447 | 0.018672 | 0.323698 | 1.0000 |
| 0.080 | 0.041491 | 0.000561 | 0.033682 | 0.272910 | 1.0000 |

## 结论

耦合 Henon 满足预注册替代标准，建议在 Part1 中替换 Lorenz-3D。
