# Hénon Unique-Information Sweep: MMI-PID vs MLP+PEID

This controlled Hénon-style readout increases a separate single-source observation channel while decreasing the explicit interaction coefficient:

$$\mathbf{z}_{t+1}=\left[1-1.4x_t^2+\kappa(\lambda)x_ty_t,\;\gamma(\lambda)y_t+\epsilon_t\right],\quad \sigma_\epsilon=0.500.$$

`lambda` maps linearly to an increasing `gamma(lambda)` and a decreasing `kappa(lambda)`. Here `gamma` runs from 0.300 to 2.000 and `kappa` runs from 0.500 to 0.100. Thus the single-source channel strengthens while the explicit interaction term weakens. The same noise draw is reused across the sweep for each seed.

![Hénon unique sweep](../../fig/henon_unique_sweep_mmi_vs_mlp_peid/henon_unique_sweep_mmi_vs_mlp_peid.png)

## Dynamic Range

| quantity | max-min |
|---|---:|
| Observed MMI-PID synergy | 0.273610 |
| MLP+PEID residual | 0.115894 |
| True-map PEID residual | 0.265914 |
| Weaker source MI | 0.539523 |

## Parameter Sweep

| lambda | gamma | kappa | MMI-PID synergy | MLP+PEID residual | true-map PEID residual | weaker source MI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.300 | 0.500 | 0.588135 ± 0.003720 | 0.134321 ± 0.042493 | 0.431683 ± 0.004195 | 0.156452 ± 0.005875 |
| 0.167 | 0.583 | 0.433 | 0.593128 ± 0.011182 | 0.068778 ± 0.024455 | 0.383747 ± 0.006965 | 0.209381 ± 0.004397 |
| 0.333 | 0.867 | 0.367 | 0.617937 ± 0.008522 | 0.044805 ± 0.015672 | 0.334839 ± 0.005306 | 0.283098 ± 0.005613 |
| 0.500 | 1.150 | 0.300 | 0.665855 ± 0.006167 | 0.034460 ± 0.013577 | 0.286034 ± 0.005783 | 0.379820 ± 0.008803 |
| 0.667 | 1.433 | 0.233 | 0.722878 ± 0.006105 | 0.028026 ± 0.011105 | 0.237890 ± 0.005258 | 0.484988 ± 0.005175 |
| 0.833 | 1.717 | 0.167 | 0.791135 ± 0.012984 | 0.023509 ± 0.010841 | 0.201959 ± 0.003675 | 0.589176 ± 0.009945 |
| 1.000 | 2.000 | 0.100 | 0.861744 ± 0.016463 | 0.018427 ± 0.008710 | 0.165769 ± 0.005305 | 0.695975 ± 0.011177 |

## Interpretation

For every row, `MMI-PID synergy - true-map PEID residual` equals the weaker single-source MI by construction of two-source MMI-PID. The curve therefore exposes the qualitative mismatch: PEID can fall as the explicit interaction weakens, while MMI-PID can still rise when the weaker single-source information grows faster.
