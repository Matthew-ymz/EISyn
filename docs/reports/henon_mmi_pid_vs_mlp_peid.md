# Hénon MMI-PID vs MLP+PEID

This report uses the classic Hénon one-step equation

$$x_{t+1}=1-1.4x_t^2+y_t,$$

with sources `x_t` and `y_t` and target `x_{t+1}`. The example is deliberately chosen because both sources carry strong single-source information about the target. Under MMI-PID, the synergy term is

$$S_{MMI}=I(x,y;x_{t+1})-\max\{I(x;x_{t+1}), I(y;x_{t+1})\},$$

so it exceeds the PEID residual by the weaker single-source information term.

![Hénon comparison](../../fig/henon_mmi_pid_vs_mlp_peid/henon_mmi_pid_vs_mlp_peid.png)

## Summary

| readout | mean ± std |
|---|---:|
| Observed MMI-PID synergy | 0.940794 ± 0.003904 |
| True-map PEID residual | 0.654859 ± 0.004847 |
| MLP+PEID residual | 0.656329 ± 0.003563 |
| Weaker single-source MI | 0.285935 ± 0.002534 |
| MMI minus true-map PEID | 0.285935 ± 0.002534 |

## Interpretation

The observed MMI-PID synergy is not isolating a learned mechanism. It is high because the weaker source has its own information about the target, and MMI redundancy uses the minimum single-source information. The MLP+PEID residual remains tied to the fitted mechanism's joint surplus after subtracting both single-source terms, so the two quantities separate on this classic additive Hénon readout.
