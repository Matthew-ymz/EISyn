# TM L sensitivity notes

- `tm_l_sweep_runs.csv` stores every repeat for each `(L, alpha)` setting.
- `tm_l_sweep_summary.csv` stores mean and standard deviation summaries used by the figures.
- The heatmap compares `Syn / EI` and absolute `Syn` over the `(alpha, L)` grid.
- `Syn / EI` entries are marked `NA` when mean `EI` is below 0.05 nats because the ratio is numerically unstable.
- The line figure shows how `Syn / EI` changes with `L`, and decomposes the pure nonlinear case `alpha = 1`.