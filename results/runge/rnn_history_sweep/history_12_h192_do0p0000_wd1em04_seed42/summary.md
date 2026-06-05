# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 12
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 3000.0
- RNN residual scale: 1.0

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.731324 | 0.58478 | 0.415441 |
| RNN | 1 | 0.730619 | 0.584158 | 0.417577 |
| TunedRidge | 1 | 0.732249 | 0.585478 | 0.414315 |
| MLP | 2 | 0.77642 | 0.620575 | 0.267463 |
| RNN | 2 | 0.774104 | 0.618712 | 0.275365 |
| TunedRidge | 2 | 0.774609 | 0.61908 | 0.270829 |
| MLP | 4 | 0.788897 | 0.630389 | 0.207844 |
| RNN | 4 | 0.786704 | 0.62861 | 0.212576 |
| TunedRidge | 4 | 0.786704 | 0.62861 | 0.212576 |
| MLP | 8 | 0.790743 | 0.631441 | 0.189427 |
| RNN | 8 | 0.788864 | 0.629988 | 0.197315 |
| TunedRidge | 8 | 0.789168 | 0.630165 | 0.194323 |
