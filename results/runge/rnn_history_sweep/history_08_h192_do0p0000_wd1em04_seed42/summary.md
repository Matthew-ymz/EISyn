# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 8
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 1000.0
- RNN residual scale: -0.08000000000000002

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.724352 | 0.57903 | 0.433651 |
| RNN | 1 | 0.724298 | 0.579104 | 0.434509 |
| TunedRidge | 1 | 0.724291 | 0.579098 | 0.434531 |
| MLP | 2 | 0.779261 | 0.623028 | 0.267065 |
| RNN | 2 | 0.780764 | 0.624378 | 0.264612 |
| TunedRidge | 2 | 0.78064 | 0.624232 | 0.265981 |
| MLP | 4 | 0.791531 | 0.632522 | 0.202202 |
| RNN | 4 | 0.792942 | 0.633762 | 0.199354 |
| TunedRidge | 4 | 0.792932 | 0.633712 | 0.200425 |
| MLP | 8 | 0.787988 | 0.629245 | 0.198409 |
| RNN | 8 | 0.788727 | 0.629925 | 0.195001 |
| TunedRidge | 8 | 0.788477 | 0.629749 | 0.196788 |
