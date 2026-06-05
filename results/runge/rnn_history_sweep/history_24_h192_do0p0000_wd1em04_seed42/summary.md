# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 24
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 3000.0
- RNN residual scale: -0.04999999999999999

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.746589 | 0.596274 | 0.370599 |
| RNN | 1 | 0.746932 | 0.596611 | 0.371465 |
| TunedRidge | 1 | 0.746932 | 0.596611 | 0.371465 |
| MLP | 2 | 0.784305 | 0.626192 | 0.246489 |
| RNN | 2 | 0.787597 | 0.628828 | 0.24145 |
| TunedRidge | 2 | 0.787495 | 0.628784 | 0.242962 |
| MLP | 4 | 0.795716 | 0.63525 | 0.196697 |
| RNN | 4 | 0.799299 | 0.63805 | 0.191188 |
| TunedRidge | 4 | 0.799462 | 0.638188 | 0.192046 |
| MLP | 8 | 0.798073 | 0.636671 | 0.179199 |
| RNN | 8 | 0.80157 | 0.639428 | 0.172772 |
| TunedRidge | 8 | 0.801536 | 0.639426 | 0.174572 |
