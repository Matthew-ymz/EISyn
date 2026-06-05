# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 2
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 100.0
- RNN residual scale: 1.0

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.709445 | 0.565504 | 0.471943 |
| RNN | 1 | 0.709416 | 0.565438 | 0.472165 |
| TunedRidge | 1 | 0.709416 | 0.565438 | 0.472165 |
| MLP | 2 | 0.774606 | 0.618124 | 0.274544 |
| RNN | 2 | 0.774509 | 0.61802 | 0.275165 |
| TunedRidge | 2 | 0.774509 | 0.61802 | 0.275165 |
| MLP | 4 | 0.78896 | 0.629538 | 0.190292 |
| RNN | 4 | 0.787434 | 0.628605 | 0.199878 |
| TunedRidge | 4 | 0.788826 | 0.629504 | 0.191196 |
| MLP | 8 | 0.795105 | 0.634792 | 0.159435 |
| RNN | 8 | 0.790084 | 0.631242 | 0.183058 |
| TunedRidge | 8 | 0.794919 | 0.634669 | 0.162052 |
