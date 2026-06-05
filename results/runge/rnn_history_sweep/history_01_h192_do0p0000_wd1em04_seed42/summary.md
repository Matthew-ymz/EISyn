# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 1
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 100.0
- RNN residual scale: 1.0

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.717034 | 0.571811 | 0.452136 |
| RNN | 1 | 0.717076 | 0.571869 | 0.452279 |
| TunedRidge | 1 | 0.717076 | 0.571869 | 0.452279 |
| MLP | 2 | 0.779988 | 0.622802 | 0.242683 |
| RNN | 2 | 0.777766 | 0.621355 | 0.257016 |
| TunedRidge | 2 | 0.780164 | 0.62303 | 0.242582 |
| MLP | 4 | 0.796185 | 0.63507 | 0.13886 |
| RNN | 4 | 0.789539 | 0.630183 | 0.186619 |
| TunedRidge | 4 | 0.795841 | 0.63489 | 0.142058 |
| MLP | 8 | 0.803443 | 0.641178 | 0.0297697 |
| RNN | 8 | 0.792632 | 0.633202 | 0.165787 |
| TunedRidge | 8 | 0.802614 | 0.64054 | 0.0535717 |
