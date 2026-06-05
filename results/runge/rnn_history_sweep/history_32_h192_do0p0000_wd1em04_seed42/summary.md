# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 32
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 3000.0
- RNN residual scale: -0.04999999999999999

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.754561 | 0.602603 | 0.348177 |
| RNN | 1 | 0.756801 | 0.604461 | 0.347745 |
| TunedRidge | 1 | 0.756801 | 0.604461 | 0.347745 |
| MLP | 2 | 0.789808 | 0.630163 | 0.234774 |
| RNN | 2 | 0.79651 | 0.635575 | 0.22659 |
| TunedRidge | 2 | 0.796577 | 0.635614 | 0.2279 |
| MLP | 4 | 0.8004 | 0.638143 | 0.191048 |
| RNN | 4 | 0.807794 | 0.644122 | 0.181627 |
| TunedRidge | 4 | 0.808166 | 0.6444 | 0.182092 |
| MLP | 8 | 0.80266 | 0.639684 | 0.174626 |
| RNN | 8 | 0.809819 | 0.645288 | 0.164937 |
| TunedRidge | 8 | 0.810166 | 0.645557 | 0.165416 |
