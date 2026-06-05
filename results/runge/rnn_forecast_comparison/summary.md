# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 4
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 1000.0
- RNN residual scale: 1.0

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.71525 | 0.570838 | 0.456082 |
| RNN | 1 | 0.715441 | 0.570881 | 0.455635 |
| TunedRidge | 1 | 0.715798 | 0.571185 | 0.45545 |
| MLP | 2 | 0.775089 | 0.618565 | 0.270895 |
| RNN | 2 | 0.77392 | 0.617649 | 0.273318 |
| TunedRidge | 2 | 0.774004 | 0.617722 | 0.272498 |
| MLP | 4 | 0.788698 | 0.629704 | 0.193171 |
| RNN | 4 | 0.787609 | 0.628887 | 0.198739 |
| TunedRidge | 4 | 0.788317 | 0.629409 | 0.193439 |
| MLP | 8 | 0.790339 | 0.630992 | 0.190553 |
| RNN | 8 | 0.788341 | 0.629554 | 0.196747 |
| TunedRidge | 8 | 0.792561 | 0.632601 | 0.184912 |
