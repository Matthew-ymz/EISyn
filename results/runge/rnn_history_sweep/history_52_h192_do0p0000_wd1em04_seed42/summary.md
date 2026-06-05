# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 52
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 3000.0
- RNN residual scale: -0.10999999999999999

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.769407 | 0.614412 | 0.30787 |
| RNN | 1 | 0.777119 | 0.62053 | 0.304525 |
| TunedRidge | 1 | 0.777119 | 0.62053 | 0.304525 |
| MLP | 2 | 0.800263 | 0.638587 | 0.211748 |
| RNN | 2 | 0.813241 | 0.649102 | 0.202247 |
| TunedRidge | 2 | 0.814152 | 0.649764 | 0.202682 |
| MLP | 4 | 0.810379 | 0.646559 | 0.173783 |
| RNN | 4 | 0.824138 | 0.657398 | 0.165321 |
| TunedRidge | 4 | 0.825593 | 0.658578 | 0.164614 |
| MLP | 8 | 0.81298 | 0.648676 | 0.159338 |
| RNN | 8 | 0.826736 | 0.659823 | 0.150943 |
| TunedRidge | 8 | 0.828268 | 0.661135 | 0.150318 |
