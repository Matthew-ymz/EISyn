# Runge RNN forecast comparison

This run compares recursive multi-step forecasts from an RNN-class transition model, the existing MLP-style transition model, and the validation-selected Ridge baseline.

## Run

- Lag: 16
- Horizons: [1, 2, 4, 8]
- Components: 60
- Best linear alpha: 3000.0
- RNN residual scale: 1.0

## Test metrics

| model | horizon | rmse | mae | corr |
| --- | ---: | ---: | ---: | ---: |
| MLP | 1 | 0.736933 | 0.589125 | 0.399204 |
| RNN | 1 | 0.735265 | 0.587791 | 0.403837 |
| TunedRidge | 1 | 0.73694 | 0.589136 | 0.399211 |
| MLP | 2 | 0.778416 | 0.62196 | 0.260988 |
| RNN | 2 | 0.777887 | 0.621617 | 0.266524 |
| TunedRidge | 2 | 0.778268 | 0.621839 | 0.261269 |
| MLP | 4 | 0.790803 | 0.631499 | 0.20481 |
| RNN | 4 | 0.79063 | 0.631362 | 0.205161 |
| TunedRidge | 4 | 0.79063 | 0.631362 | 0.205161 |
| MLP | 8 | 0.792706 | 0.632867 | 0.187059 |
| RNN | 8 | 0.792564 | 0.632758 | 0.1874 |
| TunedRidge | 8 | 0.792564 | 0.632758 | 0.1874 |
