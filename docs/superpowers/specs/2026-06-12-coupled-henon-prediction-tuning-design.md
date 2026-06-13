# Coupled Henon Prediction-Tuning Design

## Goal

Reduce the coupled Henon surrogate's one-step prediction error on the state
domain used by PEID, then evaluate whether the resulting MLP+PEID curve tracks
the frozen Oracle curve. Oracle and PEID values must not participate in model
selection.

## Diagnosis

The existing model has low held-out error on narrow natural trajectories, but
those trajectories start from `x,z in [-0.4,0.4]` and `y,w in [-0.1,0.1]`.
PEID evaluates the model over the broader intervention box
`x,z in [-1.5,1.5]`, `y,w in [-0.5,0.5]`. The dominant failure is therefore
mechanism extrapolation, not insufficient fitting of the narrow attractor.

## Data Protocol

- Generate short, true-map trajectories whose initial states cover the full
  registered intervention box. These remain natural trajectories after the
  initial state; no interventional target or Oracle value is added to training.
- Use independent trajectory seeds for train, validation, and test pools.
- Keep the final PEID intervention states separate from all three prediction
  pools.
- Reject non-finite trajectories rather than clipping the Henon dynamics.

## Model Selection

Use a configurable normalized MLP with SiLU activations, AdamW, and early
stopping. Search a small grid over hidden widths, learning rate, and weight
decay. Rank configurations only by validation prediction NRMSE:

$$
L_{\mathrm{val}}=0.7\,\mathrm{NRMSE}(x_{t+1})
+0.3\,\frac{1}{4}\sum_j\mathrm{NRMSE}(s_{j,t+1}).
$$

The emphasis on `x_tau` matches the registered readout while retaining a
whole-state accuracy constraint. Save per-target validation and test NRMSE,
best epoch, and state digests for auditability.

## Final Evaluation

- Freeze the best prediction configuration before computing PEID.
- Run 12 independent training seeds for every registered `kappa`.
- Reuse one fixed intervention-state sample across all `kappa` values.
- Compute Oracle+PEID only after all models and predictions are frozen.
- Preserve the exact displayed structural zero at `kappa=0`, while retaining
  the raw surrogate readout.

Lorenz is replaced only if the existing replacement criteria pass after this
prediction-only tuning. Otherwise Part1 remains unchanged.

