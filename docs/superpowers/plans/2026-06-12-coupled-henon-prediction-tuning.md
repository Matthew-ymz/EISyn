# Coupled Henon Prediction-Tuning Implementation Plan

**Goal:** Improve intervention-domain one-step prediction without selecting on
Oracle or PEID values, then rerun the 12-seed replacement experiment.

## Task 1: Prediction-domain data and metrics

- [x] Add tests for broad-initial-state trajectory generation and independent
  train/validation/test pools.
- [x] Add per-target NRMSE and the registered weighted validation objective.
- [x] Verify deterministic generation and finite trajectories.

## Task 2: Configurable early-stopped MLP

- [x] Add tests for explicit validation data, checkpoint restoration, and
  prediction metadata.
- [x] Implement hidden-width, learning-rate, weight-decay, epoch, and patience
  configuration on the coupled Henon path only.
- [x] Verify that the restored checkpoint beats the mean-target baseline.

## Task 3: Prediction-only search

- [x] Add a smoke test proving search output is prediction-only and does not
  invoke PEID.
- [x] Search the registered configuration grid on fixed calibration kappas and
  seeds.
- [x] Persist ranking, selected configuration, and validation/test metrics.

## Task 4: Frozen 12-seed sweep

- [x] Integrate the selected configuration and broad trajectory pools into the
  focused coupled Henon sweep.
- [x] Run the full 12-seed experiment and only then compute Oracle+PEID.
- [x] Regenerate the sweep figure, Lorenz comparison figure, and report.

## Task 5: Verification

- [x] Run focused pytest and Python compilation.
- [x] Visually inspect both figures for clipping and legend overlap.
- [x] Confirm Part1 changes only if all replacement criteria pass.
- [x] Run `git diff --check`.
