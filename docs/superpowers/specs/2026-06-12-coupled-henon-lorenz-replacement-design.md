# Coupled Henon Lorenz-Replacement Experiment Design

## Goal

Test whether a discrete-time chaotic coupled Henon map provides a clearer and
more stable Part1 four-method synergy panel than the current Lorenz-3D rho
sweep.

## Map And Readout

Use the four-dimensional map

$$
x_{t+1}=(1-\kappa)(1-a x_t^2+y_t)+\kappa x_t z_t,\qquad
y_{t+1}=b x_t,
$$

$$
z_{t+1}=(1-\kappa)(1-a z_t^2+w_t)+\kappa z_t x_t,\qquad
w_{t+1}=b z_t,
$$

with `a=1.4`, `b=0.3`, and coupling scan
`kappa = [0, 0.02, 0.04, 0.05, 0.06, 0.08]`. The convex gate keeps the map
bounded while increasing the relative contribution of the explicit product
mechanism. Pre-readout pilots verified positive largest-Lyapunov estimates
throughout this range.

The registered relation is `x+z->x_tau`. At `kappa=0`, `z` is absent from the
first subsystem's next-state equation, so the registered interaction is a
structural zero. Positive `kappa` introduces the explicit product term
`kappa*x*z`.

## Protocol

- Train one MLP per `kappa` and seed using multiple short natural trajectories.
- Use separate natural trajectories for WMS, SURD, and SHAP.
- Use independent uniform intervention states for MLP+PEID and Oracle+PEID.
- Use four seeds: `0,1,2,3`.
- Preserve the existing full-mode transport-map estimator and plotting style.
- Save raw learned readouts at `kappa=0`, but report exact structural zero in
  the main curve.

## Additional Diagnostics

For each `kappa`, save:

- Oracle PEID synergy for `x+z->x_tau`;
- MLP prediction MSE relative to the mean-target baseline;
- finite-time largest Lyapunov estimate from paired nearby trajectories;
- bounded-trajectory fraction;
- train, natural-readout, and intervention-state digests.

The parameter range is valid only while trajectories remain bounded and the
Lyapunov estimate indicates chaotic sensitivity for most positive-coupling
points.

## Replacement Criteria

Coupled Henon may replace Lorenz only if:

1. positive-coupling MLP+PEID has a clear increasing trend;
2. MLP+PEID follows Oracle+PEID qualitatively;
3. the zero-control displayed value is exactly zero and raw zero residual is
   reported separately;
4. cross-seed PEID variability is lower than the Lorenz panel on a relative
   range-normalized basis;
5. trajectories remain bounded and predominantly chaotic;
6. the resulting panel remains readable without legend or uncertainty-band
   overlap.

## Outputs

- `results/discrete_iteration_dynamics_benchmark/coupled_henon_synergy_sweep.json`
- `fig/discrete_iteration_dynamics_benchmark/coupled_henon_synergy_sweep.png`
- `fig/part1_synergy_comparison/lorenz_vs_coupled_henon.png`
- `docs/reports/coupled_henon_lorenz_replacement.md`
