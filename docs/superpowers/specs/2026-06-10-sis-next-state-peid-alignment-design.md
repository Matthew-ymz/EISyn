# SIS Next-State PEID Alignment Design

## Goal

Demonstrate on at least one classical dynamical system that PEID computed from a learned MLP transition model is close to PEID computed from the known dynamics.

The selected system is the three-node SIS motif in `scripts/classic_network_dynamics_benchmark.py`. Both channels use the same finite-time state transition

$$
\mathbf{x}(t) \longmapsto \mathbf{x}(t+\tau),
$$

rather than the vector field. This matches the future-state target used by the PEID definition.

## Fixed Protocol

- System: SIS.
- Integrator: the existing RK4 implementation.
- Integration horizon: $\tau=1.0$, equal to 50 SIS integration steps at $\Delta t=0.02$.
- Target names: `w_tau`, `x_tau`, and `y_tau`.
- Dynamics: stochastic SIS with additive process noise,

$$
d\mathbf{x}=\mathbf{f}(\mathbf{x})\,dt+0.05\,d\mathbf{W},
$$

implemented by adding Gaussian increments with standard deviation $0.05\sqrt{\Delta t}$ during every numerical integration step and clipping states to $[0,1]$.
- MLP architecture: a probabilistic two-hidden-layer SiLU network with separate conditional-mean and diagonal conditional-log-standard-deviation outputs.
- Training distribution: equal numbers of natural-trajectory states and independently uniform states from the SIS intervention domain $[0.02,0.98]^3$.
- Training targets: the known SIS dynamics integrated for exactly $\tau$ from every input state.
- PEID intervention distribution: independently uniform over the same domain.
- Transition uncertainty: the Oracle samples the stochastic differential equation over the full integration interval. The probabilistic MLP samples its learned conditional transition distribution. No noise is added by the PEID estimator or as output post-processing.
- PEID estimator: the existing transport-map estimator for full runs and the histogram estimator for smoke tests.

The stochastic channel avoids the singular continuous mutual information of a deterministic transition and represents uncertainty in the original dynamics. A deterministic MLP is insufficient because it learns only the conditional mean and therefore removes the transition uncertainty, causing substantial PEID overestimation.

## Comparison

The primary relations are the SIS state-dependent mechanisms

$$
\{w_t,x_t\}\to x_{t+\tau},
\qquad
\{w_t,y_t\}\to y_{t+\tau}.
$$

For each relation, compute

$$
\epsilon_{\mathrm{rel}}
=
\frac{\left|\operatorname{Syn}_{\mathrm{MLP}}-\operatorname{Syn}_{\mathrm{Oracle}}\right|}
{\max\left(\left|\operatorname{Syn}_{\mathrm{Oracle}}\right|,10^{-12}\right)}.
$$

The experiment passes when both relative errors are at most 20%. The report must display the Oracle value, MLP value, and relative error for both relations. Model selection must use transition prediction error only; PEID agreement cannot be used to choose epochs, $\tau$, or the training seed.

## Code Changes

Keep the existing four-system vector-field benchmark intact. Add a focused SIS next-state alignment experiment to the same module or a tightly scoped helper module, with:

1. a reusable finite-horizon transition method;
2. deterministic construction of natural, intervention-domain, and mixed training pairs;
3. stochastic SIS integration and a probabilistic MLP transition model;
4. an alignment summary containing protocol metadata and relation-level errors;
5. a focused comparison figure and Markdown report section.

Do not claim that the result generalizes to all classical systems. The supported conclusion is that MLP+PEID can reproduce known-dynamics PEID when the learned model is trained over the intervention domain and both are evaluated as the same finite-time stochastic transition channel.

## Verification

- Unit tests verify that finite-horizon targets use 50 repeated stochastic integration steps and reproduce exactly for a fixed random seed.
- Unit tests verify the mixed training sample composition and target names.
- A deterministic smoke test verifies output metadata and artifact creation.
- A full transport-map run verifies both SIS relative errors are at most 20%.
- Existing benchmark tests remain green.
- The generated figure is visually checked for clipping and overlapping legends.
