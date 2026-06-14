# ODE Synergy Candidate Screening Design

## Goal

Screen four continuous-time differential-equation mechanisms for the same
interpretable pattern shown by the Wilson-Cowan refractory experiment:

1. the scanned parameter directly opens or strengthens the registered
   two-source mechanism;
2. an MLP accurately learns the broad-domain one-step transition;
3. MLP+PEID follows Oracle PEID under matched interventions;
4. competing readouts expose a scientifically interpretable contrast.

The output is a standalone report. This experiment does not replace or edit the
current Part1 six-panel comparison.

## Candidate Systems

All systems are converted into deterministic one-step transition maps using one
explicit Euler step. The readout is the next state, not the instantaneous
derivative.

### SIS Infection Gate

$$
\dot w=-0.8w+w(1-w),\qquad
\dot x=-x+\beta w(1-x).
$$

Registered relation: `w+x->x_tau`.

At $\beta=0$, $w$ is absent from the target equation. Positive $\beta$ opens
the state-dependent infection gate.

### Lorenz Product Gate

$$
\dot x=\sigma(y-x),\qquad
\dot y=x(\rho-z)-y,\qquad
\dot z=\gamma xy-\frac{8}{3}z.
$$

Registered relation: `x+y->z_tau`.

At $\gamma=0$, neither $x$ nor $y$ affects the Euler readout of $z$. Positive
$\gamma$ directly opens the Lorenz product mechanism.

### Rossler Product Gate

$$
\dot x=-y-z,\qquad
\dot y=x+ay,\qquad
\dot z=c+z(\gamma x-b).
$$

Registered relation: `x+z->z_tau`.

At $\gamma=0$, $x$ is absent from the target equation. Positive $\gamma$
directly opens the internal Rossler product mechanism.

### Kuramoto Phase Gate

$$
\dot x=\omega_x+\kappa\sin(w-x),\qquad
\dot w=\omega_w.
$$

Registered relation: `w+x->x_tau`.

At $\kappa=0$, $w$ is absent from the target equation. Positive $\kappa$ opens
a joint phase-difference response. This is retained as a periodic-variable
boundary case because density estimation may be less stable near the phase-box
boundary.

## Matched Broad One-Step Protocol

For every candidate, parameter, and seed:

- sample one broad training pool from the registered intervention box;
- generate targets with the true Euler transition;
- train one MLP on that pool;
- sample one independent broad held-out readout pool;
- use that same held-out pool for WMS, SURD, SHAP, MLP+PEID, and Oracle PEID;
- use the same fitted MLP for SHAP and MLP+PEID;
- use the transport-map estimator with the repository-standard degree and
  configuration for all information readouts;
- preserve estimated zero-point residuals instead of replacing them by exact
  zeros;
- save state/model digests and run the existing Part1 fairness audit.

The parameter value changes only the true transition targets. Within a seed,
the broad train and readout input pools remain identical across the parameter
scan.

## Outputs

- `scripts/ode_synergy_candidate_benchmark.py`
- `tests/test_ode_synergy_candidate_benchmark.py`
- `results/ode_synergy_candidate_benchmark/*.json`
- `fig/ode_synergy_candidate_benchmark/*.png`
- `docs/reports/ode_synergy_candidate_screening.md`

## Figure Contract

Core conclusion: directly gating a known two-source ODE term produces
interpretable Oracle and MLP+PEID curves, while the other readouts expose
different observational or response-amplitude semantics.

Figure archetype: quantitative grid.

Backend: Python.

Panel map:

- a: SIS infection gate, primary biological mechanism example;
- b: Lorenz product gate, primary chaotic-system mechanism example;
- c: Rossler internal product gate, second product validation;
- d: Kuramoto phase gate, periodic-variable boundary case.

All method colors and markers are shared. One legend is placed outside the
axes. Curves are seed means and bands are population standard deviations.

## Screening Criteria

A candidate is recommended for later Part1 inclusion only when:

1. the fairness audit passes;
2. MLP prediction NRMSE is low over the broad held-out pool;
3. MLP+PEID follows the Oracle PEID trend and has acceptable absolute error;
4. the zero-point residual is small relative to active points;
5. the mechanism interpretation is clearer than the competing readouts;
6. the final curve is visually stable across seeds.

Kuramoto may remain a useful boundary case even if it fails the recommendation
criteria because periodic support exposes a known estimator challenge.
