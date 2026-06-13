# Six-System Natural-Trajectory Synergy Comparison Design

## Goal

Extend the existing four-method synergy comparison with Coupled Rössler and
Wilson–Cowan, then present Standard Map, SIS, Lorenz, Kuramoto, Wilson–Cowan,
and Coupled Rössler in one consistent six-panel figure.

The new experiments first test whether an MLP trained only on natural
trajectories preserves the mechanisms needed by the four readouts.

## Scientific Claim

Across six classical dynamical systems, the four synergy readouts respond
differently to parameter changes and natural-trajectory coverage. Coupled
Rössler provides a nonlinear coupled-chaos case, while Wilson–Cowan provides a
neural-dynamics additive-structure control whose PEID readout need not be zero.

## New Model Sweeps

### Coupled Rössler coupling sweep

Use two bidirectionally coupled Rössler oscillators:

```text
dx_i = -y_i - z_i + kappa * sin(x_j - x_i)
dy_i = x_i + 0.165 * y_i
dz_i = 2 + z_i * (x_i - 5.5)
```

Scan `kappa = [0, 0.1, 0.25, 0.5, 0.75]`.

The panel averages the two coupling relations
`{x0,x1}->dx0` and `{x0,x1}->dx1`. The internal Rössler product terms remain
present at every coupling value but are not the target of this sweep.

### Wilson–Cowan gain sweep

Use the existing three-node fork with additive decay and sigmoid drive:

```text
dw = -w + sigmoid_g(w)
dx = -x + sigmoid_g(w)
dy = -y + sigmoid_g(w)
sigmoid_g(u) = 1 / (1 + exp(-g * (u - 1)))
```

Scan `g = [1, 2, 3.5, 5.1, 7.5]`.

The panel averages `{w,x}->dx` and `{w,y}->dy`. These relations are additive
in the vector field and therefore serve as a structural-interaction control,
not as a PEID numerical-zero control.

## Natural-Trajectory Data Protocol

For each parameter value and seed:

1. Generate a training pool by concatenating multiple short, transient-inclusive
   trajectories from independent initial states. Use a short per-trajectory
   burn-in and the benchmark's small process/observation noise rather than the
   original long single-trajectory warm-up.
2. Generate a separate held-out readout pool from different initial states
   using the same natural-dynamics protocol.
3. Train the MLP only on the natural-trajectory training pool.
4. Evaluate WMS, SURD, and SHAP interaction on the same held-out
   natural-trajectory states and targets.
5. Evaluate MLP+PEID on independent uniform intervention states using the same
   MLP trained only on natural trajectories.
6. Save training, natural-readout, and PEID-readout state digests, sample counts, and trajectory
   counts in the result JSON.

Multiple short trajectories are required for Wilson–Cowan because one long
trajectory can collapse near a stable fixed point. The same pool protocol is
used for Rössler to keep the two new panels comparable.

The supervised target remains the instantaneous vector field
`state -> derivative`, matching the existing classic-network benchmark.
WMS and SURD use the held-out natural states and noisy derivative observations.
SHAP interaction uses predictions from the fitted MLP on those held-out natural
states. MLP+PEID uses independent uniform intervention states, preserving its
maximum-entropy intervention interpretation while still testing whether a
natural-trajectory-trained MLP extrapolates the mechanism.

## Four Readouts

Each new panel reports the native values of:

- observational WMS;
- observational SURD synergy;
- MLP+SHAP interaction;
- MLP+PEID synergy.

Each curve is the mean across seeds; the shaded region is population standard
deviation across seeds. Native scales are retained, so comparisons focus on
trend, residual, and stability rather than absolute cross-method magnitude.

## Figure Design

The final figure is a `2 x 3` quantitative grid:

```text
a Standard Map | b SIS          | c Lorenz
d Kuramoto     | e Wilson–Cowan | f Coupled Rössler
```

A single shared legend is placed outside the axes on the right. Method colors
and markers remain identical across all panels. No legend may overlap plotted
data or uncertainty regions. The figure is saved as a high-resolution PNG
with tight bounding-box handling.

## Code Changes

Extend `scripts/classic_network_dynamics_benchmark.py` with:

- parameterized Rössler and Wilson–Cowan spec builders;
- a reusable multi-initial-condition natural-trajectory pool generator;
- two four-method sweep runners and summary aggregators;
- two single-model plotting entry points;
- an updated combined-figure function accepting six result files;
- CLI flags for both sweeps and the six-panel combined figure.

Extend `tests/test_classic_network_dynamics_benchmark.py` with:

- parameterized-vector-field tests;
- natural-pool independence and shape tests;
- smoke tests for both new sweeps;
- a six-input combined-figure test.

Update `docs/reports/Part1.md` with the new experiment protocol, figures,
results, and interpretation.

## Verification

1. Run focused unit and smoke tests for the two builders, pool generator,
   sweep runners, and combined figure.
2. Run the complete classic-network benchmark test module.
3. Run full sweeps using four seeds when runtime permits; persist JSON results
   because recomputation is costly.
4. Generate the six-panel figure.
5. Visually inspect the final PNG for clipping, unreadable labels, and legend
   overlap.
