# Kuramoto WMS Nonnegativity Comparison Design

## Objective

Use a physically meaningful classical dynamical system to demonstrate that WMS is a signed
net quantity rather than a nonnegative synergy atom. In the synchronized regime of a
two-active-rotator Kuramoto phase model, observational source redundancy can drive WMS below
zero, while PEID evaluated under independent maximum-entropy phase interventions remains
nonnegative and recovers the joint phase mechanism.

## Model

Use the driver-response active-rotator equations

$$
\dot{\theta}_1=\omega_1+A\sin\theta_1+K\sin(\theta_2-\theta_1),\qquad
\dot{\theta}_2=\omega_2+A\sin\theta_2,
$$

with $\omega_1=1.0$, $\omega_2=0.9$, $A=0.2$, and coupling scan
$K\in\{0,0.05,0.10,0.15,0.20,0.30,0.50\}$. The active-rotator term is the
standard periodic phase potential used for excitable and driven phase oscillators. The
frequency detuning $|\Delta\omega|=|\omega_1-\omega_2|=0.1$ is plotted as a reference scale,
not asserted to be the exact locking threshold once $A\ne0$.

The registered source-target relation is identical for every method:

$$
\{\theta_{1,t},\theta_{2,t}\}\rightarrow \dot{\theta}_{1,t}.
$$

No method may average this relation with the symmetric
$\{\theta_{1,t},\theta_{2,t}\}\rightarrow \dot{\theta}_{2,t}$ relation or replace the target
with a future phase.

## Data Protocol

For each coupling and seed, generate multiple natural post-burn-in trajectories and one
independent uniform phase pool on $[-\pi,\pi)^2$.

- Train one MLP on an equal mixture of natural and uniform one-step phase states, with target
  $\dot{\theta}_1$.
- WMS and SURD use the natural trajectory pairs
  $(\theta_{1,t},\theta_{2,t},\dot{\theta}_{1,t})$.
- SHAP interaction uses the same fitted MLP and natural trajectory foreground/background.
- MLP+PEID uses the same fitted MLP on independent uniform phase interventions.
- Oracle PEID applies the same PEID estimator to the analytical Kuramoto mechanism on the
  exact same intervention states used by MLP+PEID.
- Use the same estimator family and transport-map configuration for all information
  quantities where technically applicable. Record state and model digests for auditability.

The distinct natural and intervention readout distributions are intentional. WMS measures
the observed synchronized distribution, whereas PEID measures the mechanism after an
independent maximum-entropy source intervention.

## Diagnostics

Compute the phase-locking value

$$
\operatorname{PLV}=\left|\frac{1}{N}\sum_t
e^{i(\theta_{2,t}-\theta_{1,t})}\right|
$$

from the natural trajectory. PLV supplies the physical link between phase locking, source
redundancy, and negative WMS.

For WMS, also retain its components

$$
I(\theta_1,\theta_2;\dot\theta_1),\quad
I(\theta_1;\dot\theta_1),\quad
I(\theta_2;\dot\theta_1),
$$

so the report can show that the negative value is caused by subtracting duplicated source
information rather than by a negative mutual information estimate.

## Figure

Create a two-panel Python figure.

- Panel a: PLV versus $K$, with a vertical line at the detuning scale
  $|\Delta\omega|=0.1$.
- Panel b: WMS, MLP+PEID, and Oracle PEID versus $K$, with a horizontal zero line.
- Use `Synergy / Interaction` as the panel-b y-axis label and update the existing Part1
  comparison y-axis to the same label.
- Put legends outside the axes on the right and save with a layout that prevents clipping.
- Curves show seed means and uncertainty. The report must define the interval used.

The Kuramoto comparison should replace the current panel that does not separate WMS from
MLP+PEID, rather than adding an unrelated synthetic common-driver panel.

## Success Criteria

- The source and target variables are identical across every compared method.
- At two or more strongly phase-locked points, the WMS uncertainty interval lies below zero.
- At the same points, MLP+PEID and Oracle PEID uncertainty intervals lie above zero.
- MLP+PEID follows the Oracle PEID trend within estimator uncertainty.
- PLV rises across the locking transition, supporting the redundancy interpretation.
- The MLP prediction error is reported and low enough that the PEID result is not explained
  by surrogate failure.
- Visual inspection confirms that no legend overlaps plotted data or uncertainty regions.

## Interpretation Boundary

The claim is not that WMS is incorrectly implemented. WMS equals a signed balance of
synergy and redundancy on the observational distribution, so it can legitimately be
negative. The experiment shows why WMS cannot be interpreted as a nonnegative synergy
atom. Under independent maximum-entropy interventions, PEID removes source-side redundancy
and targets the irreducible Kuramoto phase mechanism instead.
