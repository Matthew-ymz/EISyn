# Kuramoto Short-Horizon Phi Design

## Question

Does the N=64 whole-state Oracle \(\Phi^{EID}\) peak remain near the Kuramoto
transition when the finite prediction horizon is too short for the high-coupling
conditions to collapse into a strongly synchronized target distribution?

## Decision

Use one fixed global horizon per curve; never choose a horizon separately for
each coupling. The primary short-horizon curve uses `tau=0.5`; a paired
multi-horizon robustness sweep uses `tau=(0.5, 0.75, 1.0, 1.5, 2.0, 4.0)`.

For each seed, the frequency vector and uniform intervention/readout states
are generated once and reused for every coupling and every horizon. This makes
the horizon the sole treatment factor when curves are compared.

## Synchronization guard

The primary curve is acceptable only if the 99th percentile of the raw global
order parameter is below `0.8` for every coupling. The result records the
mean, 95th percentile, 99th percentile, and maximum raw order parameter, plus
the fraction of readout targets with raw order at least `0.8`.

## Readout and estimator

The source remains one two-dimensional phase block per oscillator,
`(cos(theta_i(t)), sin(theta_i(t)))`; the target remains the concatenated
whole-system phase features at `t + tau`. The Oracle degree-1 transport-map
estimator, target-shuffle null procedure, source count, sample count, and
frequency distribution are unchanged from the existing whole-state sweep.

## Interpretation rule

Peak amplitude is allowed to vary with horizon. A short-horizon result supports
the original interpretation only if its peak lies in the transition band and
not at the high-coupling endpoint; a displaced, absent, or monotone short-
horizon curve is evidence that the original peak is horizon-specific.
