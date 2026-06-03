# Runge 2015 Reproduction Comparison

Source paper: Runge et al., *Identifying causal gateways and mediators in complex spatio-temporal systems*, Nature Communications 6:8502 (2015).

## Paper Targets

- Climate experiment uses 60 Varimax-rotated SLP components from weekly global pressure data.
- Reported sample size is 3,339 weekly observations.
- Main climate-network settings include `tau_max = 4` weeks and PC significance level `alpha = 0.001`.
- The main climate network is discussed at about 20% link density.
- The paper highlights components 0, 1, 2, and 18 as major causal gateways and dominant mediators; it also notes components 26 and 48 among important mediators.

## This Run

- Daily SLP files: 64 files, 1948-2011.
- Components: 60.
- Weekly samples: 3339.
- `tau_max`: 4.
- `alpha`: 0.001.
- Tigramite backend: tigramite.
- Lagged edges: 838.
- Directed source-target pair density, including self pairs: 0.207.
- Directed source-target pair density, excluding self pairs: 0.194.

## Ranking Overlap

Paper-highlighted components: 0, 1, 2, 18.

Gateway ranking overlap:

| Component | Reproduction gateway rank |
|---:|---:|
| 0 | 7 |
| 1 | 5 |
| 2 | 1 |
| 18 | 29 |

Mediator ranking overlap:

| Component | Reproduction mediator rank |
|---:|---:|
| 0 | 4 |
| 1 | 2 |
| 2 | 3 |
| 18 | 8 |

Top gateway components in this run: 2, 57, 9, 4, 1, 58, 0, 16, 30, 52, 25, 3, 5, 21, 7.

Top mediator components in this run: 57, 1, 2, 0, 9, 4, 7, 18, 49, 11, 58, 38, 56, 30, 54.

## Assessment

The reproduction is aligned with the paper on the main structural checks: 60 components, 3,339 weekly samples, `tau_max=4`, `alpha=0.001`, and an approximately 20% directed-pair link density. The ranking result is partially consistent rather than numerically identical: components 0, 1, and 2 are again high-ranked gateways, and all four paper-highlighted components 0, 1, 2, and 18 appear in the top mediator table. Component 18 is recovered strongly as a mediator but not as a top gateway in this run.

The remaining mismatch is expected for a main-figure-level independent reproduction because exact EOF/Varimax conventions, component ordering/sign choices, preprocessing details, and the published authors' graph-thresholding/readout implementation are not fully specified in executable form.
