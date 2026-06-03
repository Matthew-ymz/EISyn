# Runge 2015 causal gateways and mediators reproduction

This report reproduces the core workflow from `Identifying causal gateways and mediators in complex spatio-temporal systems` on the local NCEP/NCAR sea-level-pressure data.

## Method

- Daily SLP fields are restricted to the configured year range, transformed to standardized daily anomalies, and latitude-area weighted.
- The weighted anomaly matrix is reduced to Varimax-rotated PCA components.
- Component scores are aggregated to weekly resolution.
- Lagged causal links are reconstructed with the configured backend; the full reproduction uses Tigramite PCMCI with ParCorr.
- Causal gateways are ranked by outgoing average causal effect (ACE), susceptibility by incoming total effect (ACS), and mediators by absolute mediated causal effect (AMCE).

## Run

- Years: 1948-2011
- Components: 60
- Weekly lag maximum: 4
- Backend: tigramite
- Daily samples: 23376
- Weekly samples: 3339
- Causal links: 838

## Top causal gateways

| component | ace | acs | direct_out_strength | direct_in_strength |
| --- | --- | --- | --- | --- |
| 2 | 8.53425 | 6.31283 | 2.76531 | 2.22375 |
| 57 | 8.03818 | 6.52989 | 3.05105 | 2.54638 |
| 9 | 6.74516 | 6.35118 | 2.04989 | 2.00657 |
| 4 | 6.56829 | 6.64581 | 1.99307 | 1.82634 |
| 1 | 6.48665 | 6.74269 | 2.37479 | 2.69618 |
| 58 | 5.59849 | 6.32438 | 1.35283 | 1.55467 |
| 0 | 5.40083 | 5.11668 | 2.39278 | 2.94615 |
| 16 | 5.28002 | 2.34006 | 2.19629 | 0.98973 |
| 30 | 5.11035 | 6.16786 | 1.22123 | 1.42334 |
| 52 | 4.9328 | 2.86137 | 1.38538 | 1.00474 |

## Top causal mediators

| component | amce | mediated_fraction |
| --- | --- | --- |
| 57 | 14.5382 | 0.0729084 |
| 1 | 13.7184 | 0.0687972 |
| 2 | 13.3158 | 0.0667782 |
| 0 | 13.1929 | 0.0661616 |
| 9 | 9.30457 | 0.0466619 |
| 4 | 7.36342 | 0.0369271 |
| 7 | 4.42108 | 0.0221714 |
| 18 | 4.16753 | 0.0208999 |
| 49 | 4.06364 | 0.0203789 |
| 11 | 4.04547 | 0.0202878 |

## Artifacts

- `causal_edges.csv`: lagged directed links.
- `gateway_scores.csv`: component ACE/ACS rankings.
- `mediator_scores.csv`: component AMCE rankings.
- `mediated_path_effects.csv`: source-mediator-target path effects.
- `component_weekly_scores.csv`: weekly rotated component scores.
- `fig/runge2015_gateways/*.png`: component maps, network, gateway ranking, and mediator ranking.
