# Runge 2015 causal gateways and mediators reproduction

This report reproduces the core workflow from `Identifying causal gateways and mediators in complex spatio-temporal systems` on the local NCEP/NCAR sea-level-pressure data.

## Method

- Daily SLP fields are restricted to the configured year range, transformed to standardized daily anomalies, linearly detrended, and latitude-area weighted.
- Varimax-rotated PCA components are fitted on monthly fields when enough monthly samples are available, then projected back to daily fields.
- Component scores are aggregated to weekly resolution.
- Tigramite/ParCorr selects candidate parents; sparse standardized OLS estimates the lagged causal regression coefficients.
- Lagged links are thresholded by the configured aggregated cross-link density.
- Causal effects use the lag-resolved Runge recursion, and gateways/mediators are ranked by ACE, ACS, and AMCE.

## Run

- Years: 1948-2011
- Components: 60
- Weekly lag maximum: 4
- Link density target: 0.2
- Backend: tigramite
- Daily samples: 23376
- Weekly samples: 3339
- Causal links: 837

## Top causal gateways

| paper_component | component | ace | acs | direct_out_strength | direct_in_strength |
| --- | --- | --- | --- | --- | --- |
| No.2 | No.2 | 0.0722743 | 0.0454446 | 0.0594318 | 0.0354186 |
| No.1 | No.1 | 0.0574933 | 0.0452482 | 0.041264 | 0.0368521 |
| No.0 | No.0 | 0.0534309 | 0.0459951 | 0.0397709 | 0.0345034 |
| No.3 | No.3 | 0.0527876 | 0.0409589 | 0.0395466 | 0.0308168 |
| No.6 | No.6 | 0.0512324 | 0.0289007 | 0.0337728 | 0.0216362 |
| No.4 | No.4 | 0.0457974 | 0.0294359 | 0.0366042 | 0.0224485 |
| No.48 | No.21 | 0.0456893 | 0.0419797 | 0.0329904 | 0.0321501 |
| No.14 | No.14 | 0.0428531 | 0.0366292 | 0.0298344 | 0.0268801 |
| No.18 | No.7 | 0.0410139 | 0.0361837 | 0.0296192 | 0.0275652 |
| No.11 | No.11 | 0.0403158 | 0.0377532 | 0.0304315 | 0.0293815 |

## Top causal mediators

| paper_component | component | amce | mediated_fraction |
| --- | --- | --- | --- |
| No.2 | No.2 | 0.00287861 | 0.968732 |
| No.1 | No.1 | 0.00185883 | 0.813559 |
| No.0 | No.0 | 0.00173793 | 0.859147 |
| No.48 | No.21 | 0.00156057 | 0.930742 |
| No.26 | No.8 | 0.00148323 | 0.806546 |
| No.3 | No.3 | 0.00140223 | 0.927236 |
| No.11 | No.11 | 0.00119148 | 0.917884 |
| No.14 | No.14 | 0.00106367 | 0.936002 |
| No.18 | No.7 | 0.00103019 | 0.842198 |
| No.6 | No.6 | 0.000983046 | 0.817943 |

## Artifacts

- `causal_edges.csv`: lagged directed links.
- `gateway_scores.csv`: component ACE/ACS rankings.
- `mediator_scores.csv`: component AMCE rankings.
- `mediated_path_effects.csv`: source-mediator-target path effects.
- `component_weekly_scores.csv`: weekly rotated component scores.
- `fig/runge2015_gateways/*.png`: component maps, network, gateway ranking, and mediator ranking.
