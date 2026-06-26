# Runge 2015 causal gateways and mediators reproduction

This report reproduces the core workflow from `Identifying causal gateways and mediators in complex spatio-temporal systems` on the local NCEP/NCAR sea-level-pressure data.

## Method

- Daily SLP fields are restricted to the configured year range; Feb 29 is removed; each gridpoint is transformed to standardized 365-day calendar-day anomalies; the anomalies are linearly detrended and latitude-area weighted.
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
- Daily samples: 23360
- Removed leap days: 16
- Weekly samples: 3337
- Causal links: 848

## Top causal gateways

| paper_component | component | ace | acs | direct_out_strength | direct_in_strength |
| --- | --- | --- | --- | --- | --- |
| No.2 | No.2 | 0.0649404 | 0.0485633 | 0.050401 | 0.0385012 |
| No.1 | No.1 | 0.0605437 | 0.0451949 | 0.0448756 | 0.0355241 |
| No.0 | No.0 | 0.0568999 | 0.0418528 | 0.0417109 | 0.0306984 |
| No.6 | No.6 | 0.053712 | 0.0307023 | 0.0382532 | 0.0229468 |
| No.3 | No.3 | 0.0480879 | 0.0452808 | 0.0339083 | 0.0344794 |
| No.4 | No.4 | 0.0455079 | 0.0274293 | 0.0352526 | 0.0213952 |
| No.57 | No.57 | 0.0438301 | 0.0235142 | 0.0326186 | 0.0174417 |
| No.13 | No.13 | 0.0419835 | 0.0314331 | 0.030719 | 0.0210608 |
| No.18 | No.7 | 0.041187 | 0.036904 | 0.0308992 | 0.0290576 |
| No.22 | No.22 | 0.0393424 | 0.0333702 | 0.0283588 | 0.0228914 |

## Top causal mediators

| paper_component | component | amce | mediated_fraction |
| --- | --- | --- | --- |
| No.2 | No.2 | 0.00251885 | 0.955289 |
| No.1 | No.1 | 0.0018437 | 0.886616 |
| No.0 | No.0 | 0.00167321 | 0.8045 |
| No.3 | No.3 | 0.00151637 | 0.855348 |
| No.6 | No.6 | 0.00115184 | 0.917884 |
| No.9 | No.9 | 0.00114262 | 0.606663 |
| No.48 | No.21 | 0.00113259 | 0.901227 |
| No.13 | No.13 | 0.00101472 | 0.791935 |
| No.18 | No.7 | 0.000984569 | 0.860023 |
| No.4 | No.4 | 0.000935099 | 0.582116 |

## Artifacts

- `causal_edges.csv`: lagged directed links.
- `gateway_scores.csv`: component ACE/ACS rankings.
- `mediator_scores.csv`: component AMCE rankings.
- `mediated_path_effects.csv`: source-mediator-target path effects.
- `component_weekly_scores.csv`: weekly rotated component scores.
- `fig/runge/2015_gateways/*.png`: component maps, network, gateway ranking, and mediator ranking.
