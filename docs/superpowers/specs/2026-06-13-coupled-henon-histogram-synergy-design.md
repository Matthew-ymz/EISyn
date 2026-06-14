# Coupled Henon Histogram Synergy Design

## Goal

Replace all information-theoretic readouts in the Coupled Henon Part1 panel
with a shared discrete histogram estimator because the current transport-map
SURD specific-MI estimate is numerically unreliable for the multimodal inverse
mapping of the Henon target.

## Estimation Protocol

- Keep the existing broad one-step train/readout state protocol and fitted MLP.
- Use six uniform-width bins per variable for the main Hénon panel.
- Scan `kappa = 0, 0.04, 0.08, 0.12, 0.16, 0.20` so the PEID increase is
  visible while retaining a bounded coupled Hénon map.
- Estimate WMS, discrete SURD, MLP+PEID, and Oracle PEID with the same six-bin
  discretization.
- Keep MLP+SHAP interaction on its native continuous response scale.
- Record four-, six-, and eight-bin sensitivity results using the same
  held-out states and targets.

## SURD Fidelity

The original SURD paper defines causality on a finite phase-space partition,
computes nonnegative specific mutual information for every target state, sorts
the source-subset specific information values, and averages the resulting
redundant, unique, and synergistic increments over the target-state
distribution. The repository's discrete two-source SURD implementation follows
this construction. Unlike the transport-map approximation, it does not clip
estimated specific MI or force joint specific MI above single-source estimates.

## Outputs

- Persist the main estimator and bin count in the Hénon JSON.
- Persist four-, six-, and eight-bin sensitivity summaries.
- Regenerate the Hénon figure and six-system Part1 figure.
- Keep the panel title concise; document the histogram estimator in the report
  and result metadata rather than in the plot title.
- Update `docs/reports/Part1.md` with the histogram protocol, current values,
  sensitivity interpretation, and SURD-method fidelity statement.
