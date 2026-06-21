# ENSO Source EI Lead Curves

## Goal

Replace the ENSO single-source EI ranking bar chart in `Part2.md` with lead-dependent curves that expose when each source contributes information.

## Figure Contract

- Conclusion: ENSO self-history dominates at short leads and decays rapidly, while smaller non-self contributions follow distinct lead-dependent trajectories with wider checkpoint uncertainty at long leads.
- Role: metric comparison over forecast lead.
- Backend: Python with the repository's existing matplotlib plotting script.
- Outputs: overwrite the existing PNG and editable SVG assets used by `Part2.md`.
- Review risks: self EI is much larger than non-self EI; dense curves and uncertainty bands must remain legible; the legend must not overlap plotted data.

## Data And Selection

Recover single-source EI by source, checkpoint seed, and lead from the existing pair-level JSONL rows. Validate that repeated copies of a source EI agree across source pairs before aggregation.

Select:

- ENSO self source;
- the five non-self sources with the highest EI averaged over seeds and leads 1 through 24;
- NPMM and TNA, added when not already in the top five.

Do not duplicate sources that satisfy more than one selection rule.

## Layout

Use two horizontal panels with a shared Lead axis:

- left: ENSO self EI with its own y-axis scale;
- right: selected non-self source EI curves with their own y-axis scale.

Each curve shows the checkpoint-seed mean. A light band shows plus or minus one checkpoint-seed standard deviation. Put the non-self legend outside the right edge and save with constrained layout and a tight bounding box.

## Report Changes

Keep the existing asset basename `unicm_enso_source_ei_rankings` to avoid breaking references. Update the Figure 3 caption and nearby prose so they describe lead curves rather than a ranking bar chart. Preserve the summary table, which still reports lead-averaged EI.

## Verification

- Run focused data-shape and source-selection checks.
- Regenerate PNG and SVG through the existing script.
- Visually inspect the PNG for curve visibility, uncertainty-band readability, clipped labels, and legend overlap.
- Confirm `Part2.md` references the regenerated asset and no stale bar-chart wording remains near Figure 3.
