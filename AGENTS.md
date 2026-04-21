# AGENTS.md

## Plotting rules for experiment figures

- When generating experiment plots, never place the legend on top of lines, markers, bars, scatter points, or shaded confidence regions unless explicitly requested.
- Prefer putting legends outside the axes, usually on the right:
  - `ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)`
- After adding an outside legend, use layout adjustments so it is not clipped:
  - prefer `constrained_layout=True`, or
  - `fig.tight_layout()`, and when saving use `bbox_inches="tight"`.
- For multi-line or dense plots, first try outside-right legend; second choice is above the plot in multiple columns.
- If the legend still overlaps or makes the figure too narrow, enlarge the figure width before moving the legend back inside.
- Before finishing, visually check that no legend overlaps any plotted data.

## Figure formats for research documents

- When exporting experiment figures that will be cited from Markdown research documents under `doc/`, prefer `png` or `pdf` as the final inserted asset format.
- Do not use `svg` as the final cited figure format in `doc/研究框架.md`-style documents unless the user explicitly asks for it or the export toolchain is known to support it end-to-end.
- If a notebook or plotting pipeline produces `svg` by default, also export a companion `png` or `pdf`, and reference that compatible asset from the Markdown document.

## Long-running experiments

- When running a long or expensive experiment, always persist the computed results to disk in a reusable machine-readable form before finishing.
- Save enough information so later visualization, summary, or document updates can reuse prior results without recomputing the full experiment.
- Prefer storing both per-run detailed outputs and a compact summary file; examples include `json`, `csv`, `npz`, or notebook-adjacent cache files under `exp/cache/`, `results/`, or another task-appropriate experiment directory.
- If an experiment is executed from a notebook, make sure the notebook either writes these cache artifacts itself or clearly reuses an existing cache on subsequent runs.
