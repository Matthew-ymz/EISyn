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

- For experiment results, default to producing the primary visualization figure first, preferably a `png` that can be viewed directly in Markdown, notebooks, and chat summaries.
- Do not export companion `pdf`, `svg`, or `tiff` files unless the user explicitly asks for publication/vector assets or the current document build requires that format.
- When updating Markdown research documents under `doc/`, reference the directly viewable visualization asset first. Use `pdf` only for final paper export or a proven LaTeX/Pandoc path.

## Long-running experiments

- For long or expensive experiments, persist reusable computed results only when recomputation would be costly or the user asks for reusable data.
- Prefer lightweight cache formats that match the code path, such as `json`, `jsonl`, `npz`, or notebook-adjacent cache files. Do not create `csv` summaries by default.
- Treat machine-readable caches as internal support artifacts; the user-facing deliverable should prioritize the visualization figure and concise interpretation.
- If an experiment is executed from a notebook, make sure it can reuse existing cache artifacts on subsequent runs when such artifacts are needed.

## PEID theory literature

- For tasks involving PEID-related theory, first use the Zotero plugin to search the local Zotero library for PEID papers and read the relevant literature before doing derivations, research-framework writing, method design, or implementation decisions.
- If Zotero is unavailable or no relevant PEID paper can be found, state that blocker clearly and continue only with the best available repository context.
