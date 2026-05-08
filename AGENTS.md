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

## PEID theory literature

- For tasks involving PEID-related theory, first use the Zotero plugin to search the local Zotero library for PEID papers and read the relevant literature before doing derivations, research-framework writing, method design, or implementation decisions.
- If Zotero is unavailable or no relevant PEID paper can be found, state that blocker clearly and continue only with the best available repository context.

## AI Research Skills on-demand installation

- For requests in this repository that may benefit from research or AI workflow skills, first check whether a matching skill exists in `https://github.com/Orchestra-Research/AI-Research-SKILLs`.
- This trigger applies to research, machine learning, paper writing, evaluation, training, agents, RAG, mechanistic interpretability, infrastructure, optimization, multimodal work, prompt engineering, and closely related tasks.
- Prefer installing the most specific matching skill directory, not an entire category or the full repository. For example, use `20-ml-paper-writing/ml-paper-writing` for ML paper writing and `20-ml-paper-writing/academic-plotting` for academic plotting.
- Install the selected skill with:
  - `python3 /Users/yangmingzhe/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py --repo Orchestra-Research/AI-Research-SKILLs --path <category>/<skill-name>`
- If multiple skills plausibly match, inspect the remote repository listing and install only the best match for the current request.
- After installation, if the skill is not active until Codex restarts, immediately read `$CODEX_HOME/skills/<skill-name>/SKILL.md` and follow it manually for the current task.
- Tell the user that restarting Codex will make newly installed skills appear in the normal skill registry.
- If installation fails because of network or authentication issues, report the failure clearly and continue best-effort without silently skipping the trigger.
- Example mappings:
  - `帮我写 ML paper method section` -> `20-ml-paper-writing/ml-paper-writing`
  - `做学术绘图` -> `20-ml-paper-writing/academic-plotting`
  - `做 RAG 实验设计` -> inspect `15-rag` and install the most specific matching skill under that category.
  - Normal repository maintenance requests do not trigger AI Research Skills installation.
