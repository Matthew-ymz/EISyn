# AGENTS.md

## Git workflow

- Unless the user explicitly asks for a separate branch, work directly on the current main branch and update it in place.
- Do not create a feature branch or worktree by default.
- After completing work on the main branch, do not ask whether to merge, open a pull request, keep the branch, or discard the work.
- Only use a separate branch, worktree, pull request, or branch-completion workflow when the user explicitly requests it.

## Delegation and heavy workflows

- Before invoking subagents or starting a materially complex workflow, first explain in writing why its scope is necessary, what lighter alternative exists, and its expected cost.
- Obtain the user's explicit approval before proceeding. If the user considers the additional rigor unnecessary, use the lighter approach and do not expand the work.

## Long-running experiments

- For long or expensive experiments, persist reusable computed results only when recomputation would be costly or the user asks for reusable data.
- Prefer lightweight cache formats that match the code path, such as `json`, `jsonl`, `npz`, or notebook-adjacent cache files. Do not create `csv` summaries by default.
- Treat machine-readable caches as internal support artifacts; the user-facing deliverable should prioritize the visualization figure and concise interpretation.
- If an experiment is executed from a notebook, make sure it can reuse existing cache artifacts on subsequent runs when such artifacts are needed.

## EI estimation

- When computing EI over continuous variable spaces, prefer TM-based estimation first.
- Use an alternative EI estimator only when TM is inapplicable, computationally prohibitive, or explicitly requested.
- If using a non-TM estimator, state the reason and document the tradeoff.

## PEID theory literature

- For tasks involving PEID-related theory, first use the Zotero plugin to search the local Zotero library for PEID papers and read the relevant literature before doing derivations, research-framework writing, method design, or implementation decisions.
- If Zotero is unavailable or no relevant PEID paper can be found, state that blocker clearly and continue only with the best available repository context.
