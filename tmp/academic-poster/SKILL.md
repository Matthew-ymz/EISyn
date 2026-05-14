---
name: academic-poster
description: "Use this skill any time the user wants to create a conference poster or academic poster from a LaTeX paper, PDF, or written research content. Trigger whenever the user mentions 'poster', 'conference poster', 'beamerposter', 'academic poster', 'poster session', or asks to turn a paper into a visual summary for a conference. Also trigger when the user has a main.tex, paper.tex, or any .tex file and wants a printable or presentable visual output. This skill handles the full workflow: reading the paper, identifying core contributions, generating a structured beamerposter LaTeX file, compiling it, and fixing layout issues automatically."
---

# Academic Poster Skill

You are an academic poster generation assistant. Your job is to produce a polished, conference-ready poster in LaTeX using the `beamerposter` package, starting from a LaTeX paper (or any structured research document).

---

## Guiding Philosophy

A great academic poster is **not** a compressed paper. It is a visual argument. Someone walking past in 10 seconds should grasp the core claim; someone stopping for 2 minutes should understand the method and results. Everything that doesn't serve one of those two audiences should be cut.

Keep this in mind throughout: prefer visuals over prose, prefer one bold claim over three hedged ones, prefer a clean diagram over a wall of equations.

---

## Workflow

Work through these stages in order. Do not skip stages.

### 1. Read and understand the source paper

Read `main.tex` (or whichever source file is provided). Also read any included files (`\input{...}`, `\include{...}`). Build a mental model of:

- The **core problem** being solved (one sentence)
- The **key contributions** (usually 2–4 bullets from the abstract/intro)
- The **method** (what is novel about how they do it)
- The **main results** (one or two headline numbers or figures)
- The **conclusion / takeaway**

List the figure files referenced (`\includegraphics`) — you will reuse them.

### 2. Identify what to include on the poster

A poster has limited space. Apply aggressive selection:

- **Always include**: problem motivation, key contributions, one method diagram, one results figure or table, conclusion + future work, references (abbreviated)
- **Include if space allows**: one equation that is central to the method (displayed, not inline), one ablation or comparison table
- **Cut**: related work section (reduce to 2–3 inline citations), proofs, derivations, implementation details, acknowledgements beyond one line

If there are existing figures in the paper, **use them directly** — do not recreate or paraphrase them. List the figure paths at the top of your working notes.

### 3. Generate a poster outline

Before writing any LaTeX, write a brief outline in plain text:

```
Poster outline:
- Header: title, authors, affiliations, logo path (if any)
- Column 1 (left): Motivation | Problem statement | Key contributions
- Column 2 (middle): Method overview | Key equation | Method diagram (fig path)
- Column 3 (right): Results figure | Results table | Conclusion | References
```

Adjust column assignments based on content weight. The method column is usually the widest. Show this outline to the user before proceeding if the task is interactive; otherwise proceed directly.

### 4. Write `poster.tex`

Use `beamerposter` with a three-column layout (or two-column if content is sparse). Follow the template in `references/beamerposter_template.md`.

Critical requirements:
- **All mathematical notation must be preserved exactly** as it appears in the source paper — copy-paste LaTeX math rather than rewriting it
- **All citations must be preserved** — use `\cite{}` as in the original; include a `\bibliography{}` call at the end
- **Use existing figure files** — reference the same paths as in `main.tex` (adjust relative paths if needed)
- **No text-heavy blocks** — any paragraph longer than 4 lines should be replaced with a bullet list or diagram
- **Visual balance** — each column should have roughly equal visual density; redistribute content if one column looks cramped
- **Nature-style writing** — short declarative sentences, active voice, no hedging, no jargon that isn't defined on the poster itself

### 5. Compile and fix

Run:
```bash
latexmk -pdf -interaction=nonstopmode poster.tex
```

Read the log. Common issues and fixes:

| Error | Fix |
|-------|-----|
| `Undefined control sequence` for beamerposter | Add `\usepackage[...]{beamerposter}` to preamble |
| Figure file not found | Adjust `\graphicspath` or fix relative path |
| Overfull `\hbox` | Shorten text or add `\small` / `\footnotesize` to that block |
| Column overflow | Move a block to an adjacent column or increase poster height |
| Bibliography not found | Ensure `.bib` file path is correct; run `bibtex poster` then `latexmk` again |

Fix errors and recompile until the PDF generates cleanly. If compilation is not available in the current environment, produce the `.tex` file with a note explaining how to compile.

### 6. Output

Deliver:
- `poster.tex` — the complete compilable LaTeX file
- A brief note on: paper size used, any figures that couldn't be located (with suggested replacements), any content that was cut and why

---

## Style Guidelines

**Fonts & sizes**: Use `\usefonttheme{serif}` for math-heavy posters. Title ~88pt, section headers ~36pt, body text ~24pt, captions ~20pt (adjust based on poster size).

**Colors**: Pick 2–3 colors. One dominant background/header color, one accent for section headers, white/light for body. Avoid red+green (colorblind). A clean dark-blue + white + gold palette works well for science.

**Equations**: Display only the 1–2 most important equations. Label them. Surround with one sentence of context above and one of interpretation below.

**Figures**: Caption should be ≤2 lines. The figure should be self-explanatory from the caption alone.

**Columns**: Use `\begin{columns}` inside `beamercolorbox` blocks. Standard widths: `0.32\linewidth` for 3 equal columns; `0.38 / 0.27 / 0.27` if method column needs more space.

---

## Reference files

- `references/beamerposter_template.md` — full minimal working beamerposter template to start from
- `references/common_fixes.md` — extended list of LaTeX compilation fixes and layout tricks
