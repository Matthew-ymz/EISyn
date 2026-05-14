# Common LaTeX / Beamerposter Compilation Fixes

## Package & Preamble Issues

### beamerposter not found
```
! LaTeX Error: File 'beamerposter.sty' not found.
```
**Fix**: Install via TeX Live: `tlmgr install beamerposter`. On Overleaf this is pre-installed. Alternatively, download `beamerposter.sty` from CTAN and place it in the same directory as `poster.tex`.

### Font encoding warning with special characters
Add `\usepackage[T1]{fontenc}` and `\usepackage[utf8]{inputenc}` to the preamble.

### `\bfseries` in math mode
Replace `\bfseries` with `\boldsymbol{}` or `\mathbf{}` inside equations.

---

## Figure Issues

### Figure file not found
```
! LaTeX Error: File 'figures/foo.pdf' not found.
```
**Fix options**:
1. Add `\graphicspath{{figures/}{../figures/}{./}}` to preamble to search multiple directories.
2. Use the absolute path (for local compilation).
3. Copy figures into the same directory as `poster.tex`.

### Figure too large / overflows column
```latex
% Instead of width=\linewidth, use:
\includegraphics[width=0.9\linewidth]{figure}
% Or clip:
\includegraphics[width=\linewidth, trim=0 0 0 0, clip]{figure}
```

### PDF figure bbox warning
Add `\usepackage{grffile}` or switch from `.pdf` to `.png` for the figure.

---

## Layout Issues

### Overfull \hbox (text overflows column)
Common causes and fixes:
- Long URLs: wrap in `\url{}` or use `\usepackage{url}` + `\sloppy`
- Long equation: use `\begin{multline}` or `\begin{split}` inside `align`
- Long word: add `\-` discretionary hyphen or use `\allowbreak`
- Tight column: slightly reduce font with `{\small ...}` around the block

### Column overflows vertically (content too tall)
Options in order of preference:
1. Move one block to the adjacent column
2. Reduce body font: add `\small` or `\footnotesize` inside the overfull block
3. Compress a bullet list: reduce `\itemsep` with `\setlength{\itemsep}{2pt}`
4. Increase poster height: change `height=...cm` in beamerposter options
5. Shrink a figure: reduce `width=0.95\linewidth` to `0.80\linewidth`

### Blocks not aligning at top across columns
Make sure all `\begin{column}` use the `[t]` option:
```latex
\begin{columns}[t]
  \begin{column}[t]{0.32\linewidth}
```

### Headline / footer not full width
Ensure `wd=\paperwidth` in the `beamercolorbox`:
```latex
\begin{beamercolorbox}[wd=\paperwidth, center]{headline}
```

---

## Bibliography Issues

### `\cite{}` shows `[?]`
The `.bib` file wasn't found or bibtex wasn't run.
```bash
# Run in order:
pdflatex poster.tex
bibtex poster
pdflatex poster.tex
pdflatex poster.tex
# OR with latexmk (recommended):
latexmk -pdf poster.tex
```

### biblatex vs. natbib conflict
Choose one bibliography system. `biblatex` is recommended for new posters.
Remove any `\usepackage{natbib}` if using `biblatex`, and vice versa.

### Too many references overflow the block
Use abbreviated style: `\usepackage[style=numeric-comp]{biblatex}` which compresses [1,2,3] into [1-3].
Alternatively, manually list only 3–5 key references using `\bibitem` directly.

---

## Beamer-specific Issues

### `\pause` or overlays cause blank pages
Remove all `\pause`, `\only<>{}`, `\uncover<>{}` commands — they don't make sense on a static poster.

### itemize bullets missing or wrong style
```latex
\setbeamertemplate{itemize item}{\textbullet}
\setbeamertemplate{itemize subitem}{--}
```

### Block title color not applying
Make sure you set colors **before** `\begin{document}`:
```latex
\setbeamercolor{block title}{fg=white, bg=navyblue}
\setbeamercolor{block body}{fg=black, bg=lightgray}
```

---

## Compilation Commands Reference

```bash
# Standard latexmk (recommended)
latexmk -pdf -interaction=nonstopmode poster.tex

# With bibtex
latexmk -pdf -bibtex poster.tex

# Force recompile everything
latexmk -pdf -gg poster.tex

# Clean auxiliary files
latexmk -c

# View log for errors
grep -A3 "^!" poster.log
```

---

## Overleaf-specific Notes

- Overleaf uses pdflatex by default; switch to XeLaTeX or LuaLaTeX in Settings if you need special fonts.
- All figures must be uploaded to the Overleaf project. Zip the figure directory and upload as a project zip.
- `.bib` file must be uploaded too.
- Compilation timeout: if the poster is large and compilation times out, compress figures (use `.jpg` instead of `.pdf` for photos).
