# Minimal Beamerposter Template

Use this as the starting skeleton for `poster.tex`. Replace all `TODO` placeholders.

```latex
\documentclass[final]{beamer}

% ---- Packages ----
\usepackage[size=a0, orientation=portrait, scale=1.4]{beamerposter}
% For landscape A0: orientation=landscape
% For smaller posters: size=a1 or custom: width=84cm,height=119cm

\usepackage{graphicx}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{booktabs}        % nice tables
\usepackage{multicol}
\usepackage{lipsum}          % remove after filling in content
\usepackage[backend=bibtex, style=numeric-comp, sorting=none]{biblatex}
\addbibresource{TODO_refs.bib}  % <-- replace with actual .bib path

% ---- Theme ----
\usetheme{default}
\usefonttheme{serif}          % good for math-heavy posters

% ---- Color definitions ----
% Adjust these to match your institution / conference style
\definecolor{navyblue}{RGB}{0, 43, 91}
\definecolor{goldaccent}{RGB}{185, 151, 91}
\definecolor{lightgray}{RGB}{245, 245, 245}

\setbeamercolor{headline}{fg=white, bg=navyblue}
\setbeamercolor{footline}{fg=white, bg=navyblue}
\setbeamercolor{block title}{fg=white, bg=navyblue}
\setbeamercolor{block body}{fg=black, bg=lightgray}
\setbeamercolor{block title alerted}{fg=white, bg=goldaccent}
\setbeamercolor{block body alerted}{fg=black, bg=white}

% ---- Header / Footer templates ----
\setbeamertemplate{headline}{
  \leavevmode
  \begin{beamercolorbox}[wd=\paperwidth]{headline}
    \vskip2cm
    \centering
    {\Huge\bfseries TODO: Poster Title\par}
    \vskip1cm
    {\Large TODO: Author One$^1$, Author Two$^2$, Author Three$^1$\par}
    \vskip0.5cm
    {\large $^1$Institution One \quad $^2$Institution Two\par}
    \vskip1.5cm
  \end{beamercolorbox}
}

\setbeamertemplate{footline}{
  \leavevmode
  \begin{beamercolorbox}[wd=\paperwidth]{footline}
    \vskip0.5cm
    \centering
    {\normalsize TODO: Conference Name, Year \quad | \quad
     TODO: contact@email.com \quad | \quad
     \textbf{Code / data:} TODO URL}
    \vskip0.5cm
  \end{beamercolorbox}
}

\setbeamertemplate{navigation symbols}{}  % hide nav buttons

% ---- Document ----
\begin{document}
\begin{frame}[t]
\vskip1cm

% ========================================================
%  THREE-COLUMN LAYOUT
% ========================================================
\begin{columns}[t]

% --------------------------------------------------------
%  COLUMN 1 — Motivation & Contributions
% --------------------------------------------------------
\begin{column}{0.32\linewidth}

  \begin{block}{Motivation}
    % 3–5 sentences or bullet points. One compelling hook sentence first.
    TODO
  \end{block}

  \begin{block}{Problem Statement}
    % One short paragraph + key equation if central to framing
    TODO
    \[
      TODO\_equation
    \]
  \end{block}

  \begin{alertblock}{Key Contributions}
    \begin{itemize}
      \item TODO contribution 1
      \item TODO contribution 2
      \item TODO contribution 3
    \end{itemize}
  \end{alertblock}

  \begin{block}{Related Work}
    % 3–4 sentences only. Use \cite{} for key refs.
    TODO \cite{ref1}. TODO \cite{ref2}.
  \end{block}

\end{column}

% --------------------------------------------------------
%  COLUMN 2 — Method (usually widest)
% --------------------------------------------------------
\begin{column}{0.36\linewidth}

  \begin{block}{Method Overview}
    % One paragraph (≤4 lines) + the key figure
    TODO

    \vskip0.5cm
    \begin{figure}
      \centering
      \includegraphics[width=0.95\linewidth]{TODO_method_figure_path}
      \caption{TODO: one-sentence caption that makes the figure self-contained.}
    \end{figure}
  \end{block}

  \begin{block}{Key Equation}
    % Only the most important equation. Add one line above + one below.
    TODO context sentence:
    \[
      TODO\_main\_equation \tag{1}
    \]
    where TODO symbol explanation. This implies TODO interpretation.
  \end{block}

\end{column}

% --------------------------------------------------------
%  COLUMN 3 — Results & Conclusion
% --------------------------------------------------------
\begin{column}{0.28\linewidth}

  \begin{block}{Results}
    % One headline result sentence, then figure or table.
    TODO headline claim.

    \begin{figure}
      \centering
      \includegraphics[width=0.95\linewidth]{TODO_results_figure_path}
      \caption{TODO: caption.}
    \end{figure}

    % OR a table:
    % \begin{table}
    %   \centering
    %   \small
    %   \begin{tabular}{lcc}
    %     \toprule
    %     Method & Metric1 & Metric2 \\
    %     \midrule
    %     Baseline & 0.00 & 0.00 \\
    %     \textbf{Ours} & \textbf{0.00} & \textbf{0.00} \\
    %     \bottomrule
    %   \end{tabular}
    %   \caption{TODO}
    % \end{table}
  \end{block}

  \begin{block}{Conclusion}
    \begin{itemize}
      \item TODO takeaway 1
      \item TODO takeaway 2
      \item \textbf{Future work:} TODO
    \end{itemize}
  \end{block}

  \begin{block}{References}
    \footnotesize
    \printbibliography[heading=none]
  \end{block}

\end{column}

\end{columns}

\end{frame}
\end{document}
```

## Customization notes

- **Poster size**: Change `size=a0` to `size=a1` for A1, or `width=...,height=...` for custom sizes (in cm).
- **Orientation**: `orientation=landscape` flips to wide-landscape format; adjust column widths accordingly (three equal columns → `0.31\linewidth` each with spacing).
- **Scale**: The `scale=1.4` factor controls base font size relative to the poster size. Increase for more readable text at distance.
- **Two-column layout**: Replace `0.32/0.36/0.28` split with two columns at `0.48\linewidth` each.
- **Logo**: Add to headline block with `\includegraphics[height=3cm]{logo.pdf}` floated left or right.
