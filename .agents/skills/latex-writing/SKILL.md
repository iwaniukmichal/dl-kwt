---
name: latex-writing
description: Write, revise, and format LaTeX documents for academic or technical work, especially theses, reports, papers. Use this skill whenever the user wants LaTeX source, asks to rewrite a section in LaTeX, needs equations, figures, tables, citations, cross-references, or wants a document to follow strict academic style conventions. Also use it when the user shares a partial LaTeX draft and expects the result to be cleaner, more consistent, more scientific, or more publication-ready.
---

# LaTeX academic writing

Use this skill to produce clean, compilable LaTeX that follows academic writing conventions and preserves factual discipline.

## Core principles

- Write in a scientific style: precise, concise, evidence-based, and explicit about uncertainty.
- Prefer reusable procedures over ad hoc formatting.
- Preserve the user's existing document structure unless there is a clear reason to improve it.
- Never invent citations, bibliography entries, results, or experimental details.
- When a claim depends on literature and no source is available, say so clearly and leave a visible placeholder instead of fabricating support.

## Writing workflow

1. Extract the user's factual content, claims, and constraints.
2. Separate established facts from intuition, interpretation, and open questions.
3. Draft or revise the prose in clear academic language.
4. Convert the prose into clean LaTeX markup.
5. Add cross-references, citations, figure/table captions, and equation labels.
6. Run a self-check against the validation checklist before finalizing.

## Content rules

### Scientific style

- Write only statements that are supported by the provided material, established knowledge, or cited sources.
- Keep the text concise. Remove filler, hype, and vague intensifiers.
- Prefer direct statements over ornamental phrasing.
- When mechanisms, causes, or interpretations are uncertain, describe the limitation explicitly.
- You can write intuitive explanation only when it is well justified and clearly framed as interpretation rather than as direct evidence.

### Citation discipline

- Cite all claims that depend on papers, articles, datasets, or external sources.
- Use in-text citation commands that match the existing project style, for example `\cite{...}` or project-specific citation macros.
- Do not fabricate citation keys.
- If a citation is required but unavailable, leave a clear marker such as `% TODO: add citation` near the claim.

## Language and typography rules

- Ensure stylistic, grammatical, spelling, and punctuation correctness.
- Use non-breaking spaces after one and two-letter words, for example `an~example`, `in~sec.~\ref{sec:results}`.
- Emphasize proper names with `\emph{...}` when the user's style requires it.
- Do not use bold text in running academic prose. Bold is acceptable only in controlled structural contexts such as a leading label in a definition list or table header.
- At the beginning of a sentence, write full forms such as `Section~\ref{...}`, `Subsection~\ref{...}`, `Figure~\ref{...}`, `Table~\ref{...}`.
- In the middle of a sentence, use abbreviations such as `see. sec.~\ref{...}`, `fig.~\ref{...}`, `tab.~\ref{...}`.
- Use `\dots` for ellipsis.
- Use `--` for parenthetical dashes and numeric ranges.
- Use a single hyphen for compound forms such as `fifty-five`.

## Structure and cross-references

- Every referenced structural element must have a label.
- Use `\label{...}` with stable, readable prefixes such as `sec:`, `subsec:`, `fig:`, `tab:`, `eq:`.
- Refer to chapters, sections, figures, tables, and equations with LaTeX references instead of hard-coded numbers.
- Use `\eqref{...}` for equations.
- Do not leave orphan references such as “see figure below” without a proper `\ref`.

## Figures and tables

### Rules

- Reference every figure and every table in the text.
- Every figure and table must include a `\label{...}`.
- Prefer `[t!]` or `[b!]` placement. Use `[h!]` only when the element must stay tightly attached to the surrounding text.
- Put table captions above the table.
- Put figure captions below the figure.
- Insert `\smallskip` between a table caption and the tabular material.
- Do not end captions with a period
- Prefer vector graphics (`.pdf`, `.svg`) for figures. Use `.png` only when it is clearly high resolution and the content is raster by nature.

### Table template

Use this as the default table pattern unless the document already uses a different house style:

```latex
\begin{table}[t!]
\centering
\caption{Title}
\label{tab:label}
\smallskip
\small
\begin{tabular}{@{}ll@{}}
\toprule
\textbf{Col 1} & \textbf{Col 2}\\
\midrule
text & text \\
text & text \\
\bottomrule
\end{tabular}
\end{table}
```

Use `tabularx` only when fixed-width wrapping is actually needed.

## Mathematics

- Put mathematical symbols, variables, and standalone numeric values appearing in prose into inline math mode.
- Interpret any instruction to put numbers in `$$...$$` as a request to use math mode, not display math. Use inline math for prose, for example `$5$`, and display environments only for standalone formulas.
- Use numbered equations for displayed equations that are not grammatically integrated into the sentence.
- Use `equation` for one numbered equation.
- Use `align` for several aligned equations with separate numbers.
- Use `equation` with `aligned` when several lines should share one number.
- If an equation is part of a sentence, punctuate it consistently with the sentence.
- Reference equations with `\eqref{...}`.

## Package policy

- Prefer packages already approved in `assets/setup.txt`.
- Add a new package only when it is necessary for the requested output.
- Avoid package sprawl.
- Prefer standard, well-supported packages over obscure alternatives unless the project already depends on them.

## Validation checklist

Before finalizing, verify all of the following:

- [ ] The prose is concise, scientific, and factually disciplined.
- [ ] Unsupported claims are removed, qualified, or marked for citation.
- [ ] Citations are present where needed and no citation keys were invented.
- [ ] One or two-letter words use non-breaking spaces where appropriate.
- [ ] Proper names are emphasized with `\emph{...}` when that convention applies.
- [ ] No bold appears in running prose.
- [ ] All figures and tables are referenced in the text.
- [ ] All figures, tables, sections, and equations that need cross-references have labels.
- [ ] Table captions are above tables; figure captions are below figures.
- [ ] `\smallskip` is present between a table caption and the table body.
- [ ] Mathematical content uses the correct inline or display mode.
- [ ] Displayed equations that should be numbered are numbered and referenced with `\eqref{...}`.
- [ ] The output is consistent with the existing document style and package stack.
