# LaTeX lecture notes — SMC² and MPC for FSA-v2

This directory contains a short set of pedagogical lecture notes
explaining the mathematical theory underlying the
`python-smc2-filtering-control` repository.

## Building the PDF

From this directory:

```
latexmk -pdf main.tex
```

To clean intermediate files:

```
latexmk -c
```

## Files

- `main.tex` — top-level document (article class, 11pt, a4paper)
- `preamble.tex` — packages, theorem environments, math macros
- `sections/01_intro.tex` … `sections/08_discussion.tex` — the eight
  sections of the document
- `figures/` — figures (currently placeholders; can be regenerated from
  `version_1/outputs/` and `version_2/outputs/`)

## Audience

Graduate students in applied mathematics, mathematical engineering,
applied stochastics, or process engineering. Familiarity with Itô
stochastic differential equations, Bayesian filtering, and basic
optimal control at the level of a one-semester course is assumed.
