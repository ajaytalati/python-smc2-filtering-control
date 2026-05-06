# LaTeX framework documentation — `smc2fc`

Pure framework documentation for the `python-smc2-filtering-control`
package: the inner bootstrap particle filter, the outer SMC² over
parameters, the JAX-native compile-once backend, the Gaussian-bridge
warm-start, control-as-inference, and receding-horizon MPC.

Model-independent. The earlier FSA-specific lecture notes are
preserved in `../LaTex_docs_outdated/` for reference.

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

- `main.tex` — top-level document (article class, 11pt, A4)
- `preamble.tex` — packages, theorem environments, math macros
- `sections/01_overview.tex` … `sections/11_future_work.tex` — the
  eleven sections of the document
- `sections/appendix_D_code_api.tex` — one-line API reference for
  every public function in `smc2fc/`

## Section map

1. Overview + reader's guide + architecture diagram
2. Generic SDE-with-observations setting (`SDEModel` / `EstimationModel`)
3. Bootstrap particle filter (SIR, Liu–West shrinkage, unbiased
   marginal likelihood)
4. OT rescue inside the filter (Sinkhorn / Nyström / barycentric
   projection / sigmoid blend)
5. SMC² over parameters: tempered SMC + HMC + adaptive diagonal
   mass matrix
6. **JAX-native SMC: once-per-run compilation** (the framework's
   key practical contribution; documents the failed NUTS experiment)
7. Gaussian bridge for warm-starts (short — SF/BW does not pay for
   itself empirically)
8. Control as inference (with the empirical hard-vs-soft chance-
   constrained finding)
9. Receding-horizon model-predictive control
10. Configuration and knobs (every `SMCConfig` field)
11. Future work — optimising the soft-HMC controller's speed/quality
    trade-off
- Appendix D — code-API quick reference

## Audience

The doc is pitched at engineers and applied researchers using or
extending the framework on a new application. Familiarity with
Bayesian filtering, importance sampling, and the basics of HMC is
assumed; specialist topics (BW geometry, low-rank Sinkhorn) are
introduced where they appear and not developed beyond what the code
does.
