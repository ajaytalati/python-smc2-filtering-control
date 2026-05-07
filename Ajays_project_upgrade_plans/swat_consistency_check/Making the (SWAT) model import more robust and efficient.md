# Pre-Porting Validation Plan: SWAT "Port-Ready" Model Factory Repository

This document outlines the strategy for creating a new, fresh repository (the **SWAT Model Factory**) designed to eliminate the "technical debt" incurred during model porting. This repository will serve as the single source of truth for production-grade SWAT models, ensuring they are numerically stable, mathematically identifiable, and self-consistent before being exported to the control framework.

## 1. Goal: The "Port-Ready" Certification
A model is considered "Port-Ready" only if it passes a rigorous suite of automated tests that verify its suitability for SMC² filtering and MPC control. The output of this repository is a verified "Port-Package."

## 2. Pillar I: Mathematical Identifiability (FIM Analysis)
To prevent "flat" parameter traces in the filter, the factory must prove the parameters can be recovered from the 4-channel observation model.
*   **Fisher Information Matrix (FIM) Test:**
    *   Implement a script `tools/analyze_identifiability.py` using JAX-autodiff.
    *   Compute the FIM $I(\theta)$ for the full 27-parameter set across Scenario A (Healthy) and Scenario C (Recovery).
    *   **Acceptance Gate:** The FIM for the chosen "Identifiable Subset" must be full rank, and the condition number must be below a threshold (e.g., $10^9$).
    *   **Output:** Automatically generate the `identifiable_subset` list for the `EstimationModel`.

## 3. Pillar II: Numerical Stability & Stiffness Audit
To prevent "NaN" explosions or the need for excessive sub-stepping in production.
*   **Stiffness Analysis:**
    *   Compute the Jacobian $J = \partial f / \partial y$ of the drift equations.
    *   Calculate the spectral radius $\rho(J)$ across the entire state space $[0, 1]^4$.
    *   **Acceptance Gate:** Determine the maximum stable step size $h_{\text{max}} = 2 / \rho(J)$.
    *   If $h_{\text{max}} < 15\text{min}$, the factory **must** flag the model as "Stiff" and mandate $N$ sub-steps in the export config.

## 4. Pillar III: Plant vs. Estimator Reconciliation
To ensure the Filter and Plant are "speaking the same language."
*   **The "Mirror" Test:**
    *   Implement `tests/test_reconciliation.py`.
    *   Initialize a `StepwisePlant` and an `EstimationModel` with the same parameters and state.
    *   Advance the plant 1 bin and the estimator 1 `propagate_fn` call with the same noise key.
    *   **Acceptance Gate:** The resulting states must be bit-equivalent (within $10^{-7}$ for float32 or $10^{-14}$ for float64).

## 5. Pillar IV: Observability & Likelihood Sanity
*   **Likelihood Sensitivity Test:**
    *   For each observation channel, verify that the log-likelihood $\ell(y | \text{obs})$ is peaked at the truth.
    *   Verify that "illegal" states (e.g., $T < 0$ or $W > 1$) result in $-\infty$ or extremely low likelihoods.
*   **Prior/Truth Alignment:**
    *   Automated check: `EstimationModel.prior_mean` must be within 2 standard deviations of `DEFAULT_PARAMS`.

## 6. Pillar V: Three-Component "Mirror" Sandbox
The factory implements the three-component architecture (Estimator, Controller, Plant) to verify the model in a closed-loop environment.

*   **1. The Estimator (`estimation.py`):**
    *   Must define the `EstimationModel` singleton with full parameter priors and likelihoods.
*   **2. The Controller (`control.py`):**
    *   Implement the `ControlSpec` and cost functional in the sandbox.
*   **3. The Plant (`_plant.py`):**
    *   Implement the `StepwisePlant` interface for half-day/full-day strides.

## 7. Pillar VI: Legacy Scenario Integration & Regression
To ensure consistency with previous research and verify against known edge cases.
*   **Reference Scenarios:** Port the legacy reference scenarios from `SWAT_model_dev/examples/swat/` into the factory.
*   **Regression Suite:** Every "Port-Package" must pass a regression test against these standard scenarios to ensure no behavioral regressions are introduced during optimization.

## 8. Pillar VII: Automated "Port-Package" Generation
Replace manual copying with an automated exporter.
*   **Export Script:** `tools/export_to_framework.py`
    *   Checks all acceptance gates (FIM, Stiffness, Bit-Equivalence, Regression).
    *   Bundles the verified suite: `estimation.py`, `control.py`, `_plant.py`.
    *   Generates a `MANIFEST.json` containing the verified $h_{\text{max}}$, identifiable subset, and stiffness warnings.

## 9. Execution Timeline (Repo: `swat-model-factory`)
1.  **Week 1:** Repository initialization and core ODE logic migration. Port legacy reference scenarios.
2.  **Week 2:** Implement FIM analysis and Identification Subset generator.
3.  **Week 3:** Implement Numerical Stability/Stiffness audit and Mirror tests.
4.  **Week 4:** Finalize the "Port-Ready" exporter script and generate the first verified package.

---
**Status:** This plan has been updated to reflect the move to a fresh "Model Factory" repository, incorporating legacy reference scenarios as a core validation pillar.
