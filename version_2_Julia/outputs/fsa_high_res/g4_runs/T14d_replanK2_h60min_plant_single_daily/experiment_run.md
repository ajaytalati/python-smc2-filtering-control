# FSA-v2 Julia GPU bench — full SMC² rolling-window filter

- **Mode:** filter only (Φ=1.0 from plant; no controller)
- **Timestamp:** 2026-05-07T22:05:51.653
- **Device:** NVIDIA GeForce RTX 5090
- **Wall time:** 297.2 s (5.0 min)
- **T_total_days:** 14, step=60 min
- **BINS_PER_DAY:** 24, WINDOW=24, STRIDE=12
- **n_strides:** 27
- **N_SMC:** 32, K_per_chain=400
- **HMC:** step_size=0.025, num_leapfrog=8, num_mcmc=5
- **Cold-start:** prior at first window; warm-start (carry posterior forward) for subsequent windows.

## Implementation

Locally-guided (Pitt-Shephard) PF on GPU. Per-particle, per-bin:
1. G1-reparametrized prior predictive (mean + state-dep cov)
2. Sequential scalar Kalman fusion across 3 Gaussian channels (HR / stress / log_steps)
3. Cholesky-3 sample from fused N(μ_fused, P_fused)
4. Predictive log-marginal accumulation + Bernoulli sleep ll

All inner loops in fp32. Outer SMC² log-weights / ESS / posterior cloud in fp64.
Parallel-chains HMC: M·(1+2d) chains in one kernel launch per leapfrog step.
