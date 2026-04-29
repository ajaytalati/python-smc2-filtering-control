# Stage A — Scalar OU LQG: filter + control demo

The simplest principled filter+control test model. Closed-form Kalman
filter, LQR feedback law, and joint LQG cost via the separation
principle. Tightest possible analytical gates.

## Model

```
dx_t = -a x_t dt + b u_t dt + σ_w dW_t
y_t  = x_t + σ_v ν_t                      (Gaussian obs each step)
J(u) = ∫₀ᵀ (q x_t² + r u_t²) dt + s x_T²  (quadratic cost)
```

Truth parameters (chosen so the LQG cost reduction vs open-loop is
detectable but the SMC² particle counts stay modest):

| param   | a   | b   | σ_w | σ_v | q   | r   | s   | T  | Δt   |
|---------|-----|-----|-----|-----|-----|-----|-----|----|------|
| value   | 1.0 | 1.0 | 0.3 | 0.2 | 1.0 | 0.1 | 1.0 | 1s | 0.05 |

20 time steps. Estimable: 4 dynamics (a, b, σ_w, σ_v) + 1 init (x_0).

## A1 — Filter

PF log-likelihood at truth, averaged over 10 seeds at K=200:

| quantity                 | value         |
|--------------------------|---------------|
| Kalman log p(y\|truth)   | **23.07** (analytical, exact)  |
| PF log p(y\|truth)       | **22.89 ± 0.68** |
| bias                     | −0.18 nats (< 1% relative) |

The locally-optimal Pitt-Shephard PF on this linear-Gaussian model is
essentially unbiased — confirming the SMC² filter pillar works
correctly on its target regime.

Tests: `tests/test_kalman_lqr_baseline.py` (5 tests),
`tests/test_scalar_ou_filter_matches_kalman.py` (2 tests).

## A2 — Open-loop control (raw schedule)

Tempered SMC over a 20-D raw-pulse schedule `u = (u_0, …, u_19) ∈ ℝ²⁰`,
target = `prior · exp(-β · J(u))`, β_max auto-calibrated.

| controller                            | mean cost    |
|----------------------------------------|--------------|
| analytical LQR (perfect state)         | 5.41         |
| MC LQG (Kalman + LQR)                  | 5.63         |
| MC open-loop u=0                       | 9.29         |
| **SMC²-mean schedule**                 | **9.63**     |
| SMC² / open-loop ratio                 | **1.036** ✓ (gate [0.95, 1.10]) |

Open-loop u=0 IS the best deterministic schedule under x_0~N(0,1) (the
expectation of LQR feedback at zero state is zero), and SMC² found it.
The 4-unit gap to LQG is the value of state feedback, unreachable by
any deterministic schedule.

Plot: [`A2_control_diagnostic.png`](A2_control_diagnostic.png)

Driver: `tools/bench_smc_control_ou.py`.

## A3 — State-feedback control

Reparameterise the control as `u_k = -K_k · x̂_k` where `x̂_k` is the
inline Kalman posterior mean and `(K_0, …, K_{T-1})` is the SMC²
particle. Run tempered SMC over the gain vector K.

| controller                                     | mean cost  |
|------------------------------------------------|------------|
| analytical LQR (perfect state)                 | 5.41       |
| MC LQG (Kalman + LQR Riccati)                  | 5.63 ± 0.11 |
| MC open-loop u=0                               | 9.29       |
| **SMC²-mean-gain controller**                  | **5.61**   |
| Riccati-gain controller (same evaluator)       | 5.56       |

Gates:
- **SMC² / MC LQG = 0.995** (gate [0.95, 1.10]) ✓
- **SMC² / open-loop = 0.603** (gate ≤ 0.7) ✓
- **K RMS error vs Riccati = 20%** (gate < 25%) ✓

The SMC²-derived gain trajectory tracks the Riccati gain shape:
both start at K_0 ≈ 2.1 and decay to K_19 ≈ 0.5 near the terminal time.

Plot: [`A3_state_feedback_diagnostic.png`](A3_state_feedback_diagnostic.png)

Driver: `tools/bench_smc_control_ou_state_feedback.py`.

## Convergence

- A1 filter: 5–10 tempering levels, 10–20 s on CPU.
- A2 open-loop: 11 tempering levels, 8 s on CPU.
- A3 state-feedback: 12 tempering levels, 8 s on CPU.

HMC acceptance ≈ 99.9% throughout. Stage A is essentially CPU-only and
runs in under a minute end-to-end after JIT warmup.
