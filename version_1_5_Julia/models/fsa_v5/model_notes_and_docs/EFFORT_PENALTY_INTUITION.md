# Effort Penalty Intuition (L2 Regularization in FSA-v5)

In the `gpu_control_v5.jl` cost function, the stimulus $\Phi$ is penalised using the integral of its square:
$$J_{effort} = \lambda_{\Phi} \int (\Phi_B^2 + \Phi_S^2) dt$$

This document explains why this quadratic form is used to represent "effort" and how it shapes the controller's behavior.

## 1. Why Square the Stimulus?

1.  **Non-linear Physiological Strain:** Doubling the intensity of a workout doesn't just double the strain; it causes disproportionately more fatigue, metabolic waste, and injury risk. The square term ($\Phi^2$) captures this "red-line" effect.
2.  **Preventing Infinite Training:** Without this penalty, the optimizer would conclude that "more is always better" and pin the stimulus to the maximum allowed value. The effort penalty forces the agent to find the *minimum* necessary stimulus to achieve its fitness goals.
3.  **Encouraging Smoothness:** Quadratic penalties (L2 norm) prefer many small actions over one large action. This leads to more realistic, consistent training plans rather than erratic "on/off" bursts.

---

## 2. Numerical Examples

### Case A: "Spike" vs. "Steady" (Convexity)
Imagine providing a total stimulus of **2.0 units** over 2 days.

*   **Steady Strategy:** 1.0 unit on Day 1, 1.0 unit on Day 2.
    *   Total Stimulus: $1.0 + 1.0 = 2.0$
    *   **Effort Cost:** $1.0^2 + 1.0^2 = \mathbf{2.0}$
*   **Spike Strategy:** 2.0 units on Day 1, 0.0 units on Day 2.
    *   Total Stimulus: $2.0 + 0.0 = 2.0$
    *   **Effort Cost:** $2.0^2 + 0.0^2 = \mathbf{4.0}$

**Insight:** The Spike costs **twice as much effort** for the same total volume. The math forces the controller to prefer consistency.

### Case B: The Price of Intensity
The "marginal cost" of turning up the dial grows as you get closer to the limit:

| Intensity ($\Phi$) | Effort Cost ($\Phi^2$) | Increase in Cost |
| :--- | :--- | :--- |
| 0.5 (Light) | 0.25 | — |
| 1.0 (Moderate) | 1.00 | +0.75 |
| 2.0 (Heavy) | 4.00 | +3.00 |
| 3.0 (Extreme) | 9.00 | +5.00 |

**Insight:** Moving from Moderate to Heavy is **4x more expensive**, creating a natural "soft ceiling" on training intensity.

### Case C: Channel Balancing (Cross-Training)
If the agent needs to generate **1.0 unit** of total stimulus:

*   **Single Channel:** $\Phi_B = 1.0, \Phi_S = 0.0 \implies$ Cost: $1.0^2 + 0^2 = \mathbf{1.0}$
*   **Balanced:** $\Phi_B = 0.5, \Phi_S = 0.5 \implies$ Cost: $0.5^2 + 0.5^2 = \mathbf{0.5}$

**Insight:** It is **50% cheaper** to do a little of both than to crush one channel. The quadratic form mathematically incentivizes cross-training.

---

## 3. Summary of Controller Behavior

Because of the $\int \Phi^2 dt$ term, the agent will:
1.  **Spread the load:** Prefer training every day at lower intensity rather than one massive session per week.
2.  **Avoid red-lining:** Stay away from the `Phi_max` limit unless the predicted reward ($A$) is very high.
3.  **Diversify:** Use both Aerobic ($\Phi_B$) and Strength ($\Phi_S$) stimulus to reach fitness targets at the lowest "metabolic price."
