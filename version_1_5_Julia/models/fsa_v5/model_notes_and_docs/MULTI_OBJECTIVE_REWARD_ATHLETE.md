# Multi-Objective Reward Structure: The Athlete Mindset

In `gpu_control_v5.jl`, the controller's objective was expanded from a simple reward on Autonomic health ($A$) to a multi-objective form:

$$J_{reward} = - \int A \, dt - \int B \, dt - \int S \, dt$$

This document explains the physiological rationale and the mathematical coupling between these variables.

---

## 1. The Core Coupling (Equation 7)
The fundamental mechanism of the FSA-v5 model is the coupling of fitness and fatigue to the autonomic "drive" via the bifurcation parameter $\mu$. Per Equation 7 in the Technical Guide:
$$\mu(B, S, F) = \mu_0 + \mu_B B + \mu_S S - \mu_F F - \mu_{FF} (F - F_{TYP})^2 - \dots$$

### The Reward Logic:
*   **$B$ and $S$ as Drive Loadings:** Aerobic Fitness ($B$) and Strength ($S$) are the primary terms that push $\mu$ into the positive, healthy regime. Maximizing $B$ and $S$ isn't just about "collecting fitness points"—it is the physical mechanism required to maintain a stable, high Autonomic health ($A$).
*   **$F$ as the Suppression:** Fatigue ($F$) is the term that subtracts from $\mu$, potentially forcing it negative and triggering autonomic collapse ($A \to 0$).

---

## 2. Temporal Dynamics and Hierarchy
The system exhibits a clear hierarchy based on the turnover rates (half-lives) of the reward variables:

| Variable | Turnover ($\tau$) | Role in the Strategy |
| :--- | :--- | :--- |
| **Alertness ($A$)** | ~5–10 days | **Fast Performance:** The high-frequency indicator of current readiness. Highly responsive to tactical training changes. |
| **Aerobic ($B$)** | 42 days | **Medium Capacity:** A slow-moving asset that provides a steady, stable "floor" for autonomic drive. |
| **Strength ($S$)** | 60 days | **Slow Capacity:** The most persistent "infrastructure" of the system. Provides the longest-term contribution to the reward integral. |

---

## 3. Emerging Optimizer Behavior
By maximizing the integrals of all three, the 100-day rolling optimizer behaves like a professional athlete rather than a short-term performer:

1.  **Investment in slow assets:** Because the optimization window is 100 days, the agent prioritizes building $S$ and $B$. It knows that these variables "stick" in the reward integral for much longer than $A$.
2.  **Endogenous A-Generation:** The agent "learns" that the most efficient way to maintain high $A$ is to build a massive foundation of $B$ and $S$ while keeping $F$ at a sustainable level. 
3.  **Longevity Focus:** Because there is no terminal date (always rolling 100 days), the agent seeks a **sustainable equilibrium altitude**. It finds the maximum training load $\Phi^*$ that builds the highest fitness/strength possible without ever compromising the "performance cash-flow" of $A$.

## 4. Summary
The expanded cost function moves the agent toward a **capacity-building** strategy. It treats $A$ as the fast performance signal and $B/S$ as the slow-moving engines that power that signal. The physics of the model (Equation 7) ensures that maximizing the slow assets is the only principled way to sustain the fast ones.
