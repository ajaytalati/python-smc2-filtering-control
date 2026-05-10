# Transition from Ad-hoc to Endogenous Fatigue Penalties

In previous versions (e.g., Banister-based models), a manual fatigue penalty was used in the cost function:
$$J_{fatigue} = \lambda_{F} \int \max(F - F_{max}, 0)^2 dt$$

In FSA-v5, this penalty is **disabled by default ($\lambda_F = 0.0$)**. This document explains the physiological and mathematical reasoning behind this decision.

---

## 1. Why Ad-hoc Penalties were needed in v1.5
In the classic linear Banister model, fitness ($B$) and fatigue ($F$) are decoupled. Technically, an agent could train at infinite intensity ($\Phi = \infty$) to gain infinite fitness. While $F$ would also be high, it did not "break" the system or prevent further gains. The $\lambda_F$ penalty was a "guardrail" required to keep the optimizer within realistic physiological bounds.

## 2. Why they are Redundant in FSA-v5
FSA-v5 replaces the linear dynamics with a non-linear Stuart-Landau/Busso hybrid. Fatigue is now managed endogenously by the physics of the model in three ways:

### A. The Busso "Fatigue Gain" ($K$ dynamics)
The Busso 2003 extension introduces dynamic fatigue gains ($K_{FB}, K_{FS}$). As the agent trains harder ($\Phi$ increases), the "damage" coefficients $K$ increase. This means that **future training sessions become more exhausting for the same amount of effort.** The model endogenously represents the state of being "overtrained" where the system loses efficiency.

### B. Coupling to Reward (Alertness $A$)
The primary objective of the controller is to maximize the integral of Alertness ($A$). In FSA-v5, fatigue is directly coupled to the survival of $A$ through the bifurcation parameter $\mu$:
$$\mu = \mu_0 + \dots - \mu_F F - \mu_{FF} (F - F_{TYP})^2$$
High fatigue ($F$) causes the healthy attractor to vanish, leading to **autonomic collapse ($A \to 0$)**. Because the optimizer wants to maximize $A$, it has a massive natural incentive to keep $F$ low. Pushing $F$ too high destroys the very "reward" the agent is seeking.

### C. Safety via Chance-Constraints ($\lambda_{chance}$)
The introduction of the soft chance-constraint ($\lambda_{chance}$) penalizes trajectories that approach the **separatrix**. Since high fatigue is the primary driver that pushes the system toward the unstable region, the safety goal is already explicitly handled by the $\lambda_{chance}$ term.

---

## 3. Summary of Controller Logic
By setting $\lambda_F = 0.0$, we allow the controller to be **principled**. It no longer relies on an arbitrary "clipping" value ($F_{max}$), but instead learns the optimal fatigue boundary based on:
1.  How much fatigue destroys the reward ($A$).
2.  How much current effort increases future fatigue sensitivity ($K$).
3.  The risk of crossing the safety separatrix ($A_{sep}$).

*Note: The code for $\lambda_F$ remains in the GPU kernel for legacy comparison and can be re-enabled via the `--ctrl-lam-f` CLI flag.*
