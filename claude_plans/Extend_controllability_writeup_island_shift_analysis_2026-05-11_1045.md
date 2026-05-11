# §17 Revisions: speed-up via τ/κ + FIM-duality witness search

> Archived from plan mode: 2026-05-11 10:45.
> Updated: 2026-05-11 12:47 — appended §17 revisions (timescale halving + FIM-duality witness search) after the §14-§16 implementation completed and the user flagged T=200 d as too long.

> **Status update (2026-05-11 evening).** Plan extended once more. §14--§16 are written and compiled in the document. New revisions in §17 below address two user-flagged issues with the §16 result: (a) T=200 d horizon is computationally unacceptable; halve τ_B, τ_S and double κ_B, κ_S to preserve island geometry while doubling the trajectory speed; (b) replace the hand-designed bang-bang witness in §16.2 with a grid search exploiting the **identification ↔ controllability duality**: trajectories with well-conditioned FIM are, by duality, controllable trajectories. Selection criterion is FIM condition number computed on the trajectory under each candidate constant Φ.

# Extend controllability writeup: healthy-island analysis under the 3 recommended modifications

## Context

The previous round (now archived at [LaTeX_writeup_controllability_of_FSA-v5_from_SEDENTARY_INIT_2026-05-11_0943.md](claude_plans/LaTeX_writeup_controllability_of_FSA-v5_from_SEDENTARY_INIT_2026-05-11_0943.md)) produced a 26-page document arguing that the FSA-v5 model is heuristically uncontrollable from `SEDENTARY_INIT` to `A_target = 0.30` within T=100 d. The document's §13 recommended three single-parameter deconditioning modifications that should restore controllability:

1. **Mod A**: μ_B− = μ_S− halved (0.10 → 0.05)
2. **Mod B**: B_dec = S_dec reduced (0.07 → 0.04)
3. **Mod C**: Hill exponent n reduced (4 → 2)

The user now wants the document EXTENDED with new section(s) that:
1. Plot the healthy island (where μ̄(0; Φ) > 0) under each of the 3 single modifications, mirroring the §10.4-style visualisation in the original tech guide.
2. **Numerically sweep ALL 5 deconditioning parameters** `(B_dec, S_dec, μ_B−, μ_S−, n)` to find a parametrisation that *shifts* the island center toward Φ ≈ (1, 1), not merely widens it around (0.3, 0.3). The single-mod analyses in (1) are the warm-up; the 5-parameter sweep is the substantive exploration.
3. Verify three qualitative properties of the recommended (shifted) parametrisation:
   - **Pathological sedentary basin still exists**: low constant Φ ≈ (0.1, 0.1) from `SEDENTARY_INIT` still drives A → 0 (because (0.1, 0.1) sits well below the shifted island).
   - **Constructive escape from `SEDENTARY_INIT`**: a hand-designed Φ(t) trajectory takes the state from `SEDENTARY_INIT` into the shifted healthy island center near (1, 1). This is the LEAN-mechanisable witness.
   - **Pathological overtraining basin (shifted) still exists**: under heavy constant Φ ≈ **(2, 2)** (not (1, 1) — because (1, 1) is now safe!), from `TRAINED_ATHLETE_INIT_v2`, A → 0 over 100 d. But a hand-designed Φ(t) starting from `TRAINED_ATHLETE_INIT_v2` can stay inside the shifted island and reach A ≈ 0.7.

The user's corrections from the previous round:
- The healthy basin shift implies the overtraining basin also shifts. The test policy for the overtraining qualitative-behaviour test is therefore Φ ≈ (2, 2), not the (1, 1) value used in the original tech guide.
- `TRAINED_ATHLETE_INIT` will most likely need to be **redefined** under the new parametrisation. The canonical values ($B_0 = 0.50, S_0 = 0.45, F_0 = 0.20, A_0 = 0.45, K_{FB,0} = 0.0615, K_{FS,0} = 0.0815$) correspond to the slow-manifold equilibrium at Φ = (0.30, 0.30) under the *current* parametrisation — they are the "trained" state when the healthy island is centred at (0.30, 0.30). Under a shifted island centred near (1, 1), the natural redefinition is the slow-manifold state at the new island center (or slightly offset toward the over-training side of the separatrix, so that a heavy Φ ≈ (2, 2) policy pushes the trajectory out of the island into overtraining collapse). The new `TRAINED_ATHLETE_INIT_v2` is therefore computed *from* the recommended parametrisation, not assumed in advance.
- The 5 deconditioning parameters must be swept jointly. The plan's earlier framing — "the 3 mods likely only widen, not shift" — was wrong: **raising** $B_{\rm dec}$ and $S_{\rm dec}$ (rather than lowering them, as I originally suggested in §13) is the natural way to shift the deconditioning threshold to larger $B$ values, which forces the island center to larger Φ. Combined with appropriate $μ_{B-}, μ_{S-}, n$ changes, the deconditioning block alone may suffice.

The user has explicitly chosen **hand-designed Φ(t) witnesses** (cleanest for LEAN4 mechanisation) for the constructive escape proofs.

## Approach

### New LaTeX content (appended to `sedentary_basin_controllability.tex`)

Three new sections, plus a small extension to App~A:

- **§14: Healthy-island analysis under single deconditioning modifications.** Three subsections (one per mod from §13: μ_dec halved, B_dec lowered, n=2). Each subsection contains: parameter setting, μ̄(0; Φ) heatmap with the closed-island contour overlaid, location/size/shape of the island, and a table comparing μ_init at SEDENTARY_INIT and μ̄(0; (0.3, 0.3)) and μ̄(0; (1, 1)) against the canonical baseline. Concludes by observing whether the island shifted toward (1, 1) — likely answer: no, it merely widens or deepens.

- **§15: 5-parameter deconditioning sweep to shift the island toward (1, 1).** This is the substantive contribution of the extension. Numerically explores the 5-dimensional parameter space $(B_{\rm dec}, S_{\rm dec}, μ_{B-}, μ_{S-}, n)$ (or 3D if B/S coupling is enforced for physiological symmetry), searching for a parametrisation that:
    - shifts the maximum of μ̄(0; Φ) from near (0.30, 0.30) toward (1.0, 1.0),
    - widens the closed island into a larger basin,
    - preserves a pathological sedentary corner ($\bar\mu(0; (0.1, 0.1)) < 0$ still),
    - and creates a pathological overtraining corner at $\bar\mu(0; (2, 2)) \ll 0$.
  
  Sweep strategy: start with the user's intuition that $B_{\rm dec}, S_{\rm dec}$ need to be **raised** (not lowered as in §13), so the deconditioning threshold moves to higher $B, S$ values, forcing the slow-manifold equilibria $B^*, S^*$ — and hence the island maximum — to higher Φ. Combined with $μ_{B-}, μ_{S-}$ tweaks for depth-of-penalty and $n$ for transition sharpness. If the deconditioning block alone is insufficient to shift the island all the way to (1, 1), §15 also explores small changes to the reward/fatigue coefficients (μ_F, μ_FF, κ_B, κ_S) as a fallback, but only after exhausting the 5-deconditioning-parameter space first.
  
  Output: a **recommended parametrisation** stated as a numerical 5-tuple (or larger if reward-side changes were needed), plus a comparison table showing μ̄ at five key Φ probes against the canonical baseline.

- **§16: Constructive controllability under the recommended parametrisation.** Three subsections, one per qualitative property. **Important:** all tests use the recommended parametrisation from §15, and the trained-athlete reference state is **`TRAINED_ATHLETE_INIT_v2`** — redefined as the slow-manifold state at (or slightly toward the overtraining edge of) the new island center, computed from the §15 recommended parameters.
  - **§16.1 — Pathological sedentary basin**: passive trajectory under Φ ≡ (0.1, 0.1) constant from SEDENTARY_INIT. Show A(100) ≈ 0 (basin preserved).
  - **§16.2 — Constructive escape from SEDENTARY_INIT**: hand-designed piecewise Φ(t) witness, integrated end-to-end, showing A(T) reaches the new island center near (1, 1). LEAN-mechanisable: the witness is a finite-segment schedule and verification reduces to checking A(t) > threshold at segment boundaries.
  - **§16.3 — Pathological overtraining basin (shifted)**: passive trajectory under **Φ ≡ (2, 2) constant** from TRAINED_ATHLETE_INIT_v2. Show A(100) → 0 (overtraining basin preserved, just shifted to higher Φ). Then exhibit a hand-designed Φ(t) starting from TRAINED_ATHLETE_INIT_v2 that *stays inside* the island and reaches A ≈ 0.7. This is the controllability-from-the-edge witness.

### Figure-generation code (extending `generate_figures.py`)

The existing `generate_figures.py` already exposes `mu_bar_state(B, S, F, p=PARAMS)` and `drift(y, phi_B, phi_S, p=PARAMS)` parameterised by a `p` dict, so passing modified parameter sets is trivial. New helper functions to add:

- `make_params(**overrides)`: build a modified copy of `PARAMS` with named overrides.
- `simulate(y0, phi_fn, T, dt, p=PARAMS)`: extend the existing `simulate` to accept `p` (currently uses default).
- `mu_bar_slow_grid(phi_grid, p, A_anchor=0.0)`: compute the slow-manifold $\bar\mu(A; Φ)$ on a (Φ_B, Φ_S) grid (evaluating the cascade equilibria for $K^*, B^*, S^*, F^*$ at the supplied $A$ — usually $A = 0$ for the transcritical-curve plot).
- `plot_island(p, ax, **kwargs)`: render one island heatmap with the $\bar\mu = 0$ contour, probe-point overlays, and (optionally) trajectory overlays.
- `sweep_decoditioning(B_dec_grid, mu_dec_grid, n_grid)`: 5D-parameter sweep that scores each parametrisation by (i) location of the μ̄ maximum in (Φ_B, Φ_S) plane, (ii) island area, (iii) whether the user's three qualitative tests pass. Returns a sorted list of candidates.
- `find_trained_athlete_v2(p)`: given a parametrisation, locate the new island center (Φ_B*, Φ_S*) and compute the slow-manifold state there to define TRAINED_ATHLETE_INIT_v2.

New figures (saved in same directory, referenced from new LaTeX sections):

- `island_mod_A.png`, `island_mod_B.png`, `island_mod_C.png` — single-mod islands (§14).
- `island_sweep_grid.png` — 3×3 or 4×4 grid of islands across the 5-parameter sweep, with the recommended cell highlighted (§15).
- `island_recommended.png` — final recommended parametrisation with the SEDENTARY_INIT and TRAINED_ATHLETE_INIT_v2 probe points marked, plus the witness trajectories overlaid (§15 + §16).
- `sedentary_basin_witness.png` — witness Φ(t) and resulting 6D trajectory from SEDENTARY_INIT under the recommended parametrisation; A reaches the new island center (§16.2).
- `overtraining_basin_witness.png` — two trajectories from TRAINED_ATHLETE_INIT_v2 side-by-side: (a) passive Φ ≡ (2, 2) → collapse, (b) hand-designed dynamic Φ(t) → controllable maintenance near A ≈ 0.7 (§16.3).

All figures generated by extending `generate_figures.py` — no new script needed (keeps everything in one reproducer).

### Hand-designed witnesses (LEAN-mechanisable form)

Each constructive witness will be a piecewise schedule of the form:
```
Φ(t) = Φ_1   for t ∈ [0, T_1]
     = Φ_2   for t ∈ [T_1, T_2]
     ...
     = Φ_N   for t ∈ [T_{N-1}, T]
```
with N small (probably 2-3 segments). The LEAN-mechanisable claim is then a finite verification: under each segment, integrate the linear ODE in closed form on the slow manifold + scalar A-equation; at each segment boundary, check A(T_k) > threshold; at the final time, check A(T) ≥ A_target.

### Honest reporting

If the 3 single modifications widen the island but don't shift it (my prior, now revised: I expect *raising* B_dec to shift it toward higher Φ, but only modestly), I'll report that clearly in §14 and motivate §15's full 5-parameter sweep. If the 5-parameter sweep successfully shifts the island toward (1, 1), §15 names the recommended parametrisation; §16 then runs the qualitative checks on it. If even the full 5-parameter sweep doesn't shift it convincingly (e.g., the island can't be moved past Φ ≈ (0.6, 0.6) within the deconditioning block alone), §15 reports honestly that reward-side changes (μ_F, μ_FF, κ_B, κ_S) are also needed, and §16 uses the best-available parametrisation. The bar for "success" is: a parametrisation where SEDENTARY_INIT (Φ ≈ 0) is in the pathological basin, TRAINED_ATHLETE_INIT_v2 (near the new island edge) can be controllably maintained, and Φ ≈ (2, 2) constant drives the system out of the basin into overtraining collapse.

## Critical files

**To extend (only edits)**:
- [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/sedentary_basin_controllability.tex](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/sedentary_basin_controllability.tex) — append §14, §15, §16, update App A.
- [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/generate_figures.py](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/generate_figures.py) — extend with `make_params`, `mu_bar_grid`, `plot_island`, and the new figure-generation functions.

**To read for reference (no edits)**:
- [version_1_5_LEAN/LaTex_docs/sections/14_v5_sedentary_collapse.tex](version_1_5_LEAN/LaTex_docs/sections/14_v5_sedentary_collapse.tex) — for the canonical island-plot style (Figure `v5_sedentary_collapse_island.png`).
- [version_1_5_Julia/models/fsa_v5/model_notes_and_docs/plot_mu_bar_landscape.jl](version_1_5_Julia/models/fsa_v5/model_notes_and_docs/plot_mu_bar_landscape.jl) — existing Julia plotter, already shows the canonical island under v5 baseline. Useful as a styling reference.

**Existing infrastructure to reuse**:
- `PARAMS`, `SEDENTARY_INIT`, `TRAINED_ATHLETE_INIT`, `hill`, `mu_bar_state`, `drift`, `simulate` — all in `generate_figures.py`. The `simulate` function takes `phi_fn: t -> (φ_B, φ_S)` and returns trajectory + μ(t) array. Piecewise schedules become two-line lambdas.

## Verification

The new sections need to anchor every numerical claim:

1. **Island heatmaps**: every μ̄(0; Φ) value plotted must be reproducible from `mu_bar_state(B_slow(A=0, Φ), S_slow(A=0, Φ), F_slow(A=0, Φ), p=modified_params)`. The script's sanity assert (which checks μ_init = -0.115 under baseline) is extended to check μ̄_baseline at the (0.3, 0.3) probe = +0.011, then the three modified values at the same probe.

2. **Constructive witness trajectories**: each witness Φ(t) is integrated end-to-end. The final A value is reported and visualised. The witness is "verified" if A(T) ≥ A_target — that's the LEAN-mechanisable claim.

3. **Overtraining basin (shifted)**: passive Φ ≡ **(2, 2)** trajectory from **TRAINED_ATHLETE_INIT_v2** under the recommended parametrisation. A(T) reported. Should be near 0 to confirm the basin is preserved at this shifted location. Note: Φ = (1, 1) is now INSIDE the healthy island under the recommended parametrisation (that is the entire point of the shift); it must NOT cause collapse. Only the heavier Φ ≈ (2, 2) policy reproduces the qualitative overtraining-collapse behaviour.

4. **Recompile + visual check**: `pdflatex` twice to resolve refs and ToC; visually verify every new figure renders correctly and is referenced by `\ref{}` in the body.

5. **End-to-end reproducibility**: a reader who hasn't seen this conversation should be able to run `python generate_figures.py` in the document directory and reproduce every figure. The script's print statements report the integral and final A value for each witness, which match the LaTeX text exactly.

## Expected outcome

The expected qualitative outcome (hedged — to be confirmed by running):

- **Mod A (μ_dec halved, 0.10 → 0.05)**: island widens around (0.3, 0.3); μ̄ at the centre roughly doubles. Probably permits escape from SEDENTARY_INIT but doesn't shift the island toward (1, 1).
- **Mod B (B_dec lowered, 0.07 → 0.04)**: counter to my §13 suggestion, this likely shifts the island *toward smaller* Φ, not larger. The user's correction is that *raising* B_dec is what shifts toward (1, 1). I'll plot both and report.
- **Mod C (n = 2 from 4)**: smoother Hill transition; intermediate effect on island size, minimal effect on location.
- **5-parameter sweep result (§15)**: my prior is that *raising* (B_dec, S_dec) to ~0.20--0.30 and possibly *raising* (μ_B−, μ_S−) modestly will shift the island center from (0.30, 0.30) to somewhere in (0.6, 0.6)--(1.0, 1.0). Whether the deconditioning block alone is sufficient to reach the user's target (1, 1) depends on the algebra; if not, reducing μ_F or μ_FF by 30--50% is the natural fallback.
- **TRAINED_ATHLETE_INIT_v2**: at the new island center the slow-manifold equilibrium has $A^*$ probably in [0.5, 0.8] depending on the parametrisation. The redefined trained-athlete state inherits its $A$ from this and its $(B, S, F, K)$ from the slow-manifold equilibrium.
- **Heavy overtraining test Φ = (2, 2)**: under the recommended parametrisation, $\bar\mu(0; (2, 2))$ should be strongly negative (similar to the canonical $\bar\mu(0; (1, 1)) = -1.61$), guaranteeing collapse from the trained state under that constant policy.

If the user's intuition holds, the recommended parametrisation has the island centered near (1, 1), all three qualitative behaviours verified (pathological sedentary AND constructive escape AND pathological overtraining AND constructive maintenance), and is the basis for a future LEAN4 controllability theorem with a hand-designed Φ(t) as the constructive witness.

---

# §17 — Revisions for the published §16 (timescale fix + FIM-duality witness)

## Why this revision

The §14--§16 write-up produced a valid constructive escape witness, but at **T = 200 d**. The user has flagged that doubling the bench horizon is computationally unacceptable. The mathematical reason for needing 200 d is the time-scale separation between the autonomic equation (decay timescale $\sim 1/|\mu| \approx 9$ d) and the chronic-capacity equations (relaxation timescales $\tau_B = 42$ d, $\tau_S = 60$ d). The recommended parametrisation of §15 fixes the *static* island geometry (μ̄(0; Φ) is positive at (1,1)) but does not address the *dynamic* problem of how fast B, S can rise from their SEDENTARY_INIT values into the new island.

The fix is to halve the chronic relaxation timescales **without changing the slow-manifold equilibria** (which would force a re-do of the §15 sweep). This is achieved by:

\[
\tau_B \mapsto \tau_B / 2 = 21\,\text{d}, \quad \kappa_B \mapsto 2\kappa_B = 0.02496, \quad \tau_S \mapsto \tau_S / 2 = 30\,\text{d}, \quad \kappa_S \mapsto 2\kappa_S = 0.01632.
\]

Verification: the slow-manifold equilibria $B^* = \tau_B \cdot \kappa_B \cdot a_B \cdot \Phi_B$ and $S^* = \tau_S \cdot \kappa_S \cdot a_S \cdot \Phi_S$ are invariant under $(\tau, \kappa) \to (\tau/2, 2\kappa)$. So the recommended-parametrisation island center (1.06, 0.78) and probe values $\bar\mu(0; (0.1, 0.1)) = -0.144$, $\bar\mu(0; (1, 1)) = +0.119$, $\bar\mu(0; (2, 2)) = -0.656$ all stay the same.

The §17 revisions:
1. **§17.1**: Re-define the *full* recommended parametrisation including the halved timescales. Call it `TRUTH_PARAMS_V5_RECOMMENDED_v2`.
2. **§17.2**: Re-run §16.1 (pathological sedentary basin) at T = 100 d under the timescale-halved parameters. Should still collapse — passive Φ = (0.1, 0.1) trajectory has no fast-rising mode.
3. **§17.3**: Replace the §16.2 hand-designed bang-bang witness with a **FIM-duality grid search** over constant Φ. The user's clarification (received after plan approval): treat the control variates $(\Phi_B, \Phi_S)$ themselves as the "dummy parameters" being identified — that *is* the duality. The FIM is therefore:
   \[
   F(\Phi_{\rm const}) \;=\; \int_0^T J_\Phi(t)^\top J_\Phi(t) \, dt, \quad J_\Phi(t) = \tfrac{\partial y(t)}{\partial \Phi}\;\in\;\mathbb{R}^{6\times 2},
   \]
   a **2×2 matrix**. Its condition number $\kappa(F)$ tells us how evenly $\Phi_B$ and $\Phi_S$ excite the 6D state along the trajectory: small $\kappa$ ⇒ both control axes have strong effect ⇒ trajectory is controllable in both directions; large $\kappa$ ⇒ one axis is much weaker (one of $\partial y/\partial \Phi_B$ or $\partial y/\partial \Phi_S$ is nearly null along the trajectory). The grid search:
   - Grid: 21×21 over $\Phi \in [0.1, 2.0]^2$. 441 cells, ~30 seconds total at T=100 d each.
   - For each candidate constant $\Phi$: integrate the ODE from SEDENTARY_INIT for T=100 d under the timescale-halved parameters; numerically compute $J_\Phi(t)$ along the trajectory via central differences (perturb $\Phi_B \pm h$, re-integrate, take difference, divide by $2h$; same for $\Phi_S$).
   - Accumulate $F(\Phi) = \sum_{t_k} J_\Phi(t_k)^\top J_\Phi(t_k) \cdot \Delta t$, score by $\kappa(F)$ via `numpy.linalg.eigh`. Lower = better.
   - Also record $A(100; \Phi)$ for verification.
   - **Identification-controllability duality claim**: the $\Phi$ minimising $\kappa(F)$ is a trajectory where both $\Phi_B$ and $\Phi_S$ have strong sensitivity on $y(\cdot)$; by duality this is also a $\Phi$ where the state is most accessible from $\Phi$. So $A(100; \Phi^*)$ should be among the highest.
4. **§17.4**: Compare the two rankings (by κ(F) vs by A(100)). If they agree (top-5 overlap is non-empty), the duality claim is empirically borne out and the FIM-condition-minimising Φ is the §16.2 witness. If they disagree, report both rankings honestly; the duality doesn't hold tightly enough for this purpose and the A(100)-maximising Φ is the witness.
5. **§17.5**: Re-run §16.3 (over-training basin + maintenance) at T = 100 d under the timescale-halved parameters. Expected: Φ = (2, 2) still collapses A; Φ = (1, 1) still maintains.

## Code extensions (extend `generate_figures.py`)

New parameter dict: `RECOMMENDED_V2 = make_params(B_dec=0.25, S_dec=0.25, mu_F=0.030, mu_FF=0.020, tau_B=21.0, tau_S=30.0, kappa_B=0.02496, kappa_S=0.01632)`.

New helpers:
- `compute_trajectory_fim(y0, phi_fn, T, dt, p, theta_keys, h=1e-3)`: numerical sensitivities of trajectory w.r.t. selected parameters via central differences. Returns the 6×6 (or however-many-θ-keys) FIM matrix integrated over the trajectory.
- `fim_condition(F)`: condition number κ(F) = λ_max / λ_min from `numpy.linalg.eigh`.
- `fig_fim_grid_search(p, T=100, n_grid=21)`: grid search, returns heat-maps of κ(F) and A(T) over (Φ_B, Φ_S), highlights the top-3 cells under each metric, returns the top-1 in each.
- `fig_witness_v2(p, phi_witness)`: same shape as the existing `fig_sedentary_basin_witness` but with constant Φ = phi_witness instead of bang-bang.

New figures (under same directory):
- `island_recommended_v2.png` — same as `island_recommended.png` but under timescale-halved parameters. Should look qualitatively identical because equilibria are preserved.
- `fim_grid_search.png` — 2-panel figure: κ(F) heat-map vs A(100) heat-map over (Φ_B, Φ_S) grid. Top picks marked. Shows the duality empirically.
- `sedentary_basin_witness_v2.png` — replaces the existing one. Left: passive (0.1, 0.1) → collapse. Right: FIM-selected constant Φ witness → escape with A(100) ≥ A_target.
- `overtraining_basin_witness_v2.png` — replaces the existing one. Same structure as before but with timescale-halved params.

## LaTeX edits

In `sedentary_basin_controllability.tex`, **revise rather than append**:
- §15.3 "Recommended parametrisation" — add a paragraph documenting the v2 update with timescale halving (4 lines).
- §16.1 (pathological sedentary basin) — re-run at T = 100 d, update the numerical result.
- §16.2 — substantial rewrite. New title: "Constructive escape witness via FIM-duality grid search". Body:
  - State the identification-controllability duality used.
  - Describe the 21×21 grid + numerical FIM computation method.
  - Present the FIM-condition heat-map and the A(100) heat-map side-by-side (`fim_grid_search.png`).
  - Identify the winner Φ* under each ranking. Comment on agreement.
  - Verify A(100; Φ*) ≥ A_target.
  - Discuss what makes this LEAN-mechanisable: the winner Φ* is a single constant pair; verification reduces to scalar ODE integration on each component, no piecewise schedule.
- §16.3 — minor update to mention T = 100 d under timescale-halved parameters; results expected to stay qualitatively identical.
- Abstract: add one sentence to the "Extension" paragraph noting the §17 revision (timescale halving + FIM-duality witness).

## Files modified

- [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/generate_figures.py](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/generate_figures.py) — add `RECOMMENDED_V2`, `compute_trajectory_fim`, `fim_condition`, `fig_fim_grid_search`, revised `fig_*_witness_v2`.
- [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/sedentary_basin_controllability.tex](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/sedentary_basin_controllability.tex) — revise §15.3, §16.1, §16.2 (largest change), §16.3, abstract.

## Verification

1. **Equilibrium invariance check** (algebraic): under the (τ, κ) → (τ/2, 2κ) substitution, recompute $\bar\mu(0; (1,1))$. It should equal +0.119 to 3 dp. Print this in the script's start-up sanity check, alongside the canonical μ_init = -0.115.

2. **Trajectory speed-up check**: under the timescale-halved parameters with constant Φ = (1, 1) from SEDENTARY_INIT, B should reach $B_{\rm dec} = 0.25$ in roughly half the time it took before (i.e., ~12 d vs ~25 d). Print this.

3. **FIM-duality plot check**: visually inspect `fim_grid_search.png`. The top-3 Φ cells by κ(F) should overlap the top-3 cells by A(100). If they disagree completely, the duality claim is wrong and the FIM-witness approach should be reported as a negative result.

4. **A(100) escape check**: under the FIM-selected witness Φ*, integrate from SEDENTARY_INIT for T = 100 d. Assert A(100) ≥ 0.30 (the escape target). If it fails, fall back to the A(100)-maximising Φ from the grid.

5. **Recompile + visual check**: pdflatex twice, verify the updated §16.2 reads cleanly and the new figures are referenced correctly.

## Out of scope (deferred)

- A formal mathematical statement of the identification ↔ controllability duality in the FSA-v5 setting. The numerical demonstration in §17 is the primary deliverable; a formal proof would belong in a follow-up section or a separate paper.
- Time-varying Φ optimisation. The §17.2 witness is constant on [0, 100]; better controllability might be achievable with a time-varying Φ but that re-introduces the bang-bang complexity the user wants to avoid.
- LEAN mechanisation of the FIM computation. Numerical sensitivities through an ODE integrator are out of Lean4's current Mathlib reach. Constant-Φ witness verification (integrate ODE under constant Φ, check A(T)) IS in reach.
