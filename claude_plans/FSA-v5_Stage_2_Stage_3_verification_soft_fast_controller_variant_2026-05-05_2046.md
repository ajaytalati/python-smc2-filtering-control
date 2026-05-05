# FSA-v5 — Stage 2 / Stage 3 verification + `soft_fast` controller variant

> Archived from plan mode: 2026-05-05 20:46.

## Where we are (1 paragraph)

The model is imported and the filter side is verified. `version_3/models/fsa_v5/` holds the FSA-v5 model files (pinned to `FSA_model_dev` SHA `7075436`). All three bench drivers are written and smoke-tested. **Stage 1 verification has passed**: the rolling-window SMC² filter on synthetic 14-day healthy-island data recovers truth on 24/27 windows (≥30 of 37 estimable params each), all 4 gates green in 47.5 minutes (run05). The bench drivers use the JAX-native compile-once SMC kernel path (matching v2 production) and v5's `propagate_fn` runs cleanly in fp32. **All that remains is Stage 2 and Stage 3 — the controller side.**

Per Ajay's redirect (2026-05-05): the `hard` (pure-SMC² importance-weighted) variant is dropped from testing — not worth the GPU time. Instead Stage 2 and Stage 3 will A/B test the **current `soft`** controller cost against a **new `soft_fast`** variant built per Gemini's optimization plan ([`Geminis_plan_speed_to_speed_up_soft_controller.md`](Geminis_plan_speed_to_speed_up_soft_controller.md)).

---

## Step 1 — Build `soft_fast` (the optimised controller cost)

The current `soft` cost (`evaluate_chance_constrained_cost_soft` in `version_3/models/fsa_v5/control_v5.py`) is the v5 main-novelty objective: a sigmoid surrogate for the basin-escape chance constraint, JAX-jittable, HMC-friendly. It works, but it's slow. Gemini's plan identifies four wall-clock costs that can be trimmed without changing the qualitative behaviour:

1. **Run the cost in fp32, not fp64.** The 5090 has ~64× more fp32 cores than fp64 cores. The cost is currently fp64 throughout (state, params, schedule, weights). Cast everything in the cost path to fp32 — except SMC log-weights and the cost scalar itself, which stay fp64 to avoid catastrophic cancellation in the outer loop.
2. **Loosen the separatrix bisection.** `_jax_find_A_sep` searches for the basin boundary to ~1e-6 precision. The soft sigmoid surrogate already smooths that boundary, so 1e-3 is plenty. Drop max iterations from 50 to 20.
3. **Sub-sample the chance-constraint check.** A moves slowly. Right now we check `A_t < A_sep(Φ_t)` every 15 minutes (1344 evaluations × n_particles per replan). Hourly is enough — 4× fewer evaluations.
4. **Trim HMC inside the controller.** The `soft_fast` bench config drops `num_mcmc_steps` 10 → 5, `hmc_num_leapfrog` 16 → 8, `n_smc` 256 → 128. The soft cost surface is smooth; HMC doesn't need exhaustive integration to find good proposals.

### What gets written

**New file `version_3/models/fsa_v5/control_v5_fast.py`** (about 200 lines). Imports the shared helpers from `control_v5.py` and exposes:

  - `_jax_find_A_sep_fast(...)` — relaxed bisection, 1e-3 tolerance, 20-iter cap.
  - `_compute_cost_internals_fast(...)` — fp32 throughout, sub-samples the chance check every `bin_stride` (default 4) bins.
  - `_cost_soft_fast_jit` — `@jax.jit` wrapper.
  - `evaluate_chance_constrained_cost_soft_fast(...)` — public API, mirrors `evaluate_chance_constrained_cost_soft` plus a `bin_stride=4` kwarg.

I will NOT modify `control_v5.py` — that file is byte-equivalent to upstream `FSA_model_dev/7075436`, and edits there would break provenance for any future re-pin. `control_v5_fast.py` lives alongside it.

**Bench wiring** in both `version_3/tools/bench_controller_only_fsa_v5.py` and `version_3/tools/bench_smc_full_mpc_fsa_v5.py`:

  - Add `--cost soft_fast` as a third option (`soft`, `hard`, `gradient_ot` already exist; `soft_fast` joins them).
  - When `--cost soft_fast`, route the cost-fn factory to `_cost_soft_fast_jit` AND apply the trimmed HMC config (`num_mcmc_steps=5`, `hmc_num_leapfrog=8`, `n_smc=128`).
  - The existing `--cost soft` path is unchanged. That's the baseline we A/B against.

**Three new tests** added to `version_3/tests/test_fsa_v5_bench_smoke.py`:

  - `test_soft_fast_jits` — compiles, runs, returns finite metrics, output is fp32.
  - `test_soft_fast_grad_finite` — `jax.grad` of the cost wrt `theta_ctrl` returns non-zero finite gradients.
  - `test_soft_fast_agrees_with_soft_at_healthy_island` — at trained-athlete + Φ=(0.30, 0.30), `soft_fast` and `soft` agree on `mean_A_integral` and `mean_effort` within 5%. (Both have zero violations at this corner, so the chance-constraint term is identically zero.)

All three must pass before any A/B run on the GPU.

**The `basin_overlay.png` plotter** (used by both Stage 2 and Stage 3 — see Step 2). Add a small helper in `version_3/tools/plot_basin_overlay.py` that takes a run's `trajectory.npz` (or the `applied_phi_per_stride` array directly) and the static contours of Figure 2 from `LaTex_docs/FSA_version_5_technical_guide.tex` (`fig:full-bifurcation`, file `figures/v5_full_bifurcation.png`) and draws the controller's chosen-Φ path overlaid on the basin geometry. Both bench drivers call this helper as their last save step, so the PNG ends up in each `experiments/runNN_<tag>/`. Build this BEFORE the overnight run — otherwise the runs finish without the diagnostic and we have to re-render from saved `trajectory.npz` (annoying but possible).

---

## Step 2 — Stage 2 A/B (controller only, no filter)

Run the controller-only bench twice on the **healthy** scenario (trained-athlete init state, baseline Φ = (0.30, 0.30), T = 14 d):

```
PYTHONPATH=.:.. python tools/bench_controller_only_fsa_v5.py \
    --cost soft      --scenario healthy --T-days 14 --replan-K 2 \
    --run-tag stage2_soft_healthy_T14_baseline

PYTHONPATH=.:.. python tools/bench_controller_only_fsa_v5.py \
    --cost soft_fast --scenario healthy --T-days 14 --replan-K 2 \
    --run-tag stage2_soft_fast_healthy_T14_optimized
```

Each writes its own `experiments/runNN_<tag>/` with manifest, posterior arrays, plots.

### Wall-clock budget — overnight, but actively monitored by me

Ajay measured **`soft` controller-only takes ~100 min per 14-day scenario**. Sweeping all three scenarios (healthy, sedentary, overtrained) for both variants is ~3 × 100 + 3 × ~30 ≈ **6–7 hours just for Step 2**, and Step 3 adds the filter on top. Total ≈ **8–12 hours**. This is an overnight run.

**Ajay has authorised overnight GPU time and expects me to actively monitor the sweep, not fire-and-forget it.** That's my job as the assistant: watch each run, catch problems early, fix them or abort, keep the GPU usefully busy.

Concrete monitoring plan:
1. **During the day**, get all smoke tests (D.1 / D.2 / D.3 from Step 1) green on CPU. Get the basin-overlay plotter built and tested. Get a single end-to-end "tiny" run done (e.g. T = 2 d, healthy scenario, `soft_fast`) to confirm the GPU pipeline starts producing artifacts and the manifest schema is well-formed. THIS catches all the typo-level failures before the overnight sweep.
2. **Build a shell launcher** `version_3/tools/launchers/run_stage23_overnight.sh` that runs the six A/B combinations sequentially: Stage 2 × {soft, soft_fast} × {healthy, sedentary, overtrained}, then Stage 3 × {soft, soft_fast} × healthy. Pipe each run's stdout to its own log under `outputs/fsa_v5/experiments/runNN_<tag>/run.log` so each can be inspected independently. Order them sensibly — `soft_fast` first per scenario so I see the cheap-fast result before the expensive baseline, and a single tiny T=2d sanity-check at the very start of the launcher so the first 10 minutes flag any obvious crash before sinking 8 hours into doomed runs.
3. **Kick off the launcher**, then keep a foreground watcher (`Bash run_in_background=True` plus a Monitor on the per-run log file) active for as long as the conversation stays open. Each run emits a heartbeat (per-window or per-replan id_cov / cost / elapsed_s line) — set the Monitor filter to fire on those plus any `Traceback`, `Error`, `OOM`, `Killed`, `assert` patterns. **Silence is not success** — if a heartbeat hasn't fired in N × expected-stride-time, something is wrong and I should investigate.
4. **When something goes wrong, intervene.** Three failure modes I should expect:
   - **Crash early** (within ~5 min) → fix the bug, re-run the affected combination + everything downstream of it. The other runs in the launcher continue while I debug.
   - **Wall-clock blowout** (run is going to take 3× expected) → kill it, diagnose, decide whether to re-run with reduced settings (smaller `--T-days`, fewer scenarios) or skip.
   - **Numerical garbage** (NaN trajectory, 0% coverage, etc.) → kill it, save the partial output for inspection, decide whether to re-run after a fix or accept the empirical evidence that the variant has a real issue.
5. **As each run finishes**, immediately call `version_3/tools/summarize_run.py outputs/fsa_v5/experiments/runNN_<tag>/` and paste the markdown summary into `outputs/fsa_v5/CHANGELOG.md` as Run 06, 07, 08, …. Don't batch the writeups for next morning — write them per-run so the audit trail is current.
6. **By morning** the CHANGELOG is up to date, the failed runs (if any) have been investigated and either fixed-and-rerun or recorded as informative failures, and Ajay can read the entries cold over coffee and know what happened.

The order of priority if something has to be cut: Stage 2 healthy A/B is non-negotiable (that's the whole point). Stage 2 sedentary + overtrained are nice-to-have if `soft_fast` passes healthy. Stage 3 only if Stage 2 passes — Stage 3 with both variants is the headline result but it costs ~3 h, so I'd only run it if Stage 2 healthy is clean.

### Pass criteria (per scenario)

A/B is acceptable if `soft_fast` is **≥3× faster** than `soft` AND its applied schedule lands **within 5%** of `soft`'s on `mean_A_integral`. The post-hoc `weighted_violation_rate` (re-evaluated with the legacy hard evaluator on the actual plant trajectory) should be ≤ 0.05 (α budget) for both.

If the A/B fails (large divergence between variants), peel back the four optimisation buckets one at a time — drop bucket 2 (full bisection precision), then bucket 4 (full bin resolution) — to identify which step broke things. This diagnostic doesn't need its own overnight run; a single targeted A/B at healthy scenario is enough to localise the bug.

### Plots produced per run

Each run drops these PNGs into its `experiments/runNN_<tag>/` folder:

- **`latent_trajectory.png`** — already implemented. 6-panel plot of B, S, F, A, K_FB, K_FS over time.
- **`applied_schedule.png`** — already implemented. Two stacked panels of Φ_B(t) and Φ_S(t) per stride.
- **`basin_overlay.png`** — NEW, the diagnostic Ajay specifically asked for. Mirrors **Figure 2 of `LaTex_docs/FSA_version_5_technical_guide.pdf`** (`fig:full-bifurcation`, the closed-island bifurcation diagram in (Φ_B, Φ_S) space). On top of the static contours we overlay the controller's chosen schedule as a path:
   - Start point (large filled dot, e.g. green) = the scenario's `baseline_phi` (Stage 2 starting condition before the first replan).
   - Path (line + arrowheads at each replan boundary) = the daily-mean (Φ_B, Φ_S) the controller actually applied, in chronological order across the 14 days.
   - End point (large filled dot, e.g. red) = the final day's daily-mean (Φ_B, Φ_S).
   - **Pass test by eye:** for every scenario, the end point should sit inside the healthy island (the closed contour interior to both axes), regardless of where the start point was. The path should walk INTO the island from outside (sedentary / overtrained start) or stay inside (healthy start).
   - Static contours can be drawn either by re-running the v5 author's bifurcation-sweep script, OR by loading a pre-rendered version of the published figure as a backdrop and overlaying our trajectory. Implementation choice during Step 1 build; the cheaper option is fine.

The `basin_overlay.png` is THE single plot that makes Stage 2 verification verifiable at a glance — anything else is supporting detail.

---

## Step 3 — Stage 3 A/B (full closed-loop MPC)

Same A/B but with the rolling-window SMC² filter in the loop. Only run the **healthy** scenario at first; Stage 3 runs are 40 min – 2 h each so we don't sweep scenarios unless something interesting comes up at healthy.

```
PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa_v5.py \
    --cost soft      --scenario healthy --T-days 14 --replan-K 2 \
    --run-tag stage3_soft_healthy_T14_baseline

PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa_v5.py \
    --cost soft_fast --scenario healthy --T-days 14 --replan-K 2 \
    --run-tag stage3_soft_fast_healthy_T14_optimized
```

**Pass = all four standard gates green for both runs:**
  1. `mean_A_integral ≥ A_target` (= 2.0 by default).
  2. `weighted_violation_rate ≤ α` (= 0.05) on the post-hoc check.
  3. Filter id-cov: posterior 90% CI covers truth on ≥30 of 37 estimable params for ≥22 of the rolling-window posteriors (matches the Stage 1 threshold).
  4. Total compute ≤ 4 hours on RTX 5090.

If Stage 2 passed but Stage 3 fails for `soft` AND `soft_fast` together, the bug is in the integration (filter posterior → controller cost wiring) — not in either variant. If only `soft_fast` fails, that's a real divergence under filter-uncertainty conditions and the optimisation backs off.

Stage 3 produces the same plots as Stage 2 (`latent_trajectory.png`, `applied_schedule.png`, `basin_overlay.png`) plus an extra **`Stage3_param_traces.png`** showing the filter posterior 90% CIs across the 27 windows for all 37 estimable params (already implemented). The `basin_overlay.png` here is even more diagnostic than at Stage 2 — under filter uncertainty the controller's schedule can drift if the posterior posterior-mean state estimate is biased; if the path stays inside the island under filter uncertainty, that's solid empirical evidence the closed-loop pipeline works.

Stage 3 runs add ~3 h on top of Step 2 (one healthy scenario, both variants), so they fit inside the same overnight launcher.

---

## When done

Write a Run 06 / Run 07 / etc. entry to `version_3/outputs/fsa_v5/CHANGELOG.md` for each A/B run, using `version_3/tools/summarize_run.py` to format the markdown. Keep the bench-side changes self-contained:

- `version_3/models/fsa_v5/control_v5_fast.py` (new file, only file added under `models/fsa_v5/`).
- Edits to `version_3/tools/bench_controller_only_fsa_v5.py` + `bench_smc_full_mpc_fsa_v5.py` (add `--cost soft_fast` branch).
- Edits to `version_3/tests/test_fsa_v5_bench_smoke.py` (3 new tests).

`control_v5.py` and the rest of `version_3/models/fsa_v5/` stay byte-equivalent to the upstream pin.
