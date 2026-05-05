# Faster way to test the SWAT controller — bypass the filter

> Written 2026-05-04 by Claude after Ajay pointed out the obvious. Filed here so future-me does not repeat the mistake.

## What I did this session (the slow way)

Every fix candidate (D1–D6) was tested by re-running the **full closed-loop bench** (`bench_smc_full_mpc_swat.py`) — which advances the plant, runs the SMC² rolling-window filter on the obs, hands the filter's posterior-mean params + smoothed state to the controller, and only THEN plans the next stride. That ate ~22–26 min per T=2 run because the filter is the expensive part (256 outer particles × 200 inner particles × ~10 tempering levels × 9 strides).

That was the wrong tool for the job. The bug was in the controller, not the filter; the filter was infrastructure I was paying for and not learning from.

## What I should have done

The controller only needs two inputs:

- `init_state` — the patient's current latent state (W, Z, a, T)
- `params`     — the dynamics parameters

In a synthetic debugging test we **know both already**. Truth params live in `version_2/models/swat/_dynamics.TRUTH_PARAMS` (re-exported via `simulation.DEFAULT_PARAMS`). The current plant state is just `plant.state` after each `plant.advance(...)`. There is no need to:

- sample observations,
- run the SMC² filter on them,
- fuse Liu-West + bridge across rolling windows,
- extract a smoothed posterior mean.

All of that exists to *recover* (params, state) when we don't know them. We always know them in a controller debug.

## The minimal test loop

```python
from version_2.models.swat._plant import StepwisePlant
from version_2.models.swat.simulation import DEFAULT_PARAMS, DEFAULT_INIT
from version_2.models.swat.control import build_control_spec
from smc2fc.control import SMCControlConfig
from smc2fc.control.tempered_smc_loop import run_tempered_smc_loop_native

plant = StepwisePlant(state=init_state)
truth = dict(DEFAULT_PARAMS)
controls = baseline_daily_schedules                  # whatever start the patient is on
ctrl_cfg = SMCControlConfig(...)                     # same as the full bench

for stride in range(n_strides):
    obs_stride = plant.advance(STRIDE_BINS, *controls)   # obs ignored

    if stride % replan_K == 0:
        spec = build_control_spec(
            n_steps      = plan_horizon * BINS_PER_DAY,
            dt           = DT,
            init_state   = plant.state.copy(),       # ← actual ground-truth current state
            params       = truth,                     # ← truth, no posterior needed
            ...,
        )
        result = run_tempered_smc_loop_native(spec, ctrl_cfg, key=jax.random.PRNGKey(stride))
        controls = decode_schedule(result.mean_schedule)
    # else: reuse previous controls (same as full bench)
```

That's it. No filter, no obs samplers running every stride, no rolling-window bridging, no posterior-extraction code path. The plant + controller are the only moving pieces.

## Estimated speedup

Per-stride cost is roughly:

| Component                     | Closed-loop bench | Controller-only |
|-------------------------------|-------------------|-----------------|
| Plant advance (1 stride)      | ~1 s              | ~1 s            |
| SMC² filter window            | ~125 s            | (skipped)       |
| Controller tempered-SMC plan  | ~0–60 s (every K) | ~0–60 s (every K) |
| Obs sampling, bridge handoff  | ~5 s              | (skipped)       |

T=2 closed-loop bench observed at ~22–26 min per run this session. A controller-only test of the same horizon should land around **5–10 min** — roughly 3× faster, and the saving grows with horizon because the filter cost is per-stride.

## What you GIVE UP by skipping the filter

Real concerns, but each is **downstream** of "does the controller find good plans?":

1. **Filter posterior uncertainty.** The real controller sees a posterior cloud over (params, state), not point estimates. If the controller is sensitive to that uncertainty, controller-only tests miss it.
2. **Posterior bias.** If the filter systematically biases the posterior mean (e.g. prior-pull on slow params), the controller-only test won't catch it.
3. **End-to-end pipeline glitches.** Window indexing, replan timing, bridge handoff bugs only manifest in the full bench.

These all matter for the **final** sign-off run. They do **not** matter when you're trying to figure out whether the controller is mathematically broken (V_c bound, integrator mismatch, prior-centre issues — all of this session's hypotheses). Those failure modes are intrinsic to the controller code and visible at truth-params/truth-state.

## Recommended workflow for the next debugging session

1. **Phase A (static review)** — same as before: read the senior files, find mismatches in `control.py`. Zero GPU.
2. **Phase B/C (controller-only sanity)** — use the minimal loop above with `truth_params` + `DEFAULT_INIT` (or the pathological start). Iterate over fix candidates here. Each test ~5–10 min.
3. **Phase D (full closed-loop)** — only when the controller-only loop passes, run the full SMC²-MPC bench to verify the integrated pipeline.

Saves ~15 min per fix candidate × N candidates × M re-runs. This session would have been 4–6× faster end-to-end.

## Where the code already lives

- Plant: `version_2/models/swat/_plant.py:StepwisePlant`. `advance(stride_bins, V_h_daily, V_n_daily, V_c_daily)` — already callable in isolation. Returns obs but you can ignore them.
- Controller spec builder: `version_2/models/swat/control.py:build_control_spec` — already takes `init_state` and `params` as keyword arguments. Bench passes posterior-mean values; truth values are equally valid.
- Tempered-SMC controller engine: `smc2fc/control/tempered_smc_loop.py:run_tempered_smc_loop_native`. Same signature.
- Schedule decoder: `bench_smc_full_mpc_swat.py` lines ~545–565 already converts `result.mean_schedule` (n_steps, 3) into per-day (V_h, V_n, V_c). Lift that block.

So a `tools/bench_controller_only_swat.py` would be ~150 LoC, mostly lifted from the full bench.

## Key lesson for me

I anchored on "reproduce Run 09 exactly so the comparison is apples-to-apples." That is the right principle for the **final** sign-off, not for **debugging**. Debugging wants the smallest test that still exercises the suspected bug. The filter was not under suspicion; running it on every iteration was paying ~125 s of GPU time per stride for zero diagnostic information.

If a future Claude session is debugging *anything* on the controller side — bounds, integrator, RBF transforms, cost composition, prior structure, tempering schedule — start with the controller-only loop. Promote to the full closed-loop bench only when the controller-only loop passes.
