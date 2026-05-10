# bench/bench_args.jl  (v5)
#
# CLI parsing for `tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl`.
#
# Entry point: `_parse_args(argv::Vector{String}) -> Dict{String,Any}`.
#
# Pure CLI logic — has zero dependencies on the framework, the model,
# or any other bench module. Can be `include`d by any caller that needs
# to parse the bench's CLI flags into a Dict.
#
# Behaviour identical to the prior in-line definition; this is a
# verbatim move, not a refactor of the parser logic.

function _parse_args(argv::Vector{String})
    # ─────────────────────────────────────────────────────────────────────
    #  CLI ARGUMENTS & DEFAULTS
    defaults = Dict{String,Any}(
        # =====================================================================
        # 1. I/O & Environment Args
        # Parameters that control logging, random numbers, and output folders.
        # =====================================================================
        
        # Random seed for reproducing results.
        "seed"            => 42,
        
        # Directory where output files (plots, logs, data) will be saved.
        "output-dir"      => "",
        
        # Enable live TensorBoard monitoring.
        # Live TensorBoard logging. When "true" (also accepts "1"/"yes",
        # mirrors the lenient --open-loop convention) the bench writes
        # per-stride scalars (latent state truth + posterior mean, Φ
        # stride means, 37-param 5/50/95 quantiles, stride wall and
        # tempering levels) to `<output-dir>/tb_events/`. Watch live by
        # running `tensorboard --logdir <SWEEP_ROOT>` from any shell —
        # the bench prints the exact command at startup. Default OFF:
        # keeps prior behaviour and avoids the TBLogger dep load for
        # runs that don't want it.
        "tensorboard"     => "true",
        
        # Collects controller-specific debug info like accept rates and timing.
        # Collect controller-HMC diagnostics per tempering level
        # (β-ladder, ChEES picker output, accept rate, ESJD, ΔlogD,
        # wall) and write them to controller_diagnostics.csv per run.
        # Default ON for the 2026-05-09 "is filter HMC needed" study.
        "collect-ctrl-diagnostics" => "true",

        # =====================================================================
        # 2. Benchmark / Model Args
        # Parameters that control the simulation length, time steps, and initial states.
        # =====================================================================
        
        # Total simulation length in days.
        "T-days"          => 14,
        
        # Length of each time step in minutes.
        # v5 production grid (was 60 in v1.5)
        "step-minutes"    => 15,
        
        # Number of simulation steps before the controller recalculates the plan.
        "replan-K"        => 2,
        
        # Set to "true" to disable active control (calculate a plan but don't apply feedback).
        "open-loop"       => "false",
        
        # Which starting state to use for the simulated person.
        # Plant initial-state preset. The flag value is the literal name of
        # the named-tuple constant in `models/fsa_v5/simulation_v5.jl`.
        #
        #   "TRAINED_ATHLETE_INIT" — const at `models/fsa_v5/simulation_v5.jl:167-174`
        #                            (tech guide §8.1: "canonical test-scenario
        #                             starting point"). DEFAULT.
        #   "DEFAULT_INIT"         — const at `models/fsa_v5/simulation_v5.jl:158-165`
        #                            (tech guide §9.10: "deconditioned but
        #                             otherwise healthy"). Forward-sim from
        #                             this under moderate Φ takes weeks to
        #                             settle.
        #
        # Threaded through: bench plant init, controller open-loop s0,
        # filter last_xhat, the bench-glue first-window fallback, and the
        # inner-PF kernel's NaN-guard (tech guide §6.4).
        "init-preset"     => "TRAINED_ATHLETE_INIT",
        
        # Starting stimulus intensity for "Base Aerobic" (Φ_B) and "Strength" (Φ_S).
        # Initial controller stimulus. Overrides the previously-hardcoded
        # `Φ_B = Φ_S = 1.0`. Threads through TWO sites in the bench:
        #   1. The CLOSED-LOOP initial plan that drives the plant for
        #      strides 1..(replan_K-1) before the controller's first
        #      replan. With default --replan-K 2 that's stride 1 only;
        #      with --replan-K 4 it's strides 1-3. In OPEN-LOOP mode
        #      (--open-loop true) the controller plans the full horizon
        #      up front, so this flag has no effect on the closed-loop
        #      plan in that case.
        #   2. The BASELINE plant rollout (the grey reference curves on
        #      the end-of-run state-traces PNG and the `state/baseline/*`
        #      / `phi/baseline_*` series in TensorBoard). Comparing MPC
        #      vs baseline becomes "MPC vs constant Φ at the user's
        #      chosen level" instead of "MPC vs constant Φ=1".
        # Range: [0, Phi_max] = [0, 3.0]. Default 0.3 
        "init-phi-B"      => 0.3,
        "init-phi-S"      => 0.3,

        # =====================================================================
        # 3. Filter Args
        # Parameters that control the state estimation (understanding what happened).
        # =====================================================================
        
        # Number of outer particles in the filter. More is slower but more accurate.
        "filt-n-smc"           => 32,    
        
        # Number of inner particles for likelihood estimation.
        "filt-k-per-chain"     => 200,   
        
        # Number of MCMC steps taken to rejuvenate the particles.
        "filt-num-mcmc"        => 3,
        
        # Base step size for the filter's MCMC sampler.
        # Filter HMC step size. Changed 2026-05-09 from 0.05 to 0.005:
        # the standalone HMC accept-rate sweep at
        # `compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/diag_hmc_accept_sweep.csv`
        # showed 0% accept at 0.05, ~50–75% at 0.005 (sweet spot).
        # Rolling-window sweep (julia_horizon_sweep_2026-05-09_filter_hmc_unified)
        # observed 62–71% accept at this default, matching the standalone sweep.
        "filt-hmc-step-size"   => 0.005,
        
        # Trajectory length for the filter's MCMC sampler.
        "filt-hmc-leapfrog"    => 4,
        
        # Max permitted temperature jump during adaptive tempering.
        "filt-max-lambda-inc"  => 0.20,
        
        # The fraction of effective sample size to target when tempering.
        "filt-target-ess-frac" => 0.5,
        
        # Max number of tempering levels allowed before the filter stops trying to bridge.
        "filt-max-temp-levels" => 30,
        
        # Strength of particle shrinkage to prevent identical clones (1.0 = off).
        # Liu-West θ-cloud shrinkage. ON by default at a=0.97 (2026-05-09):
        # without it the parameter cloud collapses by ~day 15 and posterior
        # medians freeze at biased values regardless of horizon length
        # (TIER_0_FINDINGS, H1 confirmed). 0.0 disables (original bootstrap
        # behaviour); 1.0 also a no-op. Formula:
        # θ_i := a·θ_i + (1-a)·θ_mean + √(1-a²)·jitter, applied between
        # systematic resample and HMC at every tempering level.
        "filt-liu-west-a"      => 0.97,
        
        # Smoothing bandwidth for rejuvenating particles (multivariate approach).
        # Smooth-resample (Silverman-bandwidth KDE + Liu-West correction)
        # — multivariate alternative to the per-dim Liu-West above. When
        # > 0 the per-dim Liu-West is skipped and `smooth_resample` from
        # SMC2FC_functional.Kernels is applied to the θ-cloud. 0.0 = OFF
        # (use the cheaper per-dim path). 1.0 = full Silverman.
        # Changed 2026-05-09 from 0.0 to 1.0: the no-filter-HMC and
        # filter-HMC-unified sweeps both used 1.0; this is the
        # rejuvenation knob the post-2026-05-09 default config relies on.
        "filt-smooth-resample-bw" => 1.0,
        
        # Adds some gaussian noise between time steps to keep the particle cloud healthy.
        # Multivariate Gaussian bridge between rolling windows. ON by
        # default (2026-05-09): replaces the identity-copy of the
        # previous-window posterior with a sample from N(μ, Σ̂_reg)
        # fitted via Bridge.fit_gaussian (regularised sample covariance,
        # 1e-6 Tikhonov). "true" = on, "false" = identity copy
        # (original bootstrap behaviour).
        "filt-gaussian-bridge" => "true",
        
        # Step size for calculating gradients numerically in the filter.
        # Filter-side FD-gradient step size for the HMC's central-FD
        # gradient. Was hardcoded at 1e-3 in filter_cfg pre-2026-05-09;
        # the standalone HMC accept-rate sweep at
        # `compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/diag_hmc_accept_sweep.csv`
        # shows 1e-2 is the sweet spot for the inner-PF stochastic
        # likelihood (1e-3 falls into the noise-dominated regime).
        "filt-h-fd"            => 1e-2,
        
        # Minimum bound for the adaptive MCMC leapfrog step selector (ChEES).
        # Filter-side ChEES candidate-L list bounds. Mirrors the
        # controller's `--ctrl-chees-max`. The candidate list is
        # built as powers of 2: [min, 2·min, 4·min, ..., max].
        # Defaults to [4, 8, 16, 32, 64] — shorter trajectories than
        # the controller's [16, 32, 64, 128, 256, 512] because each
        # filter HMC leapfrog step calls into the inner PF (more
        # expensive than the controller's deterministic cost
        # rollout).
        "filt-chees-min" => 4,
        
        # Maximum bound for the adaptive MCMC leapfrog step selector (ChEES).
        "filt-chees-max" => 64,
        
        # An emergency rescue feature when the filter gets completely stuck (Warning: VERY slow).
        # ─────────────────────────────────────────────────────────────────
        #  ⚠️  OT (optimal-transport) sigmoid-blend rescue weight ⚠️
        # ─────────────────────────────────────────────────────────────────
        #  DEFAULT: 0.0  (OT rescue DISABLED — fast path).
        #
        #  ⚠️  WARNING — turning this on (e.g. 0.01) causes a
        #      ~12× WALL-TIME SLOWDOWN on this stack at v1.5's config. ⚠️
        #
        #  Why it's so expensive on Julia:
        #    The framework's `gpu_ot_blend_chain!` (in
        #    `julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl:199`) wraps the
        #    OT step in a per-CHAIN Julia for-loop that does
        #
        #        log_w_arr = Array(chain_log_w)   # GPU → CPU sync + memcpy
        #        ... softmax on CPU ...
        #        b_gpu     = CuArray(b_cpu)       # CPU → GPU sync + memcpy
        #
        #    once per chain. With M=32 chains and ~5 segments per
        #    `gpu_log_density` call, that's ~320 PCIe round-trips per
        #    call. Microbench at matched config (RTX 5090, N=32, K=200,
        #    T_steps=24):
        #
        #        ot_max = 0.0   →  median 0.73 ms / call
        #        ot_max = 0.01  →  median  84.78 ms / call    (116×!)
        #
        #    Closed-loop bench at T=2d, N=16, K=100, seed=42:
        #
        #        --ot-max-weight 0.0   →  total wall  12.5 s
        #        --ot-max-weight 0.01  →  total wall 159.5 s   (12.77×)
        #
        #    Numerical effect of disabling OT at v1.5: mean A identical;
        #    7 of 10 posterior parameter medians bit-identical, the other
        #    3 differ < 22% relative inside the IQR band. The Python+JAX
        #    reference (smc2fc.filtering.gk_dpf_v3_lite) runs without an
        #    explicit OT rescue and matches the truth posterior fine.
        #    The writeup §6.3 covers this in detail.
        #
        #  When to turn it back on:
        #    Filter degeneracy at very low ESS / hard obs models. v1.5's
        #    obs (3-channel direct Gaussian on every bin) is informative
        #    enough that OT is unnecessary. v2's obs model is sparser
        #    and may benefit.
        #
        #  How to make it cheap:
        #    Replace the per-chain Julia loop in `gpu_ot_blend_chain!`
        #    with a single batched-across-chains GPU kernel (the Sinkhorn
        #    + barycentric projection are already GPU-resident; only the
        #    softmax + dispatch are on the host). That is framework work
        #    in `julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl`, out of
        #    scope here.
        # ─────────────────────────────────────────────────────────────────
        "filt-ot-max-weight"   => 0.0,

        # =====================================================================
        # 4. Controller Args
        # Parameters that dictate how the controller computes future plans.
        # =====================================================================
        
        # Number of particles evaluating different control plans.
        # Defaults are the SATURATED tier from the technical guide.
        # Controller particle count. Changed 2026-05-09 from 2048
        # (saturated tier) to 1024 (technical guide's "fast-controller"
        # recipe — ~4× wall reduction at the cost of a weaker schedule).
        # Used in both the no-filter-HMC and filter-HMC-unified 2026-05-09
        # sweeps. To recover the saturated default pass --ctrl-n-smc 2048.
        "ctrl-n-smc"      => 1024,
        
        # Number of Monte Carlo trials to compute the cost of a specific plan.
        "ctrl-n-inner"    => 64,          # MC trials per cost evaluation (default unchanged)
        
        # Number of MCMC steps the controller takes to improve its plans.
        # Controller HMC moves per tempering level. Changed 2026-05-09
        # from 16 (saturated tier) to 8 (fast-controller recipe; matches
        # ctrl-n-smc=1024 above). Pass --ctrl-num-mcmc 16 to recover the
        # saturated default.
        "ctrl-num-mcmc"   => 8,
        
        # Base step size for exploring the control plan landscape.
        "ctrl-hmc-step"   => 0.2,         # base leapfrog step size (overridden adaptively by ChEES)
        
        # Base trajectory length for exploring the control plan landscape.
        "ctrl-hmc-leap"   => 16,          # base leapfrog trajectory length (overridden adaptively by ChEES)
        
        # Max permitted MCMC steps during adaptive length selection.
        "ctrl-chees-max"  => 512,         # was 256; saturated default. List = [16, 32, 64, ..., max]
        
        # Max tempering levels allowed for the controller to form a plan.
        "ctrl-max-levels" => 40,          # was 25; saturated default
        
        # Desired number of natural bits of entropy (diversity) in the plan particles.
        "ctrl-target-nats" => 8.0,
        
        # Spread (variance) of the initial random guess for future plans.
        "ctrl-sigma-prior" => 1.5,
        
        # Max permitted temperature jump during controller tempering.
        # Controller adaptive-tempering targets — the `max_lambda_inc`
        # cap and the ESS-bisection target. Both were hardcoded inside
        # ctrl_cfg pre-2026-05-09; exposed as CLI flags 2026-05-09 to
        # support the controller-exploration study (finer β-ladder).
        # Default 0.20 / 0.5 reproduces pre-flag behaviour exactly.
        "ctrl-max-lambda-inc"   => 0.20,
        
        # Target effective sample size fraction during controller tempering.
        "ctrl-target-ess-frac"  => 0.5,
        
        # The number of "knots" or resolution points used to construct the plan schedule.
        # ─────────────────────────────────────────────────────────────────
        #  ctrl-n-anchors — RBF basis cardinality on the controller side
        # ─────────────────────────────────────────────────────────────────
        #  DEFAULT: 12 anchors (saturated tier; was 8 historically).
        #
        #  Bumping this gives the controller a richer schedule basis:
        #  more degrees of freedom for the per-bin Φ schedule, finer
        #  daily resolution. The RBF coefficients θ_ctrl are the
        #  outer-SMC² decision variable, so n_anchors is also the
        #  posterior dimension on the control side.
        #
        #  Wall-time scaling is roughly O(n_anchors²) over the full
        #  controller SMC² run, because:
        #    1. each cost-kernel thread loops over n_anchors to decode
        #       Φ(t) (linear in n_anchors per thread), AND
        #    2. the FD-gradient batch packs M chains × (1 + 2·n_anchors)
        #       perturbations into one kernel call, so the gradient
        #       batch grows with n_anchors too.
        #
        #  GPU memory scales linearly: M_max = ctrl_n_smc × (1 + 2·n_anchors).
        #  With ctrl-n-smc=256 and n_anchors=8 → M_max = 4,352. With
        #  n_anchors=16 → M_max = 8,448. The cost_per_thread buffer is
        #  (M_max × n_inner) Float32 = 4 bytes per cell, so VRAM at
        #  defaults is ~1 MB; even at the maxed config it stays under
        #  10 MB. Memory is not the binding constraint; per-thread
        #  compute is.
        #
        #  Recommended bump for richer schedules: 8 → 12 (modest) or
        #  8 → 16 (aggressive). Beyond 16, FD-gradient noise per
        #  dimension may dominate the signal — diminishing returns.
        # ─────────────────────────────────────────────────────────────────
        "ctrl-n-anchors"  => 12,

        # ─────────────────────────────────────────────────────────────────
        #  Controller Prior Center (Φ_default)
        # ─────────────────────────────────────────────────────────────────
        #  DEFAULT: 0.3 (Baseline training load).
        #
        #  The starting stimulus intensity for the controller's search.
        #  The RBF coefficients θ optimize deviations *relative* to
        #  this value.
        #
        #  Rationale:
        #    - FSA-v5 Interpretation: Φ=0.3 is the steady-state baseline.
        #      Φ=1.0 is considered heavy overtraining.
        #    - Search Efficiency: Centering the search on 0.3 allows the
        #      SMC particles to explore healthy regions immediately.
        #
        #  Usage in Code:
        #    Converted to logit-bias `c_Phi` in `gpu_control_v5.jl:292`.
        #    If θ=0, the resulting plan is a flat line at Φ_default.
        # ─────────────────────────────────────────────────────────────────
        "ctrl-phi-default" => 0.3,

        # ─────────────────────────────────────────────────────────────────
        #  Effort Penalty Coefficient (λ_Φ)
        # ─────────────────────────────────────────────────────────────────
        #  DEFAULT: 1.0 (Enabled).
        #
        #  Controls the cost of training intensity. The penalty term is:
        #    J_effort = λ_Φ · ∫ (Φ_B² + Φ_S²) dt
        #
        #  Rationale:
        #    - Metabolic Price: Doubling intensity costs 4× more effort.
        #    - Regularization: Prevents the controller from pinning Φ
        #      to Phi_max (3.0) indefinitely.
        #    - Consistency: Encourages steady, moderate training over
        #      erratic high-intensity spikes.
        #
        #  Usage in Code:
        #    Integrated in `gpu_control_v5.jl:140` and weighted in
        #    `gpu_control_v5.jl:214`. High values (e.g. 10.0) force
        #    sedentary plans; 0.0 allows "infinite effort" training.
        # ─────────────────────────────────────────────────────────────────
        "ctrl-lam-phi"    => 1.0,

        # ─────────────────────────────────────────────────────────────────
        #  Soft Fatigue Penalty Coefficient (λ_F)
        # ─────────────────────────────────────────────────────────────────
        #  DEFAULT: 0.0 (Disabled).
        #
        #  Controls the ad-hoc penalty for exceeding F_max:
        #    J_fatigue = λ_F · ∫ max(F - F_max, 0)² dt
        #
        #  Rationale:
        #    - Legacy Guardrail: In the simple Banister model, this was
        #      required to keep training volume realistic.
        #    - Endogenous v5 Penalty: In FSA-v5, high fatigue naturally
        #      destroys reward (A) and increases future fatigue gains (K).
        #      The ad-hoc penalty is redundant and disabled by default.
        #
        #  Usage in Code:
        #    Integrated in `gpu_control_v5.jl:142` and weighted in
        #    `gpu_control_v5.jl:215`. Set to 1.0 to restore legacy
        #    behavior.
        # ─────────────────────────────────────────────────────────────────
        "ctrl-lam-f"      => 0.0,

        # ─────────────────────────────────────────────────────────────────
        #  Soft Chance-Constraint Coefficient (λ_chance)
        # ─────────────────────────────────────────────────────────────────
        #  DEFAULT: 1.0 (Enabled).
        #
        #  Controls the penalty for approaching the collapse boundary
        #  (low Alertness A). The soft surrogate term is:
        #    J_chance = λ_chance · ∫ σ(β · (A_thr - A) / scale) dt
        #
        #  Rationale:
        #    - Safety: Penalizes trajectories that drift into the
        #      unstable region where A < A_thr.
        #    - Differentiability: Uses a sigmoid (σ) relaxation rather
        #      than a hard indicator function to allow HMC gradients.
        #
        #  Usage in Code:
        #    Implemented in `gpu_control_v5.jl:143` and weighted in
        #    `gpu_control_v5.jl:216`. A_thr defaults to 0.05
        #    (separatrix approximation).
        # ─────────────────────────────────────────────────────────────────
        "ctrl-lam-chance" => 1.0,
    )
    
    # ── Filter-side flag aliases (updated) ─────────────────────────
    # We now use `--filt-` as the canonical prefix for all filter arguments
    # (e.g., `--filt-n-smc`). The `filter_aliases` dict translates both old
    # unprefixed names (`--N-smc`) and old `--filter-` names (`--filter-n-smc`)
    # into the new canonical `--filt-` keys for backward compatibility.
    filter_aliases = Dict{String,String}(
        # Old exact matches (unprefixed)
        "n-smc"              => "filt-n-smc",
        "k-per-chain"        => "filt-k-per-chain",
        "num-mcmc"           => "filt-num-mcmc",
        "hmc-step-size"      => "filt-hmc-step-size",
        "hmc-leapfrog"       => "filt-hmc-leapfrog",
        "max-lambda-inc"     => "filt-max-lambda-inc",
        "target-ess-frac"    => "filt-target-ess-frac",
        "max-temp-levels"    => "filt-max-temp-levels",
        "liu-west-a"         => "filt-liu-west-a",
        "smooth-resample-bw" => "filt-smooth-resample-bw",
        "gaussian-bridge"    => "filt-gaussian-bridge",
        "ot-max-weight"      => "filt-ot-max-weight",
        "h-fd"               => "filt-h-fd",
        
        # Old 'filter-' prefixed aliases
        "filter-n-smc"              => "filt-n-smc",
        "filter-k-per-chain"        => "filt-k-per-chain",
        "filter-num-mcmc"           => "filt-num-mcmc",
        "filter-hmc-step-size"      => "filt-hmc-step-size",
        "filter-hmc-leapfrog"       => "filt-hmc-leapfrog",
        "filter-h-fd"               => "filt-h-fd",
        "filter-max-lambda-inc"     => "filt-max-lambda-inc",
        "filter-target-ess-frac"    => "filt-target-ess-frac",
        "filter-max-temp-levels"    => "filt-max-temp-levels",
        "filter-liu-west-a"         => "filt-liu-west-a",
        "filter-smooth-resample-bw" => "filt-smooth-resample-bw",
        "filter-gaussian-bridge"    => "filt-gaussian-bridge",
        "filter-ot-max-weight"      => "filt-ot-max-weight",
        "filter-chees-min"          => "filt-chees-min",
        "filter-chees-max"          => "filt-chees-max",
    )
    
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("Unrecognized arg: $a")
        key = a[3:end]
        
        key_lc = lowercase(key)
        if haskey(filter_aliases, key_lc)
            key = filter_aliases[key_lc]
        end
        
        haskey(defaults, key) || error("Unknown flag: $a")
        i += 1
        v = argv[i]
        if defaults[key] isa Int
            defaults[key] = parse(Int, v)
        elseif defaults[key] isa Float64
            defaults[key] = parse(Float64, v)
        else
            defaults[key] = v
        end
        i += 1
    end
    return defaults
end