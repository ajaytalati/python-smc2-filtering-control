# bench/bench_args.jl
#
# CLI parsing for `tools/bench_smc_full_mpc_fsa_gpu.jl`.
#
# Entry point: `_parse_args(argv::Vector{String}) -> Dict{String,Any}`.
#
# Defines the bench's `defaults` dict (every CLI flag + its default value
# + an inline-comment block per group of flags explaining provenance) and
# the `filter_aliases` dict (translates `--filter-<name>` aliases for the
# legacy unprefixed filter-side flags to their canonical keys).
#
# Extracted from the monolithic bench script 2026-05-09 as Phase 1a of the
# refactor described in
# `claude_plans/Refactor_v1_5_bench_into_5_modules_2026-05-09_2003.md`.
# Pure CLI logic — has zero dependencies on the framework, the model,
# or any other bench module. Can be `include`d by any caller that needs
# to parse the bench's CLI flags into a Dict.
#
# Behaviour identical to the prior in-line definition; this is a
# verbatim move, not a refactor of the parser logic.

function _parse_args(argv::Vector{String})
    # ─────────────────────────────────────────────────────────────────────
    #  DEFAULTS = "saturated" config recommended in the technical guide:
    #  compare_v15_julia_vs_python/docs/
    #    technical_guide_to_current_best_Julia_SMC2FC_config.pdf
    #
    #  At these defaults a T=28d run completes in ~19.5 min on RTX 5090,
    #  with mean GPU util ~50%. The saturated controller finds the
    #  long-horizon Banister-overload ramp-up (stride-21 Φ̄ ≈ 0.7).
    #
    #  PRIOR DEFAULTS (kept for reference; they were tuned for the slow
    #  pre-fix Julia path and now leave the GPU at ~38% util):
    #     N-smc=32, K-per-chain=200, ctrl-n-smc=256, ctrl-num-mcmc=8,
    #     ctrl-chees-max=256, ctrl-n-anchors=8, ctrl-max-levels=25.
    #  Use them explicitly via --N-smc 32 etc. for reproducibility of
    #  pre-saturation runs.
    # ─────────────────────────────────────────────────────────────────────
    defaults = Dict{String,Any}(
        # ── Bench / filter ──
        "T-days"          => 14,
        "step-minutes"    => 60,
        "replan-K"        => 2,
        "N-smc"           => 512,    # was 32; saturated default
        "K-per-chain"     => 1000,   # was 200; saturated default
        "num-mcmc"        => 3,
        # Filter HMC step size. Changed 2026-05-09 from 0.05 to 0.005:
        # the standalone HMC accept-rate sweep at
        # `compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/diag_hmc_accept_sweep.csv`
        # showed 0% accept at 0.05, ~50–75% at 0.005 (sweet spot).
        # Rolling-window sweep (julia_horizon_sweep_2026-05-09_filter_hmc_unified)
        # observed 62–71% accept at this default, matching the standalone sweep.
        "hmc-step-size"   => 0.005,
        "hmc-leapfrog"    => 4,
        "max-lambda-inc"  => 0.20,
        "target-ess-frac" => 0.5,
        "max-temp-levels" => 30,
        # Liu-West θ-cloud shrinkage. ON by default at a=0.97 (2026-05-09):
        # without it the parameter cloud collapses by ~day 15 and posterior
        # medians freeze at biased values regardless of horizon length
        # (TIER_0_FINDINGS, H1 confirmed). 0.0 disables (original bootstrap
        # behaviour); 1.0 also a no-op. Formula:
        # θ_i := a·θ_i + (1-a)·θ_mean + √(1-a²)·jitter, applied between
        # systematic resample and HMC at every tempering level.
        "liu-west-a"      => 0.97,
        # Smooth-resample (Silverman-bandwidth KDE + Liu-West correction)
        # — multivariate alternative to the per-dim Liu-West above. When
        # > 0 the per-dim Liu-West is skipped and `smooth_resample` from
        # SMC2FC_functional.Kernels is applied to the θ-cloud. 0.0 = OFF
        # (use the cheaper per-dim path). 1.0 = full Silverman.
        # Changed 2026-05-09 from 0.0 to 1.0: the no-filter-HMC and
        # filter-HMC-unified sweeps both used 1.0; this is the
        # rejuvenation knob the post-2026-05-09 default config relies on.
        "smooth-resample-bw" => 1.0,
        # Multivariate Gaussian bridge between rolling windows. ON by
        # default (2026-05-09): replaces the identity-copy of the
        # previous-window posterior with a sample from N(μ, Σ̂_reg)
        # fitted via Bridge.fit_gaussian (regularised sample covariance,
        # 1e-6 Tikhonov). "true" = on, "false" = identity copy
        # (original bootstrap behaviour).
        "gaussian-bridge" => "true",
        # Collect controller-HMC diagnostics per tempering level
        # (β-ladder, ChEES picker output, accept rate, ESJD, ΔlogD,
        # wall) and write them to controller_diagnostics.csv per run.
        # Default ON for the 2026-05-09 "is filter HMC needed" study.
        "collect-ctrl-diagnostics" => "true",
        # Filter-side FD-gradient step size for the HMC's central-FD
        # gradient. Was hardcoded at 1e-3 in filter_cfg pre-2026-05-09;
        # the standalone HMC accept-rate sweep at
        # `compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/diag_hmc_accept_sweep.csv`
        # shows 1e-2 is the sweet spot for the inner-PF stochastic
        # likelihood (1e-3 falls into the noise-dominated regime).
        "h-fd"            => 1e-2,
        # Filter-side ChEES candidate-L list bounds. Mirrors the
        # controller's `--ctrl-chees-max`. The candidate list is
        # built as powers of 2: [min, 2·min, 4·min, ..., max].
        # Defaults to [4, 8, 16, 32, 64] — shorter trajectories than
        # the controller's [16, 32, 64, 128, 256, 512] because each
        # filter HMC leapfrog step calls into the inner PF (more
        # expensive than the controller's deterministic cost
        # rollout).
        "filter-chees-min" => 4,
        "filter-chees-max" => 64,
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
        "ot-max-weight"   => 0.0,
        "seed"            => 42,
        "output-dir"      => "",
        "open-loop"       => "false",
        # ── Controller (the SMC²-MPC plan-finder) ──
        # Defaults are the SATURATED tier from the technical guide.
        # Controller particle count. Changed 2026-05-09 from 2048
        # (saturated tier) to 1024 (technical guide's "fast-controller"
        # recipe — ~4× wall reduction at the cost of a weaker schedule).
        # Used in both the no-filter-HMC and filter-HMC-unified 2026-05-09
        # sweeps. To recover the saturated default pass --ctrl-n-smc 2048.
        "ctrl-n-smc"      => 1024,
        "ctrl-n-inner"    => 64,          # MC trials per cost evaluation (default unchanged)
        # Controller HMC moves per tempering level. Changed 2026-05-09
        # from 16 (saturated tier) to 8 (fast-controller recipe; matches
        # ctrl-n-smc=1024 above). Pass --ctrl-num-mcmc 16 to recover the
        # saturated default.
        "ctrl-num-mcmc"   => 8,
        "ctrl-hmc-step"   => 0.2,         # base leapfrog step size (overridden adaptively by ChEES)
        "ctrl-hmc-leap"   => 16,          # base leapfrog trajectory length (overridden adaptively by ChEES)
        "ctrl-chees-max"  => 512,         # was 256; saturated default. List = [16, 32, 64, ..., max]
        "ctrl-max-levels" => 40,          # was 25; saturated default
        "ctrl-target-nats" => 8.0,
        "ctrl-sigma-prior" => 1.5,
        # Controller adaptive-tempering targets — the `max_lambda_inc`
        # cap and the ESS-bisection target. Both were hardcoded inside
        # ctrl_cfg pre-2026-05-09; exposed as CLI flags 2026-05-09 to
        # support the controller-exploration study (finer β-ladder).
        # Default 0.20 / 0.5 reproduces pre-flag behaviour exactly.
        "ctrl-max-lambda-inc"   => 0.20,
        "ctrl-target-ess-frac"  => 0.5,
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
    )
    # ── Filter-side flag aliases (added 2026-05-09) ─────────────────────────
    # Historically the bench's filter flags were *unprefixed*
    # (e.g. --N-smc) while controller flags were prefixed (--ctrl-n-smc).
    # That was confusing — at a glance you couldn't tell which SMC²
    # `--N-smc` referred to. The asymmetry is now fixed by accepting
    # `--filter-<name>` aliases for every filter-side flag. The bare
    # name is preserved for back-compat and remains the canonical key
    # in the `defaults` dict; the alias dict below just translates a
    # `--filter-X` flag to its canonical form before lookup.
    #
    # Controller flags keep their `--ctrl-` prefix; bench-level flags
    # (--T-days, --seed, --replan-K, --output-dir, --open-loop,
    # --step-minutes) take no prefix and are unambiguous.
    filter_aliases = Dict{String,String}(
        "filter-n-smc"              => "N-smc",
        "filter-k-per-chain"        => "K-per-chain",
        "filter-num-mcmc"           => "num-mcmc",
        "filter-hmc-step-size"      => "hmc-step-size",
        "filter-hmc-leapfrog"       => "hmc-leapfrog",
        "filter-h-fd"               => "h-fd",
        "filter-max-lambda-inc"     => "max-lambda-inc",
        "filter-target-ess-frac"    => "target-ess-frac",
        "filter-max-temp-levels"    => "max-temp-levels",
        "filter-liu-west-a"         => "liu-west-a",
        "filter-smooth-resample-bw" => "smooth-resample-bw",
        "filter-gaussian-bridge"    => "gaussian-bridge",
        "filter-ot-max-weight"      => "ot-max-weight",
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        startswith(a, "--") || error("Unrecognized arg: $a")
        key = a[3:end]
        # Translate filter-side aliases to their canonical names. After
        # this line, `key` always refers to a key in `defaults`.
        if haskey(filter_aliases, key)
            key = filter_aliases[key]
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
