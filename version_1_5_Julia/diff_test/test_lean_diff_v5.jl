# Julia v5 ↔ Lean4 v5 differential test.
#
# Runs every Lean-exposed v5 function on both sides and asserts
# |Δ| < 1e-6 (single-step) / 1e-4 (integrated EM step).
#
# Mirrors `test_lean_diff_v15.jl`'s structure: long-lived `fsa_v5_cli`
# subprocess, JSON-on-stdin / JSON-on-stdout protocol, pre-drawn noise
# so RNG-divergence is excluded from the comparison.
#
# Prereq: build the Lean4 binary first.
#   cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN
#   lake build
#
# Run:
#   cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia
#   julia --project=. diff_test/test_lean_diff_v5.jl
#
# Coverage (15 testsets):
#   sigmoid, c_phi, drift_v5, diffusion_v5, em_step_v5, hr_mean,
#   sleep_prob, stress_mean, steps_log_mean, volume_load_mean,
#   mu_bar, find_a_sep (3-way: -Inf / +Inf / finite),
#   a_sep_grid (shape + values), schedule_from_theta, design_matrix.

using Test
using JSON3
using Random
using StaticArrays

# Load v5 Julia model files (script-style project, not a package).
const _V5_DIR = joinpath(@__DIR__, "..", "models", "fsa_v5")
include(joinpath(_V5_DIR, "FSAv5.jl"))

using .FSAv5

const SINGLE_STEP_TOL = 1e-6
const INTEGRATED_TOL  = 1e-4

const LEAN_BIN = joinpath(@__DIR__, "..", "..", "version_1_5_LEAN",
                           ".lake", "build", "bin", "fsa_v5_cli")

# ── Long-lived subprocess client ──────────────────────────────────────
mutable struct LeanClient
    io::Base.AbstractPipe
end

function open_lean_client()
    isfile(LEAN_BIN) || error("LEAN binary not found at $LEAN_BIN — run " *
                               "`lake build` in version_1_5_LEAN/ first.")
    io = open(`$LEAN_BIN`, "r+")
    return LeanClient(io)
end

close!(c::LeanClient) = (try; close(c.io); catch; end)

function round_trip(c::LeanClient, payload::Dict)
    s = JSON3.write(payload; allow_inf = true)
    println(c.io, s)
    flush(c.io)
    line = readline(c.io)
    isempty(line) && error("LEAN binary produced no output for: $s")
    return JSON3.read(line; allow_inf = true)
end

# ── JSON encoders for v5 types ────────────────────────────────────────
# Order doesn't matter for objects (Lean parses by key) but we use
# the canonical key lists from SimulationV5 for readability.

function _params_payload(p::Dict{Symbol, <:Real})
    Dict{String, Float64}(String(k) => Float64(p[k]) for k in PARAM_KEYS_V5)
end

function _obs_params_payload(op::Dict{Symbol, <:Real})
    Dict{String, Float64}(String(k) => Float64(op[k]) for k in OBS_PARAM_KEYS_V5)
end

function _state_payload(y::AbstractVector)
    [Float64(y[i]) for i in 1:6]
end

function _phi_payload(phi::Tuple)
    Dict{String, Float64}("Phi_B" => Float64(phi[1]), "Phi_S" => Float64(phi[2]))
end

# Random 6D state inside a sensible physiological-bound box:
# B, S ∈ [0.05, 0.95]; F, A, K_FB, K_FS ∈ [0.05, 1.0].
function _random_state(rng)
    SVector{6, Float64}(
        0.05 + 0.90 * rand(rng),  # B
        0.05 + 0.90 * rand(rng),  # S
        0.05 + 0.95 * rand(rng),  # F
        0.05 + 0.95 * rand(rng),  # A
        0.05 + 0.95 * rand(rng),  # K_FB
        0.05 + 0.95 * rand(rng),  # K_FS
    )
end

function _random_phi(rng)
    (3.0 * rand(rng), 3.0 * rand(rng))   # both in [0, 3]
end

# Pre-drawn 6-vector standard-normal noise.
function _random_noise6(rng)
    SVector{6, Float64}(randn(rng) for _ in 1:6)
end


# ── Test cases ────────────────────────────────────────────────────────

@testset "FSA v5 LEAN4 diff" begin
    rng = MersenneTwister(42)
    client = open_lean_client()
    try
        params_payload     = _params_payload(TRUTH_PARAMS_V5)
        obs_params_payload = _obs_params_payload(DEFAULT_OBS_PARAMS_V5)

        # ── sigmoid ────────────────────────────────────────────────────
        @testset "sigmoid" begin
            for x in [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
                resp = round_trip(client, Dict("fn" => "sigmoid", "x" => x))
                jl = sigmoid(x)
                @test isapprox(jl, Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
            for _ in 1:5
                x = 4.0 * randn(rng)
                resp = round_trip(client, Dict("fn" => "sigmoid", "x" => x))
                @test isapprox(sigmoid(x), Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        # ── softChancePenalty ─────────────────────────────────────────
        @testset "softChancePenalty" begin
            for trial in 1:5
                val   = 2.0 * rand(rng)
                thr   = 0.5 + rand(rng)
                beta  = 50.0
                scale = 0.1
                req = Dict("fn"    => "softChancePenalty",
                            "val"   => val,
                            "thr"   => thr,
                            "beta"  => beta,
                            "scale" => scale)
                resp = round_trip(client, req)
                # Julia version from gpu_control_v5.jl:
                # soft_X · (X_thr - X)²  where
                #   soft_X = 1f0 / (1f0 + exp(-beta * (thr - val) / scale))
                # (was the bare sigmoid; multiplied by (thr-val)² so the
                # penalty grows with violation depth — matches Lean
                # Cost.lean::softChancePenalty.)
                d  = thr - val
                jl = (1.0 / (1.0 + exp(-beta * d / scale))) * d * d
                @test isapprox(jl, Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        # ── c_phi ──────────────────────────────────────────────────────
        @testset "c_phi" begin
            for (pd, pm) in [(0.5, 1.0), (1.0, 3.0), (0.1, 1.0), (2.0, 5.0)]
                resp = round_trip(client, Dict("fn" => "cPhi",
                                                 "phi_default" => pd,
                                                 "phi_max" => pm))
                jl = c_phi(pd, pm)
                @test isapprox(jl, Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        # ── drift ──────────────────────────────────────────────────────
        @testset "drift_v5" begin
            for _ in 1:5
                y   = _random_state(rng)
                phi = _random_phi(rng)
                req = Dict("fn"     => "drift",
                            "state"  => _state_payload(y),
                            "phi"    => _phi_payload(phi),
                            "params" => params_payload)
                resp = round_trip(client, req)
                jl   = drift_v5(y, TRUTH_PARAMS_V5, phi)
                lean = Float64.(resp.deriv)
                for i in 1:6
                    @test isapprox(jl[i], lean[i];
                                    atol = SINGLE_STEP_TOL,
                                    rtol = SINGLE_STEP_TOL)
                end
            end
        end

        # ── diffusion ──────────────────────────────────────────────────
        @testset "diffusion_v5" begin
            for _ in 1:5
                y   = _random_state(rng)
                req = Dict("fn"     => "diffusion",
                            "state"  => _state_payload(y),
                            "params" => params_payload)
                resp = round_trip(client, req)
                jl   = diffusion_v5(y, TRUTH_PARAMS_V5)
                lean = Float64.(resp.sigma)
                for i in 1:6
                    @test isapprox(jl[i], lean[i]; atol = SINGLE_STEP_TOL)
                end
            end
        end

        # ── em_step ────────────────────────────────────────────────────
        @testset "em_step_v5" begin
            dt = 1.0 / 96.0   # 15-min bin
            sigma_diag_jl = SVector{6, Float64}(
                TRUTH_PARAMS_V5[:sigma_B], TRUTH_PARAMS_V5[:sigma_S],
                TRUTH_PARAMS_V5[:sigma_F], TRUTH_PARAMS_V5[:sigma_A],
                TRUTH_PARAMS_V5[:sigma_K], TRUTH_PARAMS_V5[:sigma_K],
            )
            sigma_diag_payload = Float64[sigma_diag_jl...]
            for _ in 1:5
                y     = _random_state(rng)
                phi   = _random_phi(rng)
                noise = _random_noise6(rng)
                req = Dict(
                    "fn"        => "emStep",
                    "state"     => _state_payload(y),
                    "phi"       => _phi_payload(phi),
                    "params"    => params_payload,
                    "sigmaDiag" => sigma_diag_payload,
                    "dt"        => dt,
                    "noise"     => Float64[noise...],
                )
                resp = round_trip(client, req)
                jl   = em_step_v5(y, phi, TRUTH_PARAMS_V5, sigma_diag_jl, dt, noise)
                lean = Float64.(resp.next_state)
                for i in 1:6
                    @test isapprox(jl[i], lean[i];
                                    atol = INTEGRATED_TOL,
                                    rtol = SINGLE_STEP_TOL)
                end
            end
        end

        # ── obs-channel means ─────────────────────────────────────────
        @testset "hr_mean" begin
            for _ in 1:5
                y    = _random_state(rng)
                C    = cos(2π * rand(rng))
                req  = Dict("fn"         => "hrMean",
                             "state"      => _state_payload(y),
                             "C"          => C,
                             "obs_params" => obs_params_payload)
                resp = round_trip(client, req)
                @test isapprox(hr_mean(y, C, DEFAULT_OBS_PARAMS_V5),
                                Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        @testset "sleep_prob" begin
            for _ in 1:5
                y    = _random_state(rng)
                C    = cos(2π * rand(rng))
                req  = Dict("fn"         => "sleepProb",
                             "state"      => _state_payload(y),
                             "C"          => C,
                             "obs_params" => obs_params_payload)
                resp = round_trip(client, req)
                @test isapprox(sleep_prob(y, C, DEFAULT_OBS_PARAMS_V5),
                                Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        @testset "stress_mean" begin
            for _ in 1:5
                y    = _random_state(rng)
                C    = cos(2π * rand(rng))
                req  = Dict("fn"         => "stressMean",
                             "state"      => _state_payload(y),
                             "C"          => C,
                             "obs_params" => obs_params_payload)
                resp = round_trip(client, req)
                @test isapprox(stress_mean(y, C, DEFAULT_OBS_PARAMS_V5),
                                Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        @testset "steps_log_mean" begin
            for _ in 1:5
                y    = _random_state(rng)
                C    = cos(2π * rand(rng))
                req  = Dict("fn"         => "stepsLogMean",
                             "state"      => _state_payload(y),
                             "C"          => C,
                             "obs_params" => obs_params_payload)
                resp = round_trip(client, req)
                @test isapprox(steps_log_mean(y, C, DEFAULT_OBS_PARAMS_V5),
                                Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        @testset "volume_load_mean" begin
            for _ in 1:5
                y    = _random_state(rng)
                req  = Dict("fn"         => "volumeLoadMean",
                             "state"      => _state_payload(y),
                             "obs_params" => obs_params_payload)
                resp = round_trip(client, req)
                @test isapprox(volume_load_mean(y, DEFAULT_OBS_PARAMS_V5),
                                Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        # ── mu_bar ─────────────────────────────────────────────────────
        @testset "mu_bar" begin
            for _ in 1:5
                A   = 0.05 + 0.95 * rand(rng)
                phi = _random_phi(rng)
                req = Dict("fn"     => "muBar",
                            "A"      => A,
                            "phi"    => _phi_payload(phi),
                            "params" => params_payload)
                resp = round_trip(client, req)
                jl = mu_bar(A, phi, TRUTH_PARAMS_V5)
                @test isapprox(jl, Float64(resp.x);
                                atol = SINGLE_STEP_TOL,
                                rtol = SINGLE_STEP_TOL)
            end
        end

        # ── find_a_sep — three-way regime test (-Inf / +Inf / finite) ─
        @testset "find_a_sep" begin
            # Per tech guide §4.5: balanced moderate Φ=(0.30,0.30) is healthy → -Inf
            phi_healthy = (0.30, 0.30)
            resp = round_trip(client,
                Dict("fn" => "findASep",
                      "phi" => _phi_payload(phi_healthy),
                      "params" => params_payload))
            jl_healthy = find_a_sep(phi_healthy, TRUTH_PARAMS_V5)
            @test isinf(jl_healthy) && jl_healthy < 0
            @test isinf(Float64(resp.x)) && Float64(resp.x) < 0
            @test sign(jl_healthy) == sign(Float64(resp.x))

            # Per tech guide §4.5: aggressive Φ=(1.0,1.0) is collapsed → +Inf or
            # large negative A_sep depending on saddle-node geometry. The Lean
            # and Julia codepaths agree by construction; we just check parity.
            phi_collapse = (1.0, 1.0)
            resp_c = round_trip(client,
                Dict("fn" => "findASep",
                      "phi" => _phi_payload(phi_collapse),
                      "params" => params_payload))
            jl_c   = find_a_sep(phi_collapse, TRUTH_PARAMS_V5)
            lean_c = Float64(resp_c.x)
            if isinf(jl_c) || isinf(lean_c)
                @test isinf(jl_c) && isinf(lean_c) && sign(jl_c) == sign(lean_c)
            else
                @test isapprox(jl_c, lean_c;
                                atol = SINGLE_STEP_TOL,
                                rtol = SINGLE_STEP_TOL)
            end

            # Three random Φ in the bistable annulus / boundary region.
            for _ in 1:3
                phi = (0.4 + 0.5 * rand(rng), 0.05 + 0.3 * rand(rng))
                resp = round_trip(client,
                    Dict("fn" => "findASep",
                          "phi" => _phi_payload(phi),
                          "params" => params_payload))
                jl   = find_a_sep(phi, TRUTH_PARAMS_V5)
                lean = Float64(resp.x)
                if isinf(jl) || isinf(lean)
                    @test isinf(jl) && isinf(lean) && sign(jl) == sign(lean)
                else
                    @test isapprox(jl, lean;
                                    atol = SINGLE_STEP_TOL,
                                    rtol = SINGLE_STEP_TOL)
                end
            end
        end

        # ── a_sep_grid — shape and per-cell values ─────────────────────
        @testset "a_sep_grid" begin
            # 3 particles × 4 schedule bins. The 3 particles are slight
            # perturbations of TRUTH_PARAMS_V5 (so each gets its own
            # separator under its own params, exercising the
            # Bug-2-prevention signature).
            particles = [
                copy(TRUTH_PARAMS_V5),
                let p = copy(TRUTH_PARAMS_V5); p[:eta] *= 1.05; p end,
                let p = copy(TRUTH_PARAMS_V5); p[:mu_dec_B] *= 1.10; p end,
            ]
            schedule = [(0.30, 0.30), (1.0, 1.0), (0.50, 0.30), (0.20, 0.40)]

            req = Dict(
                "fn"        => "aSepGrid",
                "particles" => [_params_payload(p) for p in particles],
                "schedule"  => [_phi_payload(s) for s in schedule],
            )
            resp = round_trip(client, req)
            lean_mat = [Float64.(row) for row in resp.matrix]
            jl_mat   = a_sep_grid(particles, schedule)

            @test length(lean_mat) == 3
            @test all(length.(lean_mat) .== 4)
            @test size(jl_mat) == (3, 4)
            for i in 1:3, j in 1:4
                jl = jl_mat[i, j]
                ln = lean_mat[i][j]
                if isinf(jl) || isinf(ln)
                    @test isinf(jl) && isinf(ln) && sign(jl) == sign(ln)
                else
                    @test isapprox(jl, ln;
                                    atol = SINGLE_STEP_TOL,
                                    rtol = SINGLE_STEP_TOL)
                end
            end
        end

        # ── schedule_from_theta ───────────────────────────────────────
        @testset "schedule_from_theta" begin
            n_steps   = 24
            n_anchors = 4
            dt        = 1.0 / 96.0
            phi_max   = 3.0
            cphi_val  = c_phi(1.0, phi_max)

            # Build the design matrix on the Lean side too so both sides
            # use exactly the same input.
            dm_resp = round_trip(client, Dict(
                "fn"           => "designMatrix",
                "n_steps"      => n_steps,
                "dt"           => dt,
                "n_anchors"    => n_anchors,
                "width_factor" => 1.0,
            ))
            phi_design_lean = [Float64.(row) for row in dm_resp.matrix]
            phi_design_jl   = design_matrix(n_steps, dt, n_anchors, 1.0)

            # First check design_matrix parity (covers the design_matrix testset too).
            # Design-matrix entries are in [0, 1] so 1e-6 atol is fine.
            for t in 1:n_steps, a in 1:n_anchors
                @test isapprox(phi_design_jl[t, a],
                                phi_design_lean[t][a];
                                atol = SINGLE_STEP_TOL)
            end

            # Schedule outputs scale by phi_max = 3, putting them at
            # magnitude up to 3. Lean's `Float.toString` emits 6 significant
            # digits, so the wire-format precision floor is ~1e-6 relative
            # (i.e. ~3e-6 absolute at magnitude 3). Use rtol=1e-5 for the
            # schedule-output checks — testing tighter is testing Lean's
            # decimal printer, not the math. The math itself is the same
            # `sigmoid` already verified at 1e-6 above.
            SCHEDULE_TOL = 1e-5
            for trial in 1:3
                theta = randn(rng, 2 * n_anchors)
                req = Dict(
                    "fn"        => "scheduleFromTheta",
                    "theta"     => collect(theta),
                    "phiDesign" => [collect(phi_design_lean[t]) for t in 1:n_steps],
                    "cPhi"      => cphi_val,
                    "phiMax"    => phi_max,
                    "n_anchors" => n_anchors,
                )
                resp = round_trip(client, req)
                lean_sched = [(Float64(p.Phi_B), Float64(p.Phi_S)) for p in resp.schedule]

                jl_sched = schedule_from_theta(theta, phi_design_jl,
                                                  cphi_val, phi_max, n_anchors)
                @test length(jl_sched) == length(lean_sched) == n_steps
                for t in 1:n_steps
                    @test isapprox(jl_sched[t][1], lean_sched[t][1];
                                    atol = SCHEDULE_TOL,
                                    rtol = SCHEDULE_TOL)
                    @test isapprox(jl_sched[t][2], lean_sched[t][2];
                                    atol = SCHEDULE_TOL,
                                    rtol = SCHEDULE_TOL)
                end
            end
        end

        # design_matrix is implicitly tested above via schedule_from_theta;
        # an explicit standalone testset is omitted to avoid duplication.

        # ── GPU kernel parity (Phase 4) ───────────────────────────────────
        #
        # The CPU functions `em_step_v5` and `obs_log_weight_v5` are
        # diff-tested against Lean above at 1e-6 (fp64). The GPU kernel in
        # `gpu_pf_v5.jl::propagate_segment_kernel_v5!` re-implements the
        # same math in fp32. Calling the kernel for ndrange=1 via the
        # debug entry point `gpu_propagate_one!` lets us compare the GPU
        # output against the CPU reference at fp32 precision (~1e-4
        # relative). Same idea for `gpu_cost_one!` in the controller.

        # Tolerance for fp32 GPU vs fp64 CPU comparisons.
        # ~1e-4 absolute is what fp32 round-off + sqrt + exp accumulate to
        # over a single bin; tighter would be testing fp32, not the math.
        GPU_FP32_TOL = 1e-3

        @testset "gpu_pf_v5 single-thread parity" begin
            # Use truth params (so all 15 estimated dynamics + 22 estimated
            # obs entries come from the same dicts the CPU uses).
            dyn_keys = (:tau_B, :kappa_B, :epsilon_AB,
                         :tau_S, :kappa_S, :epsilon_AS,
                         :tau_F, :lambda_A,
                         :mu_K,
                         :mu_0, :mu_B, :mu_S, :mu_F, :mu_FF, :eta)
            params_dyn = [TRUTH_PARAMS_V5[k] for k in dyn_keys]
            params_obs = [DEFAULT_OBS_PARAMS_V5[k] for k in OBS_PARAM_KEYS_V5]
            frozen = (
                KFB_0    = FROZEN_PARAMS_V5[:KFB_0],
                KFS_0    = FROZEN_PARAMS_V5[:KFS_0],
                tau_K    = FROZEN_PARAMS_V5[:tau_K],
                B_dec    = FROZEN_PARAMS_V5[:B_dec],
                S_dec    = FROZEN_PARAMS_V5[:S_dec],
                mu_dec_B = FROZEN_PARAMS_V5[:mu_dec_B],
                mu_dec_S = FROZEN_PARAMS_V5[:mu_dec_S],
                sigma_B  = FROZEN_PARAMS_V5[:sigma_B],
                sigma_S  = FROZEN_PARAMS_V5[:sigma_S],
                sigma_F  = FROZEN_PARAMS_V5[:sigma_F],
                sigma_A  = FROZEN_PARAMS_V5[:sigma_A],
                sigma_K  = FROZEN_PARAMS_V5[:sigma_K],
            )

            for trial in 1:3
                state0  = collect(_random_state(rng))
                phi     = _random_phi(rng)
                noise   = collect(_random_noise6(rng))
                C       = cos(2π * rand(rng))
                # All gates ON so every channel contributes.
                gates   = ones(Float64, 5)
                # Random observations consistent with the state at typical
                # physiological scales (so the log-weight magnitudes are
                # reasonable and we don't hit floating-point cancellation).
                y_after_drift = collect(state0)
                obs_HR    = hr_mean(y_after_drift, C, DEFAULT_OBS_PARAMS_V5)        + 0.5 * randn(rng)
                obs_S     = stress_mean(y_after_drift, C, DEFAULT_OBS_PARAMS_V5)    + 0.5 * randn(rng)
                obs_steps = steps_log_mean(y_after_drift, C, DEFAULT_OBS_PARAMS_V5) + 0.05 * randn(rng)
                obs_VL    = volume_load_mean(y_after_drift, DEFAULT_OBS_PARAMS_V5)  + 1.0 * randn(rng)
                obs_sleep = rand(rng) < 0.5 ? 0.0 : 1.0
                obs       = [obs_HR, obs_S, obs_steps, obs_VL, obs_sleep]
                dt = 1.0 / 96.0

                # GPU call.
                gpu_out = gpu_propagate_one!(
                    state0, phi, params_dyn, params_obs, frozen,
                    obs, gates, C, noise, dt,
                )

                # CPU reference: em_step_v5 + obs_log_weight_v5.
                # Use the same merged dict the production GPU callers
                # would assemble (frozen + estimated dynamics).
                params_full = merge(TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5)
                sigma_diag = SVector{6, Float64}(
                    params_full[:sigma_B], params_full[:sigma_S],
                    params_full[:sigma_F], params_full[:sigma_A],
                    params_full[:sigma_K], params_full[:sigma_K],
                )
                cpu_next = em_step_v5(state0, phi, params_full,
                                        sigma_diag, dt, noise)

                # Per-particle obs log-weight (CPU): pass particle as a
                # 1-row matrix and use the same gate/obs layout.
                particle_mat = reshape(collect(cpu_next), 1, 6)
                obs_nt = (
                    obs_HR    = obs_HR,
                    obs_sleep = obs_sleep == 1.0,
                    obs_S     = obs_S,
                    obs_steps = obs_steps,
                    obs_VL    = obs_VL,
                )
                gates_nt = (
                    hr_present     = true,
                    sleep_present  = true,
                    stress_present = true,
                    steps_present  = true,
                    vl_present     = true,
                )
                cpu_logw = obs_log_weight_v5(particle_mat, obs_nt, gates_nt,
                                              C, params_full)[1]

                # Compare next_state component-by-component (fp32 tol).
                for i in 1:6
                    @test isapprox(gpu_out.next_state[i], cpu_next[i];
                                    atol = GPU_FP32_TOL,
                                    rtol = GPU_FP32_TOL)
                end
                # Compare log-weight (fp32 absolute tol; the magnitude can
                # be in the tens because sigma_HR is small and HR_obs has
                # a residual on the order of HR_base = 60, so ΔHR² gets
                # large fast — use a relaxed rtol).
                @test isapprox(gpu_out.log_w, cpu_logw;
                                atol = 1e-1, rtol = 1e-3)
            end
        end

        @testset "gpu_control_v5 single-thread parity" begin
            # Build a small target.
            n_steps   = 24
            n_anchors = 4
            dt        = 1.0 / 96.0
            tgt = FSAv5ControlGPUTarget(
                n_inner = 1, M_max = 1,
                n_steps = n_steps, n_anchors = n_anchors,
                n_substeps = 4, dt = dt,
                F_max = 0.40, Phi_max = 3.0, Phi_default = 1.0,
                lam_Phi = 0.0, lam_F = 1.0,
                lam_chance = 0.0,        # disable chance term for the parity test
                A_thr = 0.05, 
                B_thr = 0.20, lam_chance_B = 0.0,
                S_thr = 0.20, lam_chance_S = 0.0,
                beta_chance = 50.0, scale_chance = 0.10,
                params = TRUTH_PARAMS_V5,
                noise_seed = 1234,
            )

            for trial in 1:2
                theta = randn(rng, 2 * n_anchors)
                noise = randn(rng, n_steps, 6)
                gpu_cost = Float64(gpu_cost_one!(tgt, theta, noise))

                # CPU reference: replay the exact same arithmetic the
                # kernel does, in fp32 throughout. A pure-Julia fp32 sim
                # of the v5 SDE under the decoded schedule, accumulating
                # the same SOFT cost expression (lam_Phi=0, lam_chance=0,
                # so the cost reduces to: -A_acc + lam_F*barrier_acc).
                # This is the inline reference rather than calling the
                # CPU module's em_step_v5 because the GPU kernel uses
                # 4-substep drift and adds noise at the OUTER bin only,
                # while em_step_v5 takes a single combined step. The
                # pattern matches v1.5's gpu_control.jl cost kernel.
                p_max  = Float32(3.0)
                c_phi_v = Float32(log((1.0 / 3.0) / (2.0 / 3.0)))
                rbf = Float32.(build_rbf_design_v5(n_steps, dt, n_anchors))

                B   = Float32(TRAINED_ATHLETE_INIT.B)
                S   = Float32(TRAINED_ATHLETE_INIT.S)
                F   = Float32(TRAINED_ATHLETE_INIT.F)
                A   = Float32(TRAINED_ATHLETE_INIT.A)
                KFB = Float32(TRAINED_ATHLETE_INIT.KFB)
                KFS = Float32(TRAINED_ATHLETE_INIT.KFS)

                A_acc = 0f0
                B_acc = 0f0
                S_acc = 0f0
                bar_acc = 0f0
                F_max_f = Float32(0.40)
                lam_F_f = Float32(1.0)
                dt_f = Float32(dt)
                sub_dt = dt_f / 4f0
                sqrt_dt_f = sqrt(dt_f)
                A_TYP_f = Float32(A_TYP); F_TYP_f = Float32(F_TYP)
                p = TRUTH_PARAMS_V5
                for k in 1:n_steps
                    raw_B = c_phi_v
                    raw_S = c_phi_v
                    for a in 1:n_anchors
                        raw_B += Float32(theta[a])              * rbf[k, a]
                        raw_S += Float32(theta[n_anchors + a])  * rbf[k, a]
                    end
                    Phi_B = p_max / (1f0 + exp(-raw_B))
                    Phi_S = p_max / (1f0 + exp(-raw_S))

                    A_acc   += A * dt_f
                    B_acc   += B * dt_f
                    S_acc   += S * dt_f
                    bar_acc += max(F - F_max_f, 0f0)^2 * dt_f

                    for sub in 1:4
                        F_dev = F - F_TYP_f
                        Bn = max(B, 0f0); Bn = Bn*Bn*Bn*Bn
                        Sn = max(S, 0f0); Sn = Sn*Sn*Sn*Sn
                        Bdn = Float32(p[:B_dec]); Bdn = Bdn*Bdn*Bdn*Bdn
                        Sdn = Float32(p[:S_dec]); Sdn = Sdn*Sdn*Sdn*Sdn
                        dec_B = Float32(p[:mu_dec_B]) * Bdn / (Bn + Bdn)
                        dec_S = Float32(p[:mu_dec_S]) * Sdn / (Sn + Sdn)
                        mu_bif = Float32(p[:mu_0]) +
                                  Float32(p[:mu_B])*B + Float32(p[:mu_S])*S -
                                  Float32(p[:mu_F])*F - Float32(p[:mu_FF])*F_dev*F_dev -
                                  dec_B - dec_S
                        a_B = (1f0 + Float32(p[:epsilon_AB])*A) /
                               (1f0 + Float32(p[:epsilon_AB])*A_TYP_f)
                        a_S = (1f0 + Float32(p[:epsilon_AS])*A) /
                               (1f0 + Float32(p[:epsilon_AS])*A_TYP_f)
                        a_F = (1f0 + Float32(p[:lambda_A])*A) /
                               (1f0 + Float32(p[:lambda_A])*A_TYP_f)
                        dB = Float32(p[:kappa_B])*a_B*Phi_B - B/Float32(p[:tau_B])
                        dS = Float32(p[:kappa_S])*a_S*Phi_S - S/Float32(p[:tau_S])
                        dF = KFB*Phi_B + KFS*Phi_S - a_F/Float32(p[:tau_F])*F
                        dA = mu_bif*A - Float32(p[:eta])*A*A*A
                        dKFB = (Float32(p[:KFB_0]) - KFB)/Float32(p[:tau_K]) +
                                Float32(p[:mu_K])*Phi_B
                        dKFS = (Float32(p[:KFS_0]) - KFS)/Float32(p[:tau_K]) +
                                Float32(p[:mu_K])*Phi_S
                        B   += sub_dt*dB; S   += sub_dt*dS
                        F   += sub_dt*dF; A   += sub_dt*dA
                        KFB += sub_dt*dKFB; KFS += sub_dt*dKFS
                    end

                    B_cl = max(1f-4, min(1f0 - 1f-4, B))
                    S_cl = max(1f-4, min(1f0 - 1f-4, S))
                    F_cl = max(0f0, F); A_cl = max(0f0, A)
                    KFB_cl = max(0f0, KFB); KFS_cl = max(0f0, KFS)
                    sB = Float32(p[:sigma_B])*sqrt(B_cl*(1f0 - B_cl))
                    sS = Float32(p[:sigma_S])*sqrt(S_cl*(1f0 - S_cl))
                    sF = Float32(p[:sigma_F])*sqrt(F_cl)
                    sA = Float32(p[:sigma_A])*sqrt(A_cl)
                    sKB = Float32(p[:sigma_K])*sqrt(KFB_cl)
                    sKS = Float32(p[:sigma_K])*sqrt(KFS_cl)
                    B   += sB*sqrt_dt_f*Float32(noise[k, 1])
                    S   += sS*sqrt_dt_f*Float32(noise[k, 2])
                    F   += sF*sqrt_dt_f*Float32(noise[k, 3])
                    A   += sA*sqrt_dt_f*Float32(noise[k, 4])
                    KFB += sKB*sqrt_dt_f*Float32(noise[k, 5])
                    KFS += sKS*sqrt_dt_f*Float32(noise[k, 6])
                    B = max(1f-4, min(1f0 - 1f-4, B))
                    S = max(1f-4, min(1f0 - 1f-4, S))
                    F = max(0f0, F); A = max(0f0, A)
                    KFB = max(0f0, KFB); KFS = max(0f0, KFS)
                end
                cpu_cost = Float64(-A_acc - B_acc - S_acc + lam_F_f * bar_acc)
                @test isapprox(gpu_cost, cpu_cost;
                                atol = GPU_FP32_TOL,
                                rtol = GPU_FP32_TOL)
            end
        end

    finally
        close!(client)
    end
end
