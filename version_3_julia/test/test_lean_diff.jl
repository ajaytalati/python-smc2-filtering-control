# Julia ↔ LEAN4 binary differential test.
#
# Per the LEAN4-first charter §5: every Julia function with a LEAN4
# counterpart must agree with the LEAN binary at `1e-6` per-step
# tolerance. The LEAN binary is the formal source of truth.
#
# This file uses a long-lived subprocess to the `fsa_v5_cli` binary
# at `FSA_model_dev/lean/.lake/build/bin/fsa_v5_cli`, exchanging
# line-delimited JSON. The cli supports dispatch tags `drift`,
# `muBar`, `findASep`, `schedule`, `emStep`, `hrMean`, `sleepProb`,
# `stressMean`, `stepsLogMean`, `volumeLoadMean`.

using FSAv5
using Test
using JSON3

const SINGLE_STEP_TOL = 1.0e-6

const LEAN_BIN = "/home/ajay/Repos/FSA_model_dev/lean/.lake/build/bin/fsa_v5_cli"

# Long-lived subprocess client. Spawned once per test session.
# `open(cmd, "r+")` opens a bidirectional pipe — IO object reads
# stdout AND writes stdin. Works correctly with line-buffered text
# protocols; `pipeline(...; stdin=Pipe(), stdout=Pipe())` is for
# chaining commands and doesn't give a usable bidirectional handle.
mutable struct LeanClient
    io::Base.AbstractPipe   # IO with both read+write to the subprocess
end

function open_lean_client()
    isfile(LEAN_BIN) || error("LEAN binary not found at $LEAN_BIN. " *
                               "Run `lake build` in FSA_model_dev/lean first.")
    io = open(`$LEAN_BIN`, "r+")
    return LeanClient(io)
end

function close!(c::LeanClient)
    try; close(c.io); catch; end
end

# Send JSON, read JSON. Handles ±Inf / NaN encoding from the Lean side
# (the cli emits "Infinity"/"-Infinity"/"NaN" — JSON3 accepts these
# via `allow_inf=true`).
function round_trip(c::LeanClient, payload::Dict)
    s = JSON3.write(payload; allow_inf = true)
    println(c.io, s)
    flush(c.io)
    line = readline(c.io)
    isempty(line) && error("LEAN binary produced no output for: $s")
    return JSON3.read(line; allow_inf = true)
end

# Convert a Julia DynParams to the JSON dict the LEAN cli expects.
function _params_payload(p::FSAv5.DynParams)
    nt = NamedTuple(p)
    return Dict{String,Float64}(String(k) => Float64(v) for (k, v) in pairs(nt))
end
function _obs_params_payload(op::FSAv5.ObsParams)
    nt = NamedTuple(op)
    return Dict{String,Float64}(String(k) => Float64(v) for (k, v) in pairs(nt))
end

@testset "LEAN diff" begin
    client = open_lean_client()
    try
        p_jl   = FSAv5.TRUTH_PARAMS_V5
        op_jl  = FSAv5.DEFAULT_OBS_PARAMS_V5
        params_payload    = _params_payload(p_jl)
        obs_params_payload = _obs_params_payload(op_jl)

        # ── Drift ──────────────────────────────────────────────────────────
        @testset "drift @ healthy island" begin
            y = FSAv5.DEFAULT_INIT
            phi = BimodalPhi(0.30, 0.30)
            req = Dict("fn" => "drift",
                       "state" => [y.B, y.S, y.F, y.A, y.KFB, y.KFS],
                       "phi" => [phi.Phi_B, phi.Phi_S],
                       "params" => params_payload)
            resp = round_trip(client, req)
            dy_lean = Float64.(resp.deriv)
            dy_jl   = dynamics_drift(y, p_jl, phi)
            @test isapprox(dy_jl.B,   dy_lean[1]; atol = SINGLE_STEP_TOL)
            @test isapprox(dy_jl.S,   dy_lean[2]; atol = SINGLE_STEP_TOL)
            @test isapprox(dy_jl.F,   dy_lean[3]; atol = SINGLE_STEP_TOL)
            @test isapprox(dy_jl.A,   dy_lean[4]; atol = SINGLE_STEP_TOL)
            @test isapprox(dy_jl.KFB, dy_lean[5]; atol = SINGLE_STEP_TOL)
            @test isapprox(dy_jl.KFS, dy_lean[6]; atol = SINGLE_STEP_TOL)
        end

        # ── muBar across LaTeX §10.4 anchor points ─────────────────────────
        @testset "muBar parity at LaTeX anchors" begin
            for (A, pB, pS) in [(0.0, 0.30, 0.30), (0.0, 0.0, 0.0),
                                 (0.0, 1.0, 1.0), (0.5, 0.45, 0.30),
                                 (0.1, 0.20, 0.20)]
                req = Dict("fn" => "muBar",
                           "A" => A, "phi" => [pB, pS],
                           "params" => params_payload)
                resp = round_trip(client, req)
                mb_lean = Float64(resp.muBar)
                mb_jl   = mu_bar(A, BimodalPhi(pB, pS), p_jl)
                @test isapprox(mb_jl, mb_lean; atol = SINGLE_STEP_TOL)
            end
        end

        # ── findASep across closed-island regimes ──────────────────────────
        @testset "findASep closed-island gates" begin
            cases = [(0.30, 0.30, "healthy"),
                     (0.0,  0.0,  "sedentary"),
                     (1.0,  1.0,  "overtrn"),
                     (0.30, 0.0,  "aerobic-only"),
                     (0.0,  0.30, "strength-only"),
                     (0.45, 0.30, "bistable")]
            for (pB, pS, name) in cases
                req = Dict("fn" => "findASep", "phi" => [pB, pS],
                           "params" => params_payload)
                resp = round_trip(client, req)
                a_lean = Float64(resp.A_sep)
                a_jl   = find_a_sep(BimodalPhi(pB, pS), p_jl)
                @test isnan(a_jl) == isnan(a_lean)
                @test isinf(a_jl) == isinf(a_lean)
                if isfinite(a_jl) && isfinite(a_lean)
                    @test isapprox(a_jl, a_lean; atol = SINGLE_STEP_TOL)
                elseif isinf(a_jl) && isinf(a_lean)
                    @test sign(a_jl) == sign(a_lean)
                end
            end
        end

        # ── em_step zero-noise parity ──────────────────────────────────────
        @testset "em_step zero noise" begin
            y = FSAv5.DEFAULT_INIT
            phi = BimodalPhi(0.30, 0.30)
            sigma_diag = [p_jl.sigma_B, p_jl.sigma_S, p_jl.sigma_F,
                           p_jl.sigma_A, p_jl.sigma_K, p_jl.sigma_K]
            dt = DT_BIN_DAYS
            noise = zeros(6)
            req = Dict("fn" => "emStep",
                       "state" => [y.B, y.S, y.F, y.A, y.KFB, y.KFS],
                       "phi" => [phi.Phi_B, phi.Phi_S],
                       "params" => params_payload,
                       "sigma_diag" => sigma_diag,
                       "dt" => dt,
                       "noise" => noise)
            resp = round_trip(client, req)
            yn_lean = Float64.(resp.next_state)
            yn_jl   = em_step(y, phi, p_jl, sigma_diag, dt, noise)
            @test isapprox(yn_jl.B,   yn_lean[1]; atol = SINGLE_STEP_TOL)
            @test isapprox(yn_jl.S,   yn_lean[2]; atol = SINGLE_STEP_TOL)
            @test isapprox(yn_jl.F,   yn_lean[3]; atol = SINGLE_STEP_TOL)
            @test isapprox(yn_jl.A,   yn_lean[4]; atol = SINGLE_STEP_TOL)
            @test isapprox(yn_jl.KFB, yn_lean[5]; atol = SINGLE_STEP_TOL)
            @test isapprox(yn_jl.KFS, yn_lean[6]; atol = SINGLE_STEP_TOL)
        end

        # ── Obs channels ────────────────────────────────────────────────────
        @testset "obs channels parity" begin
            y = FSAv5.DEFAULT_INIT
            C = 0.5
            req_base = Dict("state" => [y.B, y.S, y.F, y.A, y.KFB, y.KFS],
                            "C" => C, "obs_params" => obs_params_payload)

            # hrMean
            r = round_trip(client, merge(req_base, Dict("fn" => "hrMean")))
            @test isapprox(hr_mean(y, C, op_jl), Float64(r.hr_mean); atol = SINGLE_STEP_TOL)

            # sleepProb
            r = round_trip(client, merge(req_base, Dict("fn" => "sleepProb")))
            @test isapprox(sleep_prob(y, C, op_jl), Float64(r.sleep_prob); atol = SINGLE_STEP_TOL)

            # stressMean
            r = round_trip(client, merge(req_base, Dict("fn" => "stressMean")))
            @test isapprox(stress_mean(y, C, op_jl), Float64(r.stress_mean); atol = SINGLE_STEP_TOL)

            # stepsLogMean
            r = round_trip(client, merge(req_base, Dict("fn" => "stepsLogMean")))
            @test isapprox(steps_log_mean(y, C, op_jl), Float64(r.steps_log_mean); atol = SINGLE_STEP_TOL)

            # volumeLoadMean (no C input)
            r = round_trip(client, Dict("fn" => "volumeLoadMean",
                                          "state" => req_base["state"],
                                          "obs_params" => obs_params_payload))
            @test isapprox(volume_load_mean(y, op_jl), Float64(r.vl_mean); atol = SINGLE_STEP_TOL)
        end
    finally
        close!(client)
    end
end
