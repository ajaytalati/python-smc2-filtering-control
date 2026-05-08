# Julia v1.5 ↔ Python+JAX v1.5 differential test.
#
# Cross-validation: Julia is the production stack, Python+JAX is a
# third independent implementation. This driver opens a bidirectional
# pipe to `version_1_5_Python_JAX/diff_test/diff_main.py` and round-
# trips the same 121 randomised cases the LEAN diff test exercises,
# asserting agreement at 1e-6 (single-step) / 1e-4 (integrated).
#
# Triangulates: Julia ↔ LEAN at 1e-6 (already green), Julia ↔ Python+JAX
# at 1e-6 (this file). LEAN ↔ Python+JAX agreement follows transitively.
#
# Prereqs:
#   - conda env `comfyenv` activated (or any env with JAX / numpy / smc2fc)
#   - `version_1_5_Python_JAX/` exists with the model files
#
# Run:
#   cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia
#   julia --project=. diff_test/test_python_diff_v15.jl

using Test
using JSON3
using Random
using StaticArrays

# Load v1.5's model files (script-style, no package).
const _V15_DIR = joinpath(@__DIR__, "..", "models", "fsa_high_res")
include(joinpath(_V15_DIR, "_dynamics.jl"))
include(joinpath(_V15_DIR, "simulation.jl"))
include(joinpath(_V15_DIR, "_plant.jl"))
include(joinpath(_V15_DIR, "estimation.jl"))

using .Dynamics: drift, diffusion_state_dep
using .Simulation: DEFAULT_PARAMS, params_v15_to_v1_nt, INIT_STATE
using .Estimation: obs_log_weight

const SINGLE_STEP_TOL = 1e-6
const INTEGRATED_TOL  = 1e-4

# ── Subprocess client to the Python+JAX diff bridge ───────────────────────
const _PYJAX_DIR  = joinpath(@__DIR__, "..", "..", "version_1_5_Python_JAX")
const PYTHON_BIN  = get(ENV, "FSA_DIFF_PYTHON",
                        joinpath(homedir(), "miniconda3", "envs", "comfyenv",
                                  "bin", "python"))
const DIFF_SCRIPT = joinpath(_PYJAX_DIR, "diff_test", "diff_main.py")

mutable struct PyClient
    io::Base.AbstractPipe
end

function open_py_client()
    isfile(PYTHON_BIN) || error("Python interpreter not at $PYTHON_BIN. " *
                                 "Set FSA_DIFF_PYTHON env var or activate comfyenv.")
    isfile(DIFF_SCRIPT) || error("diff_main.py not found at $DIFF_SCRIPT.")
    cmd = setenv(`$PYTHON_BIN $DIFF_SCRIPT`,
                  "PYTHONPATH" => string(_PYJAX_DIR, ":",
                                           dirname(_PYJAX_DIR)),
                  "JAX_ENABLE_X64" => "True",
                  "JAX_PLATFORMS"  => "cpu")  # cpu is fine for diff test
    io = open(cmd, "r+")
    return PyClient(io)
end

close!(c::PyClient) = (try; close(c.io); catch; end)

function round_trip(c::PyClient, payload::Dict)
    s = JSON3.write(payload; allow_inf = true)
    println(c.io, s)
    flush(c.io)
    line = readline(c.io)
    isempty(line) && error("Python diff bridge produced no output for: $s")
    return JSON3.read(line; allow_inf = true)
end

# ── Helpers (identical to LEAN diff test) ─────────────────────────────────
const PARAMS_V1_NT = params_v15_to_v1_nt(DEFAULT_PARAMS)

_params_v1_payload(p) =
    Dict{String, Float64}(String(k) => Float64(v) for (k, v) in pairs(p))

function _params_v15_payload(d::Dict{Symbol, Float64})
    Dict{String, Float64}(
        "tau_B"     => d[:tau_B],
        "tau_F"     => d[:tau_F],
        "B_inf"     => d[:B_inf],
        "F_inf"     => d[:F_inf],
        "epsilon_A" => d[:epsilon_A],
        "lambda_A"  => d[:lambda_A],
        "mu_0"      => d[:mu_0],
        "mu_B"      => d[:mu_B],
        "mu_F"      => d[:mu_F],
        "mu_FF"     => d[:mu_FF],
        "eta"       => d[:eta],
        "sigma_B"   => d[:sigma_B],
        "sigma_F"   => d[:sigma_F],
        "sigma_A"   => d[:sigma_A],
    )
end

const OBS_NOISE_PAYLOAD = Dict{String, Float64}(
    "sigma_B_obs" => DEFAULT_PARAMS[:sigma_B_obs],
    "sigma_F_obs" => DEFAULT_PARAMS[:sigma_F_obs],
    "sigma_A_obs" => DEFAULT_PARAMS[:sigma_A_obs],
)

_random_state(rng) = (B = rand(rng), F = rand(rng), A = rand(rng))
_random_noise(rng) = randn(rng, 3)

# ── Test cases (identical structure to LEAN diff test) ────────────────────
@testset "FSA v1.5 Julia ↔ Python+JAX diff" begin
    rng = MersenneTwister(42)
    client = open_py_client()
    try
        params_v1_payload  = _params_v1_payload(PARAMS_V1_NT)
        params_v15_payload = _params_v15_payload(DEFAULT_PARAMS)

        @testset "reflect_unit" begin
            for x in [-1.5, -0.3, -0.0, 0.0, 0.25, 0.5, 0.999, 1.0, 1.3, 1.7]
                resp = round_trip(client, Dict("fn" => "reflectUnit", "x" => x))
                jl = x < 0.0 ? -x : (x > 1.0 ? 2.0 - x : x)
                @test isapprox(jl, Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
            for _ in 1:5
                x = 4.0 * rand(rng) - 1.5
                resp = round_trip(client, Dict("fn" => "reflectUnit", "x" => x))
                jl = x < 0.0 ? -x : (x > 1.0 ? 2.0 - x : x)
                @test isapprox(jl, Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        @testset "apply_prior" begin
            for u in [-25.0, -2.0, -1.0, 0.0, 1.0, 2.0, 25.0]
                resp = round_trip(client, Dict("fn" => "applyPrior",
                    "kind" => "logNormal", "mu" => 0.5, "sigma" => 0.3, "u" => u))
                jl = exp(clamp(u, -20.0, 20.0))
                @test isapprox(jl, Float64(resp.x); atol = SINGLE_STEP_TOL,
                                rtol = SINGLE_STEP_TOL)
            end
            for _ in 1:5
                mu = randn(rng); sigma = abs(randn(rng)); u = randn(rng)
                resp = round_trip(client, Dict("fn" => "applyPrior",
                    "kind" => "normal", "mu" => mu, "sigma" => sigma, "u" => u))
                jl = mu + sigma * u
                @test isapprox(jl, Float64(resp.x); atol = SINGLE_STEP_TOL)
            end
        end

        @testset "params_v15_to_v1" begin
            resp = round_trip(client, Dict("fn" => "paramsV15ToV1",
                                            "params" => params_v15_payload))
            py_p = resp.params_v1
            for (k, v) in pairs(PARAMS_V1_NT)
                @test isapprox(Float64(v), Float64(py_p[k]); atol = SINGLE_STEP_TOL)
            end
        end

        @testset "drift" begin
            for _ in 1:5
                s = _random_state(rng)
                phi = 3.0 * rand(rng)
                req = Dict("fn" => "drift",
                            "state" => [s.B, s.F, s.A],
                            "phi" => phi,
                            "params" => params_v1_payload)
                resp = round_trip(client, req)
                jl = drift([s.B, s.F, s.A], PARAMS_V1_NT, phi)
                py = Float64.(resp.deriv)
                for i in 1:3
                    @test isapprox(jl[i], py[i]; atol = SINGLE_STEP_TOL,
                                    rtol = SINGLE_STEP_TOL)
                end
            end
        end

        @testset "diffusion_state_dep" begin
            for _ in 1:5
                s = _random_state(rng)
                req = Dict("fn" => "diffusion",
                            "state" => [s.B, s.F, s.A],
                            "params" => params_v1_payload)
                resp = round_trip(client, req)
                jl = diffusion_state_dep([s.B, s.F, s.A], PARAMS_V1_NT)
                py = Float64.(resp.sigma)
                for i in 1:3
                    @test isapprox(jl[i], py[i]; atol = SINGLE_STEP_TOL)
                end
            end
        end

        @testset "em_step_substepped" begin
            dt = 1.0 / 24.0
            for _ in 1:5
                s = _random_state(rng)
                phi = 1.5 * rand(rng)
                noise = _random_noise(rng)
                req = Dict("fn" => "emStep",
                            "state" => [s.B, s.F, s.A],
                            "params" => params_v1_payload,
                            "noise"  => noise,
                            "phi"    => phi,
                            "dt"     => dt,
                            "n_substeps" => 4)
                resp = round_trip(client, req)
                # Julia equivalent (substep then 1 Wiener increment + reflection)
                y = [s.B, s.F, s.A]
                sub_dt = dt / 4
                for _ in 1:4
                    y = y .+ sub_dt .* drift(y, PARAMS_V1_NT, phi)
                end
                σ = diffusion_state_dep(y, PARAMS_V1_NT)
                y_pred = y .+ σ .* sqrt(dt) .* noise
                jl = [y_pred[1] < 0.0 ? -y_pred[1] :
                       (y_pred[1] > 1.0 ? 2.0 - y_pred[1] : y_pred[1]),
                      abs(y_pred[2]), abs(y_pred[3])]
                py = Float64.(resp.next_state)
                for i in 1:3
                    @test isapprox(jl[i], py[i]; atol = INTEGRATED_TOL,
                                    rtol = SINGLE_STEP_TOL)
                end
            end
        end

        @testset "plant_step" begin
            dt = 1.0 / 24.0
            for _ in 1:5
                s = _random_state(rng)
                phi = 1.5 * rand(rng)
                sde_noise = _random_noise(rng)
                obs_noise = _random_noise(rng)
                req = Dict("fn" => "plantStep",
                            "state" => [s.B, s.F, s.A],
                            "phi" => phi,
                            "params" => params_v15_payload,
                            "obs_params" => OBS_NOISE_PAYLOAD,
                            "dt" => dt,
                            "sde_noise" => sde_noise,
                            "obs_noise" => obs_noise)
                resp = round_trip(client, req)
                y = [s.B, s.F, s.A]
                d = drift(y, PARAMS_V1_NT, phi)
                σ = diffusion_state_dep(y, PARAMS_V1_NT)
                y_pred = y .+ d .* dt .+ σ .* sqrt(dt) .* sde_noise
                next_jl = [
                    y_pred[1] < 0.0 ? -y_pred[1] :
                      (y_pred[1] > 1.0 ? 2.0 - y_pred[1] : y_pred[1]),
                    abs(y_pred[2]), abs(y_pred[3]),
                ]
                obs_jl = [
                    next_jl[1] + DEFAULT_PARAMS[:sigma_B_obs] * obs_noise[1],
                    next_jl[2] + DEFAULT_PARAMS[:sigma_F_obs] * obs_noise[2],
                    next_jl[3] + DEFAULT_PARAMS[:sigma_A_obs] * obs_noise[3],
                ]
                py_state = Float64.(resp.next_state)
                py_obs   = Float64.(resp.obs)
                for i in 1:3
                    @test isapprox(next_jl[i], py_state[i]; atol = SINGLE_STEP_TOL)
                    @test isapprox(obs_jl[i],  py_obs[i];   atol = SINGLE_STEP_TOL)
                end
            end
        end

        @testset "obs_log_weight_one" begin
            for _ in 1:5
                s = _random_state(rng)
                obs = (
                    obs_B = s.B + 0.005 * randn(rng),
                    obs_F = s.F + 0.005 * randn(rng),
                    obs_A = s.A + 0.005 * randn(rng),
                )
                req = Dict("fn" => "obsLogWeight",
                            "B" => s.B, "F" => s.F, "A" => s.A,
                            "obs" => Dict("obs_B" => obs.obs_B,
                                            "obs_F" => obs.obs_F,
                                            "obs_A" => obs.obs_A),
                            "obs_params" => OBS_NOISE_PAYLOAD)
                resp = round_trip(client, req)
                particles = reshape([s.B, s.F, s.A], 1, 3)
                lw_jl = obs_log_weight(particles, obs, DEFAULT_PARAMS)[1]
                @test isapprox(lw_jl, Float64(resp.log_w); atol = SINGLE_STEP_TOL,
                                rtol = SINGLE_STEP_TOL)
            end
        end
    finally
        close!(client)
    end
end
