"""
    test_gate3_gpu_smoke.jl

Gate 3 of the SMC2FC_functional replacement-readiness audit:
verify that the GPU code paths in `Filtering/GPUSegmentedPF.jl`
load, dispatch, and produce finite results that agree with a CPU
reference computation on tiny inputs.

This is a smoke test, not a full bench — it only confirms:
1. CUDA + KernelAbstractions are functional in this env.
2. `GPUSegmentedBuffers` constructs without error.
3. The per-chain stats kernel runs and produces correct
    `log_max`, `log_z`, `ess`, `mu_per_chain` on a controlled input
    (within fp32 tolerance vs a CPU reference).
4. The normalize-and-cumsum kernel runs and produces a per-chain
    cumulative distribution on `[0, 1]`.

If CUDA is not available the suite reports skipped; it does not fail.

# Run
```
cd julia/SMC2FC_functional
julia --project=. test/test_gate3_gpu_smoke.jl
```
"""

using Test
using Random
using SMC2FC_functional
using SMC2FC_functional.GPUSegmentedPF
using CUDA
using KernelAbstractions

@testset "Gate 3 — GPU primitives smoke test" begin
    if !CUDA.functional()
        @info "Gate 3: CUDA not functional — skipping"
        @test_skip "CUDA not available"
    else
        @info "Gate 3: CUDA functional, device = $(CUDA.name(CUDA.device()))"

        # Tiny problem size: 4 chains × 16 particles, 3 state dims.
        M_max       = 4
        K_per_chain = 16
        n_states    = 3
        Ntot        = M_max * K_per_chain

        bufs = GPUSegmentedBuffers(K_per_chain, M_max, n_states;
                                     ot_rank = 4, anchor_seed = 0)
        @test bufs isa SMC2FC_functional.GPUSegmentedPF.GPUSegmentedBuffers
        @test size(bufs.particles_a) == (Ntot, n_states)

        # Fill log-weights and particles on the host, push to device.
        rng = MersenneTwister(99)
        log_w_cpu     = randn(Float32, Ntot)
        particles_cpu = randn(Float32, Ntot, n_states)

        copyto!(bufs.log_w, log_w_cpu)
        copyto!(bufs.particles_a, particles_cpu)

        # Reset stats outputs.
        fill!(bufs.log_max, 0f0)
        fill!(bufs.log_z, 0f0)
        fill!(bufs.ess, 0f0)
        fill!(bufs.mu_per_chain, 0f0)

        # ── Run per-chain-stats kernel ──────────────────────────────────────
        bufs.stats_kernel(
            bufs.log_max, bufs.log_z, bufs.ess, bufs.mu_per_chain,
            bufs.log_w, bufs.particles_a,
            M_max, K_per_chain, n_states;
            ndrange = (M_max,),
        )
        KernelAbstractions.synchronize(CUDA.CUDABackend())

        log_max_gpu  = Array(bufs.log_max)
        log_z_gpu    = Array(bufs.log_z)
        ess_gpu      = Array(bufs.ess)
        mu_gpu       = Array(bufs.mu_per_chain)

        @test all(isfinite, log_max_gpu)
        @test all(isfinite, log_z_gpu)
        @test all(isfinite, ess_gpu)
        @test all(isfinite, mu_gpu)
        @test all(ess_gpu .>= 1f0)
        @test all(ess_gpu .<= Float32(K_per_chain) + 1e-3)

        # ── CPU reference: same maths, single-threaded Julia ────────────────
        log_max_ref = zeros(Float32, M_max)
        log_z_ref   = zeros(Float32, M_max)
        ess_ref     = zeros(Float32, M_max)
        mu_ref      = zeros(Float32, M_max, n_states)
        for m in 1:M_max
            base = (m - 1) * K_per_chain
            lm = log_w_cpu[base + 1]
            for k in 2:K_per_chain
                v = log_w_cpu[base + k]
                lm = max(lm, v)
            end
            log_max_ref[m] = lm
            sume = 0f0
            for k in 1:K_per_chain
                sume += exp(log_w_cpu[base + k] - lm)
            end
            log_z_ref[m] = log(sume)
            sum_sq = 0f0
            for k in 1:K_per_chain
                w = exp(log_w_cpu[base + k] - lm) / sume
                sum_sq += w * w
                for d in 1:n_states
                    mu_ref[m, d] += w * particles_cpu[base + k, d]
                end
            end
            ess_ref[m] = 1f0 / max(sum_sq, 1f-30)
        end

        @info "Gate 3: GPU vs CPU reference" gpu_log_max=log_max_gpu cpu_log_max=log_max_ref
        @test isapprox(log_max_gpu, log_max_ref;  atol = 1f-5)
        @test isapprox(log_z_gpu,   log_z_ref;    atol = 1f-4)
        @test isapprox(ess_gpu,     ess_ref;      rtol = 1f-3)
        @test isapprox(mu_gpu,      mu_ref;       atol = 1f-5)

        # ── Run normalize-and-cumsum kernel on the same chains ──────────────
        bufs.norm_cumsum_kernel(
            bufs.weights, bufs.cumsum_w, bufs.log_w,
            bufs.log_max, bufs.log_z,
            M_max, K_per_chain;
            ndrange = (M_max,),
        )
        KernelAbstractions.synchronize(CUDA.CUDABackend())

        cumsum_gpu = Array(bufs.cumsum_w)
        @test all(isfinite, cumsum_gpu)
        # Per-chain end-of-cumsum should be ~1 (within fp32 noise).
        for m in 1:M_max
            tail = cumsum_gpu[m * K_per_chain]
            @test isapprox(tail, 1f0; atol = 1f-3)
        end
    end
end
