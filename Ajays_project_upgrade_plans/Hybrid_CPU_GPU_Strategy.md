# Hybrid CPU/GPU Execution Strategy for SMC² in Julia

## Context & Hardware Profile
This document outlines a targeted architectural strategy for the Julia port of the `smc2fc` (Bayesian filtering and control) codebase. It is specifically designed to maximize the computational potential of bleeding-edge heterogeneous hardware:
*   **CPU:** Intel Core Ultra 9 285K (Massive IPC, extreme single-thread performance, large cache).
*   **GPU:** NVIDIA RTX 5090 (Unprecedented memory bandwidth, massive CUDA core count).

## The Core Problem: The Heterogeneous Nature of SMC²
The SMC² (Sequential Monte Carlo squared) algorithm is inherently divided into two distinct computational phases, each with diametrically opposed hardware requirements:

1.  **The Inner Loop: Particle Filtering (SDE Simulation)**
    *   **Profile:** Massively parallel, SIMD-friendly, predictable control flow. Simulating 10,000+ SDE trajectories simultaneously.
    *   **Ideal Hardware:** **GPU (RTX 5090)**. The massive core count and memory bandwidth excel at advancing independent particles in lockstep.

2.  **The Outer Loop: Parameter Rejuvenation (HMC/NUTS)**
    *   **Profile:** Highly branchy, sequential, dynamic control flow (especially NUTS with dynamic tree building).
    *   **Ideal Hardware:** **CPU (Ultra 9 285K)**. GPUs suffer catastrophic performance degradation from warp divergence when executing highly branchy code (as observed in the Python `vmap` + `while_loop` nesting cliff). CPUs, with superior branch prediction and deep caches, excel here.

## The Julia Hybrid Strategy
In Python+JAX, forcing a strict partition where data moves back and forth between device (GPU) and host (CPU) can introduce crippling latency due to JIT tracing constraints and heavy memory copies. 

Julia's multiple dispatch and native LLVM compilation allow for a **Zero-Cost Hybrid Pipeline**. We can split the workload exactly where the math dictates, minimizing PCIe bus transfers to only summary statistics and parameter vectors.

### Phase 1: GPU-Accelerated Particle Filtering
The forward simulation of the state particles ($x$) happens entirely on the RTX 5090.
*   **Data:** Particle states are stored as `CuArray`s (or `StructArray`s wrapping `CuArray`s).
*   **Execution:** SDE integration (`StochasticDiffEq.jl` or `KernelAbstractions.jl` kernels) advances all particles in parallel.
*   **Output:** The filter computes the log-likelihood estimate $\widehat{L}_N(\theta)$ for the current parameter set $\theta$.

### Phase 2: The PCIe Bottleneck Crossing (Summary Statistics Only)
The key to making a hybrid approach fast is minimizing data transfer across the PCIe bus. We **do not** transfer the massive $N \times D$ particle state matrix back to the CPU.
*   **Transfer:** We only reduce and transfer the **scalar log-likelihood** (and its gradient, if doing HMC) or the **low-dimensional parameter vector** ($\theta$) back to the host CPU. 
*   **Cost:** Transferring a scalar or a 37-dimensional vector across a PCIe Gen 5 bus takes on the order of microseconds—entirely negligible compared to the SDE integration time.

### Phase 3: CPU-Accelerated Rejuvenation (NUTS)
With the summary statistics (likelihoods and gradients) safely on the host CPU, we execute the complex NUTS rejuvenation step.
*   **Execution:** We use the Ultra 9 285K to run `AdvancedHMC.jl` (NUTS). 
*   **Advantage:** The CPU's branch predictor easily handles the dynamic tree building of NUTS. No warp divergence, no nested `vmap` penalties.
*   **Update:** The new parameter proposals ($\theta'$) are generated.

### Phase 4: Return to GPU
*   **Transfer:** The newly proposed low-dimensional parameter vectors ($\theta'$) are sent back across the PCIe bus to the GPU.
*   **Cycle Repeats:** The GPU resumes the massively parallel SDE simulations using the new parameters.

## Architectural Implementation in `SMC2FC.jl`

To achieve this in the Julia port, the architecture must support explicit data locality.

1.  **Strict Type Boundaries:** Use generic `AbstractArray` dispatch, but explicitly enforce where data lives:
    ```julia
    # Particle states live on the GPU
    struct GPUFilterState{T}
        particles::CuArray{T, 2}
        weights::CuArray{T, 1}
    end

    # Outer parameters live on the CPU
    struct CPUParameters{T}
        θ::Vector{T} 
    end
    ```

2.  **Explicit Memory Transfers:** The likelihood function provided to the CPU-based `AdvancedHMC.jl` must handle the transfer internally.
    ```julia
    function compute_likelihood(θ_cpu::Vector{Float64}, filter_state::GPUFilterState)
        # 1. Send proposed parameters to GPU (fast, low-dim transfer)
        θ_gpu = CuArray(θ_cpu) 
        
        # 2. Run massive SDE parallel simulation on GPU
        log_lik_gpu = run_particle_filter!(filter_state, θ_gpu)
        
        # 3. Bring the scalar log-likelihood back to CPU
        return Array(log_lik_gpu)[1] 
    end
    ```
    *(Note: When using Enzyme.jl or Zygote.jl for gradients, the VJP must also efficiently handle this host-device boundary).*

3.  **Opt-in Multithreading on CPU:** Even though NUTS is sequential per-chain, SMC² often evaluates multiple parameter particles in the outer loop. The Ultra 9 285K's 24 cores should be lit up using `Threads.@threads` or `Polyester.@batch` to evaluate the CPU-side of multiple $\theta$ particles simultaneously, before they each dispatch their inner SDE workload to the GPU.

## Summary of the Hybrid Advantage
By respecting the architectural strengths of both processors—using the **RTX 5090 as a raw throughput engine** for SDEs and the **Ultra 9 285K as a complex logic engine** for NUTS—we completely bypass the "NUTS-inside-vmap" cliff that crippled the Python implementation. This hybrid model represents the theoretical ceiling of performance for the SMC² algorithm on this hardware.