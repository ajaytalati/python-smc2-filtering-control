
You are mathematically and scientifically principled verification and debugging agent. YOU WILL ALWAYS READ CODE FULLY AND UNDERSTAND WHAT IT DOES SEMANTICALLY

YO NEVER PATTERN MATCH, REALLY ON DOC STRINGS OR OTHER LLM TRICKS

I want you to work on a PURE julia version of the Python/Jax code base in the (Python code) folder - /home/ajay/Repos/python-smc2-filtering-control/smc2fc

The majority of the work has been done and tested - see here  (Julia code) folder  - /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC

You will be working on the working tree of the repo - julia-port-version-1

To understand the smc2fc algorithms you will first read this very comprehensive guide which I have prepared for you (which was based on the Python code - it is inconsistent with the Julia code base) - /home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/main.pdf

That document contains one MAJOR OUTDATED DESIGN DECISION - I have chosen to NOT implement the HMC sampler on the CPU - instead I have chosen to implement the **ChEES-HMC** (Change in the Estimator of the Expected Square HMC) on the GPU 

THE PRINCIPLE NOW IS ALL THE SMC2 FILTERING AND CLOSED LOOP CONTROL MPC SHOULD BOTH RUN PURELY ON THE GPU USING ITS FLOAT 32 CORES - any code which is float 64 and/or cpu bound MUST BE FLAGGED as it will be a performance drain

---

YOUR TASK 

The FSA_v1 model (an open loop controller, i.e. with out filter or plant) has already been been converted to Julia code and work when entirely coded as float32 

The FSA_v2 model has already been converted to Julia code - what remains is debugging why the closed loop controller is not working as it should when fully coded in float 64 - i have tested it works in float64 precision 

Here is the evidence for that - /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia/tools/test_one_plan_fp64.jl

You will start by working in the directory /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia

That directory is the Julia port of the original Python FSA model here - /home/ajay/Repos/python-smc2-filtering-control/version_2/models/fsa_high_res

To understand the controller for the FSA -version 2 model you will read sections 8,9,10 of this document (it might be slightly outdated) - https://github.com/ajaytalati/python-smc2-filtering-control/blob/importing_FSA_version_5/LaTex_docs_outdated/main.pdf

To understand the latest state of the Julia code which implements the FSA version 2 model you will read the handover doc / code description doc - THIS DOC HAS CONCRETE ADVICE ON HYPOTHESIS WHICH HAVE BEEN TEST AND RULED OUT -  /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia/docs/julia_fsa_writeup.pdf

It gives full details of the files you must understand - THESE ARE PROOFs THAT THE CODE BASE WORKS UNDER SPECIFIC CIRCUMSTANCES !!!!

- /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia/tools/test_one_plan_fp64.jl

Hypothesis 6 (kernel + decoder OK on monotone reward) either B or A

- /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia/tools/test_max_A_only.jl
- /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia/tools/test_max_B_only.jl


---

Confirm that you have read all the documentation I have instructed you to, and have **==created semantic map==** of smc2 codebase

---

---

NO !!! I want you to go back to basics!!!

We will isolate the controller from the filter and just do open loop control - MUCH simpler !!!! You will work on the FSA model which preceded the one you are working on 

Look at the Julia code in this directory - /home/ajay/Repos/python-smc2-filtering-control/version_1_Julia/models/fsa_high_res

It is a NON gpu version of the Python/JAX code in its sister directory here - /home/ajay/Repos/python-smc2-filtering-control/version_1/models/fsa_high_res

I want to test that you actually understand the controller code base 

You will port the GPU_control.jl from here - /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia/models/fsa_high_res - over to - /home/ajay/Repos/python-smc2-filtering-control/version_1_Julia/models/fsa_high_res

And get that directory working so that you produce EXACTLY the same plots as in this directory - /home/ajay/Repos/python-smc2-filtering-control/version_1/outputs/fsa_high_res

---

You will now write a plan where the focus is entirely on saturating and making the full use of the RTX-5090's huge number of cores and memory. The python code was optimized to do exactly this!!!!

I want you to use the Julia profile code I gave you.

You will run very short experiments (just 5-10 mins), which are enough to allow for the Julia JIT to complete and 4-5 strides to be run so you understand the configs GPU core utilization/efficiency and the memory use - the objective is to maximize BOTH

I want you to run these experiments with as close to the same settings as the python code,

--step-minutes 60
--replan-K 2

--------------

See The new subsection (now in §5 of the writeup) - /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia/docs/julia_fsa_writeup.pdf

**Which flags to push, ranked by GPU-saturation impact:**

1. **`--N-smc`** — Most direct lever. Each filter particle spawns its own inner-PF; bigger N = bigger kernel launches. Try 32 → 64 → 128 → 256 → 512.
2. **`--K-per-chain`** — Inner-PF particles per chain. Widens per-launch thread count without amplifying host-side per-chain loops as much. Try 400 → 800 → 1600.

**Recommended novice sweep**  a 4×3 grid (N ∈ {32, 64, 128, 256} × K ∈ {400, 800, 1600}) in another, plus what to look for in `utilization.gpu`, `memory.used`, and per-stride wall.

Plus a note that if util plateaus at ~50% regardless of N×K, you've hit the per-chain host-loop ceiling diagnosed in §2.12 — the fix at that point is in the source code, not the flags.

I want you to USE YOUR COMMON SENSE,

and make me a table, over variables N  x K, 

and for each pair give me - `utilization.gpu`, `memory.used`, and per-stride wall

So this is a very simple task - you you get one with it - and the deliverable is a report of how to in terms --flag setting optimize the use of the RTX-5090 for this code repo