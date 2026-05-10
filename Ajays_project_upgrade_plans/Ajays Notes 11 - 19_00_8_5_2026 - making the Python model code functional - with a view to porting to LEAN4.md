
I want you to create a plan for rewriting the code in this directory - /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX/models/fsa_high_res

to be purely functional. Formally - "Refactor the imperative Python code into **purely functional Python**.

- Eliminate all `for/while` loops in favor of recursion or `map/reduce`.
- Ensure all functions are total (no side effects, no global state).
- Use immutable data structures (e.g., tuples instead of lists if they change)."

This is as preliminary step to porting it over into LEAN4 for software specification and verification - which you will do subsequently.

You will use the conda env comfyenv and  put your work in the NEW directory 

/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX_functional/model/fas_high_res


---

OK to begin with your time budget is wildly exaggerated !!! A competent human developer could do this in a 2-3 hours of high focused work, so as an AI you should be able to do that even quicker.

Yes all files and functions need Google level documentation.

**open questions in the plan** — most importantly:

1. Sandbox layout — `smc2fc_Jax_functional/smc2fc/` (preserves `import smc2fc`) vs `smc2fc_Jax_functional/` as the package root.  - YES
2. NUTS path testing — is `jax_native_smc_nuts.py` still in production? - YES
3. Adaptive-tempering `while` loops (3 sites) — keep on-host (default) or migrate to `jax.lax.while_loop` (would lose live prints). - YES
4. One session vs phase-by-phase review. - ONE SESSION
5. `bench_hmc_vs_nuts.py` — move out of `core/` to `tools/`? Not used anymore - irrelevant so move to tools
6. Tolerance budget — strict 1e-12 default, anywhere you'd accept 1e-8? YES STRICT code should be byte of byte ???

---

Ok I just spotted a slight rename I had to do - changed Jax to JAX here /home/ajay/Repos/python-smc2-filtering-control/smc2fc_JAX_functional 

---

So next we need to actually test that your new functional core code works as a drop in substitute for the previous non functional framework. 

This is part of larger project work which is described in A NOW outdated doc here - /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.pdf

Just read the first 6 pages - which is enough to get you up to speed !!!!

What I want you to do is to produce a short technical guide / report on what are current settings need to run the PYTHON code for FSA_v15

I want you to be able to write the Python version of this SHORT doc - /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/docs/technical_guide_to_current_best_Julia_SMC2FC_config.pdf

Thus, can you try running this Python bench script again please, but with your new functional code? You wrote all that's need to pass a different PYTHONPATH ???

/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX_DEPRECIATED_USE_FUNCTIONAL/tools/bench_smc_full_mpc_fsa_v15.py

In particular I want you to choose the same (or similar) flag settings as the Julia current best config above which runs quite quickly ~ 20 mins

So once it ran then write your version of the Python Latex current best config here using a similar file name - /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/docs/

To confirm you understand can you write a plan for this.