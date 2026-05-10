
Ok so I have had a look at the code you wrote - **Commit `392d85b`** — entire Julia port of FSA-v5 under `version_3_julia/` (30 files, +7171 lines):

And it is just far too much for a human to understand !!! And I simply could not get it to work - the controller just does not optimize the growth of A!

THE MAIN REASON FOR THIS IS BECAUSE IT WAS WRITTEN IN THE WOODEN (NON-FUNCTIONAL) PYTHON WAY - AND HENCE HAS DIFFICULT TO TRACE BUGS AND BLOATED TO  (30 files, +7171 lines) !!!

---

I want a much more simple and transparent 2-way porting between Julia and Lean4 - so I have created a much more slimmed down FSA model (the basic version 2 with a minimal observation model), and slimmed down Julia code-base this is the minimal needed to have a closed-loop MPC control model in the SMC2 framework 

There still seems to be bugs in it as it is not improving over a constant baseline training load policy.

That's not the main issue though! I want to be able to define FSA type models in LEAN and then almost automatically (by using match.jl) be able to port the code to Julia. I expect the bug will be found/traced by this process !!!!

For you to understand this I want you to now read the code in this repo - /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res

And the document here - /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/docs/julia_fsa_v15_writeup.pdf  

Also re read the LEAN4 first charter here - /home/ajay/Repos/FSA_model_dev/LaTex_docs/lean4_first_charter.pdf

Then I want you to write a plan for a new LEAN4 codebase which be in THIS REPO in a NEW LEAN4 mathlib directory  /home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/

The purpose of this LEAN4 repo you will develop will be the same as in the LEAN4 first charter -  /home/ajay/Repos/FSA_model_dev/LaTex_docs/lean4_first_charter.pdf

- START with the LaTex definition of the simple FSA version 1_5 model, 
- then define it in LEAN, 
- and then bridge the development process to the actually run-able and tested purely functional Julia code in /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res

This hence is very similar TRANSLATION work as I previously asked you to do , when porting the python code to LEAN4 - again its just TRANSLATION - now from pure functional Julia to LEAN4 inductive types and math definitions

I want the overall code base to be AS MUCH AS POSSIBLE IN LEAN4 - and hence verifiable/certifiable - the final port to Julia (using the match.jl) library should be mechanical and verifiable by a human developer


