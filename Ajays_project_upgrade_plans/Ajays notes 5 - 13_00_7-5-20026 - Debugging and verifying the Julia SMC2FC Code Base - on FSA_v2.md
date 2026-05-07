
You are mathematically and scientifically principled verification and debugging agent. 

I want you to work on a PURE julia version of the Python/Jax code base in the (Python code) folder - /home/ajay/Repos/python-smc2-filtering-control/smc2fc

The majority of the work has been done and tested - see here  (Julia code) folder  - /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC

You will be working on the working tree of the repo - julia-port-version-1

To understand the smc2fc algorithms you will first read this very comprehensive guide which I have prepared for you (which was based on the Python code - it is inconsistent with the Julia code base) - /home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/main.pdf

That document contains one MAJOR OUTDATED DESIGN DECISION - I have chosen to NOT implement the HMC sampler on the CPU - instead I have chosen to implement the **ChEES-HMC** (Change in the Estimator of the Expected Square HMC) on the GPU 

THE PRINCIPLE NOW IS ALL THE SMC2 FILTERING AND CLOSED LOOP CONTROL MPC SHOULD BOTH RUN PURELY ON THE GPU USING ITS FLOAT 32 CORES - any code which is float 64 and/or cpu bound MUST BE FLAGGED as it will be a performance drain

---
The FSA model has already been converted to Julia code - what remains is debugging why the controller is not working as it should 

You will start by working in the directory /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia

That directory is the Julia port of the original Python FSA model here - /home/ajay/Repos/python-smc2-filtering-control/version_2/models/fsa_high_res

To understand the controller for the FSA -version 2 model you will read sections 8,9,10 of this document (it might be slightly outdated) - https://github.com/ajaytalati/python-smc2-filtering-control/blob/importing_FSA_version_5/LaTex_docs_outdated/main.pdf

To understand the latest state of the Julia code which implements the FSA version 2 model you will read the handover doc / code description doc (which has exaggerated claims of matching the python performance) here -  /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia/docs/julia_fsa_writeup.pdf

---

Confirm that you have read all the documentation i have instructed you to, and have created semantic map of smc2 codebase

Then I will give you exact details of the controller bug

---

