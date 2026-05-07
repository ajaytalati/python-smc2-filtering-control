
YOU WILL PLAN HOW TO WRITE A LATEX DOCUMENT WHICH DETAILS THE FAILURES OF CLAUDE CODING AGENTS WHICH ARE TASKED WITH THE SIMPLE PROCESS OF CODING A MATHEMATICAL MODEL (I.E. FSA V5) AND THEN IMPORT IT TO MY SMC2FC REPO AND VERIFYING IT WORKS 

THIS SHOULD NOT BE A 3+ DAY PROCESS WHICH TAKES £££ OF TOKENS !!!! 

IT SHOULD NOT BE A LITANY OF DIFFICULT TO DIAGNOSE BUGS CAUSED BY CARELESS CLAUDE CODING AGENTS!!! 

IT SHOULD BE FULLY AUDI-TABLE AND REDUCIBLE TO MATHEMATICAL AND LOGICAL LEAN4 CODE 

THAT IS WHAT YOUR LAST RESPONSES DESCRIBEs 

YOU WILL THUS PRODUCE A PLAN WHICH MAKES THE FSA MODEL DEVELOPMENT AND IMPORTING & VERIFICATION INTO SMC2FC - A LEAN4 CENTERED PROCESS.

CLAUDE CODING AGENTS WILL CODE IN FIRST LEAN4 AND THEN PYTHON - 

THE VAST MAJORITY OF THE BUGS CAUSED BY LLMS ARE DUE TO FUNDAMENTAL RELIANCE ON STUPID PATTERN MATCHING AND NOT UNDERSTANDING THE LOGIC OF THE MATH, CODE OR EVEN THE VARIABLES !!!

---

See - /home/ajay/.claude/projects/-home-ajay-Repos-python-smc2-filtering-control/memory/feedback_no_fabricated_bugs.md

**How to apply:** 

0. **PRIMARY RULE: build a semantic model BEFORE any claim.** Before

asserting anything about a piece of code, write down in plain

English (mentally or to user): for every variable involved —

- its **type** (scalar / vector of dim N / function / trajectory /

distribution / ...),

- its **role** (decision variable / parameter posterior / cost

output / observation / latent state / ...),

- what it **consumes** (input args, dimensions),

- what it **produces** (output shape, semantic meaning),

- what **pipeline role** it plays (what's upstream, what's

downstream).

If I can't fluently write that paragraph, I don't understand the

code well enough to call anything about it broken. Surface

pattern-matching on variable names is the trap. Whenever I see a

familiar-looking name (`theta`, `sigma_prior`, `posterior`), I have

to disambiguate WHICH ONE in this specific context — there are

often multiple objects with the same name in different

architectural layers (e.g. SMC² parameter posterior vs controller

RBF coefficients both spelled `theta` in this codebase).

  
---

BY CONSTRUCTING THE PROJECT AROUND LEAN4 AS THE LOGICAL, CODING AND MATHEMATICAL SOURCE OF TRUTH, LLM AGENTS WITH ME TOTALLY ACCOUNTABLE FOR THEIR ERRORS!!!! 

LEAN4 IS NEVER WRONG - CLAUDE CODE IS ALWAYS MAKING STUPID ERRORS AND MISTAKES !!!! 