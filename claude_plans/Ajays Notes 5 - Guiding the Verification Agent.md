
You are a mathematically and scientifically principled verification and debugging agent. 

A previous agent has import the FSA version 5 model from the dev repo - see this branch - https://github.com/ajaytalati/FSA_model_dev/tree/claude/dev-sandbox-v4

Read this for back ground - https://github.com/ajaytalati/FSA_model_dev/blob/v4-bimodal-variable-dose-extension/LaTex_docs/FSA_version_5_technical_guide.tex

You are working in the smc2fc repo on this branch - https://github.com/ajaytalati/python-smc2-filtering-control/tree/importing_FSA_version_5

In terms of background on the smc2fc framework / repo you will read - https://github.com/ajaytalati/python-smc2-filtering-control/blob/importing_FSA_version_5/LaTex_docs/main.pdf

THE CODE IS NOT MATURE - there very likely to be MANY BUGS HERE - /home/ajay/Repos/python-smc2-filtering-control/version_3/models/fsa_v5

YOU NO NOT HAVE ANY PERMISSION TO MODIFY ANY CODE OUTSIDE OF THE VERSION 3 FOLDER !!!! YOU WILL TAKE THE SMC2FC CODE FOLDER / FRAMEWORK TO BE UNTOUCHABLE - https://github.com/ajaytalati/python-smc2-filtering-control/tree/importing_FSA_version_5/smc2fc

Your task is ensure the full filtering and control cloced loop codebase for the FSA_version_5 model produces realistic results - the bugs are in the FSA_v5 code in the version 3 folder - NOT SMC2FC !!!

As a guide which might help you should take the working FSA_version 2 model  here - https://github.com/ajaytalati/python-smc2-filtering-control/tree/importing_FSA_version_5/version_2/models/fsa_high_res

The previous agent has given an overview of it work - which is weak and very lame - THE ONLY CONSISTENT FINDING from its overnight work  IS FULL HMC IS NECESSARY - the rest of it work is nonsense - /home/ajay/Repos/python-smc2-filtering-control/claude_plans/Verification_handoff_to_fresh_agent_2026-05-06_0750.md 

See - 

/home/ajay/Repos/python-smc2-filtering-control/claude_plans/FSA-v5_Stage_2_Stage_3_verification_soft_fast_controller_variant_2026-05-05_2046.md

/home/ajay/Repos/python-smc2-filtering-control/claude_plans/Verification_handoff_to_fresh_agent_2026-05-06_0750.md

It has developed a profiler - /home/ajay/Repos/python-smc2-filtering-control/version_3/tools/profile_cost_fn.py

---

## Things You will NOT do (senior-files principle)

  
- Modify `smc2fc/` framework code (escalate to Ajay if I find a real framework bug)

- Modify `version_2/` (separate work; FSA-v5 is a fresh subtree)

- Delete or rewrite docs in `claude_plans/` (audit trail)

- Bundle unrelated framework changes into the FSA-v5 PR (the SWAT-session anti-pattern explicitly called out in the broad plan's "respect" section)

- Keep compute-heavy controller knobs in production after they show no measurable benefit at cheap test (the SWAT D2/D5 fiasco)

- State confident results quality or wall-clock estimates without verification

---

YOU WILL CREATE A new experiment logging methodology - where each run in the experiments folder is documented in a experiments_logging_md in tabular form so I Ajay can have oversight of your work - all the previous experiments have been moved to old_experiments folder - you start fresh

You will now create a NEW plan YOURSELF for my review for what work you will do to debug and verify the FSA version 5 model is working correctly. 