
You are tasked with the research engineering project of importing and verifying the latest version of the FSA model into the smc2fc repo

First read these two planning documents written by the senior agents which will guide you,

- /home/ajay/Repos/python-smc2-filtering-control/claude_plans/Importing_FSA_version_5_model_2026-05-05_0859.md
- /home/ajay/Repos/python-smc2-filtering-control/claude_plans/Importing_FSA_version_5_model_specific_notes_2026-05-05_0935.md

Make sure you understand BOTH of these and the read all the references in them.

Then you will construct the plan for YOUR work based on them,

If you find any (non trivial) contradictions you will flag them to me AJAY the user

More broadly, once the import is done we will do the verification/testing in three broad stages

1) testing the filter and plant ONLY - the controller does NOT need to be run for this - success is being able to recover the ground truth model parameters
2) testing controller ONLY - the filter does NOT need to be run for this - success to sensible qualitative targets met 
3) testing full closed loop MPC

Please give your plan with states all this