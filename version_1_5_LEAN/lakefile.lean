import Lake
open Lake DSL

-- FSA v1.5 Lean4 reference implementation — bridges to the purely-functional
-- Julia at `../version_1_5_Julia/models/fsa_high_res/` via the JSON-line
-- CLI in `Main.lean`. The Match.jl `@match` sites in the Julia map line-for-line
-- to `match … with` here. Per the LEAN4-first charter, Lean4 is the formal
-- source of truth; Julia is differentially tested at 1e-6 single-step / 1e-4
-- integrated.
--
-- v1.5's math surface is small (drift, diffusion, EM step, three small
-- @match sites). We deliberately START WITHOUT Mathlib — `Float.sqrt`,
-- `Float.exp`, `Float.log`, `Float.max`, `Float.pow` are all in Lean4 core.
-- If a future extension needs `mathlib`, it's a one-line addition to this
-- file. Compile time stays under 30 s on a fresh checkout.

package FsaV15 where
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩,
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib Fsa where
  globs := #[.andSubmodules `Fsa]

-- CLI entry points used by the Julia differential-test bridges.
-- Accept JSON on stdin, print JSON on stdout. One binary per model version
-- so the v1.5 and v5 surfaces stay decoupled (no Lean-side conditionals).
@[default_target]
lean_exe fsa_v15_cli where
  root := `Main

@[default_target]
lean_exe fsa_v5_cli where
  root := `Main_v5
