"""FSA v1.5 Python+JAX — purely-functional rewrite.

Stepping-stone for a LEAN4 port. Same math, same numerical results as
`version_1_5_Python_JAX/models/fsa_high_res/`, but every for/while loop
replaced by `jax.lax.scan` / `jax.vmap`, every dict-keyed parameter
container replaced by an immutable `typing.NamedTuple`, no
mutation-after-construction.

Verification target: bit-equivalent (≤1e-12) to the imperative source
under the same RNG keys + inputs.
"""
