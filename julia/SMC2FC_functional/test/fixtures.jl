"""
    fixtures.jl

Shared test fixtures: tiny model factories and tiny config helpers used
by every test file. Lives in `test/` so multiple test files can
`include("fixtures.jl")` without rebuilding state.

# Public factories
- `make_tiny_smc_config()`: a small `SMCConfig` suitable for fast tests.
- `make_tiny_priors()`: a 4-prior list covering all four `PriorType`
    variants — handy for transform tests.
- `make_tiny_estimation_model()`: minimal `EstimationModel` for filtering
    smoke tests (1-D scalar OU, identity obs).
"""

using SMC2FC_functional
using Random

# Guard against repeated `include("fixtures.jl")` from multiple test
# files when they're all run from `runtests.jl`.
if !@isdefined(_FIXTURES_LOADED)
const _FIXTURES_LOADED = true

"""
    make_tiny_smc_config(; n_smc=16, n_pf=32) -> SMCConfig

Tiny `SMCConfig` for fast tests. Defaults: 16 outer particles, 32 PF
particles, 2 cold-start MCMC moves, 2 leapfrog steps. Bigger than the
mathematical minimum but small enough to run in a fraction of a second.
"""
make_tiny_smc_config(; n_smc::Int = 16, n_pf::Int = 32) =
    SMCConfig(;
        n_smc_particles = n_smc,
        target_ess_frac = 0.5,
        num_mcmc_steps  = 2,
        max_lambda_inc  = 0.25,
        hmc_step_size   = 0.05,
        hmc_num_leapfrog = 2,
        n_pf_particles  = n_pf,
        bandwidth_scale = 1.0,
        ot_max_weight   = 0.0,    # disable OT rescue for speed
    )

"""
    make_tiny_priors() -> (priors_vec, priors_tuple)

A 4-element prior list covering all four `PriorType` variants
(lognormal, normal, beta, von Mises). Returns both vector and tuple
forms so tests can exercise both `constrained_to_unconstrained` methods.
"""
function make_tiny_priors()
    p_vec = PriorType[
        LogNormalPrior(0.0, 1.0),
        NormalPrior(2.0, 0.5),
        BetaPrior(2.0, 5.0),
        VonMisesPrior(0.0, 4.0),
    ]
    p_tup = (LogNormalPrior(0.0, 1.0),
             NormalPrior(2.0, 0.5),
             BetaPrior(2.0, 5.0),
             VonMisesPrior(0.0, 4.0))
    return p_vec, p_tup
end

end # if !@isdefined(_FIXTURES_LOADED)
