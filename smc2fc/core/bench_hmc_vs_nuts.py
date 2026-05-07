import time
import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.core.config import SMCConfig
from smc2fc.core.jax_native_smc import run_smc_window_native as hmc_run
from smc2fc.core.jax_native_smc_nuts import run_smc_window_native as nuts_run

class DummyModel:
    def __init__(self, n_dim):
        self.n_dim = n_dim

n_dim = 10
model = DummyModel(n_dim)

# Construct dummy T_arr
T_arr = {
    'is_log': jnp.zeros(n_dim),
    'is_ln': jnp.zeros(n_dim),
    'ln_mu': jnp.zeros(n_dim),
    'ln_sigma': jnp.ones(n_dim),
    'is_ident': jnp.ones(n_dim),
    'is_norm': jnp.ones(n_dim),
    'n_mu': jnp.zeros(n_dim),
    'n_sigma': jnp.ones(n_dim),
    'is_vm': jnp.zeros(n_dim),
    'vm_mu': jnp.zeros(n_dim),
    'vm_kappa': jnp.zeros(n_dim),
    'is_logit': jnp.zeros(n_dim),
    'is_bt': jnp.zeros(n_dim),
    'beta_a': jnp.zeros(n_dim),
    'beta_b': jnp.zeros(n_dim),
}

def dummy_log_density(u):
    # simple standard normal target
    return -0.5 * jnp.sum(u**2)

full_log_density = jax.tree_util.Partial(dummy_log_density)

cfg = SMCConfig(
    n_smc_particles=256,
    num_mcmc_steps=5,
    hmc_step_size=0.1,
    hmc_num_leapfrog=8,
    target_ess_frac=0.5
)

import copy
nuts_cfg = SMCConfig(
    n_smc_particles=256,
    num_mcmc_steps=1,
    hmc_step_size=0.1,
    hmc_num_leapfrog=8,
    target_ess_frac=0.5
)
nuts_cfg.nuts_max_num_doublings = 3

# Provide initial particles to bypass sample_from_prior
initial_particles = np.random.randn(cfg.n_smc_particles, n_dim)

print("Starting JIT (HMC)...")
t0 = time.time()
hmc_run(full_log_density, model, T_arr, cfg, initial_particles=initial_particles, seed=42)
print(f"HMC JIT + first run took {time.time() - t0:.2f}s")

print("Starting JIT (NUTS)...")
t0 = time.time()
nuts_run(full_log_density, model, T_arr, nuts_cfg, initial_particles=initial_particles, seed=42)
print(f"NUTS JIT + first run took {time.time() - t0:.2f}s")

print("\nStarting timing (HMC)...")
t0 = time.time()
_, elapsed_hmc, n_temp_hmc = hmc_run(full_log_density, model, T_arr, cfg, initial_particles=initial_particles, seed=123)
print(f"HMC run took {time.time() - t0:.3f}s (reported {elapsed_hmc:.3f}s), {n_temp_hmc} temp steps")

print("Starting timing (NUTS)...")
t0 = time.time()
_, elapsed_nuts, n_temp_nuts = nuts_run(full_log_density, model, T_arr, nuts_cfg, initial_particles=initial_particles, seed=123)
print(f"NUTS run took {time.time() - t0:.3f}s (reported {elapsed_nuts:.3f}s), {n_temp_nuts} temp steps")
