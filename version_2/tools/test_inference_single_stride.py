"""Micro-Inference Test for SWAT.
Verifies that truth parameters have high likelihood after a single stride.
"""
import jax
import jax.numpy as jnp
import numpy as np
from models.swat._plant import StepwisePlant
from models.swat.estimation import SWAT_ESTIMATION
from models.swat.simulation import DEFAULT_PARAMS
from smc2fc.transforms.unconstrained import constrained_to_unconstrained

def main():
    print("Running SWAT Micro-Inference Test...")
    
    # 1. Initialize Plant and advance 1 stride
    # Use pathological scenario
    init_state = np.array([0.5, 0.583, 0.5, 0.0], dtype=np.float64)
    plant = StepwisePlant(seed_offset=42, state=init_state.copy())
    
    # Force Ideal Recovery controls
    v_h = np.array([1.0])
    v_n = np.array([0.0])
    v_c = np.array([0.0])
    
    # Advance 3 hours (1 stride at h=15min is 12 bins)
    stride_bins = 12
    obs = plant.advance(stride_bins, v_h, v_n, v_c)
    
    # 2. Align observations
    em = SWAT_ESTIMATION
    window_obs = {
        'obs_HR': obs['obs_HR'],
        'obs_sleep': obs['obs_sleep'],
        'obs_steps': obs['obs_steps'],
        'obs_stress': obs['obs_stress'],
        'V_h': obs['V_h'],
        'V_n': obs['V_n'],
        'V_c': obs['V_c']
    }
    grid_obs = em.align_obs_fn(window_obs, t_steps=stride_bins, dt=1.0/96.0)
    grid_obs = {k: jnp.asarray(v) for k, v in grid_obs.items()}
    
    # 3. Compute Log-Likelihood of Truth
    truth_params = {k: float(v) for k, v in DEFAULT_PARAMS.items()}
    # Filter identifiable subset
    theta_truth = []
    for name in em.all_names:
        if name in truth_params:
            theta_truth.append(truth_params[name])
        else:
            # Handle offsets and other non-dynamics params
            theta_truth.append(0.0) 
    theta_truth = jnp.array(theta_truth)
    
    # Log-density factory (we need a simplified version for 1 stride)
    # or just call the components manually.
    
    # Initial state (truth)
    x = jnp.asarray(init_state)
    log_lik = 0.0
    
    # Manual 1-stride rollout
    params_vec = theta_truth
    p_dict = {name: params_vec[i] for i, name in enumerate(em.all_names)}
    for fname, fval in em.frozen_params.items():
        p_dict[fname] = fval
        
    rng_key = jax.random.PRNGKey(42)
    
    print(f"  Initial T: {x[3]:.4f}")
    
    for k in range(stride_bins):
        # Propagate (1 bin)
        # Note: plant uses EM sub-stepping 10x
        # estimation also uses EM sub-stepping 10x
        dt = 1.0/96.0
        
        # We assume the latent state is 'known' (truth) to test the likelihood logic
        x_true = obs['trajectory'][k] 
        
        # Obs likelihood at this bin
        ll_k = em.obs_log_weight_fn(x_true, grid_obs, k, params_vec)
        log_lik += ll_k
        
    print(f"  Log-Likelihood of Truth: {log_lik:.4f}")
    
    # 4. Generate random particles and compare
    n_samples = 1000
    ll_samples = []
    for i in range(n_samples):
        # Sample from prior
        # (Simplified: just perturb truth)
        theta_sample = theta_truth * (1.0 + 0.1 * np.random.randn(len(theta_truth)))
        ll_s = 0.0
        for k in range(stride_bins):
            x_true = obs['trajectory'][k]
            ll_s += em.obs_log_weight_fn(x_true, grid_obs, k, theta_sample)
        ll_samples.append(ll_s)
        
    ll_samples = np.array(ll_samples)
    rank = np.sum(ll_samples > log_lik)
    print(f"  Truth rank in {n_samples} random samples: {rank}/{n_samples}")
    
    if rank < n_samples * 0.05:
        print("  SUCCESS: Truth has high relative likelihood.")
    else:
        print("  FAILURE: Truth likelihood is low compared to random samples.")

if __name__ == "__main__":
    main()
