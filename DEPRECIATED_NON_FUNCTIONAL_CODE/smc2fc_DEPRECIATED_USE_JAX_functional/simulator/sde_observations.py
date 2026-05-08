"""
sde_observations.py — Generic Observation Generation Framework
===============================================================
Date:    15 April 2026
Version: 1.0

Generates synthetic observations from all channels in dependency order.
Model-agnostic: channels are pure functions provided by the model.
"""

import numpy as np
from smc2fc.simulator.sde_model import SDEModel


def generate_all_channels(model, trajectory, t_grid, params, aux, seed):
    """Generate observations for all channels, respecting dependencies.

    Performs a topological sort on the channel dependency graph, then
    generates each channel in order.  Each channel receives the outputs
    of all previously generated channels (its dependencies are guaranteed
    to be present).

    Parameters
    ----------
    model : SDEModel
    trajectory : ndarray (T, n_states)
    t_grid : ndarray (T,)
    params : dict
    aux : any (from model.make_aux_fn)
    seed : int

    Returns
    -------
    channel_outputs : dict {channel_name: dict_of_arrays}
    """
    rng = np.random.default_rng(seed)
    channel_seeds = {ch.name: int(rng.integers(0, 2**31))
                     for ch in model.channels}

    generated = {}
    remaining = list(model.channels)
    max_iterations = len(remaining) + 1

    for _ in range(max_iterations):
        if not remaining:
            break
        progress = False
        for ch in remaining[:]:
            if all(dep in generated for dep in ch.depends_on):
                generated[ch.name] = ch.generate_fn(
                    trajectory, t_grid, params, aux,
                    prior_channels=generated,
                    seed=channel_seeds[ch.name],
                )
                remaining.remove(ch)
                progress = True
        if not progress:
            unresolved = [c.name for c in remaining]
            raise ValueError(
                f"Circular or unresolvable channel dependencies: {unresolved}. "
                f"Already generated: {list(generated.keys())}"
            )

    return generated
