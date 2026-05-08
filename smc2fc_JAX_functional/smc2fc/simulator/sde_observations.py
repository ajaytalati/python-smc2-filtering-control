"""Generic observation-channel generation (purely-functional).

Generates synthetic observations from all of a model's channels in
dependency order. Model-agnostic: each channel is a pure function
provided by the model.

Refactor vs the imperative source:

    The original ``generate_all_channels`` used an outer ``for _ in
    range(max_iterations)`` Kahn's-style loop with mutable ``remaining
    = list(...)`` and ``remaining.remove(ch)`` mutations. The
    functional rewrite separates concerns:

    * :func:`_topo_sorted_channels` returns the channels in
      dependency order as an immutable tuple, no mutation.
    * :func:`generate_all_channels` then folds the sorted channels
      with ``functools.reduce``, building the output dict
      one-channel-at-a-time without ever mutating the input.
"""

import functools
from typing import Tuple

import numpy as np

from smc2fc.simulator.sde_model import ChannelSpec


def _topo_sorted_channels(channels: Tuple[ChannelSpec, ...]
                          ) -> Tuple[ChannelSpec, ...]:
    """Returns the channels in dependency-order via Kahn's algorithm.

    Pure: no mutation of the input. Each recursive step peels off the
    set of channels whose ``depends_on`` is a subset of the names
    already accumulated, then recurses on what remains.

    Args:
        channels: All channels declared by the model, in declaration
            order.

    Returns:
        A tuple of the same channels reordered such that every
        channel's dependencies appear earlier in the tuple.

    Raises:
        ValueError: If the dependency graph contains a cycle (i.e.
            no progress can be made on a recursive step).
    """
    def step(remaining: Tuple[ChannelSpec, ...],
             accumulated: Tuple[ChannelSpec, ...]) -> Tuple[ChannelSpec, ...]:
        if not remaining:
            return accumulated
        names_so_far = frozenset(c.name for c in accumulated)
        ready = tuple(ch for ch in remaining
                      if all(dep in names_so_far for dep in ch.depends_on))
        if not ready:
            unresolved = [c.name for c in remaining]
            raise ValueError(
                f"Circular or unresolvable channel dependencies: {unresolved}. "
                f"Already generated: {[c.name for c in accumulated]}"
            )
        ready_names = frozenset(c.name for c in ready)
        new_remaining = tuple(ch for ch in remaining
                              if ch.name not in ready_names)
        return step(new_remaining, accumulated + ready)

    return step(tuple(channels), ())


def generate_all_channels(model, trajectory, t_grid, params, aux, seed):
    """Generates observations for every channel, in dependency order.

    Pipeline:

        1. Build a per-channel seed map from a single integer seed.
        2. Topologically sort the channels by their declared
           ``depends_on`` graph.
        3. Fold the sorted channels with :func:`functools.reduce`,
           passing each channel the outputs of every previously-
           generated channel via the ``prior_channels`` kwarg.

    Args:
        model: An ``SDEModel`` whose ``channels`` attribute lists the
            model-specific observation specs.
        trajectory: Latent trajectory array of shape ``(T, n_states)``.
        t_grid: Time-grid array of shape ``(T,)``.
        params: Dict of dynamics parameters (whichever the model
            consumes).
        aux: Whatever the model's ``make_aux_fn`` returned (opaque to
            this driver).
        seed: Top-level integer seed; per-channel seeds are derived
            from it deterministically.

    Returns:
        Dict mapping each channel name to whatever its
        ``generate_fn`` returned (typically a dict of arrays).

    Raises:
        ValueError: If the channel dependency graph has a cycle.
    """
    rng = np.random.default_rng(seed)
    channel_seeds = {ch.name: int(rng.integers(0, 2**31))
                     for ch in model.channels}

    sorted_channels = _topo_sorted_channels(tuple(model.channels))

    def fold_one(generated: dict, ch: ChannelSpec) -> dict:
        """Computes one channel and returns a new dict (no mutation)."""
        return {**generated,
                ch.name: ch.generate_fn(
                    trajectory, t_grid, params, aux,
                    prior_channels=generated,
                    seed=channel_seeds[ch.name],
                )}

    return functools.reduce(fold_one, sorted_channels, {})
