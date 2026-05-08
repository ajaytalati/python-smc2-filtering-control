"""Reusable plotting helpers for SMC²-as-controller diagnostics.

Each function takes simple numpy arrays + an output path (or
matplotlib axes) and produces a publication-ready panel. Per-model
scripts compose multi-panel figures from these.

Functional refactor:

    The 7 imperative ``for`` statements in the source (one per
    matplotlib loop) are replaced by ``tuple(map(...))`` patterns.
    Matplotlib itself is intrinsically stateful, but the loop
    *statements* at function-body level are eliminated. The
    ``acceptance_gates`` evaluator uses ``functools.reduce`` to
    build the result dict without mutation of an outer ``out``
    variable.
"""

from __future__ import annotations

import functools
import os
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_cost_histogram(
    *,
    particle_costs: np.ndarray,
    references: Mapping[str, float],
    title: str = '',
    xlabel: str = 'cost',
    ax=None,
    out_path: Optional[str] = None,
) -> None:
    """Plots a histogram of per-particle costs with reference vlines.

    Args:
        particle_costs: Per-particle scalar cost, shape ``(n_smc,)``.
        references: Mapping ``{label: value}``; each value is drawn as
            a labelled vertical reference line. ``None`` values are
            skipped.
        title: Plot title (empty disables the title).
        xlabel: X-axis label.
        ax: Optional matplotlib ``Axes``; if None a new figure is
            created.
        out_path: If provided, saves the figure to disk (creating
            parent dirs as needed) and closes the figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax.figure

    ax.hist(particle_costs, bins=30, color='steelblue', alpha=0.7,
            label='SMC² per-particle cost')

    cmap = plt.get_cmap('tab10')

    def _draw_ref(idx_label_val) -> None:
        i, (label, value) = idx_label_val
        if value is None:
            return
        ax.axvline(value, linestyle='--', linewidth=2,
                   color=cmap(i % 10),
                   label=f'{label} = {value:.3g}')

    tuple(map(_draw_ref, enumerate(references.items())))

    ax.set_xlabel(xlabel)
    ax.set_ylabel('density')
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)


def plot_schedule_comparison(
    *,
    t_grid: np.ndarray,
    schedules: Mapping[str, np.ndarray],
    title: str = '',
    xlabel: str = 'time',
    ylabel: str = 'u',
    h_lines: Mapping[str, float] = {},
    v_lines: Mapping[str, float] = {},
    ax=None,
    out_path: Optional[str] = None,
) -> None:
    """Overlays multiple schedules on the same axes.

    Args:
        t_grid: Time-grid array of shape ``(n_steps,)``.
        schedules: Mapping ``{label: array(n_steps,)}``; each is
            plotted as a labelled line in a tab10 colour.
        title: Plot title (empty disables the title).
        xlabel: X-axis label.
        ylabel: Y-axis label.
        h_lines: Mapping ``{label: y_value}`` drawn as labelled
            horizontal lines.
        v_lines: Mapping ``{label: x_value}`` drawn as labelled
            vertical lines.
        ax: Optional matplotlib ``Axes``; if None a new figure is
            created.
        out_path: If provided, saves the figure to disk and closes
            it.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    cmap = plt.get_cmap('tab10')

    def _draw_schedule(idx_label_sched) -> None:
        i, (label, schedule) = idx_label_sched
        ax.plot(t_grid, np.asarray(schedule), '-', lw=1.7,
                color=cmap(i % 10), label=label, alpha=0.85)

    tuple(map(_draw_schedule, enumerate(schedules.items())))

    tuple(map(
        lambda label_y: ax.axhline(
            label_y[1], linestyle=':', alpha=0.5,
            label=f'{label_y[0]} = {label_y[1]:.3g}'),
        h_lines.items(),
    ))
    tuple(map(
        lambda label_x: ax.axvline(
            label_x[1], linestyle='--', alpha=0.4,
            label=f'{label_x[0]} = {label_x[1]:.3g}'),
        v_lines.items(),
    ))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)


def plot_trajectories(
    *,
    t_grid: np.ndarray,
    trajectories: np.ndarray,
    mean_label: str = 'mean',
    label_lines: Mapping[str, float] = {},
    title: str = '',
    xlabel: str = 'time',
    ylabel: str = 'state',
    color: str = 'steelblue',
    n_show: int = 20,
    ax=None,
    out_path: Optional[str] = None,
) -> None:
    """Sample-path overlay with the mean trajectory.

    Args:
        t_grid: Time-grid of shape ``(n_steps,)``.
        trajectories: Sample-path array of shape ``(n_traj, n_steps)``;
            the first ``n_show`` are drawn as faint individual lines
            and a thick mean line is overlaid.
        mean_label: Label for the mean trajectory.
        label_lines: Mapping ``{label: y_value}`` for horizontal
            reference lines.
        title: Plot title (empty disables the title).
        xlabel: X-axis label.
        ylabel: Y-axis label.
        color: Matplotlib colour for both the sample paths and the
            mean.
        n_show: Maximum number of sample paths to draw.
        ax: Optional matplotlib ``Axes``; if None a new figure is
            created.
        out_path: If provided, saves the figure to disk and closes
            it.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    n = min(n_show, trajectories.shape[0])

    tuple(map(
        lambda i: ax.plot(t_grid, np.asarray(trajectories[i]),
                          alpha=0.4, lw=0.7, color=color),
        range(n),
    ))
    ax.plot(t_grid, np.asarray(trajectories[:n].mean(axis=0)), '-',
            lw=2, color=color, label=mean_label)

    tuple(map(
        lambda label_y: ax.axhline(
            label_y[1], linestyle=':', alpha=0.5,
            label=f'{label_y[0]} = {label_y[1]:.3g}'),
        label_lines.items(),
    ))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)


def print_smc_step(*, step: int, lam: float, acc: float,
                   mean_cost: float) -> None:
    """Prints a one-line tempering-progress log.

    Args:
        step: Tempering-step index (1-based for human readability).
        lam: Current tempering λ.
        acc: Mean MCMC acceptance rate at this level.
        mean_cost: Posterior-mean cost at this level.
    """
    print(f"    step {step:3d}: λ={lam:.3f}  acc={acc:.3f}  "
          f"mean cost = {mean_cost:.3f}")


def evaluate_gates(
    *,
    spec, result: dict, print_table: bool = True,
) -> dict:
    """Evaluates every acceptance gate in ``spec.acceptance_gates``.

    Built declaratively via ``functools.reduce`` over the gates dict;
    no outer ``out`` variable is mutated. Optionally prints a
    pass/fail summary table to stdout.

    Args:
        spec: ``ControlSpec``-like object exposing ``name`` and an
            ``acceptance_gates`` mapping ``{gate_name: gate_fn}`` where
            each ``gate_fn(result) -> (passed: bool, value, message)``.
        result: The bench output dict that gates are evaluated on.
        print_table: If True, prints a formatted gate-by-gate table.

    Returns:
        Dict mapping gate name to ``(passed, value, message)``.
    """
    if print_table:
        print(f"  Acceptance gates for {spec.name}:")

    def _evaluate(acc_dict: dict, name_fn) -> dict:
        name, gate_fn = name_fn
        passed, value, message = gate_fn(result)
        if print_table:
            mark = '✓' if passed else '✗'
            print(f"    {name:<40s}  {mark}  {message}")
        return {**acc_dict, name: (passed, value, message)}

    out = functools.reduce(_evaluate, spec.acceptance_gates.items(), {})
    if print_table:
        all_pass = all(v[0] for v in out.values())
        print(f"  {'✓' if all_pass else '✗'} "
              f"{'all gates pass' if all_pass else 'one or more gates fail'}")
    return out
