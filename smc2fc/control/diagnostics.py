"""Reusable plotting helpers for SMC²-as-controller diagnostics.

Each function takes simple numpy arrays + an output path (or matplotlib
axes) and produces a publication-ready panel. Per-model scripts compose
multi-panel figures from these.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


def plot_cost_histogram(
    *,
    particle_costs: np.ndarray,
    references: Mapping[str, float],
    title: str = '',
    xlabel: str = 'cost',
    ax=None,
    out_path: str | None = None,
) -> None:
    """Histogram of per-particle costs with reference vertical lines.

    Args:
        particle_costs: shape (n_smc,)
        references: dict {label: value} drawn as vertical lines.
        title, xlabel: plot labels.
        ax: optional matplotlib Axes; if None creates a new figure.
        out_path: if provided, saves the figure to disk.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax.figure

    ax.hist(particle_costs, bins=30, color='steelblue', alpha=0.7,
              label='SMC² per-particle cost')

    cmap = plt.get_cmap('tab10')
    for i, (label, value) in enumerate(references.items()):
        if value is None:
            continue
        color = cmap(i % 10)
        ax.axvline(value, linestyle='--', linewidth=2,
                     color=color, label=f'{label} = {value:.3g}')

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
    out_path: str | None = None,
) -> None:
    """Overlay multiple schedules on the same axes.

    Args:
        t_grid: shape (n_steps,)
        schedules: dict {label: array(n_steps,)}
        h_lines: dict {label: y_value} drawn as horizontal lines.
        v_lines: dict {label: x_value} drawn as vertical lines.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    cmap = plt.get_cmap('tab10')
    for i, (label, schedule) in enumerate(schedules.items()):
        ax.plot(t_grid, np.asarray(schedule), '-', lw=1.7,
                  color=cmap(i % 10), label=label, alpha=0.85)

    for label, y in h_lines.items():
        ax.axhline(y, linestyle=':', alpha=0.5, label=f'{label} = {y:.3g}')
    for label, x in v_lines.items():
        ax.axvline(x, linestyle='--', alpha=0.4, label=f'{label} = {x:.3g}')

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
    out_path: str | None = None,
) -> None:
    """Sample-path overlay with mean trajectory.

    Args:
        t_grid: shape (n_steps,)
        trajectories: shape (n_traj, n_steps); plots the first n_show.
        mean_label: label for the mean trajectory.
        label_lines: dict {label: y_value} for horizontal reference lines.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.figure

    n = min(n_show, trajectories.shape[0])
    for i in range(n):
        ax.plot(t_grid, np.asarray(trajectories[i]), alpha=0.4,
                  lw=0.7, color=color)
    ax.plot(t_grid, np.asarray(trajectories[:n].mean(axis=0)), '-',
              lw=2, color=color, label=mean_label)

    for label, y in label_lines.items():
        ax.axhline(y, linestyle=':', alpha=0.5,
                     label=f'{label} = {y:.3g}')

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
    """One-line tempering-progress log."""
    print(f"    step {step:3d}: λ={lam:.3f}  acc={acc:.3f}  "
          f"mean cost = {mean_cost:.3f}")


def evaluate_gates(
    *,
    spec, result: dict, print_table: bool = True,
) -> dict:
    """Evaluate every acceptance gate in spec.acceptance_gates on result.

    Returns a dict {gate_name: (passed, value, message)}.
    """
    out = {}
    if print_table:
        print(f"  Acceptance gates for {spec.name}:")
    all_pass = True
    for name, gate_fn in spec.acceptance_gates.items():
        passed, value, message = gate_fn(result)
        out[name] = (passed, value, message)
        if not passed:
            all_pass = False
        if print_table:
            mark = '✓' if passed else '✗'
            print(f"    {name:<40s}  {mark}  {message}")
    if print_table:
        print(f"  {'✓' if all_pass else '✗'} "
              f"{'all gates pass' if all_pass else 'one or more gates fail'}")
    return out
