"""
Visualise the four candidate soft-chance penalty shapes for FSA-v5.

Saves the PNG next to this script (so it lives in `model_notes_and_docs/`
alongside the spec PDF, not in /tmp).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

beta  = 50.0
scale = 0.10

def stable_softplus(x):
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

def current_sigmoid(A, A_thr):
    return 1.0 / (1.0 + np.exp(-beta * (A_thr - A) / scale))

def quadratic_hinge(A, A_thr):
    return np.maximum(A_thr - A, 0.0) ** 2

def softplus_hinge(A, A_thr):
    return (1.0 / beta) * stable_softplus(beta * (A_thr - A))

def sigmoid_gated_quad(A, A_thr):
    gate = 1.0 / (1.0 + np.exp(-beta * (A_thr - A) / scale))
    return gate * (A_thr - A) ** 2

A_grid  = np.linspace(0.0, 0.5, 1001)
A_thrs  = [0.1, 0.3]
A_probe = 0.05

curves = [
    ("A. Current sigmoid\nσ(β(A_thr-A)/scale)",      current_sigmoid,    "tab:gray"),
    ("B. Quadratic hinge\nmax(A_thr-A,0)²",          quadratic_hinge,    "tab:blue"),
    ("C. Softplus\n(1/β)·log(1+exp(β(A_thr-A)))",    softplus_hinge,     "tab:orange"),
    ("D. Sigmoid · quadratic\nσ · (A_thr-A)²",       sigmoid_gated_quad, "tab:green"),
]

# Grid: rows = penalty shape (4), cols = A_thr value (2). Each panel
# gets its OWN y-axis scale — overlaying all four with the sigmoid
# (max 1.0) flattened the small-magnitude shapes (quadratic hinge ≤
# 0.0625, sigmoid·quad ≤ 0.0625) against zero and they didn't render.
fig, axes = plt.subplots(len(curves), 2, figsize=(13, 14), sharex=True)

for row, (label, fn, color) in enumerate(curves):
    for col, A_thr in enumerate(A_thrs):
        ax = axes[row, col]
        y = fn(A_grid, A_thr)
        ax.plot(A_grid, y, color=color, lw=2)
        ax.fill_between(A_grid, 0, y, color=color, alpha=0.15)
        ax.axvline(A_thr, color="k", lw=0.8, alpha=0.6, label=f"A_thr = {A_thr}")
        ax.axvline(A_probe, color="r", lw=0.8, alpha=0.6, label=f"A = {A_probe} probe")
        probe_val = float(fn(np.array([A_probe]), A_thr)[0])
        ax.scatter([A_probe], [probe_val], s=40, color="red", zorder=5)
        ax.annotate(
            f"{probe_val:.4g}", xy=(A_probe, probe_val),
            xytext=(8, 8), textcoords="offset points",
            fontsize=10, color="red", fontweight="bold",
        )
        if row == 0:
            ax.set_title(f"A_thr = {A_thr}", fontsize=12)
        if col == 0:
            ax.set_ylabel(label, fontsize=10)
        if row == len(curves) - 1:
            ax.set_xlabel("A")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

plt.suptitle(
    "Soft-chance penalty candidates for FSA-v5 controller\n"
    "β=50, scale=0.10 — each panel has its own y-axis scale",
    fontsize=13, y=0.995
)
plt.tight_layout()

out_path = Path(__file__).parent / "chance_penalty_options.png"
plt.savefig(out_path, dpi=130, bbox_inches="tight")

print(f"{'shape':45s}  A_thr=0.1,A=0.05    A_thr=0.3,A=0.05    ratio")
for label, fn, _ in curves:
    v1 = float(fn(np.array([A_probe]), 0.1)[0])
    v2 = float(fn(np.array([A_probe]), 0.3)[0])
    ratio = v2 / v1 if v1 > 0 else float("inf")
    flat_label = label.replace("\n", " ")
    print(f"{flat_label:45s}  {v1:.6f}            {v2:.6f}            {ratio:.2f}x")

print(f"\nSaved figure to {out_path}")
