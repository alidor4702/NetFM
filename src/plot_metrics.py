"""
Render a 3×4 training_metrics.png from a per-epoch CSV log.

Panels (row-major):
  row 1 — losses:    total L, mask L, link L, sub L
  row 2 — dynamics:  σ values, α mix gate, learning rate, grad norm
  row 3 — quality:   link AUC, sub R² (clust/tri), effective rank, per-graph visits
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _per_graph_visit_matrix(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Per-epoch visits are logged as a JSON column `visits` mapping graph->count.
    Returns (graph_names, [E, G] matrix of visit counts).
    """
    rows = [json.loads(v) for v in df["visits"].tolist()]
    all_names = sorted({k for r in rows for k in r.keys()})
    mat = np.array([[r.get(n, 0) for n in all_names] for r in rows], dtype=float)
    return all_names, mat


def plot_training_metrics(csv_path: Path, out_path: Path, run_id: str) -> None:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return
    e = df["epoch"].to_numpy()

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    fig.suptitle(f"NetFM pre-training — run {run_id}", fontsize=14)

    # row 1 — losses
    axes[0, 0].plot(e, df["loss_total"], color="C0")
    axes[0, 0].set_title("Total loss (uncertainty-weighted)")
    axes[0, 1].plot(e, df["loss_mask"], color="C1")
    axes[0, 1].set_title("Masked-feature loss (MSE)")
    axes[0, 2].plot(e, df["loss_link"], color="C2")
    axes[0, 2].set_title("Link-prediction loss (BCE)")
    axes[0, 3].plot(e, df["loss_sub"], color="C3")
    axes[0, 3].set_title("Subgraph-property loss (MSE)")

    # row 2 — dynamics
    axes[1, 0].plot(e, df["sigma_mask"], label="σ_mask")
    axes[1, 0].plot(e, df["sigma_link"], label="σ_link")
    axes[1, 0].plot(e, df["sigma_sub"], label="σ_sub")
    axes[1, 0].set_title("Per-head uncertainty σ")
    axes[1, 0].legend(loc="best", fontsize=8)

    axes[1, 1].plot(e, df["alpha"], color="C4")
    axes[1, 1].axhline(0.5, color="grey", linestyle="--", linewidth=0.7)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Mix gate α (structural ↔ SVD)")

    axes[1, 2].plot(e, df["lr"], color="C5")
    axes[1, 2].set_title("Learning rate (cosine)")
    axes[1, 2].set_yscale("log")

    axes[1, 3].plot(e, df["grad_norm"], color="C6")
    axes[1, 3].axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
    axes[1, 3].set_title("Grad norm (pre-clip)")

    # row 3 — quality
    axes[2, 0].plot(e, df["link_auc"], color="C2")
    axes[2, 0].axhline(0.5, color="grey", linestyle="--", linewidth=0.7)
    axes[2, 0].set_ylim(0.4, 1.02)
    axes[2, 0].set_title("Link AUC (train)")

    axes[2, 1].plot(e, df["sub_r2_clust"], label="clustering", color="C3")
    axes[2, 1].plot(e, df["sub_r2_tri"], label="triangles", color="C8")
    axes[2, 1].axhline(0.0, color="grey", linestyle="--", linewidth=0.7)
    axes[2, 1].set_title("Subgraph R² per target")
    axes[2, 1].legend(loc="best", fontsize=8)

    axes[2, 2].plot(e, df["eff_rank"], color="C9")
    axes[2, 2].set_title("Effective rank of embeddings")
    axes[2, 2].set_ylabel("exp(H(eig(cov)))")

    names, visit_mat = _per_graph_visit_matrix(df)
    axes[2, 3].stackplot(e, visit_mat.T, labels=names)
    axes[2, 3].set_title("Per-graph visits (cumulative per epoch)")
    axes[2, 3].legend(loc="upper left", fontsize=6, ncol=2)

    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
