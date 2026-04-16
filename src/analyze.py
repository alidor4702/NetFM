"""
NetFM Data Analysis
Computes structural, label, and feature statistics for every dataset in the
registry and writes figures + a master summary to figures/data/.
"""

import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.data import (
    DATASET_REGISTRY,
    HELDOUT_DATASETS,
    PRETRAIN_DATASETS,
    NetFMGraph,
    load_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_ROOT = PROJECT_ROOT / "figures" / "data"
CLUSTERING_SAMPLE = 2000      # nodes sampled for avg clustering on large graphs
CLUSTERING_NX_MAX_NODES = 20_000   # use full nx.average_clustering below this
DEG_HIST_BINS = 60
SEED = 0


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def degree_array(g: NetFMGraph) -> np.ndarray:
    """Return per-node degree. edge_index is undirected (both directions)."""
    row = g.edge_index[0].numpy()
    return np.bincount(row, minlength=g.num_nodes)


def adjacency_csr(g: NetFMGraph) -> csr_matrix:
    row = g.edge_index[0].numpy()
    col = g.edge_index[1].numpy()
    data = np.ones(len(row), dtype=np.int8)
    return csr_matrix((data, (row, col)), shape=(g.num_nodes, g.num_nodes))


def sampled_clustering(adj: csr_matrix, n_samples: int, rng: np.random.Generator):
    """Average local clustering on a random node sample, plus per-sample values."""
    n = adj.shape[0]
    sample = rng.choice(n, size=min(n_samples, n), replace=False)
    values = []
    for u in sample:
        neigh = adj.indices[adj.indptr[u]:adj.indptr[u + 1]]
        k = len(neigh)
        if k < 2:
            values.append(0.0)
            continue
        sub = adj[neigh][:, neigh]
        # directed both ways in adj, so triangle count = nnz / 2
        triangles = sub.nnz / 2
        possible = k * (k - 1) / 2
        values.append(triangles / possible if possible else 0.0)
    values = np.asarray(values)
    return float(values.mean()), values


def label_stats(g: NetFMGraph):
    if g.node_labels is None:
        return None
    y = g.node_labels.numpy()
    if y.ndim == 2 and y.shape[1] > 1:
        # multi-label
        pos = y.astype(np.float32).mean(axis=0)
        return {
            "type": "multilabel",
            "num_labels": int(y.shape[1]),
            "mean_positive_rate": float(pos.mean()),
            "max_positive_rate": float(pos.max()),
            "min_positive_rate": float(pos.min()),
            "labels_per_node_mean": float(y.astype(np.float32).sum(axis=1).mean()),
            "per_class_positive_rate": pos.tolist(),
        }
    y = y.reshape(-1)
    valid = y[y >= 0] if y.dtype.kind in "iu" else y
    counts = np.bincount(valid.astype(np.int64))
    counts = counts[counts > 0]
    total = counts.sum()
    p = counts / total
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "type": "single",
        "num_classes": int(g.num_classes or len(counts)),
        "observed_classes": int(len(counts)),
        "majority_fraction": float(counts.max() / total),
        "minority_fraction": float(counts.min() / total),
        "entropy_nats": entropy,
        "entropy_max": float(np.log(len(counts))) if len(counts) > 1 else 0.0,
        "class_counts": counts.tolist(),
    }


def feature_stats(g: NetFMGraph):
    if g.original_features is None:
        return None
    x = g.original_features
    if not torch.is_floating_point(x):
        x = x.float()
    x_np = x.numpy()
    nz = np.count_nonzero(x_np)
    return {
        "dim": int(x_np.shape[1]),
        "nnz_fraction": float(nz / x_np.size),
        "mean": float(x_np.mean()),
        "std": float(x_np.std()),
        "min": float(x_np.min()),
        "max": float(x_np.max()),
    }


def analyze(g: NetFMGraph, rng: np.random.Generator) -> dict:
    undirected_edges = int(g.edge_index.size(1) // 2)
    deg = degree_array(g)
    adj = adjacency_csr(g)

    n_cc, cc_labels = connected_components(adj, directed=False)
    cc_sizes = np.bincount(cc_labels)
    largest_cc = int(cc_sizes.max())

    avg_cluster, cluster_samples = sampled_clustering(adj, CLUSTERING_SAMPLE, rng)

    density = (
        2 * undirected_edges / (g.num_nodes * (g.num_nodes - 1))
        if g.num_nodes > 1 else 0.0
    )

    stats = {
        "name": g.name,
        "domain": g.domain,
        "split": g.split,
        "num_nodes": int(g.num_nodes),
        "num_edges_undirected": undirected_edges,
        "density": float(density),
        "degree_mean": float(deg.mean()),
        "degree_median": float(np.median(deg)),
        "degree_std": float(deg.std()),
        "degree_max": int(deg.max()),
        "degree_min": int(deg.min()),
        "isolated_nodes": int((deg == 0).sum()),
        "num_connected_components": int(n_cc),
        "largest_cc_fraction": float(largest_cc / g.num_nodes),
        "avg_clustering_sampled": avg_cluster,
        "clustering_sample_size": int(len(cluster_samples)),
        "has_features": g.original_features is not None,
        "features": feature_stats(g),
        "labels": label_stats(g),
    }
    stats["_degree_raw"] = deg  # dropped before json dump
    stats["_clustering_samples"] = cluster_samples
    return stats


# ---------------------------------------------------------------------------
# Per-dataset figures
# ---------------------------------------------------------------------------

def plot_degree(deg: np.ndarray, name: str, out: Path):
    fig, ax = plt.subplots(figsize=(5, 3.6))
    pos = deg[deg > 0]
    if len(pos) == 0:
        plt.close(fig)
        return
    logbins = np.logspace(np.log10(1), np.log10(pos.max()), DEG_HIST_BINS)
    ax.hist(pos, bins=logbins, color="#2b7bba", edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} · degree distribution")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_clustering(samples: np.ndarray, name: str, out: Path):
    if len(samples) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.hist(samples, bins=40, color="#e07b39", edgecolor="none")
    ax.set_xlabel("Local clustering coefficient")
    ax.set_ylabel("Count (sampled nodes)")
    ax.set_title(f"{name} · clustering (n={len(samples)} sampled)")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_labels(stats: dict, name: str, out: Path):
    lb = stats.get("labels")
    if lb is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if lb["type"] == "single":
        counts = np.array(lb["class_counts"])
        order = np.argsort(counts)[::-1]
        ax.bar(np.arange(len(counts)), counts[order], color="#4c9f70", edgecolor="none")
        ax.set_xlabel("Class (sorted by frequency)")
        ax.set_ylabel("Node count")
        ax.set_title(f"{name} · class distribution ({lb['observed_classes']} classes)")
    else:
        rates = np.array(lb["per_class_positive_rate"])
        order = np.argsort(rates)[::-1]
        ax.bar(np.arange(len(rates)), rates[order], color="#4c9f70", edgecolor="none")
        ax.set_xlabel("Label (sorted by prevalence)")
        ax.set_ylabel("Positive rate")
        ax.set_title(f"{name} · multi-label prevalence ({lb['num_labels']} labels)")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-dataset figures
# ---------------------------------------------------------------------------

DOMAIN_COLORS = {
    "citation": "#2b7bba",
    "social": "#e07b39",
    "biological": "#c0392b",
    "infrastructure": "#7f8c8d",
    "collaboration": "#4c9f70",
}


def _bar_by_domain(ax, names, values, domains, ylabel, title, log=False):
    colors = [DOMAIN_COLORS.get(d, "#888") for d in domains]
    bars = ax.bar(range(len(names)), values, color=colors, edgecolor="none")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    return bars


def plot_overview(rows: list[dict], out_dir: Path):
    names = [r["name"] for r in rows]
    domains = [r["domain"] for r in rows]
    nodes = [r["num_nodes"] for r in rows]
    edges = [r["num_edges_undirected"] for r in rows]
    density = [r["density"] for r in rows]
    avg_deg = [r["degree_mean"] for r in rows]
    clustering = [r["avg_clustering_sampled"] for r in rows]
    largest_cc = [r["largest_cc_fraction"] for r in rows]

    # sizes
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _bar_by_domain(axes[0], names, nodes, domains, "Nodes", "Node count", log=True)
    _bar_by_domain(axes[1], names, edges, domains, "Undirected edges", "Edge count", log=True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DOMAIN_COLORS.values()]
    fig.legend(handles, DOMAIN_COLORS.keys(), loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "overview_sizes.png", dpi=130)
    plt.close(fig)

    # structure
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    _bar_by_domain(axes[0, 0], names, density, domains, "Density", "Edge density", log=True)
    _bar_by_domain(axes[0, 1], names, avg_deg, domains, "Mean degree", "Average degree", log=True)
    _bar_by_domain(axes[1, 0], names, clustering, domains, "Avg clustering", "Sampled avg clustering")
    _bar_by_domain(axes[1, 1], names, largest_cc, domains, "Largest CC / N", "Largest connected component fraction")
    fig.tight_layout()
    fig.savefig(out_dir / "overview_structure.png", dpi=130)
    plt.close(fig)

    # domain + split breakdown
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    domain_counts = {}
    for d in domains:
        domain_counts[d] = domain_counts.get(d, 0) + 1
    axes[0].bar(domain_counts.keys(), domain_counts.values(),
                color=[DOMAIN_COLORS[d] for d in domain_counts])
    axes[0].set_title("Datasets per domain")
    axes[0].set_ylabel("Count")
    split_counts = {"pretrain": 0, "held_out": 0}
    for r in rows:
        split_counts[r["split"]] += 1
    axes[1].bar(split_counts.keys(), split_counts.values(),
                color=["#2b7bba", "#c0392b"])
    axes[1].set_title("Pre-train vs held-out")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "overview_domains.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_stats_json(stats: dict, path: Path):
    clean = {k: v for k, v in stats.items() if not k.startswith("_")}
    path.write_text(json.dumps(clean, indent=2))


def write_summary_csv(rows: list[dict], path: Path):
    cols = [
        "name", "domain", "split", "num_nodes", "num_edges_undirected",
        "density", "degree_mean", "degree_median", "degree_max", "isolated_nodes",
        "num_connected_components", "largest_cc_fraction",
        "avg_clustering_sampled", "has_features",
    ]
    lines = [",".join(cols)]
    for r in rows:
        line = [str(r.get(c, "")) for c in cols]
        lines.append(",".join(line))
    path.write_text("\n".join(lines) + "\n")


def write_summary_md(rows: list[dict], path: Path):
    header = (
        "| name | domain | split | nodes | edges | density | avg_deg | "
        "avg_cluster | CCs | largest_CC | features | labels |"
    )
    sep = "|" + "---|" * 12
    lines = [header, sep]
    for r in rows:
        feat = "-"
        if r["features"]:
            feat = f"{r['features']['dim']}-d"
        lb = "-"
        if r["labels"]:
            if r["labels"]["type"] == "single":
                lb = f"{r['labels']['observed_classes']} cls"
            else:
                lb = f"{r['labels']['num_labels']} (multi)"
        lines.append(
            "| {name} | {domain} | {split} | {nodes:,} | {edges:,} | "
            "{dens:.2e} | {avgd:.2f} | {clust:.3f} | {cc} | {lcc:.3f} | "
            "{feat} | {lb} |".format(
                name=r["name"], domain=r["domain"], split=r["split"],
                nodes=r["num_nodes"], edges=r["num_edges_undirected"],
                dens=r["density"], avgd=r["degree_mean"],
                clust=r["avg_clustering_sampled"],
                cc=r["num_connected_components"], lcc=r["largest_cc_fraction"],
                feat=feat, lb=lb,
            )
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    ordered = PRETRAIN_DATASETS + HELDOUT_DATASETS
    for name in ordered:
        if name not in DATASET_REGISTRY:
            continue
        g = load_dataset(name)
        stats = analyze(g, rng)

        out_dir = FIG_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_degree(stats["_degree_raw"], name, out_dir / "degree_dist.png")
        plot_clustering(stats["_clustering_samples"], name, out_dir / "clustering_dist.png")
        plot_labels(stats, name, out_dir / "label_dist.png")
        write_stats_json(stats, out_dir / "stats.json")

        print(f"  -> {name}: nodes={stats['num_nodes']:,} "
              f"edges={stats['num_edges_undirected']:,} "
              f"avg_deg={stats['degree_mean']:.2f} "
              f"cluster={stats['avg_clustering_sampled']:.3f}")

        stripped = {k: v for k, v in stats.items() if not k.startswith("_")}
        rows.append(stripped)

    plot_overview(rows, FIG_ROOT)
    write_summary_csv(rows, FIG_ROOT / "summary.csv")
    write_summary_md(rows, FIG_ROOT / "summary.md")

    print(f"\nWrote analysis for {len(rows)} datasets to {FIG_ROOT}")


if __name__ == "__main__":
    main()
