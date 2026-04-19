"""
Community detection on held-out graphs.

For each held-out graph we:
  1. Compute embeddings with each method (random, structural, svd, netfm, node2vec).
  2. Run k-means with k = number of true classes (or Louvain's number of
     communities when no labels exist).
  3. Score the resulting partition against:
       - ground-truth labels (NMI, ARI)  — when labels are available
       - Louvain communities (NMI)       — always, using NetworkX's
         community.louvain_communities on the graph itself.

Usage:
    python -m src.eval_community \
        --checkpoint outputs/training/<run_id>/encoder.pt \
        --datasets lastfm_asia,ogbn_arxiv,euro_road \
        --run-name community_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from src.baselines import (
    embed_netfm,
    embed_node2vec,
    embed_random,
    embed_structural,
    embed_svd,
)
from src.data import load_dataset


DEFAULT_METHODS = ("random", "structural", "svd", "netfm", "node2vec")
DEFAULT_DATASETS = ("lastfm_asia", "ogbn_arxiv", "euro_road")


def _get_embeddings(method, graph, checkpoint, device, seed):
    if method == "random":
        return embed_random(graph.num_nodes, d=256, seed=seed)
    if method == "structural":
        return embed_structural(graph)
    if method == "svd":
        return embed_svd(graph)
    if method == "netfm":
        return embed_netfm(graph, checkpoint, device)
    if method == "node2vec":
        return embed_node2vec(graph, device)
    raise ValueError(method)


def _louvain_partition(graph) -> np.ndarray:
    """Run Louvain on the graph's edge_index; return per-node community ids."""
    ei = graph.edge_index.cpu().numpy()
    G = nx.Graph()
    G.add_nodes_from(range(graph.num_nodes))
    mask = ei[0] < ei[1]
    G.add_edges_from(zip(ei[0, mask].tolist(), ei[1, mask].tolist()))
    communities = nx.community.louvain_communities(G, seed=0)
    labels = np.zeros(graph.num_nodes, dtype=np.int64)
    for cid, nodes in enumerate(communities):
        for n in nodes:
            labels[n] = cid
    return labels


def _kmeans(embeddings: np.ndarray, k: int, seed: int) -> np.ndarray:
    emb = embeddings.astype(np.float32)
    std = emb.std(axis=0).max()
    if std < 1e-8:
        # Degenerate embedding: assign everything to one cluster.
        return np.zeros(emb.shape[0], dtype=np.int64)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(emb)


def _plot_community(out_path: Path, title: str, metrics: dict, k: int, n: int):
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = [k for k in metrics if not np.isnan(metrics[k])]
    vals = [metrics[k] for k in keys]
    bars = ax.bar(range(len(keys)), vals, color="#4C72B0")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylim(0, max(vals + [0.01]) * 1.2)
    ax.set_ylabel("score (higher = better)")
    ax.set_title(f"{title}\n(k={k} clusters, n={n:,} nodes)", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    ap.add_argument("--outputs-root", type=str, default="outputs/testing")
    ap.add_argument("--run-name", type=str, default="community")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.run_name}"
    run_dir = Path(args.outputs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots" / "community_detection"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    checkpoint = Path(args.checkpoint)

    rows = []
    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method", "dataset", "k",
            "nmi_labels", "ari_labels", "nmi_louvain",
            "n_nodes", "n_true_classes", "n_louvain_communities",
        ])
        w.writeheader()

        for ds in datasets:
            print(f"\n{'='*70}\nDATASET: {ds}\n{'='*70}")
            graph = load_dataset(ds)
            print(f"  {graph}")

            raw_labels = getattr(graph, "node_labels", None)
            has_labels = raw_labels is not None and raw_labels.dim() == 1
            if has_labels:
                labels = raw_labels.cpu().numpy().astype(np.int64)
                k = int(labels.max() + 1)
            else:
                labels = None
                k = None

            # Louvain is networkx-pure-Python; skip on graphs with > ~1M edges
            # where it takes many minutes. NMI-vs-labels is still reported.
            LOUVAIN_MAX_EDGES = 1_000_000
            n_edges = int(graph.edge_index.shape[1])
            if n_edges <= LOUVAIN_MAX_EDGES:
                t0 = time.time()
                louvain = _louvain_partition(graph)
                n_louv = int(louvain.max() + 1)
                print(f"  louvain found {n_louv} communities in {time.time()-t0:.1f}s")
            else:
                louvain = None
                n_louv = None
                print(f"  louvain SKIP (edges={n_edges:,} > {LOUVAIN_MAX_EDGES:,})")
            if k is None:
                k = n_louv if n_louv is not None else 8  # fallback

            dataset_plot_dir = plots_dir / ds
            dataset_plot_dir.mkdir(parents=True, exist_ok=True)

            for method in methods:
                print(f"\n  [community][{ds}] method={method}")
                try:
                    emb = _get_embeddings(method, graph, checkpoint, device, args.seed)
                except Exception as e:
                    print(f"    SKIP embed: {e}")
                    continue
                t0 = time.time()
                clusters = _kmeans(emb, k, args.seed)
                dt = time.time() - t0

                nmi_l = (normalized_mutual_info_score(labels, clusters)
                         if has_labels else float("nan"))
                ari_l = (adjusted_rand_score(labels, clusters)
                         if has_labels else float("nan"))
                nmi_louv = (normalized_mutual_info_score(louvain, clusters)
                            if louvain is not None else float("nan"))

                print(f"    k-means({k}) in {dt:.1f}s  "
                      f"NMI-labels={nmi_l:.4f}  ARI-labels={ari_l:.4f}  "
                      f"NMI-louvain={nmi_louv:.4f}")

                row = {
                    "method": method, "dataset": ds, "k": k,
                    "nmi_labels": "" if np.isnan(nmi_l) else nmi_l,
                    "ari_labels": "" if np.isnan(ari_l) else ari_l,
                    "nmi_louvain": "" if np.isnan(nmi_louv) else nmi_louv,
                    "n_nodes": graph.num_nodes,
                    "n_true_classes": k if has_labels else "",
                    "n_louvain_communities": n_louv if n_louv is not None else "",
                }
                w.writerow(row)
                rows.append(row)

                metrics = {
                    "NMI vs labels": nmi_l,
                    "ARI vs labels": ari_l,
                    "NMI vs Louvain": nmi_louv,
                }
                _plot_community(
                    dataset_plot_dir / f"{method}.png",
                    f"{ds} · {method} · community detection",
                    metrics, k=k, n=graph.num_nodes,
                )

    print(f"\nwrote {csv_path}")
    print(f"wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
