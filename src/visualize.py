"""
NetFM Graph Explorer

A native Qt dashboard (PySide6 + pyqtgraph) for visualizing NetFM datasets
or any user-provided graph file in 2D / 3D.

Usage:
    pip install -e .[viz]
    python -m src.visualize
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAYOUT_CACHE = PROJECT_ROOT / "figures" / "layouts"

DEFAULT_SAMPLE = 2000
SMALL_LIMIT = 3000
SEED = 0

# 20-entry qualitative palette (bright on dark background)
PALETTE = [
    "#5bc0de", "#f0ad4e", "#d9534f", "#5cb85c", "#9b59b6",
    "#e67e22", "#1abc9c", "#ecf0f1", "#f39c12", "#e74c3c",
    "#2ecc71", "#3498db", "#8e44ad", "#16a085", "#27ae60",
    "#d35400", "#2980b9", "#7f8c8d", "#bdc3c7", "#f8c471",
]

BG_DARK = "#07080c"
PANEL_BG = "#0d1017"
PANEL_BORDER = "#232b38"
TEXT = "#d6e4ee"
MUTED = "#6c7a8c"
ACCENT = "#5ad1b4"       # terminal teal
ACCENT_WARM = "#f2a154"  # amber accent

EDGE_INTRA = "#4a8fe6"
EDGE_CROSS = "#e05a5a"
EDGE_HIGHLIGHT = "#f2d35a"
NODE_HIGHLIGHT = "#f2d35a"

MONO = "'JetBrains Mono', 'SF Mono', 'Menlo', 'Consolas', 'monospace'"


# ---------------------------------------------------------------------------
# Known class-name registries (so tooltips/legends show real names)
# ---------------------------------------------------------------------------

CLASS_NAMES: dict[str, list[str]] = {
    "cora": [
        "Case_Based", "Genetic_Algorithms", "Neural_Networks",
        "Probabilistic_Methods", "Reinforcement_Learning",
        "Rule_Learning", "Theory",
    ],
    "citeseer": ["Agents", "AI", "DB", "IR", "ML", "HCI"],
    "pubmed": [
        "Diabetes_Experimental", "Diabetes_Type1", "Diabetes_Type2",
    ],
}


# Per-dataset node noun (singular). Cora/CiteSeer/PubMed/arxiv don't ship
# per-node titles in their public distributions — the best we can do without
# external joins is to call them by what they represent.
NODE_TYPE: dict[str, str] = {
    "cora": "paper", "citeseer": "paper", "pubmed": "paper",
    "ogbn_arxiv": "paper", "ogbn_mag": "paper",
    "ppi": "protein", "ogbn_proteins": "protein",
    "facebook_ego": "user", "twitch_en": "streamer", "lastfm_asia": "listener",
    "coauthor_cs": "author", "dblp_snap": "author",
    "power_grid": "station", "as733": "router", "euro_road": "junction",
}


# Per-dataset edge relation. All of these are single-relation graphs in the
# public distributions we load, so each dataset has one semantic edge type.
EDGE_TYPE: dict[str, str] = {
    "cora": "cites", "citeseer": "cites", "pubmed": "cites",
    "ogbn_arxiv": "cites", "ogbn_mag": "cites",
    "ppi": "interacts_with", "ogbn_proteins": "interacts_with",
    "facebook_ego": "friends_with", "twitch_en": "mutual_follow",
    "lastfm_asia": "friends_with",
    "coauthor_cs": "coauthored_with", "dblp_snap": "coauthored_with",
    "power_grid": "connected_to", "as733": "peers_with",
    "euro_road": "road_to",
}


def class_name(dataset: str, class_id: int) -> str:
    names = CLASS_NAMES.get(dataset)
    if names and 0 <= class_id < len(names):
        return names[class_id]
    return f"class {class_id}"


def node_label(dataset: str, original_id: int, names: Optional[list] = None) -> str:
    """Primary display name for one node. Prefers `names[i]` if the dataset
    shipped human names; otherwise falls back to '<noun> <id>'."""
    if names is not None and 0 <= original_id < len(names) and names[original_id]:
        return str(names[original_id])
    noun = NODE_TYPE.get(dataset, "node")
    return f"{noun} {int(original_id)}"


def edge_relation(dataset: str) -> str:
    return EDGE_TYPE.get(dataset, "connects_to")


def community_label(dataset: str, comm_id: int, class_labels: Optional[np.ndarray],
                     communities: np.ndarray) -> str:
    """Name a community by its dominant class when class labels are available:
    e.g. 'cluster 3 · Neural_Networks (72%)'. Falls back to 'cluster N' otherwise."""
    if class_labels is None:
        return f"cluster {int(comm_id)}"
    mask = communities == comm_id
    if not mask.any():
        return f"cluster {int(comm_id)}"
    within = class_labels[mask]
    within = within[within >= 0]
    if len(within) == 0:
        return f"cluster {int(comm_id)}"
    counts = np.bincount(within)
    top = int(np.argmax(counts))
    frac = counts[top] / len(within)
    nm = class_name(dataset, top)
    if frac >= 0.5:
        return f"cluster {int(comm_id)} · {nm} ({int(frac * 100)}%)"
    if frac >= 0.3:
        return f"cluster {int(comm_id)} · mostly {nm}"
    return f"cluster {int(comm_id)} · mixed"


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class GraphBundle:
    """Everything the viewer needs for one dataset render."""
    name: str
    domain: str
    split: str
    full_nodes: int
    keep_ids: np.ndarray          # original indices in source graph
    edge_index: np.ndarray        # remapped to [0, n)
    labels: np.ndarray            # per-node integer label
    label_kind: str               # "class" | "community" | "none"
    coords: np.ndarray            # (n, dim)
    dim: int
    colors: np.ndarray            # (n, 4) rgba in [0,1]
    legend: list[tuple[str, str]] # list of (label_str, hex_color)
    communities: np.ndarray       # Louvain assignments (for community layout)
    sample_method: str
    layout_name: str
    node_names: Optional[list] = None  # optional human-readable node names
    stats: dict = field(default_factory=dict)
    class_labels: Optional[np.ndarray] = None  # original class labels, if shipped


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def hex_to_rgba(h: str, alpha: int = 255):
    h = h.lstrip("#")
    return int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


def palette_for(labels: np.ndarray):
    uniq = sorted(np.unique(labels).tolist())
    colors = np.zeros((len(labels), 4), dtype=np.float32)
    legend: list[tuple[str, str]] = []
    for i, u in enumerate(uniq):
        hx = PALETTE[i % len(PALETTE)]
        r, g, b, _ = hex_to_rgba(hx)
        colors[labels == u] = [r / 255, g / 255, b / 255, 1.0]
        legend.append((str(int(u)), hx))
    return colors, legend


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _csr(edge_index: np.ndarray, n: int):
    row = edge_index[0]
    order = np.argsort(row, kind="stable")
    row_s = row[order]
    col_s = edge_index[1][order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, row_s + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, col_s


def sample_ego(ei, n, target, rng):
    deg = np.bincount(ei[0], minlength=n)
    top = np.argsort(deg)[::-1][:max(10, n // 1000)]
    seed = int(rng.choice(top))
    indptr, indices = _csr(ei, n)
    visited = {seed}
    frontier = [seed]
    while len(visited) < target and frontier:
        nxt = []
        for u in frontier:
            for v in indices[indptr[u]:indptr[u + 1]]:
                v = int(v)
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
                    if len(visited) >= target:
                        break
            if len(visited) >= target:
                break
        frontier = nxt
    return np.array(sorted(visited), dtype=np.int64)


def sample_random_walk(ei, n, target, rng, walk_len=20):
    indptr, indices = _csr(ei, n)
    deg = np.diff(indptr)
    nz = np.where(deg > 0)[0]
    if len(nz) == 0:
        return rng.choice(n, size=min(target, n), replace=False)
    seen: set[int] = set()
    attempts = 0
    while len(seen) < target and attempts < target * 20:
        u = int(rng.choice(nz))
        for _ in range(walk_len):
            seen.add(u)
            if len(seen) >= target:
                break
            s, e = indptr[u], indptr[u + 1]
            if e == s:
                break
            u = int(indices[rng.integers(s, e)])
        attempts += 1
    return np.array(sorted(list(seen))[:target], dtype=np.int64)


def sample_community(ei, n, target, rng):
    try:
        import community as louvain
        import networkx as nx
    except ImportError:
        return sample_random_walk(ei, n, target, rng)
    if n > 100_000:
        return sample_random_walk(ei, n, target, rng)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(ei.T.tolist())
    part = louvain.best_partition(G, random_state=SEED)
    by: dict[int, list[int]] = {}
    for u, c in part.items():
        by.setdefault(c, []).append(u)
    picked: list[int] = []
    for comm in sorted(by.values(), key=len, reverse=True):
        picked.extend(comm)
        if len(picked) >= target:
            break
    return np.array(sorted(picked[:target]), dtype=np.int64)


SAMPLERS: dict[str, Callable] = {
    "ego": sample_ego,
    "community": sample_community,
    "random_walk": sample_random_walk,
}


def induce_subgraph(ei: np.ndarray, keep: np.ndarray):
    max_id = int(ei.max()) + 2
    remap = -np.ones(max_id, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    mask = (remap[ei[0]] >= 0) & (remap[ei[1]] >= 0)
    sub = np.stack([remap[ei[0][mask]], remap[ei[1][mask]]], axis=0)
    return sub


# ---------------------------------------------------------------------------
# Community detection helper
# ---------------------------------------------------------------------------

def detect_communities(ei: np.ndarray, n: int) -> np.ndarray:
    try:
        import community as louvain
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(ei.T.tolist())
        part = louvain.best_partition(G, random_state=SEED)
        return np.array([part.get(i, 0) for i in range(n)], dtype=np.int64)
    except Exception:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(ei.T.tolist())
        comp = {u: i for i, cc in enumerate(nx.connected_components(G)) for u in cc}
        return np.array([comp.get(i, 0) for i in range(n)], dtype=np.int64)


# ---------------------------------------------------------------------------
# Layout algorithms
# ---------------------------------------------------------------------------

def _nx_graph(ei: np.ndarray, n: int):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(ei.T.tolist())
    return G


def _normalize(coords: np.ndarray) -> np.ndarray:
    coords = coords - coords.mean(axis=0)
    span = np.max(np.abs(coords))
    if span > 1e-9:
        coords = coords / span
    return coords.astype(np.float32)


def layout_spring(ei, n, dim, spacing=1.0, **_):
    import networkx as nx
    G = _nx_graph(ei, n)
    # spring layout density mostly depends on k relative to sqrt(n)
    # (stronger repulsion) and iterations (convergence). Boost both with
    # spacing so higher values actually push nodes apart, not just scale.
    k = spacing * 2.0 / max(np.sqrt(n), 1.0)
    iters = min(600, 150 + int(40 * spacing))
    pos = nx.spring_layout(G, dim=dim, seed=SEED, iterations=iters, k=k)
    coords = np.array([pos[i] for i in range(n)], dtype=np.float32)
    return _normalize(coords) * spacing


def layout_kamada_kawai(ei, n, dim, spacing=1.0, **_):
    import networkx as nx
    G = _nx_graph(ei, n)
    try:
        pos = nx.kamada_kawai_layout(G, dim=dim)
    except TypeError:
        if dim == 2:
            pos = nx.kamada_kawai_layout(G)
        else:
            p2 = nx.kamada_kawai_layout(G)
            pos = {k: np.array([v[0], v[1], 0.0]) for k, v in p2.items()}
    coords = np.array([pos[i] for i in range(n)], dtype=np.float32)
    return _normalize(coords) * spacing


def layout_spectral(ei, n, dim, spacing=1.0, **_):
    from scipy.sparse import csr_matrix, diags, eye
    from scipy.sparse.linalg import eigsh
    row, col = ei[0], ei[1]
    A = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    A = (A + A.T)
    A.data = np.ones_like(A.data)
    d = np.asarray(A.sum(axis=1)).flatten()
    d[d == 0] = 1.0
    d_inv = 1.0 / np.sqrt(d)
    L = eye(n) - diags(d_inv) @ A @ diags(d_inv)
    try:
        vals, vecs = eigsh(L, k=min(dim + 1, n - 1), which="SM")
    except Exception:
        rng = np.random.default_rng(SEED)
        return rng.normal(size=(n, dim)).astype(np.float32)
    coords = vecs[:, 1:dim + 1]
    if coords.shape[1] < dim:
        coords = np.hstack([coords, np.zeros((n, dim - coords.shape[1]))])
    # Whiten each axis so clusters aren't squashed into a line by a dominant
    # eigenvector, then scale by spacing.
    std = coords.std(axis=0) + 1e-9
    coords = coords / std
    return _normalize(coords) * spacing


def layout_circular(ei, n, dim, spacing=1.0, **_):
    if dim == 2:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coords = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)
    elif n == 1:
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    else:
        # Fibonacci sphere
        i = np.arange(n, dtype=np.float64)
        phi = np.pi * (3.0 - np.sqrt(5))
        y = 1 - (i / (n - 1)) * 2
        r = np.sqrt(np.maximum(0.0, 1 - y * y))
        theta = phi * i
        coords = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1).astype(np.float32)
    return coords * spacing


def layout_community(ei, n, dim, communities=None, spacing=1.0, **_):
    """Cluster-first layout: place each community at a well-separated center
    on a circle/sphere, then spring-embed each community around its center.
    This gives the most *interpretable* visual: groups are obvious, and edges
    within vs. across groups are visually distinct.
    """
    import networkx as nx
    if communities is None:
        communities = detect_communities(ei, n)
    n_comms = int(communities.max()) + 1
    # Inter-cluster radius grows with both the number of communities
    # (so they don't overlap) and with spacing. This is the knob that
    # actually changes what you *see*.
    inter_r = (2.5 + 1.8 * spacing) * max(1.0, np.sqrt(n_comms) / 2.0)
    # Intra-cluster radius shrinks a little so clusters look tight and
    # inter-cluster separation dominates.
    intra_r = 0.45 + 0.15 * spacing

    centers = layout_circular(None, n_comms, dim, spacing=1.0) * inter_r
    coords = np.zeros((n, dim), dtype=np.float32)
    for c in range(n_comms):
        nodes_c = np.where(communities == c)[0]
        if len(nodes_c) == 0:
            continue
        mask = np.isin(ei[0], nodes_c) & np.isin(ei[1], nodes_c)
        local_ei = ei[:, mask]
        if len(nodes_c) == 1 or local_ei.shape[1] == 0:
            local_pos = np.zeros((len(nodes_c), dim), dtype=np.float32)
        else:
            remap = -np.ones(n, dtype=np.int64)
            remap[nodes_c] = np.arange(len(nodes_c))
            local_ei_remapped = np.stack(
                [remap[local_ei[0]], remap[local_ei[1]]], axis=0
            )
            G = nx.Graph()
            G.add_nodes_from(range(len(nodes_c)))
            G.add_edges_from(local_ei_remapped.T.tolist())
            # Shorter spring run since clusters are small.
            pos = nx.spring_layout(G, dim=dim, seed=SEED, iterations=80,
                                    k=1.5 / max(np.sqrt(len(nodes_c)), 1.0))
            local_pos = np.array([pos[i] for i in range(len(nodes_c))],
                                  dtype=np.float32)
            # normalize each cluster so they're the same size regardless of n
            span = np.max(np.abs(local_pos)) + 1e-9
            local_pos = local_pos / span * intra_r
        coords[nodes_c] = centers[c] + local_pos
    return coords.astype(np.float32)


LAYOUTS: dict[str, Callable] = {
    "spring": layout_spring,
    "kamada_kawai": layout_kamada_kawai,
    "spectral": layout_spectral,
    "circular": layout_circular,
    "community": layout_community,
}


# ---------------------------------------------------------------------------
# Dataset / custom-file loading
# ---------------------------------------------------------------------------

def _try_load_names(dataset: str, num_nodes: int) -> Optional[list]:
    """Optional sidecar: if ``figures/data/<dataset>/node_names.{json,txt}``
    exists, treat it as the human-readable node name list. Otherwise we
    return None and the viewer falls back to ``<noun> <id>``.

    This exists because the standard Planetoid (Cora/CiteSeer/PubMed) and
    OGB distributions don't ship per-node titles — the only honest way to
    show real names is to let the user provide them.
    """
    import json
    base = PROJECT_ROOT / "figures" / "data" / dataset
    for fn in ("node_names.json", "node_names.txt"):
        p = base / fn
        if not p.exists():
            continue
        try:
            if fn.endswith(".json"):
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    return [data.get(str(i)) or data.get(i) for i in range(num_nodes)]
                if isinstance(data, list) and len(data) >= num_nodes:
                    return list(data)
            else:
                lines = p.read_text().splitlines()
                if len(lines) >= num_nodes:
                    return [ln.strip() for ln in lines[:num_nodes]]
        except Exception:
            pass
    return None


def load_registry_dataset(name: str):
    from src.data import load_dataset
    g = load_dataset(name)
    return {
        "name": g.name,
        "domain": g.domain,
        "split": g.split,
        "num_nodes": g.num_nodes,
        "edge_index": g.edge_index.numpy(),
        "labels": g.node_labels.numpy() if g.node_labels is not None else None,
        "node_names": _try_load_names(g.name, g.num_nodes),
    }


def load_custom_file(path: str):
    """Accept edge list (.txt/.csv), GraphML, GEXF."""
    import networkx as nx
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".graphml", ".xml"):
        G = nx.read_graphml(p)
    elif ext == ".gexf":
        G = nx.read_gexf(p)
    elif ext in (".gpickle", ".pkl"):
        G = nx.read_gpickle(p)
    else:
        # fallback: edge list
        delim = "," if ext == ".csv" else None
        G = nx.read_edgelist(p, delimiter=delim, nodetype=str)
    G = nx.convert_node_labels_to_integers(G, label_attribute="orig")
    n = G.number_of_nodes()
    ei = np.array(list(G.edges()), dtype=np.int64).T if G.number_of_edges() else np.zeros((2, 0), dtype=np.int64)
    if ei.size:
        # ensure undirected both directions
        ei = np.concatenate([ei, ei[::-1]], axis=1)
    # pull labels if stored on nodes under common keys
    labels = None
    for key in ("label", "class", "y", "community"):
        vals = nx.get_node_attributes(G, key)
        if vals and len(vals) == n:
            try:
                labels = np.array([int(vals[i]) for i in range(n)])
                break
            except (TypeError, ValueError):
                pass
    return {
        "name": p.stem,
        "domain": "custom",
        "split": "custom",
        "num_nodes": n,
        "edge_index": ei,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _cache_path(name: str, layout: str, method: str, size: int, dim: int,
                spacing: float) -> Path:
    s = f"{spacing:.2f}".replace(".", "p")
    return LAYOUT_CACHE / f"{name}_{layout}_{method}_n{size}_d{dim}_s{s}.npz"


def compute_stats(ei: np.ndarray, n: int, communities: np.ndarray,
                  labels: np.ndarray, label_kind: str) -> dict:
    """Graph-level stats for the sampled subgraph (shown in Statistics tab)."""
    m = ei.shape[1] // 2
    deg = np.bincount(ei[0], minlength=n).astype(np.int64)
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0.0
    # Connected components via networkx (cheap at sample sizes)
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(ei.T.tolist())
        cc = list(nx.connected_components(G))
        n_cc = len(cc)
        largest = max(len(c) for c in cc) if cc else 0
        # sampled clustering
        sample = min(n, 1000)
        idx = np.random.default_rng(SEED).choice(n, size=sample, replace=False)
        avg_clust = float(nx.average_clustering(G, nodes=idx.tolist()))
    except Exception:
        n_cc, largest, avg_clust = 0, 0, 0.0
    n_iso = int((deg == 0).sum())
    n_classes = int(len(np.unique(labels)))
    n_comms = int(len(np.unique(communities)))
    return {
        "nodes": int(n),
        "edges": int(m),
        "density": float(density),
        "degree_mean": float(deg.mean()) if n else 0.0,
        "degree_median": float(np.median(deg)) if n else 0.0,
        "degree_max": int(deg.max()) if n else 0,
        "isolated_nodes": n_iso,
        "connected_components": int(n_cc),
        "largest_cc": int(largest),
        "largest_cc_fraction": float(largest / n) if n else 0.0,
        "avg_clustering_sampled": float(avg_clust),
        f"num_{label_kind}": n_classes,
        "num_communities": n_comms,
    }


def build_bundle(source: dict, dim: int, sample_method: str, sample_size: int,
                 layout_name: str, color_by: str, spacing: float = 1.5,
                 progress: Callable[[str], None] = lambda _s: None) -> GraphBundle:
    progress("sampling")
    rng = np.random.default_rng(SEED)
    ei = source["edge_index"]
    full_n = source["num_nodes"]

    if full_n > sample_size:
        keep = SAMPLERS[sample_method](ei, full_n, sample_size, rng)
    else:
        keep = np.arange(full_n, dtype=np.int64)
        sample_method = "full"

    sub = induce_subgraph(ei, keep)
    n = len(keep)

    progress("community detection")
    communities = detect_communities(sub, n)

    cache = _cache_path(source["name"], layout_name, sample_method, n, dim, spacing)
    if cache.exists():
        progress("loading cached layout")
        coords = np.load(cache)["coords"]
    else:
        progress(f"computing {layout_name} layout")
        coords = LAYOUTS[layout_name](sub, n, dim, communities=communities,
                                       spacing=spacing)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, coords=coords)

    # node labels
    src_labels = source.get("labels")
    class_labels: Optional[np.ndarray] = None
    if src_labels is not None and src_labels.ndim == 1:
        class_labels = src_labels[keep].astype(np.int64)

    if color_by == "class" and class_labels is not None:
        labels = class_labels
        label_kind = "class"
    elif color_by == "degree":
        deg = np.bincount(sub[0], minlength=n)
        bins = np.linspace(0, deg.max() + 1, 8)
        labels = np.digitize(deg, bins).astype(np.int64)
        label_kind = "degree"
    else:
        labels = communities
        label_kind = "community"

    colors, legend = palette_for(labels)

    progress("computing statistics")
    stats = compute_stats(sub, n, communities, labels, label_kind)

    src_names = source.get("node_names")

    return GraphBundle(
        name=source["name"], domain=source["domain"], split=source["split"],
        full_nodes=full_n, keep_ids=keep, edge_index=sub, labels=labels,
        label_kind=label_kind, coords=coords, dim=dim, colors=colors,
        legend=legend, communities=communities, sample_method=sample_method,
        layout_name=layout_name, node_names=src_names, stats=stats,
        class_labels=class_labels,
    )


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT};
    font-family: {MONO};
    font-size: 12px;
}}
QFrame#sidebar, QFrame#legendPanel {{
    background-color: {PANEL_BG};
    border: 1px solid {PANEL_BORDER};
    border-radius: 2px;
}}
QLabel {{
    color: {TEXT};
}}
QLabel[role="section"] {{
    color: {ACCENT};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding-top: 4px;
    border-bottom: 1px dashed {PANEL_BORDER};
    padding-bottom: 2px;
    margin-bottom: 2px;
}}
QLabel[role="muted"] {{
    color: {MUTED};
    font-size: 11px;
}}
QLabel[role="title"] {{
    color: {ACCENT};
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2px;
}}
QLabel[role="subtitle"] {{
    color: {MUTED};
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background-color: {BG_DARK};
    color: {TEXT};
    border: 1px solid {PANEL_BORDER};
    border-radius: 2px;
    padding: 5px 8px;
    min-height: 22px;
    selection-background-color: {ACCENT};
    selection-color: {BG_DARK};
}}
QComboBox:focus, QSpinBox:focus, QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_BG};
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: {BG_DARK};
    border: 1px solid {PANEL_BORDER};
    outline: 0;
}}
QPushButton {{
    background-color: transparent;
    color: {ACCENT};
    font-weight: 700;
    letter-spacing: 1px;
    border: 1px solid {ACCENT};
    border-radius: 2px;
    padding: 7px 12px;
    text-transform: uppercase;
}}
QPushButton:hover {{ background-color: {ACCENT}; color: {BG_DARK}; }}
QPushButton:disabled {{ border-color: {PANEL_BORDER}; color: {MUTED}; }}
QPushButton[role="secondary"] {{
    color: {TEXT};
    border: 1px solid {PANEL_BORDER};
    font-weight: 500;
    letter-spacing: 0px;
    text-transform: none;
}}
QPushButton[role="secondary"]:hover {{ background-color: {PANEL_BORDER}; color: {TEXT}; }}
QPushButton:checked {{
    background-color: {ACCENT};
    color: {BG_DARK};
    border-color: {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {PANEL_BORDER};
    border-radius: 0;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 10px; height: 14px;
    margin: -6px 0;
    border-radius: 0;
    border: 1px solid {ACCENT};
}}
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {PANEL_BORDER};
    background: {BG_DARK};
    border-radius: 0;
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {PANEL_BORDER};
    border-radius: 0;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTabWidget::pane {{
    border: 1px solid {PANEL_BORDER};
    border-radius: 2px;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 6px 14px;
    border: 1px solid {PANEL_BORDER};
    border-bottom: 0;
    margin-right: 2px;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 10px;
    font-weight: 700;
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    background: {PANEL_BG};
    border-color: {ACCENT};
}}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
QSplitter::handle {{ background: {PANEL_BORDER}; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
QToolTip {{
    background: {PANEL_BG};
    color: {TEXT};
    border: 1px solid {ACCENT};
    padding: 4px 6px;
}}
"""


# Qt widget classes are created lazily inside _make_qt_classes() so that
# `import visualize` works without PySide6 installed (e.g. on a headless
# server — for unit testing the data pipeline).

def _make_qt_classes():
    """Returns a namespace of Qt-dependent classes after importing the libs."""
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    pg.setConfigOption("background", BG_DARK)
    pg.setConfigOption("foreground", TEXT)
    pg.setConfigOption("antialias", True)

    # ------------------------------------------------------------------
    # Legend with actual color swatches
    # ------------------------------------------------------------------
    class LegendPanel(QtWidgets.QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("legendPanel")
            self.setMinimumWidth(220)
            self._layout = QtWidgets.QVBoxLayout(self)
            self._layout.setContentsMargins(14, 14, 14, 14)
            self._layout.setSpacing(6)
            self._header = QtWidgets.QLabel("LEGEND")
            self._header.setProperty("role", "section")
            self._layout.addWidget(self._header)
            self._content_box = QtWidgets.QVBoxLayout()
            self._content_box.setSpacing(4)
            container = QtWidgets.QWidget()
            container.setLayout(self._content_box)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(container)
            self._layout.addWidget(scroll, 1)

        def _clear(self):
            while self._content_box.count():
                it = self._content_box.takeAt(0)
                w = it.widget()
                if w:
                    w.deleteLater()

        def _section(self, title: str):
            lab = QtWidgets.QLabel(f"▎ {title}")
            lab.setProperty("role", "section")
            self._content_box.addWidget(lab)

        def _row(self, color_hex: str, text: str, note: str = ""):
            row = QtWidgets.QWidget()
            row.setStyleSheet("background: transparent;")
            hl = QtWidgets.QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(8)
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background-color: {color_hex};"
                f" border: 1px solid rgba(255,255,255,0.18);"
                f" border-radius: 0;"
            )
            hl.addWidget(swatch)
            label = QtWidgets.QLabel(text)
            hl.addWidget(label, 1)
            if note:
                side = QtWidgets.QLabel(note)
                side.setProperty("role", "muted")
                hl.addWidget(side)
            self._content_box.addWidget(row)

        def _info(self, rows: list[tuple[str, str]]):
            for k, v in rows:
                line = QtWidgets.QLabel(f"<span style='color:{MUTED}'>"
                                        f"{k}</span>  {v}")
                line.setTextFormat(QtCore.Qt.RichText)
                self._content_box.addWidget(line)

        def set_bundle(self, b: GraphBundle, label_name_fn):
            self._clear()
            self._section("DATASET")
            self._info([
                ("name", b.name),
                ("domain", b.domain),
                ("split", b.split),
            ])
            self._section("SAMPLING")
            self._info([
                ("nodes", f"{len(b.keep_ids):,} / {b.full_nodes:,}"),
                ("edges", f"{b.edge_index.shape[1] // 2:,}"),
                ("method", b.sample_method),
                ("layout", b.layout_name),
                ("dim", f"{b.dim}D"),
            ])
            self._section(f"NODES BY {b.label_kind.upper()}")
            counts = np.bincount(b.labels.astype(np.int64))
            for label_id, hx in b.legend[:30]:
                lid = int(label_id)
                if b.label_kind == "community":
                    txt = community_label(b.name, lid,
                                           b.class_labels, b.communities)
                else:
                    txt = label_name_fn(lid, b.label_kind, b.name)
                note = f"{int(counts[lid]):,}" if lid < len(counts) else ""
                self._row(hx, txt, note)
            if len(b.legend) > 30:
                more = QtWidgets.QLabel(f"+ {len(b.legend) - 30} more")
                more.setProperty("role", "muted")
                self._content_box.addWidget(more)
            rel = edge_relation(b.name)
            self._section(f"EDGES · {rel.upper()}")
            self._row(EDGE_INTRA, f"{rel} (intra-{b.label_kind})")
            self._row(EDGE_CROSS, f"{rel} (cross-{b.label_kind})")
            self._row(EDGE_HIGHLIGHT, "highlighted (hover)")
            self._content_box.addStretch(1)

    # ------------------------------------------------------------------
    # Stats panel — graph-level statistics (Statistics tab)
    # ------------------------------------------------------------------
    class StatsPanel(QtWidgets.QFrame):
        """Graph statistics shown as charts, not just numbers."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("legendPanel")
            v = QtWidgets.QVBoxLayout(self)
            v.setContentsMargins(10, 10, 10, 10)
            v.setSpacing(4)
            self._container = QtWidgets.QVBoxLayout()
            self._container.setSpacing(8)
            inner = QtWidgets.QWidget()
            inner.setLayout(self._container)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(inner)
            v.addWidget(scroll, 1)

        def _clear(self):
            while self._container.count():
                it = self._container.takeAt(0)
                w = it.widget()
                if w:
                    w.deleteLater()

        def _section(self, title):
            lab = QtWidgets.QLabel(f"▎ {title}")
            lab.setProperty("role", "section")
            self._container.addWidget(lab)

        @staticmethod
        def _fmt(v):
            if isinstance(v, float):
                if v == 0:
                    return "0"
                if abs(v) < 1e-3:
                    return f"{v:.1e}"
                return f"{v:.3f}".rstrip("0").rstrip(".")
            if isinstance(v, int):
                if v >= 1_000_000:
                    return f"{v/1e6:.2f}M"
                if v >= 1_000:
                    return f"{v/1e3:.1f}k"
                return f"{v:,}"
            return str(v)

        def _tile(self, label: str, value: str, accent: str = ACCENT):
            w = QtWidgets.QFrame()
            w.setStyleSheet(
                f"QFrame {{ background: {BG_DARK};"
                f" border: 1px solid {PANEL_BORDER}; border-radius: 2px; }}"
                f" QFrame:hover {{ border-color: {accent}; }}"
            )
            vl = QtWidgets.QVBoxLayout(w)
            vl.setContentsMargins(10, 8, 10, 8)
            vl.setSpacing(2)
            big = QtWidgets.QLabel(value)
            big.setStyleSheet(
                f"color: {accent}; font-size: 18px; font-weight: 700;"
                f" font-family: {MONO}; letter-spacing: 1px;"
                f" border: none;"
            )
            small = QtWidgets.QLabel(label.upper())
            small.setStyleSheet(
                f"color: {MUTED}; font-size: 9px; letter-spacing: 2px;"
                f" border: none;"
            )
            vl.addWidget(big)
            vl.addWidget(small)
            return w

        def _tile_grid(self, tiles: list, cols: int = 3):
            grid_w = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(grid_w)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setSpacing(6)
            for i, tile in enumerate(tiles):
                grid.addWidget(tile, i // cols, i % cols)
            self._container.addWidget(grid_w)

        def _make_plot(self, title: str, height: int = 150):
            pw = pg.PlotWidget()
            pw.setBackground(BG_DARK)
            pw.setMinimumHeight(height)
            pw.setMaximumHeight(height + 30)
            pw.setTitle(
                f"<span style='color:{MUTED}; font-family:{MONO};"
                f" font-size:10px; letter-spacing:2px;'>{title.upper()}</span>"
            )
            pi = pw.getPlotItem()
            pi.setMenuEnabled(False)
            for ax_name in ("left", "bottom"):
                ax = pi.getAxis(ax_name)
                ax.setPen(pg.mkPen(PANEL_BORDER))
                ax.setTextPen(pg.mkPen(MUTED))
                ax.setStyle(tickFont=QtGui.QFont("Menlo", 8))
            pi.showGrid(x=False, y=True, alpha=0.15)
            pw.setStyleSheet(
                f"border: 1px solid {PANEL_BORDER}; border-radius: 2px;"
            )
            return pw

        def _bar_plot(self, title: str, xs, ys, colors=None, labels=None):
            pw = self._make_plot(title)
            xs = np.asarray(xs, dtype=np.float64)
            ys = np.asarray(ys, dtype=np.float64)
            if colors is None:
                brushes = [pg.mkBrush(ACCENT) for _ in xs]
            else:
                brushes = [pg.mkBrush(*hex_to_rgba(c, 235)) for c in colors]
            width = 0.78
            for i, (x, y, br) in enumerate(zip(xs, ys, brushes)):
                bg = pg.BarGraphItem(x=[x], height=[y], width=width, brush=br,
                                      pen=pg.mkPen(BG_DARK, width=1))
                pw.addItem(bg)
            if labels is not None:
                ax = pw.getPlotItem().getAxis("bottom")
                ax.setTicks([list(zip(xs, labels))])
                # rotate-ish: shrink font for many labels
                if len(labels) > 6:
                    ax.setStyle(tickFont=QtGui.QFont("Menlo", 7))
            self._container.addWidget(pw)
            return pw

        def _hist_plot(self, title: str, values: np.ndarray, bins: int = 24,
                       log_x: bool = False):
            pw = self._make_plot(title, height=140)
            v = np.asarray(values)
            if len(v) == 0:
                self._container.addWidget(pw)
                return pw
            if log_x:
                vpos = v[v > 0]
                if len(vpos) == 0:
                    self._container.addWidget(pw)
                    return pw
                edges = np.logspace(np.log10(vpos.min()),
                                    np.log10(vpos.max() + 1), bins)
                counts, edges = np.histogram(vpos, bins=edges)
                centers = np.sqrt(edges[:-1] * edges[1:])
                widths = edges[1:] - edges[:-1]
                pw.setLogMode(x=True, y=False)
            else:
                counts, edges = np.histogram(v, bins=bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths = (edges[1] - edges[0]) * 0.9 * np.ones_like(centers)
            bg = pg.BarGraphItem(
                x=centers, height=counts, width=widths,
                brush=pg.mkBrush(*hex_to_rgba(ACCENT, 210)),
                pen=pg.mkPen(BG_DARK, width=1),
            )
            pw.addItem(bg)
            self._container.addWidget(pw)
            return pw

        def set_bundle(self, b: GraphBundle):
            self._clear()
            s = b.stats
            n, m = s.get("nodes", 0), s.get("edges", 0)

            # --- Header tiles ---
            self._section("OVERVIEW")
            self._tile_grid([
                self._tile("nodes", self._fmt(n)),
                self._tile("edges", self._fmt(m)),
                self._tile("density", self._fmt(s.get("density", 0.0)),
                           accent=ACCENT_WARM),
                self._tile("avg degree",
                           self._fmt(s.get("degree_mean", 0.0))),
                self._tile("components",
                           self._fmt(s.get("connected_components", 0)),
                           accent=ACCENT_WARM),
                self._tile("clustering",
                           self._fmt(s.get("avg_clustering_sampled", 0.0))),
            ], cols=3)

            # --- Degree distribution histogram ---
            self._section("DEGREE DISTRIBUTION")
            deg = np.bincount(b.edge_index[0], minlength=len(b.coords))
            if len(deg) and deg.max() > 1:
                self._hist_plot("degrees (log scale)", deg, bins=24, log_x=True)
            else:
                self._hist_plot("degrees", deg, bins=12)

            # --- Label distribution bars ---
            counts = np.bincount(b.labels.astype(np.int64))
            legend_colors = [hx for _, hx in b.legend]
            short, long = [], []
            for lid in range(len(counts)):
                if b.label_kind == "class":
                    nm = class_name(b.name, lid)
                    short.append(nm[:12])
                    long.append(nm)
                elif b.label_kind == "community":
                    full = community_label(b.name, lid,
                                            b.class_labels, b.communities)
                    short.append(f"c{lid}")
                    long.append(full)
                else:
                    short.append(f"b{lid}")
                    long.append(f"bucket {lid}")
            # sort bars by count desc for easier reading when many communities
            order = np.argsort(-counts)
            show = order[:min(15, len(order))]
            self._section(f"NODES BY {b.label_kind.upper()}")
            self._bar_plot(
                f"{b.label_kind} sizes (top {len(show)})",
                np.arange(len(show)),
                counts[show],
                colors=[legend_colors[i] if i < len(legend_colors) else ACCENT
                        for i in show],
                labels=[short[i] for i in show],
            )
            # annotated rows under the chart so users can see the full
            # "cluster N · Neural_Networks (72%)" name beside the short tick.
            for i in show:
                line = QtWidgets.QLabel(
                    f"<span style='color:{legend_colors[i] if i < len(legend_colors) else TEXT}'>"
                    f"■</span>  "
                    f"<span style='color:{TEXT}'>{long[i]}</span>  "
                    f"<span style='color:{MUTED}'>— {int(counts[i]):,}</span>"
                )
                line.setTextFormat(QtCore.Qt.RichText)
                line.setWordWrap(True)
                self._container.addWidget(line)

            # --- Connectivity tile row ---
            self._section("CONNECTIVITY")
            self._tile_grid([
                self._tile("isolated nodes",
                           self._fmt(s.get("isolated_nodes", 0))),
                self._tile("largest CC",
                           self._fmt(s.get("largest_cc", 0)),
                           accent=ACCENT_WARM),
                self._tile("lCC / n",
                           self._fmt(s.get("largest_cc_fraction", 0.0))),
                self._tile("max degree",
                           self._fmt(s.get("degree_max", 0))),
                self._tile("communities",
                           self._fmt(s.get("num_communities", 0)),
                           accent=ACCENT_WARM),
                self._tile("sampled",
                           f"{len(b.keep_ids):,}/{b.full_nodes:,}"),
            ], cols=3)

            # --- Meta footer ---
            self._section("SAMPLE")
            meta = QtWidgets.QLabel(
                f"<span style='color:{MUTED}'>source</span> {b.name} "
                f"· <span style='color:{MUTED}'>domain</span> {b.domain}<br>"
                f"<span style='color:{MUTED}'>method</span> {b.sample_method} "
                f"· <span style='color:{MUTED}'>layout</span> {b.layout_name} "
                f"({b.dim}D)"
            )
            meta.setTextFormat(QtCore.Qt.RichText)
            meta.setWordWrap(True)
            self._container.addWidget(meta)
            self._container.addStretch(1)

    # ------------------------------------------------------------------
    # Selected-node panel — updates on click (Selected tab)
    # ------------------------------------------------------------------
    class SelectedPanel(QtWidgets.QFrame):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("legendPanel")
            v = QtWidgets.QVBoxLayout(self)
            v.setContentsMargins(14, 14, 14, 14)
            v.setSpacing(6)
            self._container = QtWidgets.QVBoxLayout()
            self._container.setSpacing(4)
            inner = QtWidgets.QWidget()
            inner.setLayout(self._container)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(inner)
            v.addWidget(scroll, 1)
            self._bundle: Optional[GraphBundle] = None
            self.clear()

        def _reset(self):
            while self._container.count():
                it = self._container.takeAt(0)
                w = it.widget()
                if w:
                    w.deleteLater()

        def _section(self, title):
            lab = QtWidgets.QLabel(f"▎ {title}")
            lab.setProperty("role", "section")
            self._container.addWidget(lab)

        def _pair(self, k, v):
            line = QtWidgets.QLabel(
                f"<span style='color:{MUTED}'>{k}</span>  "
                f"<span style='font-weight:600'>{v}</span>")
            line.setTextFormat(QtCore.Qt.RichText)
            line.setWordWrap(True)
            self._container.addWidget(line)

        def clear(self):
            self._reset()
            hint = QtWidgets.QLabel(
                "▎ NO SELECTION\n\n"
                "Click any node in the graph to see its details here."
            )
            hint.setProperty("role", "muted")
            hint.setStyleSheet(
                f"color: {MUTED}; font-family: {MONO};"
                f" font-size: 11px; line-height: 1.4em;"
            )
            hint.setWordWrap(True)
            self._container.addWidget(hint)
            self._container.addStretch(1)

        def set_bundle(self, b: GraphBundle):
            self._bundle = b
            self.clear()

        def set_selected(self, i: int):
            b = self._bundle
            if b is None:
                return
            self._reset()
            edges = b.edge_index
            orig_id = int(b.keep_ids[i])
            deg = int((edges[0] == i).sum())
            lab = int(b.labels[i])
            comm = int(b.communities[i])
            if b.label_kind == "class":
                lab_txt = class_name(b.name, lab)
            elif b.label_kind == "community":
                lab_txt = community_label(b.name, lab,
                                           b.class_labels, b.communities)
            else:
                lab_txt = f"bucket {lab}"
            comm_txt = community_label(b.name, comm,
                                        b.class_labels, b.communities)
            rel = edge_relation(b.name)

            self._section("NODE")
            self._pair("name", node_label(b.name, orig_id, b.node_names))
            if b.node_names is None:
                note = QtWidgets.QLabel(
                    f"<span style='color:{MUTED}'>"
                    f"// no titles shipped for {b.name}. drop a file at "
                    f"figures/data/{b.name}/node_names.json "
                    f"(list or id→name map) to show real names.</span>"
                )
                note.setTextFormat(QtCore.Qt.RichText)
                note.setWordWrap(True)
                self._container.addWidget(note)
            self._pair("id (original)", f"{orig_id:,}")
            self._pair("id (sample)", f"{i:,}")
            self._pair(b.label_kind, lab_txt)
            # class always shown if it's available (even when coloring by
            # community, it's useful context)
            if b.class_labels is not None and b.label_kind != "class":
                self._pair("class", class_name(b.name, int(b.class_labels[i])))
            self._pair("community", comm_txt)
            self._pair("degree", f"{deg:,} × {rel}")

            # neighbor list (limit to 20)
            neigh_idx = edges[1, edges[0] == i]
            self._section(f"NEIGHBORS — {rel.upper()} ({len(neigh_idx):,})")
            for j in neigh_idx[:20]:
                j = int(j)
                nid = int(b.keep_ids[j])
                jlab = int(b.labels[j])
                if b.label_kind == "class":
                    j_kind = class_name(b.name, jlab)
                elif b.label_kind == "community":
                    j_kind = community_label(b.name, jlab,
                                              b.class_labels, b.communities)
                else:
                    j_kind = f"bucket {jlab}"
                self._pair(
                    node_label(b.name, nid, b.node_names),
                    j_kind,
                )
            if len(neigh_idx) > 20:
                more = QtWidgets.QLabel(f"… + {len(neigh_idx) - 20:,} more")
                more.setProperty("role", "muted")
                self._container.addWidget(more)
            self._container.addStretch(1)

    # ------------------------------------------------------------------
    # InfoPanel — tab container (Legend / Statistics / Selected)
    # ------------------------------------------------------------------
    class InfoPanel(QtWidgets.QTabWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMinimumWidth(260)
            self.legend = LegendPanel()
            self.stats = StatsPanel()
            self.selected = SelectedPanel()
            self.addTab(self.legend, "Legend")
            self.addTab(self.stats, "Statistics")
            self.addTab(self.selected, "Selected")

        def set_bundle(self, b: GraphBundle, label_name_fn):
            self.legend.set_bundle(b, label_name_fn)
            self.stats.set_bundle(b)
            self.selected.set_bundle(b)

        def show_selected(self, i: int):
            self.selected.set_selected(i)
            self.setCurrentWidget(self.selected)

    # ------------------------------------------------------------------
    # 2D view
    # ------------------------------------------------------------------
    class GraphView2D(pg.PlotWidget):
        node_clicked = QtCore.Signal(int)  # emits the bundle-local node index

        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setBackground(BG_DARK)
            self.getPlotItem().hideAxis("bottom")
            self.getPlotItem().hideAxis("left")
            self.getPlotItem().getViewBox().setAspectLocked(True)
            self.getPlotItem().setMenuEnabled(False)
            self.getPlotItem().setMouseEnabled(True, True)

            self._bundle: Optional[GraphBundle] = None
            self._scatter: Optional[pg.ScatterPlotItem] = None
            self._edge_intra = None
            self._edge_cross = None
            self._edge_highlight = None
            self._selection_ring = None
            self._tooltip = pg.TextItem("", color=TEXT, anchor=(0, 1),
                                         fill=pg.mkBrush(22, 28, 36, 230),
                                         border=pg.mkPen(PANEL_BORDER))
            self._tooltip.setZValue(500)
            self.addItem(self._tooltip)
            self._tooltip.hide()

            self._static_labels: list[pg.TextItem] = []

            self.scene().sigMouseMoved.connect(self._on_mouse_moved)
            self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        def set_bundle(self, b: GraphBundle, node_size: float, edge_width: float,
                       edge_alpha: float, show_labels: bool):
            self.getPlotItem().clear()
            self.addItem(self._tooltip)
            self._bundle = b
            coords = b.coords
            edges = b.edge_index
            labels = b.labels

            # edges split intra/cross
            intra_mask = labels[edges[0]] == labels[edges[1]]
            self._edge_intra = self._add_edges(coords, edges[:, intra_mask],
                                               EDGE_INTRA, edge_width, edge_alpha)
            self._edge_cross = self._add_edges(coords, edges[:, ~intra_mask],
                                               EDGE_CROSS, edge_width, edge_alpha)

            # highlight layer on top (empty at first)
            self._edge_highlight = pg.PlotDataItem(
                [], [], connect="pairs",
                pen=pg.mkPen(color=(*hex_to_rgba(EDGE_HIGHLIGHT)[:3], 255),
                             width=edge_width + 1.5))
            self._edge_highlight.setZValue(200)
            self.addItem(self._edge_highlight)

            # nodes
            brushes = [pg.mkBrush(c[0] * 255, c[1] * 255, c[2] * 255, 245)
                       for c in b.colors]
            deg = np.bincount(edges[0], minlength=len(coords)).astype(np.float32)
            sizes = node_size + 2.0 * np.log1p(deg)
            self._scatter = pg.ScatterPlotItem(
                pos=coords, size=sizes, brush=brushes,
                pen=pg.mkPen(color="#0b1014", width=0.8), hoverable=False,
            )
            self._scatter.setZValue(300)
            self.addItem(self._scatter)

            # static labels if small
            self._static_labels = []
            if show_labels and len(coords) <= 200:
                for i, kid in enumerate(b.keep_ids):
                    t = pg.TextItem(
                        text=node_label(b.name, int(kid), b.node_names),
                        color=MUTED, anchor=(0.5, -0.5),
                    )
                    t.setPos(coords[i, 0], coords[i, 1])
                    self.addItem(t)
                    self._static_labels.append(t)

            # selection ring (drawn on click; starts empty)
            self._selection_ring = pg.ScatterPlotItem(
                pos=np.zeros((0, 2)), size=24, symbol="o",
                pen=pg.mkPen(NODE_HIGHLIGHT, width=2.5),
                brush=pg.mkBrush(0, 0, 0, 0),
            )
            self._selection_ring.setZValue(400)
            self.addItem(self._selection_ring)

            # Fit view to spacing (1.1x padding); spacing actually changes
            # the visible spread now because autoRange used to re-normalize.
            r = float(np.max(np.abs(coords))) if len(coords) else 1.0
            self.getPlotItem().getViewBox().setRange(
                xRange=(-r * 1.1, r * 1.1), yRange=(-r * 1.1, r * 1.1),
                padding=0,
            )

        def _add_edges(self, coords, edges, color_hex, width, alpha):
            if edges.size == 0:
                return None
            x = np.empty(2 * edges.shape[1])
            y = np.empty(2 * edges.shape[1])
            x[0::2] = coords[edges[0], 0]; x[1::2] = coords[edges[1], 0]
            y[0::2] = coords[edges[0], 1]; y[1::2] = coords[edges[1], 1]
            r, g, b, _ = hex_to_rgba(color_hex)
            item = pg.PlotDataItem(
                x, y, connect="pairs",
                pen=pg.mkPen(color=(r, g, b, int(alpha)), width=width))
            item.setZValue(100)
            self.addItem(item)
            return item

        # -------- hover --------
        def _on_mouse_moved(self, scene_pos):
            if self._bundle is None or self._scatter is None:
                return
            vb = self.getPlotItem().getViewBox()
            if not self.sceneBoundingRect().contains(scene_pos):
                self._tooltip.hide()
                self._edge_highlight.setData([], [])
                return
            mouse_point = vb.mapSceneToView(scene_pos)
            mx, my = mouse_point.x(), mouse_point.y()

            coords = self._bundle.coords
            diffs = coords - np.array([mx, my])
            dists = np.linalg.norm(diffs, axis=1)
            # pixel threshold in data coords
            pix = vb.viewPixelSize()
            px_thresh = 12 * max(pix[0], pix[1])
            i = int(np.argmin(dists))
            if dists[i] < px_thresh:
                self._show_node_tooltip(i, mx, my)
                return

            # no node → try edges
            edge_idx = self._nearest_edge(mx, my, px_thresh * 0.8)
            if edge_idx is not None:
                self._show_edge_tooltip(edge_idx, mx, my)
            else:
                self._tooltip.hide()
                self._edge_highlight.setData([], [])

        def _show_node_tooltip(self, i, mx, my):
            b = self._bundle
            edges = b.edge_index
            deg = int((edges[0] == i).sum())
            label_val = int(b.labels[i])
            if b.label_kind == "class":
                label_txt = class_name(b.name, label_val)
            elif b.label_kind == "community":
                label_txt = community_label(b.name, label_val,
                                             b.class_labels, b.communities)
            else:
                label_txt = f"bucket {label_val}"
            name_txt = node_label(b.name, int(b.keep_ids[i]), b.node_names)
            comm_txt = community_label(
                b.name, int(b.communities[i]), b.class_labels, b.communities)
            lines = [
                f"<b>{name_txt}</b>",
                f"{b.label_kind}: {label_txt}",
                f"degree: {deg} × {edge_relation(b.name)}",
                f"community: {comm_txt}",
            ]
            self._tooltip.setHtml("<br>".join(lines))
            self._tooltip.setPos(mx, my)
            self._tooltip.show()
            # highlight its edges
            mask = (edges[0] == i) | (edges[1] == i)
            sub = edges[:, mask]
            if sub.size:
                coords = b.coords
                x = np.empty(2 * sub.shape[1])
                y = np.empty(2 * sub.shape[1])
                x[0::2] = coords[sub[0], 0]; x[1::2] = coords[sub[1], 0]
                y[0::2] = coords[sub[0], 1]; y[1::2] = coords[sub[1], 1]
                self._edge_highlight.setData(x, y)
            else:
                self._edge_highlight.setData([], [])

        def _nearest_edge(self, mx, my, thresh):
            b = self._bundle
            edges = b.edge_index
            if edges.shape[1] == 0:
                return None
            p = b.coords[edges[0]]
            q = b.coords[edges[1]]
            r = np.array([mx, my])
            ab = q - p
            t = np.einsum("ij,ij->i", r - p, ab) / (
                np.einsum("ij,ij->i", ab, ab) + 1e-12)
            t = np.clip(t, 0.0, 1.0)
            proj = p + (ab.T * t).T
            d = np.linalg.norm(proj - r, axis=1)
            i = int(np.argmin(d))
            if d[i] < thresh:
                return i
            return None

        def _show_edge_tooltip(self, edge_i, mx, my):
            b = self._bundle
            u, v = int(b.edge_index[0, edge_i]), int(b.edge_index[1, edge_i])
            lu, lv = int(b.labels[u]), int(b.labels[v])
            kind = "intra" if lu == lv else "cross"
            if b.label_kind == "class":
                nu, nv = class_name(b.name, lu), class_name(b.name, lv)
            elif b.label_kind == "community":
                nu = community_label(b.name, lu, b.class_labels, b.communities)
                nv = community_label(b.name, lv, b.class_labels, b.communities)
            else:
                nu, nv = f"{lu}", f"{lv}"
            un = node_label(b.name, int(b.keep_ids[u]), b.node_names)
            vn = node_label(b.name, int(b.keep_ids[v]), b.node_names)
            rel = edge_relation(b.name)
            lines = [
                f"<b>{rel}</b>",
                f"{un} → {vn}",
                f"{kind}-{b.label_kind}  ({nu} ↔ {nv})",
            ]
            self._tooltip.setHtml("<br>".join(lines))
            self._tooltip.setPos(mx, my)
            self._tooltip.show()
            # highlight this edge
            x = [b.coords[u, 0], b.coords[v, 0]]
            y = [b.coords[u, 1], b.coords[v, 1]]
            self._edge_highlight.setData(x, y)

        # -------- click --------
        def _on_mouse_clicked(self, ev):
            if self._bundle is None:
                return
            # Only act on single left-clicks
            try:
                if ev.button() != QtCore.Qt.LeftButton:
                    return
                if hasattr(ev, "double") and ev.double():
                    return
            except Exception:
                pass
            vb = self.getPlotItem().getViewBox()
            mouse_point = vb.mapSceneToView(ev.scenePos())
            mx, my = mouse_point.x(), mouse_point.y()
            coords = self._bundle.coords
            diffs = coords - np.array([mx, my])
            dists = np.linalg.norm(diffs, axis=1)
            pix = vb.viewPixelSize()
            px_thresh = 16 * max(pix[0], pix[1])
            i = int(np.argmin(dists))
            if dists[i] < px_thresh:
                self._selection_ring.setData(
                    pos=np.array([[coords[i, 0], coords[i, 1]]])
                )
                self.node_clicked.emit(i)

    # ------------------------------------------------------------------
    # 3D view
    # ------------------------------------------------------------------
    class GraphView3D(gl.GLViewWidget):
        node_clicked = QtCore.Signal(int)

        def __init__(self, parent=None):
            super().__init__(parent=parent)
            self.setBackgroundColor(pg.mkColor(BG_DARK))
            self.setMouseTracking(True)
            # Ensure Qt delivers hover events even with no mouse button pressed
            # (mac/Qt6 prefers HoverMove events over mouseMove in that case).
            self.setAttribute(QtCore.Qt.WA_Hover, True)
            self._bundle: Optional[GraphBundle] = None
            self._scatter = None
            self._edge_intra = None
            self._edge_cross = None
            self._edge_highlight = None
            self._label_items: list = []
            self._press_pos: Optional[tuple[float, float]] = None
            self._selection_marker = None

            self._tooltip = QtWidgets.QLabel("", self)
            self._tooltip.setStyleSheet(
                f"color: {TEXT};"
                f"background-color: rgba(22, 28, 36, 230);"
                f"border: 1px solid {PANEL_BORDER};"
                f"padding: 6px 8px; border-radius: 4px;"
                f"font-family: monospace; font-size: 11px;"
            )
            self._tooltip.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            self._tooltip.hide()

        def set_bundle(self, b: GraphBundle, node_size: float, edge_width: float,
                       edge_alpha: float, show_labels: bool):
            # clear previous items
            for it in list(self.items):
                self.removeItem(it)
            self._label_items = []
            self._bundle = b

            coords = b.coords.astype(np.float32)
            edges = b.edge_index
            labels = b.labels
            intra_mask = labels[edges[0]] == labels[edges[1]]

            # DO NOT re-normalize coords here — doing so cancels the user's
            # spacing slider. The layouts already scale coords by `spacing`
            # into [-spacing, spacing]. We just center them at origin.
            coords = coords - coords.mean(axis=0)
            b.coords = coords  # keep in sync for hover projection

            self._add_edges(coords, edges[:, intra_mask], EDGE_INTRA,
                            edge_width, edge_alpha, "intra")
            self._add_edges(coords, edges[:, ~intra_mask], EDGE_CROSS,
                            edge_width, edge_alpha, "cross")

            deg = np.bincount(edges[0], minlength=len(coords)).astype(np.float32)
            sizes = node_size + 1.5 * np.log1p(deg)
            self._scatter = gl.GLScatterPlotItem(
                pos=coords, size=sizes, color=b.colors, pxMode=True,
            )
            self._scatter.setGLOptions("translucent")
            self.addItem(self._scatter)

            if show_labels and len(coords) <= 150 and hasattr(gl, "GLTextItem"):
                for i, kid in enumerate(b.keep_ids):
                    try:
                        t = gl.GLTextItem(
                            pos=coords[i],
                            text=node_label(b.name, int(kid), b.node_names),
                            color=(200, 210, 220, 200),
                        )
                        self.addItem(t)
                        self._label_items.append(t)
                    except Exception:
                        break

            # Camera distance: fit a unit graph snugly; spacing pushes nodes
            # out beyond that and the user sees them genuinely more spread.
            self.opts["distance"] = 4.0
            self.opts["fov"] = 50
            self.update()

        def _add_edges(self, coords, edges, color_hex, width, alpha, kind):
            if edges.size == 0:
                setattr(self, f"_edge_{kind}", None)
                return
            verts = np.empty((2 * edges.shape[1], 3), dtype=np.float32)
            verts[0::2] = coords[edges[0]]
            verts[1::2] = coords[edges[1]]
            r, g, b, _ = hex_to_rgba(color_hex)
            item = gl.GLLinePlotItem(
                pos=verts, mode="lines", antialias=True,
                color=(r / 255, g / 255, b / 255, alpha / 255),
                width=float(width),
            )
            item.setGLOptions("translucent")
            self.addItem(item)
            setattr(self, f"_edge_{kind}", item)

        # -------- 3D hover via screen-space projection --------
        @staticmethod
        def _qmat_to_np(m) -> np.ndarray:
            # Try the fast path first: Qt stores 16 floats column-major.
            try:
                data = m.data()
                arr = np.array(list(data), dtype=np.float64)
                if arr.size == 16:
                    return arr.reshape(4, 4, order="F")
            except Exception:
                pass
            # Fallback: element-wise. Some bindings expose (row, col), others
            # implement __call__(row, col) or a .row()/.column() API.
            out = np.empty((4, 4), dtype=np.float64)
            for r in range(4):
                for c in range(4):
                    try:
                        out[r, c] = float(m[r, c])
                    except Exception:
                        try:
                            out[r, c] = float(m(r, c))
                        except Exception:
                            out[r, c] = 1.0 if r == c else 0.0
            return out

        def _project(self, pts: np.ndarray) -> np.ndarray:
            try:
                V = self._qmat_to_np(self.viewMatrix())
                P = self._qmat_to_np(self.projectionMatrix())
            except Exception:
                return np.full((len(pts), 3), np.inf)
            homog = np.hstack([pts, np.ones((len(pts), 1))])
            eye = homog @ V.T
            clip = eye @ P.T
            w = clip[:, 3:4]
            w = np.where(np.abs(w) < 1e-9, 1.0, w)
            ndc = clip[:, :3] / w
            sw, sh = self.width(), self.height()
            screen = np.empty_like(ndc)
            screen[:, 0] = (ndc[:, 0] + 1) * 0.5 * sw
            screen[:, 1] = (1 - (ndc[:, 1] + 1) * 0.5) * sh
            # points behind the camera (OpenGL convention: eye-space z > 0 is behind)
            behind = eye[:, 2] > 0
            screen[behind, 0] = np.inf
            return screen

        def _update_hover(self, cx: float, cy: float):
            if self._bundle is None or self._scatter is None:
                return
            screen = self._project(self._bundle.coords)
            dists = np.hypot(screen[:, 0] - cx, screen[:, 1] - cy)
            i = int(np.argmin(dists))
            if dists[i] < 18.0:
                b = self._bundle
                edges = b.edge_index
                deg = int((edges[0] == i).sum())
                label_val = int(b.labels[i])
                if b.label_kind == "class":
                    label_txt = class_name(b.name, label_val)
                elif b.label_kind == "community":
                    label_txt = community_label(b.name, label_val,
                                                 b.class_labels, b.communities)
                else:
                    label_txt = f"bucket {label_val}"
                name_txt = node_label(b.name, int(b.keep_ids[i]), b.node_names)
                comm_txt = community_label(
                    b.name, int(b.communities[i]),
                    b.class_labels, b.communities)
                self._tooltip.setText(
                    f"{name_txt}\n"
                    f"{b.label_kind}: {label_txt}\n"
                    f"community: {comm_txt}\n"
                    f"degree: {deg} × {edge_relation(b.name)}"
                )
                self._tooltip.adjustSize()
                tx = int(min(cx + 14, self.width() - self._tooltip.width() - 4))
                ty = int(min(cy + 14, self.height() - self._tooltip.height() - 4))
                self._tooltip.move(max(tx, 0), max(ty, 0))
                self._tooltip.raise_()
                self._tooltip.show()
            else:
                self._tooltip.hide()

        def mouseMoveEvent(self, ev):
            super().mouseMoveEvent(ev)
            pos = ev.position() if hasattr(ev, "position") else ev.localPos()
            self._update_hover(float(pos.x()), float(pos.y()))

        def hoverMoveEvent(self, ev):
            # Covers macOS where mouseMoveEvent may not fire without a button.
            try:
                super().hoverMoveEvent(ev)
            except AttributeError:
                pass
            pos = ev.position() if hasattr(ev, "position") else ev.pos()
            self._update_hover(float(pos.x()), float(pos.y()))

        def event(self, ev):
            # Qt6 hover events arrive as QHoverMove on the widget itself.
            if ev.type() == QtCore.QEvent.HoverMove:
                pos = ev.position() if hasattr(ev, "position") else ev.pos()
                self._update_hover(float(pos.x()), float(pos.y()))
            return super().event(ev)

        def leaveEvent(self, ev):
            super().leaveEvent(ev)
            self._tooltip.hide()

        # -------- click selection (distinct from camera drag) --------
        def mousePressEvent(self, ev):
            super().mousePressEvent(ev)
            pos = ev.position() if hasattr(ev, "position") else ev.localPos()
            self._press_pos = (float(pos.x()), float(pos.y()))

        def mouseReleaseEvent(self, ev):
            super().mouseReleaseEvent(ev)
            pos = ev.position() if hasattr(ev, "position") else ev.localPos()
            x, y = float(pos.x()), float(pos.y())
            if self._press_pos is not None:
                dx, dy = x - self._press_pos[0], y - self._press_pos[1]
                moved = (dx * dx + dy * dy) ** 0.5
            else:
                moved = 999
            self._press_pos = None
            if moved > 4.0 or self._bundle is None or self._scatter is None:
                return
            try:
                if ev.button() != QtCore.Qt.LeftButton:
                    return
            except Exception:
                pass
            screen = self._project(self._bundle.coords)
            dists = np.hypot(screen[:, 0] - x, screen[:, 1] - y)
            i = int(np.argmin(dists))
            if dists[i] < 20.0:
                self._show_selection(i)
                self.node_clicked.emit(i)

        def _show_selection(self, i: int):
            # replace the previous marker with a single larger dot
            if self._selection_marker is not None:
                try:
                    self.removeItem(self._selection_marker)
                except Exception:
                    pass
            b = self._bundle
            pos_arr = np.array([b.coords[i]], dtype=np.float32)
            self._selection_marker = gl.GLScatterPlotItem(
                pos=pos_arr, size=20.0,
                color=(1.0, 0.7, 0.3, 1.0), pxMode=True,
            )
            self._selection_marker.setGLOptions("translucent")
            self.addItem(self._selection_marker)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    class Sidebar(QtWidgets.QFrame):
        render_requested = QtCore.Signal()

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("sidebar")
            self.setMinimumWidth(310)
            self.setMaximumWidth(360)

            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(10)

            title = QtWidgets.QLabel("▮ NETFM ▮ GRAPH EXPLORER")
            title.setProperty("role", "title")
            sub = QtWidgets.QLabel("  // interactive 2D/3D network viewer")
            sub.setProperty("role", "subtitle")
            root.addWidget(title)
            root.addWidget(sub)

            # dataset
            root.addWidget(self._section("DATASET"))
            self.dataset_combo = QtWidgets.QComboBox()
            self._populate_datasets()
            self.dataset_combo.currentTextChanged.connect(self._on_dataset_change)
            root.addWidget(self.dataset_combo)

            self.custom_btn = QtWidgets.QPushButton("Load custom file…")
            self.custom_btn.setProperty("role", "secondary")
            self.custom_btn.clicked.connect(self._on_custom_file)
            root.addWidget(self.custom_btn)
            self.custom_path_lbl = QtWidgets.QLabel("")
            self.custom_path_lbl.setProperty("role", "muted")
            root.addWidget(self.custom_path_lbl)
            self.custom_file: Optional[str] = None

            # dim
            root.addWidget(self._section("VIEW"))
            dim_row = QtWidgets.QHBoxLayout()
            self.dim_2d = QtWidgets.QPushButton("2D")
            self.dim_3d = QtWidgets.QPushButton("3D")
            for b in (self.dim_2d, self.dim_3d):
                b.setCheckable(True)
                b.setProperty("role", "secondary")
            self.dim_3d.setChecked(True)
            self.dim_2d.clicked.connect(lambda: self._set_dim(2))
            self.dim_3d.clicked.connect(lambda: self._set_dim(3))
            dim_row.addWidget(self.dim_2d)
            dim_row.addWidget(self.dim_3d)
            w = QtWidgets.QWidget(); w.setLayout(dim_row)
            root.addWidget(w)

            # layout — default to community because it's the most readable
            self.layout_combo = QtWidgets.QComboBox()
            self.layout_combo.addItems(list(LAYOUTS.keys()))
            self.layout_combo.setCurrentText("community")
            root.addWidget(QtWidgets.QLabel("Layout algorithm"))
            root.addWidget(self.layout_combo)

            # color by — default to community so the clustering stands out
            self.color_combo = QtWidgets.QComboBox()
            self.color_combo.addItems(["community", "class", "degree"])
            root.addWidget(QtWidgets.QLabel("Color by"))
            root.addWidget(self.color_combo)

            # sampling
            root.addWidget(self._section("SAMPLING"))
            self.sample_combo = QtWidgets.QComboBox()
            self.sample_combo.addItems(list(SAMPLERS.keys()))
            root.addWidget(QtWidgets.QLabel("Method"))
            root.addWidget(self.sample_combo)

            self.sample_spin = QtWidgets.QSpinBox()
            self.sample_spin.setRange(50, 20000)
            self.sample_spin.setSingleStep(100)
            self.sample_spin.setValue(DEFAULT_SAMPLE)
            root.addWidget(QtWidgets.QLabel("Sample size"))
            root.addWidget(self.sample_spin)

            # layout spacing — spreads nodes further apart (slider / 10 = factor)
            self.spacing_slider = self._make_slider(
                5, 250, 40, "Node spacing  (×0.1)",
            )
            root.addWidget(self.spacing_slider[0])

            # appearance
            root.addWidget(self._section("APPEARANCE"))
            self.node_slider = self._make_slider(4, 50, 18, "Node size")
            self.edge_slider = self._make_slider(1, 6, 2, "Edge width")
            self.alpha_slider = self._make_slider(10, 255, 60, "Edge opacity")
            for w in self.node_slider[0], self.edge_slider[0], self.alpha_slider[0]:
                root.addWidget(w)

            self.labels_chk = QtWidgets.QCheckBox("Show node labels (small graphs)")
            self.labels_chk.setChecked(True)
            root.addWidget(self.labels_chk)

            # render
            root.addStretch(1)
            self.render_btn = QtWidgets.QPushButton("Render graph")
            self.render_btn.clicked.connect(lambda: self.render_requested.emit())
            root.addWidget(self.render_btn)

            self.status = QtWidgets.QLabel("")
            self.status.setProperty("role", "muted")
            self.status.setWordWrap(True)
            root.addWidget(self.status)

            self._dim = 3
            self._on_dataset_change(self.dataset_combo.currentText())

        def _section(self, title):
            lab = QtWidgets.QLabel(title)
            lab.setProperty("role", "section")
            return lab

        def _make_slider(self, mn, mx, val, label):
            wrapper = QtWidgets.QWidget()
            vl = QtWidgets.QVBoxLayout(wrapper)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(3)
            hdr = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel(label)
            val_lbl = QtWidgets.QLabel(str(val))
            val_lbl.setProperty("role", "muted")
            hdr.addWidget(name); hdr.addStretch(); hdr.addWidget(val_lbl)
            w = QtWidgets.QWidget(); w.setLayout(hdr)
            vl.addWidget(w)
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(mn, mx); sl.setValue(val)
            sl.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
            vl.addWidget(sl)
            return wrapper, sl

        def _populate_datasets(self):
            try:
                from src.data import PRETRAIN_DATASETS, HELDOUT_DATASETS
                names = PRETRAIN_DATASETS + HELDOUT_DATASETS
            except Exception:
                names = []
            self.dataset_combo.clear()
            self.dataset_combo.addItems(names)
            self.dataset_combo.addItem("— custom file —")

        def _on_dataset_change(self, name):
            is_custom = name == "— custom file —"
            self.custom_btn.setVisible(is_custom)
            self.custom_path_lbl.setVisible(is_custom and bool(self.custom_file))
            if not is_custom:
                self.status.setText(f"Selected: {name}")

        def _on_custom_file(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open graph file", "",
                "Graphs (*.txt *.csv *.graphml *.gexf *.gpickle *.pkl);;All files (*)",
            )
            if path:
                self.custom_file = path
                self.custom_path_lbl.setText(Path(path).name)
                self.custom_path_lbl.setVisible(True)
                self.status.setText(f"Custom file: {Path(path).name}")

        def _set_dim(self, d):
            self._dim = d
            self.dim_2d.setChecked(d == 2)
            self.dim_3d.setChecked(d == 3)

        def current_options(self):
            name = self.dataset_combo.currentText()
            source_kind = "custom" if name == "— custom file —" else "registry"
            return {
                "dataset": name,
                "source_kind": source_kind,
                "custom_path": self.custom_file,
                "dim": self._dim,
                "layout": self.layout_combo.currentText(),
                "color_by": self.color_combo.currentText(),
                "sample_method": self.sample_combo.currentText(),
                "sample_size": self.sample_spin.value(),
                "spacing": self.spacing_slider[1].value() / 10.0,
                "node_size": self.node_slider[1].value(),
                "edge_width": self.edge_slider[1].value(),
                "edge_alpha": self.alpha_slider[1].value(),
                "show_labels": self.labels_chk.isChecked(),
            }

        def set_busy(self, busy, text=""):
            self.render_btn.setDisabled(busy)
            self.render_btn.setText("Rendering…" if busy else "Render graph")
            if text:
                self.status.setText(text)

    # ------------------------------------------------------------------
    # Main window
    # ------------------------------------------------------------------
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("NetFM · Graph Explorer")
            self.resize(1500, 900)
            self.setStyleSheet(_STYLE)

            splitter = QtWidgets.QSplitter(self)
            splitter.setHandleWidth(6)
            self.setCentralWidget(splitter)

            self.sidebar = Sidebar()
            splitter.addWidget(self.sidebar)

            self.graph_container = QtWidgets.QWidget()
            self.graph_layout = QtWidgets.QVBoxLayout(self.graph_container)
            self.graph_layout.setContentsMargins(6, 6, 6, 6)
            self.placeholder = QtWidgets.QLabel(
                "Pick a dataset on the left and click <b>Render graph</b>.\n"
                "Large graphs are auto-sampled.")
            self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
            self.placeholder.setStyleSheet(f"color: {MUTED}; font-size: 14px;")
            self.graph_layout.addWidget(self.placeholder)
            splitter.addWidget(self.graph_container)

            self.info_panel = InfoPanel()
            splitter.addWidget(self.info_panel)

            splitter.setSizes([340, 900, 300])
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setStretchFactor(2, 0)

            self.current_view = None
            self.sidebar.render_requested.connect(self._render)

        def _swap_view(self, new_widget):
            while self.graph_layout.count():
                it = self.graph_layout.takeAt(0)
                w = it.widget()
                if w:
                    w.deleteLater()
            self.graph_layout.addWidget(new_widget, 1)
            self.current_view = new_widget

        def _render(self):
            opts = self.sidebar.current_options()
            try:
                if opts["source_kind"] == "custom":
                    if not opts["custom_path"]:
                        self.sidebar.set_busy(False, "⚠ no custom file chosen")
                        return
                    self.sidebar.set_busy(True, "loading custom file…")
                    QtWidgets.QApplication.processEvents()
                    source = load_custom_file(opts["custom_path"])
                else:
                    self.sidebar.set_busy(True, f"loading {opts['dataset']}…")
                    QtWidgets.QApplication.processEvents()
                    source = load_registry_dataset(opts["dataset"])

                def prog(msg):
                    self.sidebar.status.setText(msg)
                    QtWidgets.QApplication.processEvents()

                bundle = build_bundle(
                    source, opts["dim"], opts["sample_method"],
                    opts["sample_size"], opts["layout"], opts["color_by"],
                    spacing=opts["spacing"], progress=prog,
                )

                view = GraphView2D() if opts["dim"] == 2 else GraphView3D()
                view.set_bundle(
                    bundle,
                    node_size=opts["node_size"],
                    edge_width=opts["edge_width"],
                    edge_alpha=opts["edge_alpha"],
                    show_labels=opts["show_labels"],
                )
                view.node_clicked.connect(self.info_panel.show_selected)
                self._swap_view(view)
                self.info_panel.set_bundle(bundle, self._label_name)

                self.sidebar.set_busy(
                    False, f"rendered {len(bundle.keep_ids):,} nodes · "
                           f"{bundle.edge_index.shape[1] // 2:,} edges")
            except Exception as ex:
                import traceback
                traceback.print_exc()
                self.sidebar.set_busy(False, f"⚠ {type(ex).__name__}: {ex}")

        def _label_name(self, label_id: int, kind: str, dataset: str) -> str:
            if kind == "class":
                return class_name(dataset, int(label_id))
            if kind == "community":
                return f"cluster {int(label_id)}"
            return f"bucket {int(label_id)}"

    return {
        "LegendPanel": LegendPanel,
        "StatsPanel": StatsPanel,
        "SelectedPanel": SelectedPanel,
        "InfoPanel": InfoPanel,
        "GraphView2D": GraphView2D,
        "GraphView3D": GraphView3D,
        "Sidebar": Sidebar,
        "MainWindow": MainWindow,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NetFM Graph Explorer")
    parser.add_argument("--dataset", default=None,
                        help="Skip the dashboard and render this dataset immediately")
    parser.add_argument("--dim", type=int, default=3, choices=[2, 3])
    parser.add_argument("--layout", default="spring", choices=list(LAYOUTS.keys()))
    parser.add_argument("--sample", default="ego", choices=list(SAMPLERS.keys()))
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE)
    args = parser.parse_args()

    try:
        import pyqtgraph  # noqa: F401
        from pyqtgraph.Qt import QtWidgets  # noqa: F401
    except ImportError:
        print("Visualizer requires PySide6 + pyqtgraph."
              " Install with: pip install -e .[viz]")
        sys.exit(1)

    classes = _make_qt_classes()
    MainWindow = classes["MainWindow"]

    from pyqtgraph.Qt import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()

    if args.dataset is not None:
        # pre-fill sidebar and kick off a render
        combo = win.sidebar.dataset_combo
        idx = combo.findText(args.dataset)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        win.sidebar._set_dim(args.dim)
        win.sidebar.layout_combo.setCurrentText(args.layout)
        win.sidebar.sample_combo.setCurrentText(args.sample)
        win.sidebar.sample_spin.setValue(args.size)
        win.sidebar.render_requested.emit()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
