"""
NetFM Feature Engineering (GPU-accelerated)

Channel 1 — Structural features (6-dim, per-graph z-scored):
  degree, clustering, PageRank, triangles, k-core, eigenvector centrality.
  * Matrix ops via scipy.sparse (C-level, fast).
  * Iterative ops (PageRank, eigenvector centrality) run on GPU via torch sparse.
  * k-core via NetworkX's linear-time implementation.
Channel 2 — SVD-compressed original features (d-dim):
  torch.svd_lowrank on GPU; zero-pad when F < d; zeros when no features.

Cached to data/processed/.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import networkx as nx
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data import (
    DATA_ROOT,
    DATASET_REGISTRY,
    HELDOUT_DATASETS,
    PRETRAIN_DATASETS,
    NetFMGraph,
    load_dataset,
)


PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
STRUCT_DIM = 6
DEFAULT_SVD_DIM = 256


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Sparse graph construction
# ---------------------------------------------------------------------------

def _build_sparse(edge_index, num_nodes: int) -> csr_matrix:
    """Symmetric unweighted adjacency CSR, no self-loops, duplicates merged."""
    ei = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.asarray(edge_index)
    mask = ei[0] != ei[1]
    src = ei[0, mask]
    dst = ei[1, mask]
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    data = np.ones(len(row), dtype=np.float32)
    A = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    A.data = np.minimum(A.data, 1.0)
    A.eliminate_zeros()
    return A


def _scipy_to_torch_csr(A: csr_matrix, device: torch.device) -> torch.Tensor:
    crow = torch.from_numpy(A.indptr.astype(np.int64))
    col = torch.from_numpy(A.indices.astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    return torch.sparse_csr_tensor(crow, col, val, size=A.shape).to(device)


# ---------------------------------------------------------------------------
# Individual feature kernels
# ---------------------------------------------------------------------------

def _degree(A: csr_matrix) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).flatten().astype(np.float64)


def _triangles_chunked(A: csr_matrix, chunk: int = 2048) -> np.ndarray:
    """
    Per-node triangle count via (A² ⊙ A) row-sum / 2, processed in row chunks.
    Memory bounded by one chunk of A² at a time.
    """
    n = A.shape[0]
    tri = np.zeros(n, dtype=np.float64)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        A_block = A[start:end]
        AA_block = A_block @ A                       # sparse × sparse → sparse
        prod = A_block.multiply(AA_block)            # elementwise on A's sparsity
        tri[start:end] = np.asarray(prod.sum(axis=1)).flatten()
    return tri / 2.0


def _clustering_from_tri(deg: np.ndarray, tri: np.ndarray) -> np.ndarray:
    denom = deg * (deg - 1.0)
    return np.where(denom > 0, 2.0 * tri / denom, 0.0)


def _kcore(A: csr_matrix) -> np.ndarray:
    """K-core numbers via NetworkX's linear O(m) implementation."""
    G = nx.from_scipy_sparse_array(A, edge_attribute=None)
    G.remove_edges_from(nx.selfloop_edges(G))
    core = nx.core_number(G)
    n = A.shape[0]
    return np.array([core.get(i, 0) for i in range(n)], dtype=np.float64)


def _pagerank_gpu(
    A_torch: torch.Tensor,
    deg: torch.Tensor,
    alpha: float = 0.85,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> torch.Tensor:
    """PageRank via power iteration on GPU sparse matvec."""
    n = A_torch.size(0)
    device = A_torch.device
    deg_safe = torch.where(deg > 0, deg, torch.ones_like(deg))
    dangling_mask = (deg == 0)
    pr = torch.full((n,), 1.0 / n, device=device, dtype=torch.float32)
    for _ in range(max_iter):
        x = pr / deg_safe
        # A @ x (sparse matvec); A is symmetric so A == A.T
        new = (A_torch @ x.unsqueeze(1)).squeeze(1)
        dangling = pr[dangling_mask].sum() if dangling_mask.any() else torch.tensor(0.0, device=device)
        pr_new = (1.0 - alpha) / n + alpha * new + alpha * dangling / n
        if torch.norm(pr_new - pr, p=1).item() < n * tol:
            pr = pr_new
            break
        pr = pr_new
    return pr


def _eigenvector_centrality_gpu(
    A: csr_matrix,
    device: torch.device,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> np.ndarray:
    """Leading-eigenvector centrality per connected component on GPU."""
    n = A.shape[0]
    ec = np.zeros(n, dtype=np.float64)
    n_comp, labels = connected_components(A, directed=False)
    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        sub = A[idx][:, idx]
        sub_t = _scipy_to_torch_csr(sub, device)
        ns = sub.shape[0]
        x = torch.ones(ns, device=device, dtype=torch.float32) / np.sqrt(ns)
        for _ in range(max_iter):
            y = (sub_t @ x.unsqueeze(1)).squeeze(1)
            norm = torch.norm(y, p=2)
            if norm.item() < 1e-12:
                break
            y = y / norm
            if torch.norm(y - x, p=1).item() < ns * tol:
                x = y
                break
            x = y
        ec[idx] = x.detach().cpu().numpy()
    # Convention: non-negative
    return np.abs(ec)


# ---------------------------------------------------------------------------
# Structural features (combined pipeline)
# ---------------------------------------------------------------------------

def compute_structural_features(
    graph: NetFMGraph,
    device: torch.device | None = None,
    verbose: bool = True,
) -> np.ndarray:
    device = device or _pick_device()
    n = graph.num_nodes
    t0 = time.time()
    if verbose:
        print(f"  building sparse A ({n} nodes, {graph.edge_index.size(1)} directed edges)...")
    A = _build_sparse(graph.edge_index, n)
    A_torch = _scipy_to_torch_csr(A, device)

    feats = np.zeros((n, STRUCT_DIM), dtype=np.float64)

    # 1. Degree
    t = time.time()
    deg = _degree(A)
    feats[:, 0] = deg
    if verbose:
        print(f"  [1/6] degree ({time.time() - t:.2f}s)")

    # 4. Triangles  (used by clustering too, so compute first)
    t = time.time()
    tri = _triangles_chunked(A)
    feats[:, 3] = tri
    if verbose:
        print(f"  [4/6] triangles ({time.time() - t:.2f}s)")

    # 2. Clustering (from deg + tri)
    t = time.time()
    feats[:, 1] = _clustering_from_tri(deg, tri)
    if verbose:
        print(f"  [2/6] clustering ({time.time() - t:.2f}s)")

    # 3. PageRank (GPU)
    t = time.time()
    deg_t = torch.from_numpy(deg.astype(np.float32)).to(device)
    pr = _pagerank_gpu(A_torch, deg_t)
    feats[:, 2] = pr.detach().cpu().numpy()
    if verbose:
        print(f"  [3/6] pagerank on {device} ({time.time() - t:.2f}s)")

    # 5. K-core (CPU, linear-time networkx)
    t = time.time()
    feats[:, 4] = _kcore(A)
    if verbose:
        print(f"  [5/6] k-core ({time.time() - t:.2f}s)")

    # 6. Eigenvector centrality (GPU, per component)
    t = time.time()
    feats[:, 5] = _eigenvector_centrality_gpu(A, device)
    if verbose:
        print(f"  [6/6] eigenvector centrality on {device} ({time.time() - t:.2f}s)")

    # Per-graph z-score; zero-variance features → 0
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std_safe = np.where(std > 0, std, 1.0)
    feats = (feats - mean) / std_safe
    feats[:, std == 0] = 0.0

    if verbose:
        print(f"  structural features: {feats.shape} ({time.time() - t0:.2f}s total)")
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# SVD-compressed original features (GPU)
# ---------------------------------------------------------------------------

def compute_svd_features(
    graph: NetFMGraph,
    d: int = DEFAULT_SVD_DIM,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    SVD-compress original_features to [num_nodes, d].
    X ≈ U[:, :k] × Σ[:k]  where k = min(d, F); zero-pad if F < d.
    Runs on GPU via torch.svd_lowrank when CUDA is available.
    """
    n = graph.num_nodes
    if graph.original_features is None:
        return np.zeros((n, d), dtype=np.float32)
    device = device or _pick_device()

    X = graph.original_features.detach().to(device=device, dtype=torch.float32)
    if X.numel() == 0 or X.size(1) == 0:
        return np.zeros((n, d), dtype=np.float32)

    F = X.size(1)
    k = min(d, F, min(X.shape))
    # svd_lowrank uses randomized SVD; niter oversampling helps accuracy
    U, S, _ = torch.svd_lowrank(X, q=k, niter=4)
    compressed = U * S.unsqueeze(0)  # [N, k]
    out = torch.zeros((n, d), dtype=torch.float32, device=device)
    out[:, :k] = compressed
    return out.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _struct_path(name: str) -> str:
    return os.path.join(PROCESSED_DIR, f"{name}_structural.npz")


def _svd_path(name: str, d: int) -> str:
    return os.path.join(PROCESSED_DIR, f"{name}_svd_d{d}.npz")


def compute_or_load_features(
    graph: NetFMGraph,
    d: int = DEFAULT_SVD_DIM,
    force: bool = False,
    device: torch.device | None = None,
) -> NetFMGraph:
    """Populate graph.structural_features and graph.svd_features, caching to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    device = device or _pick_device()

    sp = _struct_path(graph.name)
    vp = _svd_path(graph.name, d)

    if not force and os.path.exists(sp):
        struct = np.load(sp)["arr"]
    else:
        print(f"  computing structural features for {graph.name}...")
        struct = compute_structural_features(graph, device=device)
        np.savez_compressed(sp, arr=struct)

    if not force and os.path.exists(vp):
        svd = np.load(vp)["arr"]
    else:
        print(f"  computing SVD features (d={d}) for {graph.name}...")
        svd = compute_svd_features(graph, d=d, device=device)
        np.savez_compressed(vp, arr=svd)

    graph.structural_features = torch.from_numpy(struct)
    graph.svd_features = torch.from_numpy(svd)
    return graph


def load_features(name: str, d: int = DEFAULT_SVD_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Load cached features by dataset name. Raises if not precomputed."""
    sp = _struct_path(name)
    vp = _svd_path(name, d)
    if not os.path.exists(sp) or not os.path.exists(vp):
        raise FileNotFoundError(
            f"features for {name} not cached (expected {sp} and {vp}); "
            f"run `python -m src.features {name}` first."
        )
    return np.load(sp)["arr"], np.load(vp)["arr"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    force = False
    d = DEFAULT_SVD_DIM
    device_str: str | None = None
    names: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--force":
            force = True
        elif a == "--d":
            d = int(args[i + 1]); i += 1
        elif a == "--device":
            device_str = args[i + 1]; i += 1
        elif a == "--pretrain":
            names.extend(PRETRAIN_DATASETS)
        elif a == "--heldout":
            names.extend(HELDOUT_DATASETS)
        else:
            names.append(a)
        i += 1
    if not names:
        names = list(DATASET_REGISTRY.keys())

    device = torch.device(device_str) if device_str else _pick_device()
    print(f"computing features for {len(names)} dataset(s), d={d}, force={force}, device={device}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")
    print(f"cache dir: {PROCESSED_DIR}")
    print("-" * 60)

    for name in names:
        print(f"\n[{name}]")
        g = load_dataset(name)
        compute_or_load_features(g, d=d, force=force, device=device)
        print(
            f"  ready: struct={list(g.structural_features.shape)} "
            f"svd={list(g.svd_features.shape)}"
        )
